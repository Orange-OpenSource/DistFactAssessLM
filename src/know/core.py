# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from collections import abc
import copy
from dataclasses import dataclass, field
import itertools
from math import prod
from typing import Any, Callable, Iterable

import numpy as np
import io

import torch
from tqdm import tqdm
from glob_core import Mongoable, SameType, cache_return
from kb.core import Date, Entity, Relation, Triple, TripleComp
from kb.wikidata import TempWikidata
from lm.core import CredibilityFunction, LanguageModel
from utils.general import logsumexp_by_group
from verb.core import Template, correct_determiner_errors
import plotext as plt

class HyperTunable:
    def parameters(self) -> list[str]:
        """Return the path of each hyperparameter. Example of a path: "hyperparams.learning_rate"
        """
        raise NotImplementedError
    
    def hyper_clone(self, new_hyperparams: dict[str, Any]) -> HyperTunable:
        """Returns a fresh copy of this object with new hyperparameters.

        Args:
            new_hyperparams (dict[str, Any]): Dictionary of hyperparameter paths and their new value
        """
        raise NotImplementedError
    
    def gridsearch(self, hyperparams2values: dict[str, list], eval_fn : Callable[[HyperTunable], float]) -> dict:
        lengths = [len(x) for x in hyperparams2values.values()]
        total_tests = prod(lengths)
        def iterate_hyperparams():
            all_configs = itertools.product(*[range(l) for l in lengths])
            for config in all_configs:
                hyperparams = {}
                for i, (param, values) in enumerate(hyperparams2values.items()):
                    hyperparams[param] = values[config[i]]
                yield hyperparams
        scores = []
        for hyperparams in tqdm(iterate_hyperparams(), "Hypertuning", total_tests):
            score = eval_fn(self.hyper_clone(hyperparams))
            scores.append({tuple(hyperparams.items()) : score})
        return scores

class KnowMeasure(Mongoable, metaclass=ABCMeta):
    def measure_temp(self, lm : LanguageModel, temp : Template) -> KnowMeasureReturn:
        raise NotImplementedError

    def measure_fact(self, lm : LanguageModel, fact : Triple, empty_templates : list[Template] | None = None) -> KnowMeasureReturn:
        if empty_templates is None:
            raise NotImplementedError
        
        know_measures = [self.measure_temp(lm, temp.copy().inject(fact.subject, fact.object)) for temp in empty_templates]
        results = [x.result for x in know_measures]
        does_knows = [x.does_know for x in know_measures]
        kwargs = dict(know_measure_info = self.infos(), lm_name=lm.lm_name, temp=[measure.temp for measure in know_measures], 
                                         result=np.array(results, dtype=np.float32).mean(0),
                                         does_know=np.array(does_knows, dtype=np.float32).mean(0),
                                         additional_data={
                                             'individual_results' : results,
                                             'individual_does_know' : does_knows
                                         })
        if isinstance(know_measures[0], DistKnowMeasureReturn):
            kwargs['neighbors'] = [measure.neighbors for measure in know_measures]
            kwargs['cred_true'] = [measure.cred_true for measure in know_measures]
            kwargs['cred_dist'] = [measure.cred_dist for measure in know_measures]
            kwargs['dist_temp'] = [measure.dist_temp for measure in know_measures]
            know_measure = DistKnowMeasureReturn(**kwargs)
        else:
            know_measure = KnowMeasureReturn(**kwargs)
        return know_measure
    
    @property
    def threshold(self) -> float:
        return 0



@dataclass
class KnowMeasureReturn(Mongoable):
    know_measure_info : dict
    lm_name : str
    temp : Template 
    result : tuple[float]
    does_know : bool = None
    tag : str = None
    additional_data : dict = field(default_factory=dict)

@dataclass
class DistKnowMeasure(KnowMeasure):
    
    agg : DistKnowAggMetric
    obj_dist_finder : ObjectDistractorFinder
    cred_func : CredibilityFunction
    compute_cred_on_object : bool = True
    use_aliases : bool = True
    use_aliases_subject : bool = False

    def measure_from_credibility(self, cred_true : float, cred_dist : tuple[float]) -> float:
        res = self.agg.compute(cred_true, cred_dist)
        return res

    def measure_from_know_return(self, know_return : KnowMeasureReturn):
        know_return : DistKnowMeasureReturn = copy.copy(know_return)
        know_return.result = self.agg.compute(know_return.cred_true, know_return.cred_dist)
        know_return.know_measure_info = self.infos()
        return know_return


    def infos(self) -> dict:
        return {
            'cls' : self.class_id(),
            'agg' : self.agg.infos(),
            'dist_finder' : self.obj_dist_finder.infos(),
            'cred_func' : self.cred_func.class_id(),
            'compute_cred_on' : str(self.compute_cred_on_object)
        }

    def _compute_creds_agg(self, lm: LanguageModel, temp : Template, distractors : list[Entity]):
        objects_all_names = self.obj_dist_finder.kb.get_all_names_of_entity([temp.object] + distractors)
        subject_all_names = self.obj_dist_finder.kb.get_all_names_of_entity([temp.triple.subject])[0]
        if len(subject_all_names) == 0:
            subject_all_names = [temp.triple.subject.label]
        if not self.use_aliases_subject:
            subject_all_names = subject_all_names[:1]

        if not self.use_aliases:
            objects_all_names = [x[:1] for x in objects_all_names]
        objects_num_names = torch.tensor([len(x)*len(subject_all_names) for x in objects_all_names], dtype=torch.long)
        objects_all_names_flatten = list(itertools.chain(*objects_all_names))
        del objects_all_names
        all_templates = [temp.copy() for _ in range(len(objects_all_names_flatten)*len(subject_all_names))]
        distractors_and_true_with_repetition = np.repeat([temp.object] + distractors, objects_num_names)
        for t, dist, dist_label in zip(all_templates, distractors_and_true_with_repetition, objects_all_names_flatten):
            for subject_name in subject_all_names:
                t.delete(TripleComp.SUBJECT)
                t.delete(TripleComp.OBJECT)
                t.inject(sub=Entity(temp.subject.id, subject_name), obj=Entity(dist.id, dist_label))
            # correct_determiner_errors(t)
        with torch.no_grad():
            compute_on = None if not self.compute_cred_on_object else [' ' + x + lm.hf_tokenizer.eos_token for x in objects_all_names_flatten]
            creds = lm.credibility_text([x.text + lm.hf_tokenizer.eos_token for x in all_templates], self.cred_func, batch_size=64, compute_on=compute_on) # Reduce batch_size if out-of-memory
            
        creds = creds.cpu()
        # Aggregate all names of entities
        creds_agg = logsumexp_by_group(creds, objects_num_names).float().numpy()

        return creds_agg, all_templates[objects_num_names[0]:]

    def measure_temp(self, lm: LanguageModel, temp: Template) -> DistKnowMeasureReturn:
        assert temp.full, "The input template needs to be full (no blanks). Template found : %s" % temp
        distractors = list(self.obj_dist_finder.find_from_template(temp, self.agg.max_n))
        
        creds_agg, all_templates = self._compute_creds_agg(lm, temp, distractors)
        cred_true, cred_dist = creds_agg[0], creds_agg[1:].tolist()
        res = self.agg.compute(cred_true, cred_dist)
        return DistKnowMeasureReturn(self.infos(), lm.lm_name, temp, distractors, res, float(cred_true), cred_dist, all_templates)


class CredKnowMeasure(KnowMeasure):
    def __init__(self, cred_func : CredibilityFunction, compute_on : TripleComp = None) -> None:
        super().__init__()
        self.cred_func = cred_func
        self.compute_on = compute_on

    def infos(self) -> dict:
        return {
            'cred_func' : self.cred_func.class_id(),
            'compute_on' : self.compute_on
        }
    
    def measure_temp(self, lm: LanguageModel, temp: Template) -> float:
        co = None
        if self.compute_on is not None:
            assert temp.ends_with == self.compute_on
            temp = temp.copy()
            co = [' ' + temp.triple.get_comp(self.compute_on).label]
        
        res = lm.credibility_text([temp.text], cred_func=self.cred_func, compute_on=co)[0].item()
        return KnowMeasureReturn(self.infos(), lm.lm_name, temp, res)
    
@dataclass
class DistKnowMeasureReturn(Mongoable):
    know_measure_info : dict
    lm_name : str
    temp : Template 
    neighbors : list[Entity] 
    result : tuple[float]
    cred_true : tuple[float]
    cred_dist : tuple[float]
    dist_temp: list[Template]
    does_know: bool = None
    tag : str = None
    additional_data : dict = field(default_factory=dict)


    def plot(self):
        plt.clear_figure()
        plt.scatter(self.know_measure.metric.n, self.result)
        plt.xlabel('n') 
        plt.ylabel("%s@n" % self.know_measure.metric.__class__.__name__)
        plt.ylim(0,1)
        # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.plot_size(80, 30)
        plt.show()

    def __repr__(self) -> str:
        verb_text = self.temp.text
        verb_text = verb_text.replace(self.temp.object.label, "\033[4m" + self.temp.object.label + "\033[0m")
        verb_text = verb_text.replace(self.temp.subject.label, "\033[4m" + self.temp.subject.label + "\033[0m")
        
        res = """Object: %s
Verbalization: %s
LM: %s
Knowledge base: %s
Neighbors: %s
        """ % (self.temp.object, verb_text, self.lm_name, self.know_measure_info, self.neighbors)
        return res
    


class ObjectDistractorFinder(Mongoable, metaclass=ABCMeta):
    def __init__(self, kb : TempWikidata) -> None:
        self.kb = kb

    def _to_json(self) -> dict | list:
        return dict(kb=self.kb)

    def infos(self) -> dict:
        return dict(cls=self.class_id(), kb_cls=self.kb.class_id(), kb_time=self.kb.time)

    @abstractmethod
    def find(self, triple : Triple, t : Date, n : int) -> Iterable[Entity]:
        pass

    def find_from_template(self, template : Template, n : int) -> Iterable[Entity]:
        return self.find(template.triple, template.time, n)

    def load(self) -> None:
        pass

class DistKnowAggMetric(Mongoable, metaclass=ABCMeta):
    def __init__(self, n : int | tuple[int]) -> None:
        super().__init__()
        if not isinstance(n, abc.Iterable):
            n = (n,)
        self.n = tuple(sorted(n))
        self.max_n = int(max(n))

    def _to_json(self) -> dict | list:
        return dict(n=self.n)

    def infos(self) -> dict:
        return dict(cls=self.class_id(), n=self.n)

    @abstractmethod
    def compute(self, know_correct : float, know_dist : list[float]) -> tuple[float]:
        pass


class StrictMetric(DistKnowAggMetric):
    def __init__(self, n : int | tuple[int]) -> None:
        super().__init__(n)
        self._n_cp = list(self.n)[::-1]

    def compute(self, know_correct: float, know_dist: list[float]) -> tuple[float]:
        n_cp = list(self._n_cp)
        if len(n_cp) == 0:
            return tuple()
        to_check = n_cp.pop()-1
        success = True
        l = []
        for i, k in enumerate(know_dist):
            if to_check is None:
                if len(n_cp):
                    to_check = n_cp.pop()-1
                else:
                    break
            if success and k >= know_correct:
                success = False
            if i == to_check:
                l.append(float(success))
                to_check = None
            
        l += [None]*(len(n_cp)+int(to_check is not None))
        return l

        
class MeanMetric(DistKnowAggMetric):
    def __init__(self, n : int | tuple[int]) -> None:
        super().__init__(n)
        self._n_min_1 = np.array(self.n) - 1
        self._max = self._n_min_1.max()
    
    def compute(self, know_correct: float, know_dist: list[float]) -> float | tuple[float] | None:
        l = np.array(know_dist)[:self._max+1]
        success = know_correct > l
        res = (success.cumsum() / np.arange(1, success.shape[0]+1))[self._n_min_1[self._n_min_1 < success.shape[0]]]
        res = res.tolist()
        res += [None]*(len(self.n) - len(res))
        return res
    

class LogMeanMetric(DistKnowAggMetric):
    def __init__(self, n : int | tuple[int]) -> None:
        super().__init__(n)
        self._n_min_1 = np.array(self.n) - 1
        self._max = self._n_min_1.max()
    
    def compute(self, know_correct: float, know_dist: list[float]) -> float | tuple[float] | None:
        l = np.array(know_dist)[:self._max+1]
        errors = know_correct <= np.array(l)
        res = (1 - (np.log(errors.cumsum() + 1) / np.log(np.arange(1, errors.shape[0]+1)+1)))[self._n_min_1[self._n_min_1 < errors.shape[0]]]
        res = res.tolist()
        res += [None]*(len(self.n) - len(res))
        return res
    

class PosMetric(DistKnowAggMetric):
    def __init__(self, n : int | tuple[int]) -> None:
        super().__init__(n)
        self._n_min_1 = np.array(self.n) - 1
        self._max = self._n_min_1.max()
    
    def compute(self, know_correct: float, know_dist: list[float]) -> float | tuple[float] | None:
        l = np.array(know_dist)[:self._max+1]
        success = (know_correct > np.array(l)).astype(np.float32)
        res = success[self._n_min_1[self._n_min_1 < success.shape[0]]]
        res = res.tolist()
        res += [None]*(len(self.n) - len(res))
        return res
    
class WeightMetric(DistKnowAggMetric):
    def __init__(self, n : int | tuple[int]) -> None:
        super().__init__(n)
        self._n_min_1 = np.array(self.n) - 1
        self._max = self._n_min_1.max()
    
    def compute(self, know_correct: float, know_dist: list[float]) -> float | tuple[float] | None:
        l = np.concatenate([[know_correct], know_dist[:self._max+1]])
        res = know_correct - np.logaddexp.accumulate(l)[1:]
        res = res[self._n_min_1[self._n_min_1 < res.shape[0]]]
        res = res.tolist()
        res += [None]*(len(self.n) - len(res))
        return res
    


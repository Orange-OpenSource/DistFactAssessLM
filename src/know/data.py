# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable
import hashlib
from itertools import groupby
import pickle
import random
import shutil
from typing import Hashable

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob_core import Blueprint, SaveableNotFoundError, JSONable
from globals import STORAGE_FOLDER
from kb.core import Date, Entity, InfDate, Interval, Literal, TimedTriple, TimedTripleQuery, Triple, TripleQuery
from kb.wikidata import TempWikidata, WikidataPopularity, WikidataPrepStage, wikidata_is_valid_at
from know.core import DistKnowMeasure
from lm.core import LanguageModel
from utils.general import dotdict, load_json
from utils.beamsearch import select_mask_list
from verb.core import TemplateVerbalizer, VerbalizeConfig
from verb.options import DateIndicator, Tense
import os.path as osp


def rep_sample(df : pd.DataFrame, col, n, random_seed=421):
    if n > len(df):
        return df
    a = df.groupby(col, observed=True)[col].count().astype(float)
    b = (a*len(a))**-1
    b = b.to_frame()
    b.columns = ['p']
    b = b.reset_index()
    df2 = df.merge(b, on=col)
    df2.set_index(df.index, inplace=True)
    df2 = df2.sample(n, weights=df2['p'], random_state=random_seed)
    return df.loc[df2.index]

def list_triples_to_dataframe(list_triples : list[TimedTriple]) -> pd.DataFrame:
    raise NotImplementedError

def triple_has_temporal_neighbors(wd : TempWikidata, triple : Triple):
    query = TimedTripleQuery(subject=triple.subject, relation=triple.relation)
    triples : list[TimedTriple] = list(wd.find(query))
    triples.remove(triple)
    for t in triples:
        if t.valid_between.start is not None and t.valid_between.end is not None:
            return True
    return False

def batched(it : Iterable, batch_size : int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []

class WikidataSample(JSONable):
    def __init__(self, time : str, stage : WikidataPrepStage, triples : tuple[TimedTriple], popularity : list[float], subject_popularity='balanced', valid_at='present', random_seed=421, use_temporal_triples=False) -> None:
        super().__init__()
        self.n_triples = len(triples)
        self.subject_popularity = subject_popularity
        self.triples = triples
        self.time = time
        self.stage = stage
        self.popularity = popularity
        self.random_seed = random_seed
        self.valid_at = valid_at
        self.use_temporal_triples = use_temporal_triples

    @property
    def config(self) -> dict:
        return dict(n_triples=self.n_triples, subject_popularity=self.subject_popularity, time=self.time, stage=self.stage, valid_at=self.valid_at)

    @staticmethod
    def _build_from_config_temporal_triples(n_triples : int, wd: TempWikidata, wikipop : WikidataPopularity):
        res_triples = []
        for batch in batched(tqdm(wikipop.iterate_subjects(pop_order="descending"), total=wikipop.number_of_subjects()), 1024):
            entities = [x[0] for x in batch]
            triples = list(wd.find(TimedTripleQuery(subject=entities)))
            for _, group in groupby(triples, key=lambda x : (x.subject, x.relation)):
                tr_temp = []
                for tr in group:
                    tr : TimedTriple
                    st,ed = tr.valid_between.start, tr.valid_between.end
                    if not tr.valid_between.is_point() and ((st is not None and ed is not None) or (st is not None and ed is None)):
                        tr_temp.append(tr)
                if len(tr_temp) < 2:
                    continue
                tr = random.choice(tr_temp)
                res_triples.append(tr)
                if len(res_triples) == n_triples:
                    return res_triples
        return res_triples


    @staticmethod
    def build_from_config(n_triples : int, subject_popularity : str, time : str, stage : WikidataPrepStage, valid_at : Date | str, random_seed=421, force_rebuild=False, use_temporal_triples=False):
        assert subject_popularity in ('none', 'balanced')
        # if subject_popularity == 'balanced':
        #     use_temporal_triples = False # temporal_triples not supported for balanced popularity
        if not force_rebuild:
            config = dotdict(dict(n_triples=n_triples, subject_popularity=subject_popularity, time=time, stage=stage, random_seed=random_seed))
            try:
                sample = WikidataSample.from_id(WikidataSample._identifier(config))
                print('WikidataSample retrieved from cache!')
                return sample
            except SaveableNotFoundError:
                pass
        wikidata = TempWikidata(time, stage)
        wikipop = WikidataPopularity(time)
        if use_temporal_triples:
            random.seed(random_seed)
            triples = WikidataSample._build_from_config_temporal_triples(n_triples, wikidata, wikipop)
            return WikidataSample(time, stage, triples, None, subject_popularity, random_seed, use_temporal_triples)
        if subject_popularity == 'balanced':
            subject_pop = pd.DataFrame(tqdm(wikipop.iterate_subjects(pop_order="descending"), total=wikipop.number_of_subjects()), columns=['ent', 'pop'])
            subject_pop['pop_cut'] = pd.cut(subject_pop['pop'], bins=10)
            sample = rep_sample(subject_pop, 'pop_cut', n_triples, random_seed).index
            entities = subject_pop.loc[sample].set_index('ent')
        elif subject_popularity == 'none':
            subject_pop = []
            for ent, p in wikipop.iterate_subjects(pop_order="descending"):
                if len(subject_pop) == n_triples:
                    break
                subject_pop.append((ent, p))
            subject_pop = pd.DataFrame(subject_pop, columns=['ent', 'pop'])
            entities = subject_pop.set_index('ent')
        # elif subject_popularity == 'balanced_sub+obj':
        #     pass
        triples = [t for t in wikidata.find(TimedTripleQuery(subject=entities.index)) if not isinstance(t.object, Literal)]
        triples_idx = pd.Series((x.subject.id for x in triples), name='sub').to_frame().groupby('sub').sample(1, random_state=random_seed).index.to_list()
        random.seed(random_seed)
        triples_idx = np.concatenate([triples_idx[:n_triples], 
                                     random.sample(triples_idx, max(0, n_triples - len(triples_idx)))]).astype(int)
        triples = [triples[idx] for idx in triples_idx]
        popularity = entities.loc[[t.subject for t in triples], 'pop'].tolist()
        return WikidataSample(time, stage, triples, popularity, subject_popularity, random_seed, use_temporal_triples)

    def _to_json(self) -> dict | list:
        return {
            'time' : self.time,
            'stage' : self.stage,
            'subject_popularity' : self.subject_popularity,
            'triples' : self.triples,
            "popularity" : self.popularity,
            "random_seed" : self.random_seed,
            'valid_at' : self.valid_at,
            'use_temporal_triples' : self.use_temporal_triples
        }

    # @classmethod
    # def _from_json(cls: type[SameType], d: dict | list) -> SameType:
    #     d['stage'] = WikidataPrepStage(d['stage'])
    #     d['triples'] = [TimedTriple._from_json(x) for x in d['triples']]
    #     return WikidataSample(**d)

    def _identifier(self) -> Hashable:
        return self.n_triples, self.subject_popularity, self.stage, self.time, self.random_seed


class AlternativeQualityExperiment(Blueprint):
    SAVE_FOLDER = osp.join(STORAGE_FOLDER, 'alternative_quality_experiment')
    def __init__(self, dist_know_measure : DistKnowMeasure, lm : LanguageModel, triple_set : WikidataSample, verbalizer : TemplateVerbalizer) -> None:
        self.dist_know_measure = dist_know_measure
        self.lm = lm
        self.triple_set = triple_set
        self.verbalizer = verbalizer
        self.present = self.dist_know_measure.obj_dist_finder.kb.time_date
    
    @property
    def folder_path(self) -> str:
        folder = hashlib.sha256(pickle.dumps(self._identifier()))
        return osp.join(AlternativeQualityExperiment.SAVE_FOLDER, folder)

    def eval_one_triple(self, triple : TimedTriple):
        st, et = triple.valid_between.start, triple.valid_between.end
        validity = wikidata_is_valid_at(st, et, self.present, self.present)
        if validity == "valid":
            templates = self.verbalizer.verbalize(triple)
        elif st is not None and et is not None:
            t = Date(st.id + (et.id - st.id) / 2).change_level(st.level)
            tense = Tense.PAST if t < self.present else Tense.FUTURE
            indic = DateIndicator(t)
            templates = self.verbalizer.verbalize(triple, config=VerbalizeConfig(verb_tense=tense, temporal_indicator=indic))
        
        res = [self.dist_know_measure.measure_temp(self.lm, temp) for temp in templates]
        return res
    
    def _identifier(self) -> Hashable:
        d = {
            'lm' : self.lm.lm_name,
            'tok' : self.lm.tok_name,
            'time' : self.dist_know_measure.obj_dist_finder.kb.time,
            'triple_set_config' : self.triple_set.config,
        }
        return d

    def load_progress(self) -> None:
        config_file = osp.join(self.folder_path, "config.json")
        eval_file = osp.join(self.folder_path, "eval.jsonl")
        if not osp.exists(config_file) or not osp.exists(eval_file):
            return
        config = load_json(config_file)
        triplet_set = config['triple_set_config']
        triplet_set['stage'] = WikidataPrepStage(triplet_set['stage'])
        triplet_set = WikidataSample.build_from_config(**triplet_set)

    @abstractmethod
    def _build(self) -> None:
        self.load_progress()
    
    @abstractmethod
    def _destroy(self) -> None:
        shutil.rmtree(self.folder_path, ignore_errors=True)

    @abstractmethod
    def built(self) -> bool:
        """Was this object built ?

        Returns:
            bool
        """
        return osp.exists(osp.join(self.folder_path, "finished"))


def collect_triples(wd : TempWikidata, valid_at=False, presence_of_temporal_distractors=False):
    triples = []
    sub_rel2triple = defaultdict(list)
    wiki_pop = WikidataPopularity(wd.time, wd.mongodb_url)
    wiki_pop.load()
    for tr in wd.find(TimedTripleQuery(valid_at=(valid_at if not presence_of_temporal_distractors else None))):
        tr : TimedTriple
        if type(tr.object) is not Entity:
            continue
        if presence_of_temporal_distractors:
            sub_rel2triple[(tr.subject.id, tr.relation.id)].append(tr)
        else:
            triples.append(tr)
    if presence_of_temporal_distractors:
        triples = []
        for _,v in sub_rel2triple.items():
            present_triple = None
            idx_present = None
            for i, tr in enumerate(v):
                if wikidata_is_valid_at(tr.valid_between.start, tr.valid_between.end, valid_at, present=wd.time_date) == 'valid':
                    present_triple = tr
                    idx_present = i
                    break
            if present_triple is None:
                continue
            
            # Removing triples with no start or no end
            potential_distractor = [tr for (i,tr) in enumerate(v) if i != idx_present 
                                    and tr.valid_between.start is not None
                                    and tr.valid_between.end is not None and
                                    wikidata_is_valid_at(tr.valid_between.start, tr.valid_between.end, valid_at, present=wd.time_date) == 'invalid']
            if len(potential_distractor):
                triples.append(present_triple)
    return triples

def sample_balanced_facts(wd : TempWikidata, n : int, valid_at=None, presence_of_temporal_distractors=False) -> pd.DataFrame:
    wiki_pop = WikidataPopularity(wd.time, wd.mongodb_url)
    wiki_pop.load()
    triples = collect_triples(wd, valid_at, presence_of_temporal_distractors)
    
    subject_popularity = np.array(wiki_pop.get_popularity((x.subject for x in triples)), dtype=np.float32)
    object_popularity = np.array(wiki_pop.get_popularity((x.object for x in triples)), dtype=np.float32)

    popularity = subject_popularity + object_popularity

    # Remove NaN values
    mask = ~np.isnan(popularity)
    popularity = popularity[mask]
    triples = select_mask_list(triples, mask)

    # Inject label + description in entities.
    wd.inject_info(triples)
    df = pd.DataFrame(data={
        'triples' : triples,
        'popularity' : popularity
    })
    df['popularity_bins'] = pd.cut(popularity, bins=10)
    df = rep_sample(df, 'popularity_bins', n)
    df.drop(columns="popularity_bins", inplace=True)
    return df

# Example for sample_balanced_facts function:
# wd = TempWikidata("20210104", WikidataPrepStage.PREPROCESSED)
# df = sample_balanced_facts(wd, 5000, wd.time_date)
# df.to_pickle(osp.join(STORAGE_FOLDER, 'facts_balanced_5000.pkl'))
# print(df.head())
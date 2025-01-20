# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from __future__ import annotations
from collections import Counter, defaultdict
import itertools
import os
import pickle
import random
import shutil
import time
from typing import Any, Hashable, Iterable, Type

import numpy as np
import scipy
import tqdm
from lm.core import LanguageModel
from utils.beamsearch import MultiChoicesLimiter, TokenLimiter, TokenLimitersCombinator, beam_search
from utils.general import TimeItContextManager, dump_json, is_subseq, load_json, read_list, sample_from_list_of_collections, save_list, topk_with_indices, uniquifier
from glob_core import Blueprint, Mongoable, SameType
from globals import STORAGE_FOLDER
from kb.core import Date, Entity, Literal, TimedTriple, TimedTripleQuery, Triple, TripleComp, TripleQuery
from kb.wikidata import INSTANCE_OF_RELATION, SUBCLASS_OF_RELATION, TempWikidata, WikidataPrepStage, WikidataTypeSearchEngine, wikidata_is_valid_at
from know.core import ObjectDistractorFinder
import os.path as osp
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib

from utils.general import date_is_correct
from verb.core import Template


class RandomDistractorFinder(ObjectDistractorFinder, Blueprint):
    def __init__(self, kb: TempWikidata, pure=False) -> None:
        super().__init__(kb)
        self.wtse : WikidataTypeSearchEngine = WikidataTypeSearchEngine(self.kb)
        self.kb_raw = TempWikidata(kb.time, WikidataPrepStage.ALMOST_RAW)
        self.pure = pure

    def load(self) -> None:
        self.wtse.load()

    def find(self, triple: Triple, t : Date, n: int) -> Iterable[Entity]:
        all_labels = self.kb_raw.get_all_names_of_entity([triple.object])[0]
        to_filter = set(x.object for x in self.kb.find(TimedTripleQuery(triple.subject, triple.relation, None, valid_at=t)))
        to_filter.update(y for x in self.kb.find_from_label(all_labels).values() for y in x if y != triple.object)
        n += len(to_filter)
        object_types = self.wtse.get_all_types_of_entity(triple.object)
        if len(object_types) == 0:
            # If no types are found just sample random entities of any type
            object_types = self.wtse.all_types
        if not self.pure:
            sets = [self.wtse.get_all_entities_of_type(ty) for ty in object_types]
            size = sum(len(x) for x in sets)
            distractors = sample_from_list_of_collections(sets, min(n, size))
        else:
            distractors = self.kb.sample_subjects(n, contains_label=True)
        
        return uniquifier(x for x in distractors if x not in to_filter)[:n]
        

    def _build(self) -> None:
        self.wtse._build()

    def _destroy(self) -> None:
        self.wtse._destroy()

    def built(self) -> bool:
        return self.wtse.built()

    

def _identity(x):
    return x

class Distractor(Entity):
    def __init__(self, ent : Entity, is_temp : bool = False) -> None:
        super().__init__(ent.id, ent.label, ent.description)
        self.is_temp = is_temp
    
    def _to_json(self) -> dict | list:
        d = dict(ent=Entity(self.id, self.label, self.description))
        d['is_temp'] = self.is_temp
        return d


class SimilarityDistractorFinder(ObjectDistractorFinder, Blueprint, Mongoable):
    existing_instances = {}
    _LOAD_ATTRIBUTES = ['_index', '_ent_ids', '_features_sparse', "_vocab", "_entities_in_index_name2idx", "_type2ents", "_ent2types", "_type2len"]
    def __init__(self, time : str, use_reverse_features: bool = False, eps : float = 0, 
                 use_temporal_distractors=False, dist_in_corr_removal=False) -> None:
        assert date_is_correct(time), "time must be in the following format : YYYYMMDD, found : %s" % time
        super().__init__(TempWikidata(time, WikidataPrepStage.ALMOST_RAW))
        self.kb_retrieve_subject = TempWikidata(time, WikidataPrepStage.PREPROCESSED)
        self.kb_raw = TempWikidata(time, WikidataPrepStage.ALMOST_RAW)
        self.kb : TempWikidata
        for kb in self.kb_retrieve_subject, self.kb:
            assert kb.built(), "%s needs to be built first!" % kb
        self.use_reverse_features = use_reverse_features
        self._save_folder = osp.join(STORAGE_FOLDER, "alternative_finders", "sim_alt_finder__%s__rev_feat=%s" % (time, use_reverse_features))
        self._index = None
        self._ent_ids = None
        self._features_sparse = None
        self._vocab = None
        self._entities_in_index_name2idx = None
        self._type2ents = None
        self._ent2types = None
        self.eps = eps
        self.use_temporal_distractors = use_temporal_distractors
        self.dist_in_corr_removal = dist_in_corr_removal
        self.time = time

    def infos(self) -> dict:
        d = super().infos()
        d.update(use_reverse_features=self.use_reverse_features, eps=self.eps, 
                 use_temporal_distractors=self.use_temporal_distractors, dist_in_corr_removal=self.dist_in_corr_removal)
        return d

    def _build(self) -> None:
        # assert sys.version + '.' + sys.version_info >= "3.7", "%s build process is only available for >=3.7 python versions" % self.__class__.__name__

        # Code scrapped from old codebase : src/_old_codebase/build/find_neighbors/scripts/index_tfidf_entities.py

        shutil.rmtree(self._save_folder, ignore_errors=True)
        os.makedirs(self._save_folder, exist_ok=False)


        print('Collecting entity bag-of-words (relation-object couples, relations, and objects)...')
        print('From %s' % self.kb)
        ent2bow = defaultdict(set)

        class Subjects:
            def __init__(self, it : Iterable, progress_bar : tqdm.tqdm) -> None:
                self.it = it
                self.progress_bar = progress_bar
            def __iter__(self):
                for x in self.it:
                    yield x
                    self.progress_bar.update()
        
        progress = tqdm.tqdm(total=self.kb_retrieve_subject.number_of_subjects(),)
        subjects = Subjects(self.kb_retrieve_subject.iterate_subjects(spawn_process=True), progress)

        type2ents = defaultdict(list)

        for triple in self.kb.find(TripleQuery(subject=subjects)):
            triple : Triple
            if isinstance(triple.object, Literal):
                continue
            ent2bow[triple.subject.id].update(str(x) for x in (triple.subject.id, 
                                                               triple.relation.id, triple.object.id, 
                                                               "%s_%s" % (triple.relation.id, triple.object.id)))
            if self.use_reverse_features:
                ent2bow[triple.object.id].update(str(x) for x in (triple.subject.id, 
                                                               triple.relation.id, triple.object.id, 
                                                               "%s_%s" % (triple.subject.id, triple.relation.id)))
            if triple.relation in [INSTANCE_OF_RELATION, SUBCLASS_OF_RELATION]:
                type2ents[triple.object.id].append(triple.subject.id)
        tfidf = TfidfVectorizer(analyzer=_identity, norm=None)

        ent_all_ids, ent_all_vectors = tuple(zip(*ent2bow.items()))

        print('Building TF-IDF vectors...')
        features_sparses = tfidf.fit_transform(ent_all_vectors)
        print('TF-IDF Matrix Shape : %s' % str(features_sparses.shape))
        dump_json(osp.join(self._save_folder, "tfidf_vectorizer.json"), tfidf.vocabulary_)
        

        save_list(ent_all_ids, osp.join(self._save_folder, 'ent_ids.txt'))
        entid2idx = {x:i for i,x in enumerate(ent_all_ids)}
        scipy.sparse.save_npz(osp.join(self._save_folder, 'features_sparses.npz'), features_sparses)
        print('Index creation...')
        for ent_type, ent_ids in tqdm.tqdm(type2ents.items(), "Indexing entities by type", total=len(type2ents)):
            ent_indices = [entid2idx[id_] for id_ in ent_ids]
            # Initialize NMSLib index
            index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)

            # Add data points to the index
            index.addDataPointBatch(features_sparses[ent_indices])

            # Create the index
            index.createIndex({'post': 2}, print_progress=True)
            index.saveIndex(osp.join(self._save_folder, 'index__%s.bin' % ent_type), save_data=True)
            print()
        print('Finished.')
        
        with open(osp.join(self._save_folder, 'type2ents.pkl'), 'wb') as f:
            pickle.dump(type2ents, f)

    def _destroy(self) -> None:
        shutil.rmtree(self._save_folder, ignore_errors=True)

    def built(self) -> bool:
        for to_check in ('type2ents.pkl', 'ent_ids.txt', 'features_sparses.npz'):
            if not osp.exists(osp.join(self._save_folder, to_check)):
                return False
        return True
    
    def load(self) -> None:
        attrs = SimilarityDistractorFinder._get_sim_dist_finder_if_exists(self.kb.time, self.use_reverse_features)
        if attrs is not None:
            print('Getting index and co. from another instance of %s...' % self.__class__.__name__)
            for k,v in attrs.items():
                setattr(self, k,v)
        else:
            self._load_index()

    def _load_vocab(self):
        print('Loading Vocabulary...')
        instance = SimilarityDistractorFinder._get_sim_dist_finder_if_exists(self.kb.time, self.use_reverse_features, True)
        instance._vocab = load_json(osp.join(self._save_folder, "tfidf_vectorizer.json"))
        self._vocab = instance._vocab

    def _load_index(self):
        if not self.built():
            raise Exception('Error : Index does not exist ("%s" folder does not exist or is incomplete).\nCall the setup() function to create it.' % self._save_folder)
        
        self._index = {}
        filenames = os.listdir(self._save_folder)
        for filename in tqdm.tqdm(filenames, "Loading Index", total=len(filenames)):
            if not filename.startswith('index__') or not filename.endswith('.bin'):
                continue
            type = filename[len("index__"):-4]
            index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
            index.loadIndex(osp.join(self._save_folder, filename), load_data=True)
            index.setQueryTimeParams({'efSearch': 2000})
            self._index[type] = index

        print('Loading entity names...')
        self._ent_ids = read_list(osp.join(self._save_folder, 'ent_ids.txt'))
        self._entities_in_index_name2idx = { k : i for i,k in enumerate(self._ent_ids)}

        print('Loading TF-IDF sparse matrix...')
        self._features_sparse = scipy.sparse.load_npz(osp.join(self._save_folder,'features_sparses.npz'))

        print('Load Type to Entities dictionary...')
        with open(osp.join(self._save_folder, 'type2ents.pkl'), 'rb') as f:
            self._type2ents = pickle.load(f)
        self._ent2types = defaultdict(list)
        for type, entities in self._type2ents.items():
            for entity in entities:
                self._ent2types[entity].append(type)
        self._type2len = [(x,len(y)) for x,y in self._type2ents.items()]
        self._type2len = sorted(self._type2len, key=lambda x : x[1], reverse=True)


        # Save in existing instances
        SimilarityDistractorFinder.existing_instances[(self.kb.time, self.use_reverse_features)] = self

        print('Finished Loading!')

    
    def _get_vectors(self, ids : str | Iterable[str], print_warning=False) -> np.ndarray | list[np.ndarray]:
        if is_single := isinstance(ids, str):
            ids = [ids]
        vectors = []
        for ent_id in ids:
            idx = self._entities_in_index_name2idx.get(ent_id, None)
            if idx is None:
                if print_warning:
                    print('WARNING : %s ID not found in the index' % ent_id)
                return None
            vector = self._features_sparse[idx]
            vectors.append(vector)
        if is_single:
            return vectors[0]
        return vectors

    def _get_nearest(self, emb_vector : np.ndarray, types : list[str], k=10):
        if len(types) == 0:
            return [], [] 
        all_ids, all_distances, all_types = [], [], []
        for t in types:
            ids, distances = self._index[t].knnQueryBatch(emb_vector, k=k)[0]
            all_ids.append(ids)
            all_distances.append(distances)
            all_types.append(np.full((len(ids),), t))
        all_ids = np.concatenate(all_ids)
        all_distances = np.concatenate(all_distances)
        all_types = np.concatenate(all_types)
        all_entity_ids = np.array([self._type2ents[t][i] for i,t in zip(all_ids, all_types)])

        _, unique_indices = np.unique(all_entity_ids, return_index=True, axis=0)
        all_entity_ids, all_distances = all_entity_ids[unique_indices], all_distances[unique_indices]
        _, nearest_indices = topk_with_indices(-all_distances, k)

        nearest_entity_ids, nearest_distances = all_entity_ids[nearest_indices], all_distances[nearest_indices]
        return nearest_entity_ids, nearest_distances
    
    def get_valid_triples_idx(self, triples):
        valid_ids = []
        for i, triple in enumerate(triples):
            idx = self._entities_in_index_name2idx.get(triple.object.id, None)
            if idx is not None:
                valid_ids.append(i)
        return np.array(valid_ids)

    def find(self, triple: Triple, t : Date, n: int) -> Iterable[Entity]:
        """Find n distractors for the given triple, at time t.
        
        Given a triple (s,r,o), all the triples constructed using the retrieved distractors noted (s,r,d) 
        are not valid at time t in the current knowledge base.

        Args:
            triple (Triple): Fact to find distractors for
            t (Date): Reference time. If t == "present", t is interpreted as equal to the present time of the current knowledge base (self.kb.time_date)
            n (int): Number of distractors to retrieve

        Returns:
            Iterable[Entity]: Iterator of Distractors (Entities)
        """
        assert self._index is not None, "This object was not loaded. Please call the %s.load() function first" % self.__class__.__name__
        
        if self.eps != 0 and self._vocab is None:
            print('Loading vocabulary to compute neighbors for eps != 0...')
            self._load_vocab()

        vector = self._get_vectors(triple.object.id)
        if self.eps != 0:
            to_enhance = [triple.subject.id, triple.relation.id, triple.object.id, "%s_%s" % (triple.subject.id, triple.relation.id)]
            for enh in to_enhance:
                idx = self._vocab.get(enh, None)
                if idx is not None:
                    vector[0,idx] += self.eps
        all_labels = self.kb_raw.get_all_names_of_entity([triple.object])[0]
        to_filter = set(x.object.id for x in self.kb.find(TimedTripleQuery(triple.subject, triple.relation, None, valid_at=t)))
        to_filter.update(y.id for x in self.kb.find_from_label(all_labels).values() for y in x if y != triple.object)
        n_nearest = n+len(to_filter)
        if self.dist_in_corr_removal:
            n_nearest += n_nearest // 2
        
        types = self._ent2types[triple.object.id]
        if len(types) == 0:
            types = [t for t,_ in self._type2len[:10]]
        neighbor_ids, _ = self._get_nearest(vector, k=n_nearest, types=types)

        temporal_dist = []
        if self.use_temporal_distractors:
            # Retrieve temporal distractors and put closest to t first
            temporal_triples : list[TimedTriple] = list(x for x in self.kb.find(TimedTripleQuery(triple.subject, triple.relation, None, None)))
            if t == 'present':
                t_ = self.kb.time_date
            else:
                t_ = t
            temporal_dist = [(x.object, x.valid_between.midpoint()) 
                             for x in temporal_triples 
                             if not x.valid_between.is_point() \
                                and wikidata_is_valid_at(x.valid_between.start, x.valid_between.end, t_, present=self.kb.time_date) == 'invalid']
            ref = t_.level_completion('mean').id
            dist_period = [(obj, np.abs(ref-mid.id)) for obj, mid in temporal_dist if mid is not None]
            temporal_dist = [obj for obj, mid in temporal_dist if mid is None]
            if len(dist_period):
                dist_period, _ = list(zip(*sorted(dist_period, key=lambda x : x[1])))
                dist_period = list(dist_period)
            else:
                dist_period = []
            temporal_dist = dist_period + temporal_dist
            temporal_dist = [Distractor(x, True) for x in temporal_dist]
            
            
        neighbors = temporal_dist + [Distractor(Entity(x), False) for x in neighbor_ids if x not in to_filter]
        if self.use_temporal_distractors:
            neighbors = uniquifier(neighbors)
        self.kb.inject_info(neighbors)

        if self.dist_in_corr_removal:
            neighbors_labels = [neigh.label.strip().split(' ') for neigh in neighbors]
            label = triple.object.label.strip().split(' ')
            neighbors = [neigh for i, neigh in enumerate(neighbors) if not is_subseq(neighbors_labels[i], label) and not is_subseq(label, neighbors_labels[i])]
        return neighbors[:n]
    
    def find_from_template(self, template: Template, n: int) -> Iterable[Entity]:
        return self.find(template.triple, template.time, n)

    @staticmethod
    def _get_sim_dist_finder_if_exists(time : str, use_reverse_features : bool, get_instance=False) -> dict | SimilarityDistractorFinder:
        instance = SimilarityDistractorFinder.existing_instances.get((time, use_reverse_features))
        if get_instance:
            return instance
        if instance is None:
            return None
        attrs = {}
        for attr in SimilarityDistractorFinder._LOAD_ATTRIBUTES:
            attrs[attr] = getattr(instance, attr)
        return attrs

    def _to_json(self) -> dict | list:
        return dict(time=self.time, use_reverse_features=self.use_reverse_features, eps=self.eps, use_temporal_distractors=self.use_temporal_distractors)

    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls(**d)

    def _identifier(self) -> Any:
        return self._to_json()
    

class ApproxIdealDistractorFinder(ObjectDistractorFinder, Blueprint):
    _SAVE_FOLDER_PATH = osp.join(STORAGE_FOLDER, "ApproxIdealDistractorFinder_data")
    def __init__(self, kb: TempWikidata, lm : LanguageModel) -> None:
        super().__init__(kb)
        self.lm = lm
        self.label2entity : dict[str, list[Entity]] = None
        self.entity2types : dict[Entity, list[Entity]]= None
        self.max_length_entity = 100
        self.type2token_limiter : dict[Entity, TokenLimiter] = None
        self.kb_raw = TempWikidata(self.kb.time, WikidataPrepStage.ALMOST_RAW)

    @property
    def _save_path(self):
        return osp.join(self._SAVE_FOLDER_PATH, self.kb.collection_name)
    
    def _generate_data(self) -> dict:
        type2entities = defaultdict(list)
        progress_bar = tqdm.tqdm(desc="Iterate through all 'instance of' triples", 
                                 total=self.kb.number_of_subjects())
        last_visited_subject = None
        entities = []
        for triple in self.kb.find(TripleQuery(relation=[INSTANCE_OF_RELATION, SUBCLASS_OF_RELATION])):
            sub = triple.subject
            if last_visited_subject != sub:
                last_visited_subject = sub
                progress_bar.update()
            type2entities[triple.object].append(sub)
            entities.append(sub)
        label2entity = defaultdict(list)
        for ent, labels in zip(entities, self.kb_raw.get_all_names_of_entity(entities)):
            for label in labels:
                label2entity[label].append(ent)
        label2entity = dict(label2entity)
        type2entities = dict(type2entities)
        return label2entity, type2entities
    
    @property
    def _save_path_label2ent(self):
        return osp.join(self._save_path, 'label2ent.pkl')
    
    @property
    def _save_path_type2ent(self):
        return osp.join(self._save_path, 'type2ent.pkl')
    
    @property
    def _save_path_parser(self):
        return osp.join(self._save_path, 'parser.pkl')

    def _build(self) -> None:
        print("Building %s..." % self.__class__.__name__)
        # Remove entities with no label
        with TimeItContextManager('Generating label2ent'):
            label2ent, type2ents = self._generate_data()
        os.makedirs(self._save_path, exist_ok=True)
        with open(self._save_path_label2ent, 'wb') as file:
            pickle.dump(label2ent, file, pickle.DEFAULT_PROTOCOL)

        with open(self._save_path_type2ent, 'wb') as file:
            pickle.dump(type2ents, file, pickle.DEFAULT_PROTOCOL)
        print("Building finished.")

    def load(self) -> None:
        print('Loading %s...' % self.__class__.__name__)
        t1 = time.time()
        if not self.built():
            raise Exception('Build this object first before loading it!')
        with TimeItContextManager("Load label2entity"):
            with open(self._save_path_label2ent, 'rb') as f:
                self.label2entity = pickle.load(f)
        with TimeItContextManager("Load type2entity"):
            with open(self._save_path_type2ent, 'rb') as f:
                type2entities = pickle.load(f)
        with TimeItContextManager('Build parser for each type'):
            self.type2token_limiter = {}
            self.entity2types = defaultdict(list)

            # Clean token strings
            # tokenstr2id = defaultdict(list)
            # vocab_size = len(self.lm.hf_tokenizer.get_vocab())
            # id2tokenstr = [None] * vocab_size
            # for token_str, token_idx in self.lm.hf_tokenizer.get_vocab().items():
            #     token_str_clean = self.lm.hf_tokenizer.convert_tokens_to_string([token_str])
            #     tokenstr2id[token_str_clean].append(token_idx)
            #     id2tokenstr[token_idx] = token_str_clean
            all_tokens = []
            self.type2entities = type2entities
            for type, entities in type2entities.items():
                poss = [''.join(x) for x in itertools.product(
                    [" "],
                    [x.label for x in entities]
                )]
                tokens = self.lm.hf_tokenizer(poss).input_ids
                all_tokens.extend(tokens)
                parser = MultiChoicesLimiter(tokens, self.lm.hf_tokenizer.eos_token_id)
                self.type2token_limiter[type] = parser
                for ent in entities:
                    self.entity2types[ent].append(type)
            else:
                # For entities with no case: Include everything
                parser = MultiChoicesLimiter(all_tokens, self.lm.hf_tokenizer.eos_token_id)
                self.type2token_limiter[None] = parser
        
        print('Loading finished. %.2fsec' % (time.time() - t1))
        

    def _destroy(self) -> None:
        shutil.rmtree(self._save_path, ignore_errors=True)

    def built(self) -> bool:
        return osp.exists(self._save_path_label2ent)

    def find(self, triple: Triple, t: Date, n: int) -> Iterable[Entity]:
        raise NotImplementedError

    def find_from_template(self, template: Template, n: int) -> Iterable[Entity]:
        if self.type2token_limiter is None:
            self.load()
        prompt = template.apply_str(template.subject.label, '').rstrip(' ')
        inputs = self.lm.hf_tokenizer([prompt], return_tensors="pt", add_special_tokens=False, return_token_type_ids=False, padding=False).to(self.lm.device)
        triple = template.triple

        all_labels = self.kb_raw.get_all_names_of_entity([triple.object])[0]
        to_filter = set(x.object for x in self.kb.find(TimedTripleQuery(triple.subject, triple.relation, None, valid_at=template.time)))
        to_filter.update(y for x in self.kb.find_from_label(all_labels).values() for y in x if y != triple.object)

        # AND operator between token limiter of each type of the object.
        types = self.entity2types.get(template.triple.object, [None])
        token_limiter = TokenLimitersCombinator([self.type2token_limiter[ty] for ty in types])
        best_sequences, _ = beam_search(self.lm.hf_model, inputs.input_ids, beam_width=n+len(to_filter), 
                                        max_new_tokens=self.max_length_entity, token_limiter=token_limiter,
                                        eos_token_id=self.lm.hf_tokenizer.eos_token_id)
        
        # Remove initial space at the start to get entities labels
        entities_generated = [self.lm.hf_tokenizer.decode(sequence, skip_special_tokens=True)[1:] for sequence in best_sequences]
        entities_found = [self.label2entity[ent][0] for ent in entities_generated if ent in self.label2entity]
        
        # Remove correct answers (we want distractors, i.e., incorrect answers)
        
        entities_found = [x for x in entities_found if x not in to_filter]

        return uniquifier(entities_found)[:n]

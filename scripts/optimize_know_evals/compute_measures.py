# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

import argparse
import pickle
import re
import time
import traceback

import numpy as np

import os.path as osp
import os

from tqdm import tqdm
from globals import STORAGE_FOLDER
from kb.core import Entity, Relation, Triple, TripleComp
from know.MinKAssess import KaRR
from know.generation import BERTScoreKM, GreddyCheck, LLMAsAJudgeKM, PrecisionAtN, RougeLKnowMeasure
from lm.core import LanguageModel
from know.core import DistKnowMeasure, KnowMeasure, MeanMetric, CredKnowMeasure
from know.distractor_find import SimilarityDistractorFinder, RandomDistractorFinder, ApproxIdealDistractorFinder
from lm.core import LogProbability
from kb.wikidata import TempWikidata, WikidataPrepStage
from utils.general import load_json
from verb.wikifactdiff import WikiFactDiffVerbalizer

NUM_ITERATIONS = 6+21
parser = argparse.ArgumentParser()

parser.add_argument('--iteration', type=int, default=0)
parser.add_argument('--print_num_iterations', action='store_true')
parser.add_argument('--date', type=str, default='20210104')

args = parser.parse_args()
date = args.date.strip()
assert re.match(r'^[0-9]{8}$',date), "The date must be in format YYYYMMDD"

if args.print_num_iterations:
    print(NUM_ITERATIONS)
    exit(0)

know_measures_list = ['distractor', 'greedy_check', 'precision@n', 'probability', 'bert_score', 'rouge_l', 'llm_as_a_judge', 'karr']
verbalizer = WikiFactDiffVerbalizer()
raw_wd = TempWikidata(date, WikidataPrepStage.ALMOST_RAW)
wd = TempWikidata(date, WikidataPrepStage.PREPROCESSED)


custom_templates = {}
for rel, templates in verbalizer._rel2temp.items():
    temps = []
    for temp, _ in templates:
        if temp.ends_with == TripleComp.OBJECT:
            temps.append(temp)
        if len(temps) == 5:
            break
    custom_templates[rel] = temps

temp = load_json(osp.join(osp.dirname(__file__), '../../src/know/MinKAssess/wikidata_triples_to_human_scores.json'))
triple2score = []
entities = []
for k,v in temp:
    sub_id, rel_id, obj_id = k
    triple = Triple(Entity(sub_id), Relation(rel_id), Entity(obj_id))
    triple2score.append((triple, v))
    entities.append(triple.subject)
    entities.append(triple.object)
wd.inject_info(triple2score)
triple2score = [(x,y) for x,y in triple2score if x.subject._label is not None and x.object._label is not None]

lm = LanguageModel.from_pretrained_name('gpt2-xl', 'auto')

def iterate_know_measures():
    agg = MeanMetric(np.arange(1,201))
    logprob = LogProbability()

    dist_finders = [
        SimilarityDistractorFinder(date),
        SimilarityDistractorFinder(date, use_temporal_distractors=True),
        RandomDistractorFinder(wd),
        ApproxIdealDistractorFinder(wd, lm)
    ]
    for dist_find in dist_finders:
        know_measure = DistKnowMeasure(agg, dist_find, logprob, compute_cred_on_object=True, use_aliases=True)
        yield know_measure


    yield CredKnowMeasure(logprob, compute_on=TripleComp.OBJECT)
    additional_aliases = {ent : aliases for ent, aliases in zip(entities, wd.get_all_names_of_entity(entities))}
    yield KaRR(lm, custom_templates, additional_aliases)
    yield GreddyCheck(raw_wd)
    for num_beams in [1,2,5,10,20,30,50,100]:
        yield RougeLKnowMeasure(raw_wd, num_beams)
        yield BERTScoreKM(raw_wd, num_beams)

    for n in [1,2,5,10,20,40,60,80,100,120,140,160,200]:
        yield PrecisionAtN(raw_wd, n)
    for num_beams in [1,2,5,10]:
        yield LLMAsAJudgeKM(raw_wd, num_beams, oai_model="gpt-35-turbo-16k-0613")

    yield KaRR(lm, None, additional_aliases)

def compute_know_results(know_measure : KnowMeasure):
    know_returns = []
    if isinstance(know_measure, DistKnowMeasure):
        know_measure.obj_dist_finder.load()
    for fact, _ in tqdm(triple2score, 'Measuring Knowledge...'):
        t1 = time.time()
        res = know_measure.measure_fact(lm, fact, custom_templates[fact.relation])
        res.additional_data['execution_time'] = time.time() - t1
        know_returns.append(res)
    return know_returns


it = iterate_know_measures()
try:
    for i in range(args.iteration + 1):
        km = next(it)
except StopIteration:
    print('ERROR: Iteration %s does not exist' % args.iteration)
    exit(0)

folder_save = osp.join(STORAGE_FOLDER, 'optimize_know_evals_results')
os.makedirs(folder_save, exist_ok=True)
result_path = osp.join(folder_save, 'iteration_%s.pkl' % args.iteration)

if osp.exists(result_path):
    print('Already Done!')
    exit(0)

know_returns = compute_know_results(km)

pickle.dump(know_returns, open(result_path, 'wb'))

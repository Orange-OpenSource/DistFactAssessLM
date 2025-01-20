# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

import argparse
import os.path as osp
import pickle
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from globals import STORAGE_FOLDER
from kb.core import TripleComp
from kb.wikidata import TempWikidata, WikidataPrepStage
from know.core import DistKnowMeasure, MeanMetric, StrictMetric
from know.distractor_find import ApproxIdealDistractorFinder, RandomDistractorFinder, SimilarityDistractorFinder
from lm.core import LanguageModel, LogProbability
from verb.core import VerbalizeConfig
from verb.wikifactdiff import WikiFactDiffVerbalizer


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EleutherAI/pythia-14m')
parser.add_argument('--strategy', type=str, default='random', choices=['idf', 'temp_idf', 'random', 'AoID', 'pure_random'])
parser.add_argument('--facts_path', type=str, default=osp.join(STORAGE_FOLDER, 'facts_balanced_5000.pkl'))
parser.add_argument('--force', action='store_true')
parser.add_argument('--date', type=str, default='20210104')
args = parser.parse_args()

model_name = args.model
strategy_name = args.strategy
facts_path = args.facts_path
facts_path_filename = Path(facts_path).stem
date = args.date.strip()
assert re.match(r'^[0-9]{8}$',date), "The date must be in format YYYYMMDD"


save_folder = osp.join(STORAGE_FOLDER, 'compare_retrieval_strategies_results__%s' % facts_path_filename)
os.makedirs(save_folder, exist_ok=True)
file_path = osp.join(save_folder, "%s__%s.pkl" % (model_name.replace('/', '_'), strategy_name))

if osp.exists(file_path) and not args.force:
    print('Already Done!')
    exit(0)

wd = TempWikidata(date, WikidataPrepStage.PREPROCESSED)
lm = LanguageModel.from_pretrained_name(model_name, 'auto')

mean_agg = MeanMetric(np.arange(1, 101))
strict_agg = StrictMetric(np.arange(1,101))
if strategy_name == 'AoID':
    dist_finder = ApproxIdealDistractorFinder(wd, lm)
elif strategy_name == 'idf':
    dist_finder = SimilarityDistractorFinder(wd.time)
elif strategy_name == 'temp_idf':
    dist_finder = SimilarityDistractorFinder(wd.time, use_temporal_distractors=True)
elif strategy_name == 'random':
    dist_finder = RandomDistractorFinder(wd)
elif strategy_name == 'pure_random':
    dist_finder = RandomDistractorFinder(wd, pure=True)
assert dist_finder.built(), "Distractor finder %s must be built first before executing this script!" % dist_finder
dist_finder.load()

cred_func = LogProbability()
know_mean = DistKnowMeasure(mean_agg, dist_finder, cred_func, compute_cred_on_object=True)
know_strict = DistKnowMeasure(strict_agg, dist_finder, cred_func, compute_cred_on_object=True)


facts : pd.DataFrame = pd.read_pickle(facts_path)
# Small sample is enough
facts = facts.iloc[:1000]
know_measures = []

verbalizer = WikiFactDiffVerbalizer()
config = VerbalizeConfig(
    max_num_verbs=1,
    verb_tense=None,
    ends_with=TripleComp.OBJECT
)

for _, (triple, popularity) in tqdm(facts.iterrows(), "Knowledge measuring", total=len(facts)):
    temp = verbalizer.verbalize(triple, config, skip_failed_conjugation=True)
    if len(temp) == 0:
        continue
    temp = temp[0]
    try:
        measure = know_mean.measure_temp(lm, temp)
        know_measures.append(measure)

        measure = know_strict.measure_from_know_return(measure)
        know_measures.append(measure)
    except:
        print('ERROR: Template=%s' % temp.text)


with open(file_path, 'wb') as f:
    pickle.dump(know_measures, f)

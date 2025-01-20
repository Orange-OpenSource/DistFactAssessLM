# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

import argparse
import re

from globals import STORAGE_FOLDER
from kb.wikidata import TempWikidata, WikidataPrepStage
from know.data import sample_balanced_facts
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--type', required=True, choices=['random', 'tempdist'],
                    help="What type of facts to sample")
parser.add_argument('--date', required=True,
                    help="From which Wikidata dump date facts are sampled.")
parser.add_argument('--num_facts', type=int, default=5000,
                    help='Number of facts to sample')
args = parser.parse_args()
tempdist = args.type == 'tempdist'
date = args.date.strip()
num_facts = args.num_facts
assert re.match(r'^[0-9]{8}$',date), "The date must be in format YYYYMMDD"
assert num_facts > 0, "num_facts needs to be strictly positive"

wd = TempWikidata(date, WikidataPrepStage.PREPROCESSED)
df = sample_balanced_facts(wd, num_facts, wd.time_date, presence_of_temporal_distractors=tempdist)
df.to_pickle(osp.join(STORAGE_FOLDER, f'facts_balanced_{num_facts}.pkl'))
print(df.head())
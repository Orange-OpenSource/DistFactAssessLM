# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from know.data import WikidataSample
from kb.wikidata import WikidataPrepStage


sample = WikidataSample.build_from_config(2000, 'balanced', "20210104", WikidataPrepStage.PREPROCESSED, force_rebuild=False)
# for t, p in zip(sample.triples, sample.popularity):
#     print(t,p)
from verb.wikifactdiff import WikiFactDiffVerbalizer, WFDVerbVersion
from verb.core import VerbalizeConfig


verbalizer = WikiFactDiffVerbalizer(WFDVerbVersion.V2)
import random

from tqdm import tqdm
from kb.core import Date
from verb.options import DateIndicator, Tense
from kb.wikidata import Wikidata

wiki = Wikidata("20210104", WikidataPrepStage.ALMOST_RAW)
random.seed(12412)

wiki.inject_info(x for t in sample.triples for x in (t.relation, t.object))
triples_raw = [str((t.subject.id, t.relation.id, t.object.id)) for t in sample.triples]
triples = [str(t) for t in sample.triples]
present_verb_config = VerbalizeConfig(max_num_verbs=5, verb_tense=Tense.PRESENT, temporal_indicator=None)
past_verb_config = VerbalizeConfig(max_num_verbs=5, verb_tense=Tense.PAST, temporal_indicator=DateIndicator(Date.from_string('2020-01-01')))
verbs = []
context = []
n_errors = 0
for i,t in enumerate(tqdm(sample.triples)):
    config = present_verb_config if i % 2 == 0 else past_verb_config
    ctx = 'present' if i % 2 == 0 else 'past'
    context.append(ctx)
    verb = verbalizer.verbalize(t, config, skip_failed_conjugation=True)
    if len(verb) == 0:
        n_errors += 1
        verbs.append(None)
        continue 
    verb = random.choice(verb)
    verbs.append(verb)
import pandas as pd

df = pd.DataFrame(data=dict(triples_raw=triples_raw, triples=triples, popularity=sample.popularity, context=context, verbalization=verbs))
df['errors'] = ''
pd.options.display.max_colwidth = None
df.head()

import os.path as osp
df.to_csv(osp.join(osp.dirname(__file__), 'taxonomy_hichem.csv'))

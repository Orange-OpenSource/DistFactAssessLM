# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from know.MinKAssess.karr import KaRR
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'gpt2'
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

karr = KaRR(model, tokenizer, device)

# Testing the fact: (France, capital, Paris)
# You can find other facts by looking into Wikidata
fact = ('Q142', 'P36', 'Q90')

karr, does_know = karr.compute(fact)
print('Fact %s' % str(fact))
print('KaRR = %s' % karr)
ans = 'Yes' if does_know else 'No'
print('According to KaRR, does the model knows this fact? Answer: %s' % ans)
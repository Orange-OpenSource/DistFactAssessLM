# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from wikidataintegrator import wdi_core

def get_wikidata_info(qid):
    my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id=qid)
    all_info = my_first_wikidata_item.get_wd_json_representation()
    print([item['value'] for item in all_info['aliases']['en']])
    return all_info

def get_wikidata_aliases(qid):
    my_first_wikidata_item = wdi_core.WDItemEngine(wd_item_id=qid)
    all_info = my_first_wikidata_item.get_wd_json_representation()
    aliases = [all_info['labels']['en']['value']]
    for lang in all_info['aliases'].keys():
        if lang == 'en':
            aliases += [item['value'] for item in all_info['aliases'][lang]]
    return aliases
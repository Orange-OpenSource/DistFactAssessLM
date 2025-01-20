# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import recall_score
from kb.core import Entity, Relation, Triple
from know.core import KnowMeasure
from lm.core import LanguageModel
from utils.general import load_json
from scipy.stats import kendalltau
import os.path as osp

@dataclass
class DongEvalRes:
    kendall : float
    p_value : float
    recall : float
    # var : float
    # std : float
    # sp : float
    # delta_p : float
    # I need more data for these last metrics


# evaldata format:
# [
#     {
#         "fact" : ("Q1", "P11", "Q2"),
#         "know" : 0.8
#     }
# ]

def preprocess_eval_data():
    excel_path = osp.join(osp.dirname(__file__), 'probing_data_gpt2-xl.xlsx')
    avg_rating_path = osp.join(osp.dirname(__file__), 'avg_raterdict_202408.json')
    return pd.read_excel()

class DongEvaluation:
    evaldata = None
    
    def __init__(self, know_measure : KnowMeasure) -> None:
        self.know_measure = know_measure

    def calibrate(self) -> dict:
        pass

    def eval(self, lm : LanguageModel) -> DongEvalRes:
        if DongEvaluation.evaldata is None:
            DongEvaluation.evaldata = preprocess_eval_data(DongEvaluation.EVALDATA_PATH)
        to_triple = lambda fact : Triple(Entity(fact[0]), Relation(fact[1]), Entity(fact[2]))
        facts = [to_triple(x["fact"]) for x in DongEvaluation.evaldata]
        gold_know = [x['know'] for x in DongEvaluation.evaldata]
        pred_know = [self.know_measure.measure_fact(lm, f) for f in facts]

        gold_not_known = [x < 0.5 for x in gold_know]
        pred_not_known = [not x.does_know for x in pred_know]
        
        kendall, p_value = kendalltau(gold_know, [x.result for x in pred_know])
        recall = recall_score(gold_not_known, pred_not_known)

        return DongEvalRes(kendall, p_value, recall)


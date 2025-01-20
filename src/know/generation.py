# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from globals import STORAGE_FOLDER
from kb.core import TripleComp
from kb.wikidata import Wikidata
from know.core import KnowMeasure, KnowMeasureReturn
from lm.core import LanguageModel
from utils.beamsearch import beam_search
from utils.general import get_chatgpt_response
from verb.core import Template
import bert_score
from diskcache import FanoutCache
import os.path as osp

cache = FanoutCache(osp.join(STORAGE_FOLDER, 'diskcache'))



def lcs(X, Y):
    m = len(X)
    n = len(Y)
    
    # Create a table to store lengths of longest common subsequence.
    L = [[0] * (n + 1) for i in range(m + 1)]
    
    # Build the L[m+1][n+1] table in bottom-up fashion.
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

def rouge_l(candidate, reference):
    lcs_length = lcs(candidate, reference)
    m = len(reference)
    n = len(candidate)
    
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
    
    precision = lcs_length / n
    recall = lcs_length / m
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def rouge_l_multiple_references(candidate, references):
    max_precision = 0.0
    max_recall = 0.0
    max_f1_score = 0.0
    
    for reference in references:
        precision, recall, f1_score = rouge_l(candidate, reference)
        if f1_score > max_f1_score:
            max_precision = precision
            max_recall = recall
            max_f1_score = f1_score
    
    return max_precision, max_recall, max_f1_score


class RougeLKnowMeasure(KnowMeasure):
    def __init__(self, wd : Wikidata, num_beams : int) -> None:
        super().__init__()
        self.num_beams = num_beams
        self.wd = wd
    
    def measure_temp(self, lm: LanguageModel, temp: Template) -> KnowMeasureReturn:
        object = temp.triple.object
        temp = temp.copy()
        all_labels = self.wd.get_all_names_of_entity([object])[0]
        temp.delete(TripleComp.OBJECT)
        input_ids = lm.hf_tokenizer(temp.apply_str(temp.subject.label, '').rstrip(' '), return_tensors='pt').input_ids.to(lm.device)
        max_length = lm.hf_tokenizer(all_labels, return_tensors='pt', padding=True).input_ids.shape[-1] + 5
        best_sequences, _ = beam_search(lm.hf_model, input_ids, beam_width=self.num_beams, max_new_tokens=max_length, eos_token_id=lm.hf_tokenizer.eos_token_id, batch_size=50)
        best_sequences = [lm.hf_tokenizer.decode(x) for x in best_sequences]

        score = max(rouge_l_multiple_references(cand, all_labels)[-1] for cand in best_sequences)
        return KnowMeasureReturn(self.infos(), lm.lm_name, temp, score)
    
    def infos(self) -> dict:
        return dict(cls=self.class_id(), num_beams=self.num_beams)


class GreddyCheck(KnowMeasure):
    def __init__(self, wd : Wikidata) -> None:
        super().__init__()
        self.wd = wd
    
    def measure_temp(self, lm: LanguageModel, temp: Template) -> KnowMeasureReturn:
        object = temp.triple.object
        temp = temp.copy()
        all_labels = self.wd.get_all_names_of_entity([object])[0]
        temp.delete(TripleComp.OBJECT)
        input_ids = lm.hf_tokenizer(temp.apply_str(temp.subject.label, '').rstrip(' '), return_tensors='pt').input_ids.to(lm.device)
        max_length = lm.hf_tokenizer(all_labels, return_tensors='pt', padding=True).input_ids.shape[-1] + 5
        best_sequences, _ = beam_search(lm.hf_model, input_ids, beam_width=1, max_new_tokens=max_length, eos_token_id=lm.hf_tokenizer.eos_token_id, batch_size=50)
        text_generated = lm.hf_tokenizer.decode(best_sequences[0])

        score = max(int(text_generated.startswith(label)) for label in all_labels)
        temp.inject(None, object)
        return KnowMeasureReturn(self.infos(), lm.lm_name, temp, score)
    
    def infos(self) -> dict:
        return dict(cls=self.class_id())


# Compute BERTScore for all generated answers against all gold references
def compute_bert_scores(generated_answers, gold_references):
    # Create lists to hold the repeated generated answers and the corresponding references
    all_generated = []
    all_references = []
    
    for generated_answer in generated_answers:
        all_generated.extend([generated_answer] * len(gold_references))
        all_references.extend(gold_references)
    
    # Compute BERTScore for all pairs
    _, _, F1 = bert_score.score(all_generated, all_references, lang="en", verbose=False)
    
    # Process the results to get the max F1 score for each generated answer
    num_references = len(gold_references)
    all_scores = []
    
    for i in range(0, len(F1), num_references):
        max_f1 = F1[i:i + num_references].max().item()
        all_scores.append(max_f1)
    
    return all_scores

class BERTScoreKM(KnowMeasure):
    def __init__(self, wd : Wikidata, num_beams : int) -> None:
        super().__init__()
        self.wd = wd
        self.num_beams = num_beams
    
    def measure_temp(self, lm: LanguageModel, temp: Template) -> KnowMeasureReturn:
        object = temp.triple.object
        all_labels = self.wd.get_all_names_of_entity([object])[0]
        temp = temp.copy()
        temp.delete(TripleComp.OBJECT)
        max_length = lm.hf_tokenizer(all_labels, return_tensors='pt', padding=True).input_ids.shape[-1] + 5
        input_ids = lm.hf_tokenizer(temp.apply_str(temp.subject.label, '').rstrip(' '), return_tensors='pt').input_ids.to(lm.device)
        
        best_sequences, _ = beam_search(lm.hf_model, input_ids, beam_width=self.num_beams, max_new_tokens=max_length, eos_token_id=lm.hf_tokenizer.eos_token_id, batch_size=50)
        best_sequences = [lm.hf_tokenizer.decode(x) for x in best_sequences]

        score = max(compute_bert_scores(best_sequences, all_labels))
        temp.inject(None, object)
        return KnowMeasureReturn(self.infos(), lm.lm_name, temp, score)

    def infos(self) -> dict:
        return dict(cls=self.class_id(), num_beams=self.num_beams)

LLM_AS_A_JUDGE_CHATGPT_PROMPT = """You need to check whether the prediction of a question-answering system to a query is correct. You should make the judgment based on a list of ground truth answers provided to you in a form of a list of aliases of the gold answer. Your response should be "correct" if the prediction is correct or "incorrect" if the prediction is wrong.

Query: The author of The Taming of the Shrew (published in 2002) is ____
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: W Shakespeare
Correctness: correct

Query: The author of The Taming of the Shrew (published in 2002) is ____
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: Roma Gill and W Shakespeare
Correctness: correct

Query: The author of The Taming of the Shrew (published in 2002) is ____
Ground truth: ["William Shakespeare", "Roma Gill"]
Prediction: Roma Shakespeare
Correctness: incorrect

Query: The country where Maharashtra Metro Rail Corporation Limited is located is ____
Ground truth: ["India"]
Prediction: Maharashtra
Correctness: incorrect

Query: The job of Song Kang-ho in Parasite (2019) is ____
Ground truth: ["actor"]
Prediction: He plays the role of Kim Ki-taek, the patriarch of the Kim family.
Correctness: correct

Query: The era to which Michael Oakeshott belongs is ____
Ground truth: ["20th-century philosophy"]
Prediction: 20th century.
Correctness: correct

Query: The department where Edward Tise (known for Full Metal Jacket (1987)) was is ____
Ground truth: ["sound department"]
Prediction: 2nd Infantry Division, United States Army
Correctness: incorrect

Query: The wine region to which Finger Lakes AVA belongs is ____
Ground truth: ["New York wine"]
Prediction: Finger Lakes AVA
Correctness: incorrect

Query: {}
Ground truth: {}
Prediction: {}
Correctness:"""

@cache.memoize(typed=True, expire=None, tag='llm_as_a_judge')
def llm_as_a_judge(query : str, candidate : str, references : list[str], oai_model: str="gpt-3.5-turbo") -> bool:
    response = get_chatgpt_response(LLM_AS_A_JUDGE_CHATGPT_PROMPT.format(
        query, candidate, references
    ), model=oai_model)
    if response is None:
        response = ""
    if 'incorrect' in response:
        return False
    elif 'correct' in response:
        return True
    else:
        return False

class LLMAsAJudgeKM(KnowMeasure):
    def __init__(self, wd : Wikidata, num_beams : int, oai_model : str = "gpt-3.5-turbo") -> None:
        super().__init__()
        self.wd = wd
        self.num_beams = num_beams
        self.oai_model = oai_model
    
    def measure_temp(self, lm: LanguageModel, temp: Template) -> KnowMeasureReturn:
        object = temp.triple.object
        temp = temp.copy()
        all_labels = self.wd.get_all_names_of_entity([object])[0]
        max_length = lm.hf_tokenizer(all_labels, return_tensors='pt', padding=True).input_ids.shape[-1] + 5
        temp.delete(TripleComp.OBJECT)
        input_ids = lm.hf_tokenizer(temp.apply_str(temp.subject.label, '').rstrip(' '), return_tensors='pt').input_ids.to(lm.device)
        
        best_sequences, _ = beam_search(lm.hf_model, input_ids, beam_width=self.num_beams, max_new_tokens=max_length, eos_token_id=lm.hf_tokenizer.eos_token_id, batch_size=50)
        best_sequences = [lm.hf_tokenizer.decode(x) for x in best_sequences]

        score = max(llm_as_a_judge(temp.apply_str(temp.subject.label, '____'), cand, all_labels, oai_model=self.oai_model) for cand in best_sequences)
        temp.inject(None, object)
        return KnowMeasureReturn(self.infos(), lm.lm_name, temp, int(score))

    def infos(self) -> dict:
        return dict(cls=self.class_id(), num_beams=self.num_beams)
    

class PrecisionAtN(KnowMeasure):
    def __init__(self, wd : Wikidata, n : int) -> None:
        super().__init__()
        self.wd = wd
        self.n = n
    
    def measure_temp(self, lm: LanguageModel, temp: Template) -> KnowMeasureReturn:
        object = temp.triple.object
        temp = temp.copy()
        all_labels = self.wd.get_all_names_of_entity([object])[0]
        temp.delete(TripleComp.OBJECT)
        input_ids = lm.hf_tokenizer(temp.apply_str(temp.subject.label, '').rstrip(' '), return_tensors='pt').input_ids.to(lm.device)
        
        _, top_tokens = lm.hf_model(input_ids).logits[0,-1].topk(self.n, dim=-1)
        score = max(int(lm.hf_tokenizer.decode(x) in all_labels) for x in top_tokens)

        temp.inject(None, object)
        return KnowMeasureReturn(self.infos(), lm.lm_name, temp, int(score))

    def infos(self) -> dict:
        return dict(cls=self.class_id(), n=self.n)
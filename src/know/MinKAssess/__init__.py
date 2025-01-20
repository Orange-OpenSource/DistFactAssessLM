# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from kb.core import Entity, Relation, Triple
from know.core import KnowMeasure, KnowMeasureReturn
from lm.core import LanguageModel
from verb.core import Template
from know.MinKAssess.karr import KaRR as KaRROrigin



# _cache_karr : dict[str, KaRROrigin] = {}
# class KaRR(KnowMeasure):
#     def __init__(self, thresh=22) -> None:
#         super().__init__()
#         self.thresh = thresh

#     def measure_temp(self, lm: LanguageModel, temp: Template) -> KnowMeasureReturn:
#         raise NotImplementedError
    
#     def measure_fact(self, lm: LanguageModel, fact: Triple) -> KnowMeasureReturn:
#         karr = _cache_karr.get(lm.lm_name, KaRROrigin(lm.hf_model, lm.hf_tokenizer, lm.device, self.thresh))
#         _cache_karr[karr.model_name] = karr
#         return karr.compute(fact.to_sro_str())
    

class KaRR(KnowMeasure):
    def __init__(self, lm : LanguageModel, custom_rel_to_templates : dict[Relation, list[Template]] = None, 
                 additional_aliases : dict[Entity, list[str]] = None, use_aliases = True) -> None:
        super().__init__()
        custom_rel2alias = None
        self.using_custom_templates = custom_rel_to_templates is not None
        if custom_rel_to_templates is not None:
            custom_rel2alias = {k.id:[t.apply_str('[X]', '[Y]') for t in v] for k,v in custom_rel_to_templates.items()}
        if additional_aliases is not None:
            additional_aliases = {ent.id: list_str for ent, list_str in additional_aliases.items()}
        self.minkarr = KaRROrigin(lm.hf_model, lm.hf_tokenizer, lm.device, custom_rel2alias=custom_rel2alias, 
                                  use_aliases=use_aliases, additional_aliases=additional_aliases)
                
        self.lm = lm

    def measure_temp(self, lm : LanguageModel, temp : Template) -> KnowMeasureReturn:
        raise NotImplementedError

    def measure_fact(self, lm : LanguageModel, fact : Triple, *args, **kwargs) -> KnowMeasureReturn:
        assert lm == self.lm, "Required: lm == self.lm. It's ugly I know, I'll fix later"
        res, does_know = self.minkarr.compute(tuple(x.id for x in fact.to_sro()))
        return KnowMeasureReturn(self.infos(), self.lm.lm_name, None, result=res, does_know=does_know, tag='fact_level')
    
    @property
    def threshold(self) -> float:
        return self.minkarr.thresh
    
    def infos(self) -> dict:
        return dict(cls=self.class_id(), lm_infos=self.lm.infos(), use_aliases=self.minkarr.use_aliases, using_custom_relations=self.using_custom_templates)

    
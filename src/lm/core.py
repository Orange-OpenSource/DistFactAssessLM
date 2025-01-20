# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from __future__ import annotations
from collections import abc
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import torch
from typing import Generator, Iterable, Union
from abc import abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding, PreTrainedModel, PreTrainedTokenizer, GPT2LMHeadModel
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from glob_core import Mongoable, MongoableEnum, Precision, SameType
from utils.general import create_switch_contextmanager, inf_none_gen

class CredibilityFunction(Mongoable):
    """A credibility function takes as input the logits produced by a transformer-based language model given input texts. 
    It returns a score reflecting the credibility (or plausibility) of these texts according to the language model.
    """
    def compute(self, logits : Union[Iterable[Tensor] | Tensor], input_ids : Union[Iterable[Tensor] | Tensor], attention_mask : Union[Iterable[Tensor] | Tensor] = None) -> Tensor:
        """Compute this function

        Args:
            logits (Union[Iterable[Tensor] | Tensor]): The logits produced by a language model (e.g. output of LanguageModel.batch_forward). 
            This argument could be a Tensor of shape (N,S,V) where V is the vocabulary size, or an iterable of smaller batches of shape (N_i, S_i, V). 
            
            input_ids (Union[Iterable[Tensor] | Tensor]) : Input tokens

            attention_mask (Union[Iterable[Tensor] | Tensor], optional): The attention mask. This argument could be a Tensor or an iterable of Tensors. 
            Defaults to None (No attention mask --> compute the credibility on the whole sequence dimension).

        Returns:
            Tensor: Credibility in the form of a unidimensional tensor
        """
        result = None
        check_args = self._check_args(logits, input_ids, attention_mask)
        if(check_args == CredFuncArgsStatus.ITERABLE_ALL):
            result = self._compute_iterable(logits, input_ids, attention_mask)
        elif(check_args == CredFuncArgsStatus.TENSOR_ALL):
            result = self._compute_tensor(logits, input_ids, attention_mask)
        elif(check_args == CredFuncArgsStatus.ERROR):
            result = None
            raise TypeError(f"The arguments are not of the same type, logits:\n{type(logits)} " 
                            f"input_ids: {type(input_ids)}, attention_mask: {type(attention_mask)}")
        return result
    
    def _check_args(self, logits : Union[Iterable[Tensor] | Tensor], input_ids : Union[Iterable[Tensor] | Tensor], attention_mask : Union[Iterable[Tensor] | Tensor]) -> CredFuncArgsStatus:
        """Check the validity of the input arguments of compute(logits, input_ids, attention_mask).
        Only valid combination is:
        - All arguments are iterable
        - All arguments are tensors

        Args:
            logits (Union[Iterable[Tensor] | Tensor]): Logits
            input_ids (Union[Iterable[Tensor] | Tensor]) : Input tokens
            attention_mask (Union[Iterable[Tensor] | Tensor]): Attention mask

        Returns:
            CredFuncArgsStatus
        """      
        if isinstance(logits, torch.Tensor) and isinstance(input_ids, torch.Tensor) and (isinstance(attention_mask, torch.Tensor) or attention_mask is None):
            credFuncArgsStatus = CredFuncArgsStatus.TENSOR_ALL 
        elif isinstance(logits, abc.Iterable) and isinstance(input_ids, abc.Iterable) and (isinstance(attention_mask, abc.Iterable) or attention_mask is None):
            credFuncArgsStatus = CredFuncArgsStatus.ITERABLE_ALL
        
        else:
            credFuncArgsStatus = CredFuncArgsStatus.ERROR
                
        return credFuncArgsStatus

    @abstractmethod
    def _compute_tensor(self, logits : Tensor, input_ids : Tensor, attention_mask : Tensor = None) -> Tensor:
        """Compute this function given logits and attention mask

        Args:
            logits (Tensor): Logits tensor of dimension (N,S,V)
            input_ids (Tensor, optional): Input tokens tensor of dimension (N,S). 
            attention_mask (Tensor, optional): Attention mask tensor of dimension (N,S). 
            Defaults to None (compute the credibility on the whole sequence dimension).

        Returns:
            Tensor: Credibility in the form of a unidimensional tensor
        """
        pass

    def _compute_iterable(self, logits : Iterable[Tensor], input_ids : Iterable[Tensor], attention_mask : Iterable[Tensor] = None) -> Tensor:
        """Compute this function given logits and attention mask

        Args:
            logits (Tensor): Iterable of Logits tensors of dimension (N_i,S_i,V)
            input_ids (Tensor, optional): Input tokens tensors of dimension (N_i,S_i). 
            attention_mask (Tensor, optional): Iterable of attention mask tensors of dimension (N_i,S_i). 
            Defaults to None (compute the credibility on the whole sequence dimension).

        Returns:
            Tensor: Credibility in the form of a unidimensional tensor
        """
        results = []
        if attention_mask is None:
            attention_mask = inf_none_gen()
        for logit, input_id, am in zip(logits, input_ids, attention_mask): 
            results.append(self._compute_tensor(logit, input_id, am))
        result = torch.cat(results, 0)
        return result

    def __repr__(self) -> str:
        # A string representation of a Credibility function : The name of the class
        return self.__class__.__name__
    
    def _to_json(self) -> dict | list:
        return {}
    
    def _identifier(self) -> dict | list:
        return self._to_json()
    
    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls()
    
class CredFuncArgsStatus(MongoableEnum):
    """Input type in CredibilityFunction.compute(input_ids, attention_mask)
    """
    ITERABLE_ALL = -1
    ERROR = 0
    TENSOR_ALL = 1

class NegPerplexity(CredibilityFunction):
    """The perplexity function
    """

    def __init__(self) -> None:
        super().__init__()
        self._logprob_func = LogProbability()
    
    def _compute_tensor(self, logits: Tensor, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
        
        if attention_mask is not None:
            mask_sum = attention_mask[:,1:].sum(-1)
        else:
            mask_sum = input_ids.shape[-1]-1
        perp = torch.exp(-1 / mask_sum * self._logprob_func._compute_tensor(logits, input_ids, attention_mask))
        return -perp


class LogProbability(CredibilityFunction):
    """The log-probability function
    """
    def _compute_tensor(self, logits: Tensor, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        logsoftmax = torch.log_softmax(logits, dim=-1)
        log_probs = torch.gather(logsoftmax[:, :-1, :], 2, input_ids[:, 1:, None]).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask[:,1:]
        else:
            mask = 1
        prob = (log_probs*mask).sum(-1)
        return prob

class ExactMatch(CredibilityFunction):
    """The ExactMatch function
    """
    def _compute_tensor(self, logits: Tensor, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        predicted_ids = logits[:, :-1, :].argmax(-1)
        if attention_mask is not None:
            mask = attention_mask[:,1:].bool()
        else:
            mask = torch.scalar_tensor(True, dtype=torch.bool, device=logits.device)
        prob = ((predicted_ids == input_ids[:, 1:]) | ~mask).all(-1)
        return prob
    
class LanguageModel(Mongoable):
    """Causal language model class which is composed of a tokenizer and a transformer 
    """
    enable_cache_credibility_text = False
    def __init__(self, hf_model : PreTrainedModel, hf_tokenizer : PreTrainedTokenizer, precision : Precision = None) -> None:
        """Initialize language model

        Args:
            hf_model: HuggingFace Causal Language Model
            hf_tokenizer: Huggingface Tokenizer
            precision (Precision, optional): The precision in which hf_model is encoded. 
            Defaults to None in which case all the precision is infered from the model's parameters.
            
        WARNING : It's rare, but it is technically possible for a model to be encoded in 
        one precision for some parameters and another precision for other parameters. We ignore this special case and suppose 
        that a model is encoded in a unique precision everywhere.
        """
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        param = next(iter(self.hf_model.state_dict().values()))
        self.device = param.device
        if precision is None:
            precision : Precision = Precision.to_precision(param.dtype)
        self.precision = precision
        self.hf_model.eval()
        self.hf_tokenizer.padding_side = "right"
        if self.hf_tokenizer.pad_token is None:
            self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        self.from_pretrained_kwargs = None
        self.lm_edits_post_loading = []

    def forward(self, *args, **kwargs) -> CausalLMOutputWithCrossAttentions:
        """This function points to the HuggingFace forward implementation for causal language models 
        (see https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.forward)

        Returns:
            CausalLMOutputWithCrossAttentions : contains potentially logits, loss, hidden_states, etc.
        """
        return self.hf_model.forward(*args, **kwargs)

    def apply_edits(self, edits : list) -> None:
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> CausalLMOutputWithCrossAttentions:
        """This function points to LanguageModel.forward 

        Returns:
            CausalLMOutputWithCrossAttentions : contains potentially logits, loss, hidden_states, etc.
        """
        return self.forward(*args, **kwargs)
    
    @staticmethod
    def from_pretrained(lm_kwargs : dict, tok_kwargs : dict) -> LanguageModel:
        """This method initializes a language model using the HuggingFace's from_pretrained functions of tokenizers and transformers.
        
        This is the recommended way to initialize a LanguageModel.

        Args:
            lm_kwargs (dict): Named arguments for AutoModelForCausalLM.from_pretrained (arguments' definitions 
            in https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained)
            
            tok_kwargs (dict): Named arguments for AutoTokenizer.from_pretrained (arguments' definitions 
            in https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained)

        Returns:
            LanguageModel
        """
        hf_tokenizer = AutoTokenizer.from_pretrained(**tok_kwargs)
        hf_model = AutoModelForCausalLM.from_pretrained(**lm_kwargs)
        model = LanguageModel(hf_model, hf_tokenizer)
        model.from_pretrained_kwargs = lm_kwargs, tok_kwargs
        return model

    @staticmethod
    def from_pretrained_name(name : str, device_map=None) -> LanguageModel:
        if name == 'random':
            return RandomLM(AutoTokenizer.from_pretrained('gpt2'))
        if device_map is None:
            device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
        lm_dict = {
            'pretrained_model_name_or_path' : name,
            'device_map' : device_map,
            'torch_dtype' : torch.bfloat16
        }
        tok_dict = {
            'pretrained_model_name_or_path' : name
        }
        return LanguageModel.from_pretrained(lm_dict, tok_dict)


    def batch_forward(self, input_ids : Tensor, attention_mask : Tensor = None, batch_size : int = None, streaming=None) -> CausalLMOutputWithCrossAttentions | Generator[Tensor| tuple[CausalLMOutputWithCrossAttentions, Tensor, Tensor], None, None]:
        """Batched inference given input tokens and attention mask.

        Args:
            input_ids (Tensor): Input tokens tensor of dimension (N,S)
            attention_mask (Tensor, optional): Attention mask tensor of dimension (N,S). Defaults to None.
            batch_size (int, optional): Batch size. Defaults to None in which case this function acts as LanguageModel.forward.
            streaming (bool, optional): If not None, this function becomes a generator that yield results as they are ready. Defaults to None.
            The possible values are:
                - "logits" : Yield logits only
                - "all" : Yield logits, input_ids, and attention_mask
        Returns:
            CausalLMOutputWithCrossAttentions : contains potentially logits, loss, hidden_states, etc.
        """
        if streaming is not None:
            return self._batch_forward_gen(input_ids, attention_mask, batch_size, streaming)
        batch_size = input_ids.shape[0] if batch_size is None else batch_size
        results = []
        for i in range(0, input_ids.shape[0], batch_size):
            batch = input_ids[i:i+batch_size]
            att_batch = attention_mask[i:i+batch_size] if attention_mask is not None else None
            out = self.forward(input_ids=batch, attention_mask=att_batch)
            results.append(out)
        keys = [k for k,x in results[0].items() if isinstance(x, torch.Tensor)]
        result = {key : torch.cat([res[key] for res in results]) for key in keys}
        return CausalLMOutputWithCrossAttentions(**result)
    
    def _batch_forward_gen(self, input_ids : Tensor, attention_mask : Tensor = None, batch_size : int = None, streaming=None) -> \
    Generator[Tensor| tuple[tuple[int,int], CausalLMOutputWithCrossAttentions, Tensor, Tensor], None, None]:
        batch_size = input_ids.shape[0] if batch_size is None else batch_size
        for i in range(0, input_ids.shape[0], batch_size):
            st,end = i, i+batch_size
            batch = input_ids[st:end].to(self.device)
            att_batch = attention_mask[st:end].to(self.device) if attention_mask is not None else None
            out = self.forward(input_ids=batch, attention_mask=att_batch)
            if streaming == 'all':
                yield (st,end), out, batch, att_batch
            elif streaming == 'logits':
                yield out.logits

    def credibility(self, input_ids : Tensor, cred_func : CredibilityFunction, 
                    attention_mask : Tensor = None, compute_on : Tensor = None, batch_size : int = None) -> Tensor:
        """Compute the credibility of a series of texts using a specific credibility function.

        Args:
            input_ids (Tensor): Input tokens of texts. It is a tensor of dimension (N,S)
            cred_func (CredibilityFunction): A credibility function
            attention_mask (Tensor, optional): Attention mask tensor of dimension (N,S). Defaults to None.
            batch_size (int, optional): Split input_ids and attention_mask in batches of size batch_size. Defaults to None in which case this function does no batch split.

        Returns:
            Tensor: A unidimensional tensor of size input_ids.shape[0]
        """
        creds = []
        for (st, end), out, input_ids, attention_mask in self.batch_forward(input_ids, attention_mask, batch_size, streaming="all"):
            mask = attention_mask if compute_on is None else compute_on[st:end].to(input_ids.device)
            cred = cred_func.compute(out.logits, input_ids, mask)
            creds.append(cred)
        return torch.cat(creds)

    def credibility_text(self, texts : tuple[str], cred_func : CredibilityFunction, compute_on : list[str] = None, batch_size : int = None) -> Tensor:
        """Compute the credibility of a series of texts using a specific credibility function.

        Args:
            texts (Iterable[str]): Iterable of texts
            cred_func (CredibilityFunction): A credibility function
            batch_size (int, optional): Split input_ids and attention_mask in batches of size batch_size. Defaults to None in which case this function does no batch split.

        Returns:
            Tensor: A unidimensional tensor of size len(texts)
        """ 
        t = self.tokenize(texts)
        inputs_ids, attention_mask = t['input_ids'], t['attention_mask']
        res = None
        if compute_on is not None:
            compute_on = self.hf_tokenizer(compute_on)['input_ids']
            res = torch.zeros_like(inputs_ids, dtype=torch.bool)
            for i, (input_ids, c) in enumerate(zip(inputs_ids, compute_on)):
                n = len(c)
                c = torch.tensor(c, dtype=input_ids.dtype, device=input_ids.device)
                for j in range(len(input_ids)):
                    if (input_ids[j:j+n] == c).all():
                        res[i, j:j+n] = True
                        break
                else:
                    raise ValueError('Subsequence %s not found in %s' % (c, input_ids))

        
        return self.credibility(inputs_ids, cred_func, attention_mask, res, batch_size)
    
    def batch_forward_text(self, texts : Iterable[str], batch_size : int = None, streaming=None) -> CausalLMOutputWithCrossAttentions:
        """Batched inference given texts.

        Args:
            texts (Iterable[str]): Iterable of texts
            batch_size (int, optional): Split texts in batches of size batch_size. Defaults to None in which case this function does no batch split.
            streaming (bool, optional): See batch_forward documentation

        Returns:
            CausalLMOutputWithCrossAttentions : contains potentially logits, loss, hidden_states, etc.
        """
        
        t = self.tokenize(texts)
        inputs_ids, attention_mask = t['input_ids'], t['attention_mask']
        return self.batch_forward(inputs_ids, attention_mask, batch_size, streaming)
    
    def tokenize(self, texts : Iterable[str]) -> BatchEncoding:
        """Tokenize the given texts using the HuggingFace's tokenizer

        Args:
            texts (Iterable[str]): Iterable of texts to tokenize

        Returns:
            BatchEncoding: Tokenizer output (input_ids, attention_mask, etc.)
        """
        return self.hf_tokenizer(texts, return_tensors='pt', padding=True)
    
    def __repr__(self) -> str:
        return "%s(model=%s, tokenizer=%s)" % (self.__class__.__name__, self.lm_name, self.tok_name)
    
    @property
    def lm_name(self) -> str:
        """Language model name

        Returns:
            str: name
        """
        return self.hf_model.config.name_or_path

    
    @property 
    def tok_name(self) -> str:
        """Tokenizer name

        Returns:
            str: name
        """
        return self.hf_tokenizer.name_or_path
    
    def _to_json(self) -> dict | list:
        assert self.from_pretrained_kwargs is not None, "Saving models loaded without loading the model using from_pretrained function is not supported"
        return self.from_pretrained_kwargs, tuple(x._identifier() for x in self.lm_edits_post_loading)

    def _identifier(self) -> dict | list:
        return self._to_json()
    
    @classmethod
    def _from_json(cls: type[SameType], d: dict | list) -> SameType:
        return cls.from_pretrained(*d)


class RandomLM(LanguageModel):
    """A random language model that assigns to each token in a given text a uniform probability of 1/|V| where V is the vocabulary set.
    """
    def __init__(self, hf_tokenizer: PreTrainedTokenizer, precision: Precision = None, exact=False) -> None:
        if precision is None:
            self.precision = Precision.FLOAT32
        else:
            self.precision = precision
        a = torch.nn.Module()
        a.register_parameter('dummy', torch.nn.Parameter(torch.scalar_tensor(0, dtype=self.precision.to_torch_dtype())))
        a.register_parameter('vocab_size', torch.nn.Parameter(torch.scalar_tensor(hf_tokenizer.vocab_size, dtype=torch.int32), requires_grad=False))
        super().__init__(a, hf_tokenizer, precision)
        self.exact = exact
        # self.hf_tokenizer = hf_tokenizer
        # self.hf_model = None
        # self.device = 'cpu'
        # self.from_pretrained_kwargs = None
        # self.lm_edits_post_loading = []
            
    def forward(self, *args, **kwargs) -> CausalLMOutputWithCrossAttentions:
        ids = kwargs.get('input_ids')
        if ids is None:
            ids = args[0]
        f = torch.zeros if self.exact else torch.randn
        logits = f((ids.shape[0], ids.shape[1], len(self.hf_tokenizer.vocab)), dtype=self.precision.to_torch_dtype())
        return CausalLMOutputWithCrossAttentions(logits=logits)
            
    @staticmethod
    def from_pretrained(tok_kwargs : dict) -> RandomLM:
        hf_tokenizer = AutoTokenizer.from_pretrained(**tok_kwargs)
        model = RandomLM(hf_tokenizer, None)
        model.from_pretrained_kwargs = (tok_kwargs,)
        return model
    
    @property
    def lm_name(self) -> str:
        return 'random'

class PromptedLanguageModel(LanguageModel):
    def __init__(self, lm : LanguageModel, prompt : str) -> None:
        super().__init__(lm.hf_model, lm.hf_tokenizer, lm.precision)
        self._lm = lm
        self.prompt = prompt
        self.prompt_tokens = self.hf_tokenizer(prompt, return_tensors='pt').input_ids
    
    def forward(self, *args, **kwargs) -> CausalLMOutputWithCrossAttentions:
        bound_arguments = inspect.signature(GPT2LMHeadModel.forward).bind(self, *args, **kwargs)
        # bound_arguments.apply_defaults()
        vars = bound_arguments.arguments
        vars.pop('self')
        input_ids = vars.pop('input_ids')
        attention_mask = vars.pop('attention_mask', None)

        l = _decode_and_add_prompt(input_ids, attention_mask, self.hf_tokenizer, self.prompt)
        org_seq_len = input_ids.shape[1]
        enc = self.hf_tokenizer(l, padding=True, return_tensors='pt')

        input_ids, attention_mask = enc.input_ids.to(self.device), enc.attention_mask.to(self.device)
        cur_seq_len = input_ids.shape[1]
        if cur_seq_len < org_seq_len:
            attention_mask = torch.cat([attention_mask, torch.zeros((attention_mask.shape[0], max(0, org_seq_len - cur_seq_len)), 
            dtype=attention_mask.dtype, layout=attention_mask.layout, device=attention_mask.device)], axis=1)
            input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], max(0, org_seq_len - cur_seq_len)), self.hf_tokenizer.eos_token_id,
            dtype=input_ids.dtype, layout=input_ids.layout, device=input_ids.device)], axis=1)
        return self._lm.forward(input_ids=input_ids, attention_mask=attention_mask, **vars)


def _decode_and_add_prompt(input_ids : Tensor, attention_mask : Tensor, tokenizer : PreTrainedTokenizer, prompt : str) -> list[str]:
    if attention_mask is not None:
        input_ids = [x[:i] for i,x in zip(attention_mask.sum(-1), input_ids)]
    starts_with_bos = [x[0] == tokenizer.bos_token for x in input_ids]
    input_ids = [(x[1:] if starts_with_bos[i] else x) for i, x in enumerate(input_ids)]
    
    l = [tokenizer.decode(x) for x in input_ids]
    
    for i in range(len(l)):
        l[i] = prompt + l[i]
        if starts_with_bos[i]:
            l[i] = tokenizer.bos_token + l[i]
    return l

cache_cred_returns = create_switch_contextmanager(LanguageModel, "enable_cache_credibility_text")

@dataclass
class CredReturn(Mongoable):
    lm_config : tuple[dict]
    text : str
    value : float
    cred_func : str
    
    def _identifier(self) -> dict | list:
        return {
            'lm_config' : self.lm_config,
            'text' : self.text,
            'cred_func' : self.cred_func
        }

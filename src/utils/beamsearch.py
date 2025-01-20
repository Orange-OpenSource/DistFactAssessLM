# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
import itertools
from typing import Iterable
import torch
from transformers import PreTrainedTokenizer

from multi_choices_parser import MultiChoicesParser, end_symb
from functools import lru_cache

class TokenLimiter(ABC):
    @abstractmethod
    def step(self, token : int) -> None:
        """Inject one token into the limiter which influences the state of the limiter

        Args:
            token (int): Token ID from the vocabulary of the tokenizer
        """
        pass
    
    @abstractmethod
    def authorized_tokens(self) -> list[int]:
        """Get the list authorized of authorized tokens for the next step 
        """
        pass

    @abstractmethod
    def copy(self) -> "TokenLimiter":
        """Return a stateful copy of this limiter.
        """
        pass

    @abstractmethod
    def is_at_initial_state(self) -> bool:
        """Is the limiter at its initial state?
        """
        pass

class TokenLimitersCombinator(TokenLimiter):
    def __init__(self, token_limiters : list[TokenLimiter]) -> None:
        super().__init__()
        self.token_limiters = tuple(token_limiters)

    def step(self, token: int) -> None:
        for tl in self.token_limiters:
            tl.step(token)

    @lru_cache(maxsize=4096)
    def authorized_tokens(self) -> list[int]:
        return list(set(tok for token_limiter in self.token_limiters for tok in token_limiter.authorized_tokens()))

    def copy(self) -> TokenLimiter:
        return TokenLimitersCombinator(tuple(tl.copy() for tl in self.token_limiters))

    def is_at_initial_state(self) -> bool:
        return all(tl.is_at_initial_state() for tl in self.token_limiters)
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TokenLimitersCombinator):
            return False
        return set(self.token_limiters) == set(value.token_limiters)
    
    def __hash__(self) -> int:
        return hash(self.token_limiters)



class MultiChoicesLimiter(TokenLimiter):
    def __init__(self, tokens : list[list[int]], eos_token_id : int) -> None:
        super().__init__()
        self.parser = MultiChoicesParser([tokens])
        self.eos_token_id = eos_token_id

    def step(self, token: int) -> None:
        if token == self.eos_token_id:
            token = (end_symb, )
        self.parser.step(token)

    @lru_cache(maxsize=4096)
    def authorized_tokens(self) -> list[int]:
        return [x[0] if x is not end_symb else self.eos_token_id for x in self.parser.next()]

    def copy(self) -> TokenLimiter:
        cp = copy(self)
        cp.parser = cp.parser.copy()
        return cp

    def is_at_initial_state(self) -> bool:
        return self.parser.where_am_i is self.parser.tree
    

    # Defining __eq__ and __hash__ for LRU cache for authorized_tokens method
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MultiChoicesLimiter):
            return False
        return self.eos_token_id == value.eos_token_id \
            and self.parser == value.parser
    
    def __hash__(self) -> int:
        return sum(hash(x) for x in (self.eos_token_id, self.parser))

class DoesNothingLimiter(TokenLimiter):
    def step(self, token: int) -> None:
        return 

    def authorized_tokens(self) -> list[int]:
        return slice(None, None, None)

    def copy(self) -> TokenLimiter:
        return self

    def is_at_initial_state(self) -> bool:
        return True

def select_mask_list(l : list, mask : Iterable[bool]) -> list:
    """Select a portion of a list using a boolean mask"""
    return [x for x,b in zip(l, mask) if b]

def select_index_list(l : list, index : Iterable[int]) -> list:
    """Select a portion of a list using a boolean mask"""
    return [l[idx] for idx in index]

class ListTokensLimiter(TokenLimiter):
    def __init__(self, list_tokens : list[int]) -> None:
        super().__init__()
        self.list_tokens = list_tokens

    def step(self, token: int) -> None:
        return 

    def authorized_tokens(self) -> list[int]:
        return self.list_tokens

    def copy(self) -> TokenLimiter:
        return self

    def is_at_initial_state(self) -> bool:
        return True
    

def compute_scores(previous_scores : torch.Tensor, new_log_probs : torch.Tensor, 
                   score_fn : str, num_tokens_produced : int):
    if score_fn == 'logprob':
        new_scores = previous_scores + new_log_probs
    elif score_fn == 'perplexity':
        if num_tokens_produced == 0:
            logprobs = new_log_probs
        else:
            logprobs = torch.log(previous_scores)*-num_tokens_produced + new_log_probs
        new_scores = torch.exp(-1 / (num_tokens_produced+1) * logprobs)
    return new_scores

def enforce_token_limiter(token_limiters : list[TokenLimiter], log_probs : torch.Tensor):
    for token_limiter, log_prob in zip(token_limiters, log_probs):
        mask = torch.ones_like(log_prob, dtype=torch.bool)
        mask[token_limiter.authorized_tokens()] = False
        log_prob[mask] = -torch.inf

def batched_inference_for_next_token_probs(hf_model, input_ids : torch.LongTensor, batch_size : int) -> torch.Tensor:
    log_probs = []
    for i in range(0, input_ids.shape[0], batch_size):
        inp = input_ids[i:i+batch_size]
        try:
            out = torch.log_softmax(hf_model(inp).logits[:,-1,:], dim=-1)
        except NotImplementedError:
            # Special case for random LM
            vocab_size = hf_model.state_dict()['vocab_size'].item()
            dtype = hf_model.state_dict()['dummy'].dtype
            out = torch.randn((inp.shape[0], vocab_size), device=input_ids.device, dtype=dtype)
        log_probs.append(out)
    return torch.cat(log_probs, dim=0)

@torch.no_grad()
def beam_search(hf_model, input_ids : torch.LongTensor, beam_width : int, max_new_tokens : int, eos_token_id: int, score_fn='logprob', token_limiter : TokenLimiter = None, batch_size=32):
    assert len(input_ids.shape) == 2 and input_ids.shape[0] == 1, "Requirement: input_ids.shape == (1,S) for some integer S"
    finished = torch.tensor([False], device=input_ids.device)
    unfinished = ~finished
    initial_input_length = input_ids.shape[1]
    scores = torch.tensor([0.0], device=input_ids.device)
    token_limiter = DoesNothingLimiter() if token_limiter is None else token_limiter
    assert token_limiter.is_at_initial_state(), "Requirement: token limiter must be at its initial state"
    token_limiters = [token_limiter]

    # past_key_values = None
    num_new_tok = 0

    while unfinished.any() and num_new_tok < max_new_tokens:
        # NOTE: Finished hypotheses are guaranteed to be at the start of input_ids (if they exist) ... (1)
        # Take finished inputs only
        inputs_unfinished = input_ids[unfinished]
        num_finished = finished.sum().item()

        # Inference (TODO: Accelerate inference using past_key_values)
        log_probs_next_tok_unfinished = batched_inference_for_next_token_probs(hf_model, inputs_unfinished, batch_size)

        token_limiters_unfinished = select_mask_list(token_limiters, unfinished)
        enforce_token_limiter(token_limiters_unfinished, log_probs_next_tok_unfinished)
        
        # top-tokens computation (this step needs (1) to work properly)
        probs_unfinished, top_tokens_unfinished = log_probs_next_tok_unfinished.topk(beam_width, dim=-1)
        probs_unfinished, top_tokens_unfinished = probs_unfinished.view(-1), top_tokens_unfinished.view(-1)
        idx_in = torch.arange(num_finished, num_finished + inputs_unfinished.shape[0], 
                              device=inputs_unfinished.device).repeat_interleave(beam_width)

        # Compute new scores of each hypotheses
        pre_scores_unfinished = compute_scores(scores[idx_in], probs_unfinished, score_fn, num_new_tok)
        pre_scores = torch.cat([scores[finished], pre_scores_unfinished])

        # Keep the best hypotheses (input_ids, scores, token_limiters)
        _, top_idx = pre_scores.topk(beam_width)
        top_idx = top_idx[~torch.isinf(pre_scores[top_idx])] # Remove -inf values
        scores = pre_scores[top_idx]
        input_ids = torch.cat([input_ids[finished], input_ids[idx_in]])[top_idx]
        token_limiters = select_index_list(token_limiters[:num_finished] + 
                                           [token_limiters[idx].copy() for idx in idx_in],
                                           top_idx)

        # Append best tokens and EOS for finished hypotheses + update token limiters
        eos = torch.full((num_finished, ), eos_token_id, device=input_ids.device)
        best_tokens = torch.cat([eos, top_tokens_unfinished])[top_idx]
        input_ids = torch.cat([input_ids, best_tokens.unsqueeze(-1)], dim=1)
        for tok, token_limiter in zip(best_tokens, token_limiters):
            token_limiter.step(tok.item())

        # Put finished hypotheses at the begining
        finished = input_ids[:,-1] == eos_token_id
        unfinished = ~finished
        input_ids = torch.cat([input_ids[finished], input_ids[unfinished]])
        scores = torch.cat([scores[finished], scores[unfinished]])
        token_limiters = select_mask_list(token_limiters, finished) + select_mask_list(token_limiters, unfinished)
        finished = input_ids[:,-1] == eos_token_id
        unfinished = ~finished

        num_new_tok += 1
    
    # Order best sequences by score
    best_sequences = input_ids[:, initial_input_length:]
    sort_idx = torch.argsort(scores, descending=True)
    scores = scores[sort_idx]
    best_sequences = best_sequences[sort_idx]
    return best_sequences, scores



if __name__ == '__main__':
    from lm.core import LanguageModel
    
    # Example usage:
    lm = LanguageModel.from_pretrained_name('gpt2')
    
    prompt = "the capital of France is"
    input_ids = lm.hf_tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # token_limiter = ListTokensLimiter([x.item() for x in lm.tokenize([' the beautiful city of Paris']).input_ids[0]])
    token_limiter = None

    best_sequences, best_scores = beam_search(lm.hf_model, input_ids, beam_width=3, max_new_tokens=10, eos_token_id=lm.hf_tokenizer.eos_token_id, score_fn='logprob', token_limiter=token_limiter)
    for seq, score in zip(best_sequences, best_scores):
        print("the capital of France is%s" % lm.hf_tokenizer.decode(seq))
        print('Score =', score)

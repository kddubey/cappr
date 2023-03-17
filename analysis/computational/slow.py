'''
Slow mirror of `fast`.

**(currently) USED ONLY FOR TESTING AND BENCHMARKING PURPOSES**.
Every function here has the same specification as the identically-named one in
`fast` if it exists.
'''
from __future__ import annotations
from typing import Mapping, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from lm_classification import classify
from lm_classification.utils import batch

import utils


def _keys_values_prompts(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                         prompts: Sequence[str],
                         num_completions_per_prompt: Union[int, Sequence[int]]):
    '''
    Only used for testing `fast._keys_values_prompts`.
    '''
    if not tokenizer.padding_side == 'right':
        raise ValueError('Gotta use right padding to ensure position IDs are '
                         'correct.')
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError('prompts must be a Sequence of strings.')
    if isinstance(num_completions_per_prompt, Sequence):
        if not len(prompts) == len(num_completions_per_prompt):
            raise ValueError( 'If num_completions_per_prompt is a Sequence, '
                              'then it must be the same length as prompts. Got '
                             f'lengths {len(num_completions_per_prompt)}, '
                             f'{len(prompts)}.')
    if isinstance(num_completions_per_prompt, int):
        ## for code simplicity, just repeat it
        num_completions_per_prompt = [num_completions_per_prompt
                                      for _ in range(len(prompts))]
    prompts_repeated = [prompt
                        for prompt, num_repeats
                        in zip(prompts, num_completions_per_prompt)
                        for _ in range(num_repeats)]
    encodings = tokenizer(prompts_repeated, return_tensors='pt',
                          padding=True).to(utils.DEVICE)
    with torch.no_grad():
        out = model(**encodings)

    offsets = encodings.attention_mask.sum(dim=1)

    ## Need (next-token) logits from prompts, i.e., last non-pad prompt token,
    ## since that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    last_nonpad_token_logits = (out.logits
                                .take_along_dim(_last_nonpad_token_idxs, dim=1))

    return out.past_key_values, encodings, offsets, last_nonpad_token_logits


@utils.cat_logits_encodings
@utils.batchify(batchable_arg='texts', push_up_arg='tokenizer',
                progress_bar_desc='logits (slow)')
def _logits_texts(model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                  texts: Sequence[str], batch_size: int=32
                 ) -> tuple[torch.Tensor, BatchEncoding]:
    encodings = (tokenizer(texts, return_tensors='pt', padding=True)
                 .to(utils.DEVICE))
    with torch.no_grad():
        out = model(**encodings)
    return out.logits, encodings


def _prompts_offsets(tokenizer: AutoTokenizer,
                     prompts: Sequence[str],
                     num_completions_per_prompt: Union[int, Sequence[int]]
                    ) -> torch.Tensor:
    if (not isinstance(num_completions_per_prompt, int) and
        not isinstance(num_completions_per_prompt, torch.Tensor)):
        num_completions_per_prompt = torch.tensor(num_completions_per_prompt)
    return (tokenizer(prompts, return_tensors='pt', padding=True)
            .attention_mask
            .repeat_interleave(num_completions_per_prompt, dim=0)
            .sum(dim=1)
            .to(utils.DEVICE))


def _logits_completions_given_prompts(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: Sequence[str],
        completions: Sequence[str],
        end_of_prompt: str=' ',
        batch_size: int=32):
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError('prompts must be a Sequence of strings.')
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError('completions must be a Sequence of strings.')
    texts = [prompt + end_of_prompt + completion
             for prompt in prompts
             for completion in completions]
    logits, encodings = _logits_texts(model, tokenizer, texts,
                                      batch_size=batch_size)
    ## Need these indices to slice completion tokens
    encodings['offsets'] = _prompts_offsets(tokenizer, prompts,
                               num_completions_per_prompt=len(completions))
    return logits, encodings


def _logits_completions_given_prompts_examples(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        examples: Sequence[classify.Example],
        batch_size: int=32):
    texts = [example.prompt + example.end_of_prompt + completion
             for example in examples
             for completion in example.completions]
    logits, encodings = _logits_texts(model, tokenizer, texts,
                                      batch_size=batch_size)
    ## Need these indices to slice completion tokens
    prompts = [example.prompt for example in examples]
    num_completions_per_prompt = [len(example.completions)
                                  for example in examples]
    encodings['offsets'] = _prompts_offsets(tokenizer, prompts,
                        num_completions_per_prompt=num_completions_per_prompt)
    return logits, encodings


def _logits_to_log_probs_completions(logits: torch.Tensor,
                                     encodings: Mapping[str, torch.Tensor]
    ) -> list[list[float]]:
    log_probs = utils.logits_to_log_probs(logits, encodings['input_ids'],
                                          input_ids_start_idx=1,
                                          logits_end_idx=-1)
    last_idx_non_pad = encodings['attention_mask'].sum(dim=1)
    ## i.e., # of tokens per text
    return [log_probs_prompt_completion[completion_start:completion_end]
            .tolist()
            for log_probs_prompt_completion, completion_start, completion_end
            in zip(log_probs, encodings['offsets']-1, last_idx_non_pad-1)]


def log_probs_conditional(prompts: Sequence[str],
                          completions: Sequence[str],
                          model_name: str,
                          end_of_prompt: str=' ',
                          batch_size: int=32):
    model, tokenizer = utils.load_model_and_tokenizer(model_name)
    logits, encodings = _logits_completions_given_prompts(
                            model, tokenizer, prompts, completions,
                            end_of_prompt=end_of_prompt, batch_size=batch_size)
    log_probs_completions = _logits_to_log_probs_completions(logits, encodings)
    return list(batch.constant(log_probs_completions, size=len(completions)))


def log_probs_conditional_examples(examples: Sequence[classify.Example],
                                   model_name: str, batch_size: int=32):
    model, tokenizer = utils.load_model_and_tokenizer(model_name)
    logits, encodings = _logits_completions_given_prompts_examples(
                            model, tokenizer, examples, batch_size=batch_size)
    log_probs_completions = _logits_to_log_probs_completions(logits, encodings)
    num_completions_per_prompt = [len(example.completions)
                                  for example in examples]
    return list(batch.variable(log_probs_completions,
                               sizes=num_completions_per_prompt))

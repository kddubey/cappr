'''
Perform prompt-completion classification using models from OpenAI's text
completion API.
'''
from __future__ import annotations
from typing import Sequence

import tiktoken

from lm_classification.utils import batch, classify, wrap
from lm_classification.example import Example
from lm_classification import openai


def token_logprobs(texts: Sequence[str], model: openai.api.Model,
                   ask_if_ok: bool=False) -> list[list[float]]:
    '''
    Returns a list `log_probs` where `log_probs[i]` is the value of
    `'log_probs' -> 'token_logprobs'` (from the OpenAI Completion endpoint) for
    `texts[i]` using `model`.
    '''
    choices = openai.api.gpt_complete(texts, ask_if_ok=ask_if_ok, model=model,
                                      ## rest must be hard-coded
                                      max_tokens=0, logprobs=1, echo=True)
    return [choice['logprobs']['token_logprobs'] for choice in choices]


def slice_completions(completions: Sequence[str],
                      log_probs: Sequence[Sequence[float]],
                      model: openai.api.Model) -> list[list[float]]:
    '''
    Returns a list `log_probs_completions` where `log_probs_completions[i]` is a
    list of conditional log-probablities for each token in `completions[i]`,
    extracted by slicing `log_probs[i]`.
    '''
    if len(completions) != len(log_probs):
        raise ValueError( 'Different number of completions and log_probs: '
                         f'{len(completions)}, {len(log_probs)}.')
    tokenizer = tiktoken.encoding_for_model(model)
    completion_lengths = [len(tokens)
                          for tokens in tokenizer.encode_batch(completions)]
    return [log_probs_text[-num_completion_tokens:]
            for num_completion_tokens, log_probs_text
            in zip(completion_lengths, log_probs)]


@wrap.add_doc_after(openai.docstrings.ASK_IF_OK)
@wrap.add_doc_before(classify.docstrings.LOG_PROBS_CONDITIONAL)
def log_probs_conditional(prompts: Sequence[str], completions: Sequence[str],
                          model: openai.api.Model, end_of_prompt: str=' ',
                          ask_if_ok: bool=False):
    ## str / non-Sequence[str] inputs silently, wastefully, and irreparably fail
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError('prompts must be a Sequence of strings.')
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError('completions must be a Sequence of strings.')
    ## Flat list of prompts and their completions. Will post-process
    texts = [prompt + end_of_prompt + completion
             for prompt in prompts
             for completion in completions]
    log_probs = token_logprobs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Since log_probs is a flat list, we'll need to batch them by the size and
    ## order of completions to fulfill the spec.
    return [slice_completions(completions, log_probs_batch, model)
            for log_probs_batch
            in batch.constant(log_probs, size=len(completions))]


@wrap.add_doc_after(openai.docstrings.ASK_IF_OK)
@wrap.add_doc_before(classify.docstrings.LOG_PROBS_CONDITIONAL_EXAMPLES)
def log_probs_conditional_examples(examples: Sequence[Example],
                                   model: openai.api.Model,
                                   ask_if_ok: bool=False
                                   ) -> list[list[list[float]]]:
    ## Flat list of prompts and their completions. Will post-process
    texts = [example.prompt + example.end_of_prompt + completion
             for example in examples
             for completion in example.completions]
    log_probs_all = token_logprobs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Flatten completions in same order as examples were flattened
    completions_all = [completion for example in examples
                       for completion in example.completions]
    log_probs_completions_all = slice_completions(completions_all,
                                                  log_probs_all, model)
    ## Batch by completions to fulfill the spec
    num_completions_per_prompt = [len(example.completions)
                                  for example in examples]
    return list(batch.variable(log_probs_completions_all,
                               sizes=num_completions_per_prompt))


@wrap.add_doc_after(openai.docstrings.ASK_IF_OK)
@classify.predict_proba
def predict_proba(prompts: Sequence[str], completions: Sequence[str],
                  model: openai.api.Model, end_of_prompt: str=' ',
                  ask_if_ok: bool=False):
    return log_probs_conditional(prompts, completions, model,
                                 end_of_prompt=end_of_prompt,
                                 ask_if_ok=ask_if_ok)


@wrap.add_doc_after(openai.docstrings.ASK_IF_OK)
@classify.predict_proba_examples
def predict_proba_examples(examples: Sequence[Example], model: openai.api.Model,
                           ask_if_ok: bool=False):
    return log_probs_conditional_examples(examples, model, ask_if_ok=ask_if_ok)

'''
Perform prompt-completion classification: for a given prompt and completion,
what's the probability that the completion follows the prompt?

Only supports LMs which you gotta pay for in
[OpenAI's text completion API](https://platform.openai.com/docs/models/gpt-3).
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import tiktoken

from lm_classification.utils import api, batch


_DUMMY_FLOATS = [0, -0.5, -1]
## for checking inputs which are supposed to be functions of this type of data


end_of_prompt = '\n\n###\n\n'
## https://platform.openai.com/docs/guides/fine-tuning/data-formatting


def gpt_log_probs(texts: Sequence[str], model: api.Model,
                  ask_if_ok: bool=False) -> list[list[float]]:
    '''
    Returns a list `log_probs` where `log_probs[i]` is the value of
    `'log_probs' -> 'token_logprobs'` (from the OpenAI Completion endpoint) for
    `texts[i]` using `model`.

    If `ask_if_ok`, then you'll be notified of the cost of this call, and then
    prompted to give the go-ahead.
    '''
    choices = api.gpt_complete(texts, ask_if_ok=ask_if_ok, model=model,
                               ## rest must be hard-coded
                               max_tokens=0, logprobs=1, echo=True)
    return [choice['logprobs']['token_logprobs'] for choice in choices]


def log_probs_completions(completions: Sequence[str],
                          log_probs: Sequence[Sequence[float]],
                          model: api.Model) -> list[list[float]]:
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


def log_probs_conditional(prompts: Sequence[str], completions: Sequence[str],
                          model: api.Model, end_of_prompt: str=' ',
                          ask_if_ok: bool=False):
    '''
    Returns a list `log_probs_completions` where `log_probs_completions[i][j]`
    is a list of the `model`'s estimates of log-probablities of each token in
    `completions[j]`, conditional on previous tokens in the completion and
    `prompts[i]`.

    If `ask_if_ok`, then you'll be notified of the cost of this call, and then
    prompted to give the go-ahead.
    '''
    ## str / non-Sequence[str] inputs silently, wastefully, and irreparably fail
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError('prompts must be a Sequence of strings.')
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError('completions must be a Sequence of strings.')
    ## Flat list of prompts and their completions. Will post-process
    texts = [prompt + end_of_prompt + completion
             for prompt in prompts
             for completion in completions]
    log_probs = gpt_log_probs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Since log_probs is a flat list, we'll need to batch them by the size and
    ## order of completions to fulfill the spec.
    return [log_probs_completions(completions, log_probs_batch, model)
            for log_probs_batch
            in batch.constant(log_probs, size=len(completions))]


def _check_prior(prior: Optional[Sequence[float]]=None):
    '''
    Raises an error if `prior` is not `None` or a 1-D `Sequence` which sums to
    1.
    '''
    if prior is None: ## it's a uniform prior, no need to check anything
        return None
    if not isinstance(prior, Sequence):
        raise TypeError('prior must be a Sequence.')
    if len(np.shape(prior)) != 1:
        raise ValueError('prior must be 1-D.')
    prior_arr = np.array(prior, dtype=float) ## try casting to float
    if not np.isclose(prior_arr.sum(), 1, rtol=0, atol=1e-6):
        raise ValueError('prior must sum to 1 (tol 1e-6).')


def _check_func(func: Callable[[Sequence[float]], float]):
    '''
    Raises an error is `func` is not a function of `Sequence[float]` or it
    returns `None`.
    '''
    try:
        out = func(_DUMMY_FLOATS)
    except Exception as e:
        raise ValueError( 'func is not a function of Sequence[float]. Got this '
                         f'error on input {_DUMMY_FLOATS}: {e}.')
    else:
        if out is None:
            raise ValueError( 'func must return a float. It returned None for '
                             f'this input: {_DUMMY_FLOATS}.')


@dataclass(frozen=True)
class Example:
    '''
    Represents a single test example for a prompt-completion text classification
    task. This data structure is useful when different examples from the same
    dataset may belong to different classes.
    This applies to, e.g., [COPA](https://people.ict.usc.edu/~gordon/copa.html).

    `prompt`: cointains the text to classify, perhaps with instructions

    `completions`: possible completions/answers to the `prompt`

    `prior`: (optional) a probability distribution over `completions`.

    `end_of_prompt`: (default: `' '`) the string used to join the `prompt` and
    each completion.
    '''
    prompt: str
    completions: Sequence[str]
    prior: Optional[Sequence[float]]=None
    end_of_prompt: str=' '

    def __post_init__(self):
        ## Check inputs here so that fxns of Example don't need to check
        if not isinstance(self.prompt, str):
            raise TypeError('prompt must be a string.')
        if (isinstance(self.completions, str) or
            not isinstance(self.completions, Sequence)):
            raise TypeError('completions must be a Sequence of strings.')
        _check_prior(self.prior)
        if self.prior is not None and len(self.completions) != len(self.prior):
            raise ValueError( 'completions and prior are different lengths: '
                             f'{len(self.completions)}, {len(self.prior)}.')


def log_probs_conditional_examples(examples: Sequence[Example],
                                   model: api.Model,
                                   ask_if_ok: bool=False,
                                  ) -> list[list[list[float]]]:
    '''
    Returns a list `log_probs_completions` where `log_probs_completions[i][j]`
    is a list of the `model`'s estimates of log-probablities of each token in
    `examples[i].completions[j]`, conditional on previous tokens in the
    completion and `examples[i].prompt`.

    If `ask_if_ok`, then you'll be notified of the cost of this call, and then
    prompted to give the go-ahead.
    '''
    ## Flat list of prompts and their completions. Will post-process
    texts = [example.prompt + example.end_of_prompt + completion
             for example in examples
             for completion in example.completions]
    log_probs_all = gpt_log_probs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Flatten completions in same order as examples were flattened
    completions_all = [completion for example in examples
                       for completion in example.completions]
    log_probs_completions_all = log_probs_completions(completions_all,
                                                      log_probs_all, model)
    ## Batch by completions to fulfill the spec
    completions_sizes = [len(example.completions) for example in examples]
    return list(batch.variable(log_probs_completions_all,
                               sizes=completions_sizes))


def agg_log_probs(log_probs: Sequence[Sequence[Sequence[float]]],
                  func: Callable[[Sequence[float]], float]=np.mean
                 ) -> list[list[float]]:
    '''
    Returns a list, `likelihoods`, where `likelihoods[i][j]` is
    `np.exp(func(log_probs[i][j]))`.
    '''
    ## TODO: any elegant way to vectorize? Problem is that `log_probs` can be
    ## ragged along the 2nd *and* 3rd dimensions.
    return [[np.exp(func(log_probs_class))
             for log_probs_class in log_probs_classes]
            for log_probs_classes in log_probs]


def posterior_prob(likelihoods: np.ndarray, axis: int,
                   prior: Optional[Sequence[float]]=None,
                   normalize: bool=True):
    '''
    Returns an array, `posteriors`, where `posteriors[i]` is the (normalized)
    probability distribution of `likelihoods[i] * prior`. If `prior is None`,
    then a uniform prior is applied, i.e., `posteriors[i]` is simply a
    (normalized) copy of `likelihoods[i]`.

    Set `axis` to the axis over which the distribution is defined, e.g., `0` if
    likelihoods is 1-D. 
    '''
    likelihoods = np.array(likelihoods)
    if prior is None:
        if normalize:
            return likelihoods/likelihoods.sum(axis=axis, keepdims=True)
        return likelihoods
    _check_prior(prior)
    posteriors_unnorm = likelihoods * prior
    if normalize:
        return posteriors_unnorm/posteriors_unnorm.sum(axis=axis, keepdims=True)
    return posteriors_unnorm


def predict_proba(prompts: Sequence[str], completions: Sequence[str],
                  model: api.Model, prior: Optional[Sequence[float]]=None,
                  end_of_prompt: str=' ',
                  func: Callable[[Sequence[float]], float]=np.mean,
                  ask_if_ok: bool=False):
    '''
    Returns an array with shape `(len(prompts), len(completions))` called
    `pred_probs`, where `pred_probs[i, j]` is a `model`'s estimate of the
    probability of `completions[j]` given `prompts[i] + end_of_prompt`.

    If `ask_if_ok`, then you'll be notified of the cost of this call, and then
    prompted to give the go-ahead.
    '''
    if prior is not None and len(completions) != len(prior):
        raise ValueError( 'completions and prior are different lengths: '
                         f'{len(completions)}, {len(prior)}.')
    ## Check prior and func here so that we don't hit the API for nothing
    _check_prior(prior)
    _check_func(func)
    log_probs_all = log_probs_conditional(prompts, completions,
                                          end_of_prompt=end_of_prompt,
                                          model=model, ask_if_ok=ask_if_ok)
    likelihoods = agg_log_probs(log_probs_all, func=func)
    ## If there's only 1 completion, normalizing will cause the prob to
    ## trivially be 1! So let's not normalize in that case, and hope the user
    ## knows what they're doing
    return posterior_prob(likelihoods, axis=1, prior=prior,
                          normalize=len(completions) > 1)


def predict_proba_examples(examples: Sequence[Example],
                           model: api.Model,
                           func: Callable[[Sequence[float]], float]=np.mean,
                           ask_if_ok: bool=False):
    '''
    Returns a list, `pred_probs`, where `pred_probs[i][j]` is a `model`'s
    estimate of the probability of `examples[i].completions[j]` given
    `examples[i].prompt + examples[i].end_of_prompt`.

    If the number of completions per example is a constant `k`, then an array
    with shape `(len(examples), k)` is returned instead.

    If `ask_if_ok`, then you'll be notified of the cost of this call, and then
    prompted to give the go-ahead.
    '''
    ## Check func here so that we don't hit the API for nothing
    _check_func(func)
    log_probs_all = log_probs_conditional_examples(examples, model=model,
                                                   ask_if_ok=ask_if_ok)
    likelihoods_all = agg_log_probs(log_probs_all, func=func)
    ## If an example has just 1 completion, normalizing will cause the prob to
    ## trivially be 1! So let's not normalize in that case, and hope the user
    ## knows what they're doing
    completions_sizes = [len(example.completions) for example in examples]
    should_normalize = [size > 1 for size in completions_sizes]
    pred_probs = [posterior_prob(likelihoods, axis=0, prior=example.prior,
                                 normalize=normalize)
                  for likelihoods, example, normalize
                  in zip(likelihoods_all, examples, should_normalize)]
    ## For convenience sake, convert to array if possible
    if len(set(completions_sizes)) == 1:
        return np.array(pred_probs)
    else:
        return pred_probs

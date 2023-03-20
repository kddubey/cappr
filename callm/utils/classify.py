"""
Transform conditional completion log-probabilites to a probability distribution
over completions.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from callm.example import Example
from callm.utils import check, wrap


class docstrings:
    """
    Docstrings common to functions which compute conditional completion
    log-probabilities.
    """

    LOG_PROBS_CONDITIONAL = """
    Returns a list `log_probs_completions` where `log_probs_completions[i][j]` is a list
    of the `model`'s estimates of log-probablities of each token in `completions[j]`,
    conditional on previous tokens in the completion and `prompts[i]`.
    """

    LOG_PROBS_CONDITIONAL_EXAMPLES = """
    Returns a list `log_probs_completions` where `log_probs_completions[i][j]` is a list
    of the `model`'s estimates of log-probablities of each token in
    `examples[i].completions[j]`, conditional on previous tokens in the completion and
    `examples[i].prompt`.
    """


def agg_log_probs(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> list[list[float]]:
    """
    Returns a list, `likelihoods`, where `likelihoods[i][j]` is
    `np.exp(func(log_probs[i][j]))`.
    """
    ## TODO: any elegant way to vectorize? Problem is that `log_probs` can be
    ## ragged along the 2nd *and* 3rd dimensions.
    return [
        [np.exp(func(log_probs_class)) for log_probs_class in log_probs_classes]
        for log_probs_classes in log_probs
    ]


def posterior_prob(
    likelihoods: np.ndarray,
    axis: int,
    prior: Optional[Sequence[float]] = None,
    normalize: bool = True,
):
    """
    Returns an array, `posteriors`, where `posteriors[i]` is the (normalized)
    probability distribution of `likelihoods[i] * prior`. If `prior is None`, then a
    uniform prior is applied, i.e., `posteriors[i]` is simply a (normalized) copy of
    `likelihoods[i]`.

    Set `axis` to the axis over which the distribution is defined, e.g., `0` if
    likelihoods is 1-D.
    """
    likelihoods = np.array(likelihoods)
    if prior is None:
        if normalize:
            return likelihoods / likelihoods.sum(axis=axis, keepdims=True)
        return likelihoods
    check.prior(prior)
    posteriors_unnorm = likelihoods * prior
    if normalize:
        return posteriors_unnorm / posteriors_unnorm.sum(axis=axis, keepdims=True)
    return posteriors_unnorm


def _check_func(func: Callable[[Sequence[float]], float]):
    """
    Raises an error is `func` is not a function of `Sequence[float]` or it returns
    `None`.
    """
    example_input = [0, -0.5, -1]
    if func is None:
        return np.mean
    try:
        out = func(example_input)
    except Exception as e:
        raise ValueError(
            "func is not a function of Sequence[float]. Got this "
            f"error on input {example_input}: {e}."
        )
    else:
        if out is None:
            raise ValueError(
                "func must return a float. It returned None for "
                f"this input: {example_input}."
            )
        return func


def predict_proba(conditional_func):
    """
    TODO: docstring
    """
    docstring = """
    Returns an array with shape `(len(prompts), len(completions))` called `pred_probs`,
    where `pred_probs[i, j]` is a `model`'s estimate of the probability of
    `completions[j]` given `prompts[i] + end_of_prompt`.
    """

    @wrap.add_doc_before(docstring)
    @wrap.wraps_but_keep_wrapper_return_ann(conditional_func)
    def wrapper(
        prompts, completions, model, prior=None, func=None, **kwargs
    ) -> npt.NDArray[np.floating]:
        func = _check_func(func)
        log_probs_completions = conditional_func(prompts, completions, model, **kwargs)
        likelihoods = agg_log_probs(log_probs_completions, func=func)
        ## If there's only 1 completion, normalizing will cause the prob to
        ## trivially be 1! So let's not normalize in that case, and hope the
        ## user knows what they're doing
        return posterior_prob(
            likelihoods, axis=1, prior=prior, normalize=len(completions) > 1
        )

    return wrapper


def predict_proba_examples(conditional_examples_func):
    """
    TODO: docstring
    """
    docstring = """
    Returns a list, `pred_probs`, where `pred_probs[i][j]` is a `model`'s estimate of
    the probability of `examples[i].completions[j]` given
    `examples[i].prompt + examples[i].end_of_prompt`.

    If the number of completions per example is a constant `k`, then an array with shape
    `(len(examples), k)` is returned instead.
    """

    @wrap.add_doc_before(docstring)
    @wrap.wraps_but_keep_wrapper_return_ann(conditional_examples_func)
    def wrapper(
        examples: Sequence[Example], model, func=None, **kwargs
    ) -> Union[list[list[float]], npt.NDArray[np.floating]]:
        func = _check_func(func)
        log_probs_completions = conditional_examples_func(examples, model, **kwargs)
        likelihoods_all = agg_log_probs(log_probs_completions, func=func)
        ## If an example has just 1 completion, normalizing will cause the prob
        ## to trivially be 1! So let's not normalize in that case, and hope the
        ## user knows what they're doing
        num_completions_per_prompt = [len(example.completions) for example in examples]
        should_normalize = [num > 1 for num in num_completions_per_prompt]
        pred_probs = [
            posterior_prob(
                likelihoods, axis=0, prior=example.prior, normalize=normalize
            )
            for likelihoods, example, normalize in zip(
                likelihoods_all, examples, should_normalize
            )
        ]
        ## For convenience sake, convert to array if possible
        if len(set(num_completions_per_prompt)) == 1:
            return np.array(pred_probs)
        else:
            return pred_probs

    return wrapper

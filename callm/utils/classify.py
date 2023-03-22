"""
Transform conditional completion log-probabilites to a probability distribution
over completions.
"""
from __future__ import annotations
from functools import wraps
from typing import Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from callm.example import Example
from callm.utils import check


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


def predict_proba(conditional_func):
    """
    TODO: docstring
    """

    @wraps(conditional_func)
    def wrapper(
        prompts: Sequence[str], completions: Sequence[str], *args, prior=None, **kwargs
    ) -> npt.NDArray[np.floating]:
        log_probs_completions = conditional_func(prompts, completions, *args, **kwargs)
        likelihoods = agg_log_probs(log_probs_completions)
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

    @wraps(conditional_examples_func)
    def wrapper(
        examples: Sequence[Example], *args, **kwargs
    ) -> Union[list[list[float]], npt.NDArray[np.floating]]:
        log_probs_completions = conditional_examples_func(examples, *args, **kwargs)
        likelihoods_all = agg_log_probs(log_probs_completions)
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

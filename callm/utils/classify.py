"""
Transform conditional completion token log-probabilites to a probability distribution
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
    return [
        [
            np.exp(func(log_probs_completion))
            for log_probs_completion in log_probs_completions
        ]
        for log_probs_completions in log_probs
    ]


def agg_log_probs_from_constant_completions(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> npt.NDArray[np.floating]:
    """
    Returns an array, `likelihoods`, where `likelihoods[i,j]` is
    `np.exp(func(log_probs[i][j]))`.

    `func` must take an `axis` keyword argument.

    This function is a fast version of `agg_log_probs`, but only applies to `log_probs`
    which are from a constant set of completions, i.e., `log_probs` is the output from a
    `log_probs_conditional` function, **NOT** from a `log_probs_condtional_examples`
    function.
    """
    num_completions_per_prompt = [
        len(log_probs_completions) for log_probs_completions in log_probs
    ]
    num_completions_per_prompt_set = set(num_completions_per_prompt)
    if not len(num_completions_per_prompt_set) == 1:
        raise ValueError(
            "log_probs does not have a constant number of completions, i.e., there are "
            "indices i, j such that len(log_probs[i]) != len(log_probs[j]). Please use "
            "the slower function, agg_log_probs, instead."
        )
    ## Say, e.g., we have 2 completions, ['a b', 'c d e'], and 2 prompts.
    ## Then log_probs looks like:
    ## [ [ [a1, b1],      (token log-probs for the completion 1 | prompt 1)
    #      [c1, d1, e1]], (token log-probs for the completion 2 | prompt 1)
    ##   [ [a2, b2],      (token log-probs for the completion 1 | prompt 2)
    ##     [c2, d2, e2]]  (token log-probs for the completion 2 | prompt 2)
    ## ]
    ## We can re-shape this "jagged" list as a list of (non-jagged) arrays:
    ## [ array([[a1, b1]],
    ##         [[a2, b2]]),
    ##   array([[c1, d1, e1]],
    ##         [[c2, d2, e2]])
    ## ]
    num_completions_per_prompt = list(num_completions_per_prompt_set)[0]
    array_list = [
        np.array(  ## raises jagged/inhomogeneous ValueError if non-constant # tokens
            [
                log_probs_completions[completion_idx]
                for log_probs_completions in log_probs
            ]
        )
        for completion_idx in range(num_completions_per_prompt)
    ]
    ## Now we can apply the vectorized function to each array in the list
    likelihoods: npt.NDArray[np.floating] = np.exp(
        np.array([func(array, axis=1) for array in array_list])
    )
    ## likelihoods looks like:
    ## array([[likelihood_a1b1,   likelihood_a2b2  ],
    ##        [likelihood_c1d1e1, likelihood_c2d2e2]
    ##       ])
    ## Transpose it to fulfill the spec
    return likelihoods.T


def posterior_prob(
    likelihoods: npt.ArrayLike[float],
    axis: int,
    prior: Optional[Union[Sequence[float], npt.ArrayLike[float]]] = None,
    normalize: Union[bool, Sequence[bool]] = True,
    check_prior: bool = True,
) -> npt.NDArray[np.floating]:
    """
    Returns an array, `posteriors`, where `posteriors[i]` is a (`normalize[i]`d)
    probability distribution computed as `likelihoods[i] * prior`. If `prior is None`,
    then a uniform prior is applied, i.e., `posteriors[i]` is simply a (`normalize[i]`d)
    copy of `likelihoods[i]`.

    Set `axis` to the axis over which the distribution is defined, e.g., `0` if
    likelihoods is 1-D.
    """
    ## Input checks and preprocessing
    likelihoods = np.array(likelihoods)  ## it should not be jagged/inhomogenous
    if not isinstance(normalize, Sequence):
        ## For code simplicity, just repeat it
        num_repeats = 1 if len(likelihoods.shape) == 1 else likelihoods.shape[0]
        ## If it's 1-D, there's only a single probability to normalize
        normalize = [normalize] * num_repeats
    elif len(normalize) != len(likelihoods):
        raise ValueError(
            "If normalize is a Sequence, it must have the same length as likelihoods. "
            f"Got {len(normalize)}, {len(likelihoods)}."
        )
    normalize = np.array(normalize, dtype=bool)
    if prior is not None and check_prior:
        check.prior(prior)

    ## Apply Bayes' rule, w/ optional normalization per row
    if prior is None:
        posteriors_unnorm = likelihoods
    else:
        posteriors_unnorm = likelihoods * prior
    marginals = posteriors_unnorm.sum(axis=axis, keepdims=True)
    marginals[~normalize] = 1  ## denominator of 1 <=> no normalization
    return posteriors_unnorm / marginals


def predict_proba(conditional_func):
    """
    TODO: docstring
    """

    @wraps(conditional_func)
    def wrapper(
        prompts: Sequence[str], completions: Sequence[str], *args, prior=None, **kwargs
    ) -> npt.NDArray[np.floating]:
        log_probs_completions = conditional_func(prompts, completions, *args, **kwargs)
        likelihoods = agg_log_probs_from_constant_completions(log_probs_completions)
        ## If there's only 1 completion, normalizing will cause the probability to
        ## trivially be 1! So let's not normalize in that case, and hope the user knows
        ## what they're doing
        normalize = len(completions) > 1
        return posterior_prob(likelihoods, axis=1, prior=prior, normalize=normalize)

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
        likelihoods = agg_log_probs(log_probs_completions)
        ## If an example has just 1 completion, normalizing will cause the probability
        ## to trivially be 1! So let's not normalize in that case, and hope the user
        ## knows what they're doing
        num_completions_per_prompt = [len(example.completions) for example in examples]
        normalize = [num > 1 for num in num_completions_per_prompt]
        num_completions_per_prompt_set = set(num_completions_per_prompt)
        if len(num_completions_per_prompt_set) != 1:
            ## Can't be vectorized :-(
            return [
                posterior_prob(
                    likelihoods_ex,
                    axis=0,
                    prior=example.prior,
                    normalize=normalize_ex,
                    check_prior=False,  ## already checked when constructing each example
                )
                for likelihoods_ex, example, normalize_ex in zip(
                    likelihoods, examples, normalize
                )
            ]
        ## Vectorize!
        if all([example.prior is None for example in examples]):
            prior = None
        else:
            ## For coding simplicity, just supply a non-None prior
            prior = np.array(
                [
                    example.prior
                    or [1 / len(example.completions)] * len(example.completions)
                    for example in examples
                ]
            )
        ## prior cannot be jagged b/c every example has the same # of completions
        return posterior_prob(
            likelihoods,
            axis=1,
            prior=prior,
            normalize=normalize,
            check_prior=False,  ## already checked when constructing each example
        )

    return wrapper

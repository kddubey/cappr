"""
Transform completion token log-probabilites to a probability distribution over
completions.
"""
from __future__ import annotations
from functools import wraps
from typing import Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from cappr.utils import _check


def _agg_log_probs(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> list[list[float]]:
    """
    Aggregate using a nested list comprehension.
    """
    return [
        [
            np.exp(func(log_probs_completion))
            for log_probs_completion in log_probs_completions
        ]
        for log_probs_completions in log_probs
    ]


def _agg_log_probs_from_constant_completions(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> npt.NDArray[np.floating]:
    """
    Aggregate using a vectorized numpy function `func`.
    """
    num_completions_per_prompt = [
        len(log_probs_completions) for log_probs_completions in log_probs
    ]
    num_completions_per_prompt_set = set(num_completions_per_prompt)
    if not len(num_completions_per_prompt_set) == 1:
        raise ValueError(
            "log_probs does not have a constant number of completions, i.e., there are "
            "indices i, j such that len(log_probs[i]) != len(log_probs[j])."
        )
    ## Say, e.g., we have 2 completions, ['a b', 'c d e'], and 2 prompts.
    ## Then log_probs looks like:
    ## [ [ [a1, b1],      (token log-probs for completion 1 | prompt 1)
    #      [c1, d1, e1]], (token log-probs for completion 2 | prompt 1)
    ##   [ [a2, b2],      (token log-probs for completion 1 | prompt 2)
    ##     [c2, d2, e2]]  (token log-probs for completion 2 | prompt 2)
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


def agg_log_probs(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> list[list[float]]:
    """
    Aggregate token log-probabilities along the last dimension into probabilities.

    Parameters
    ----------
    log_probs : Sequence[Sequence[Sequence[float]]]
        nested sequences where token log-probabilities are in the last dimension
    func : Callable[[Sequence[float]], float], optional
        function which aggregates a sequence of token log-probabilities into a single
        log-probability. If the function is vectorized, it must take an ``axis``
        argument. By default, `numpy.mean`

    Returns
    -------
    probs: list[list[float]]
        Lists of probabilities where::

            probs[i][j] = numpy.exp(func(log_probs[i][j]))
    """
    try:
        return _agg_log_probs_from_constant_completions(log_probs, func)
    except (
        ValueError,  ## log_probs is jagged
        TypeError,  ## func doesn't take an axis argument
    ):
        return _agg_log_probs(log_probs, func)


def posterior_prob(
    likelihoods: npt.ArrayLike[float],
    axis: int,
    prior: Optional[Union[Sequence[float], npt.ArrayLike[float]]] = None,
    normalize: Union[bool, Sequence[bool]] = True,
    check_prior: bool = True,
) -> npt.NDArray[np.floating]:
    """
    Compute posterior probabilities from likelihoods and a prior.

    Parameters
    ----------
    likelihoods : npt.ArrayLike[float]
        2-D array of probabilities of data given a hypothesis
    axis : int
        the axis along which the probability distribution should be defined, e.g.,
        `axis=0` if `likelihoods` is 1-D
    prior : Optional[Union[Sequence[float], npt.ArrayLike[float]]], optional
        a probability distribution over the `axis` of `likelihoods`, by default None
    normalize : Union[bool, Sequence[bool]], optional
        whether or not to return a normalized probability distribtution for each row, by
        default True (normalize all rows)
    check_prior : bool, optional
        whether or not to check that the `prior` is indeed a probability distribution,
        by default True

    Returns
    -------
    posterior_probs : npt.NDArray[np.floating]
        2-D array of probabilities of a hypothesis given data. Its shape is the same as
        `likelihood.shape`

    Raises
    ------
    ValueError
        if `normalize` is a sequence whose length is different than that of
        `likelihoods`
    """
    ## Input checks and preprocessing
    likelihoods = np.array(likelihoods)  ## it should not be jagged/inhomogenous
    if not isinstance(normalize, (Sequence, np.ndarray)):
        ## For code simplicity, just repeat it
        ## If likelihoods is 1-D, there's only a single probability distr to normalize
        num_repeats = 1 if len(likelihoods.shape) == 1 else likelihoods.shape[0]
        normalize = [normalize] * num_repeats
    elif len(normalize) != len(likelihoods):
        raise ValueError(
            "If normalize is a Sequence, it must have the same length as likelihoods. "
            f"Got {len(normalize)}, {len(likelihoods)}."
        )
    normalize = np.array(normalize, dtype=bool)
    if check_prior:
        _check.prior(prior)

    ## Apply Bayes' rule, w/ optional normalization per row
    if prior is None:
        posteriors_unnorm = likelihoods
    else:
        posteriors_unnorm = likelihoods * prior
    marginals = posteriors_unnorm.sum(axis=axis, keepdims=True)
    marginals[~normalize] = 1  ## denominator of 1 <=> no normalization
    return posteriors_unnorm / marginals


def _predict_proba(log_probs_conditional):
    """
    Decorator which converts a `log_probs_condtional` function call into a
    `predict_proba` call. The decorated `predict_proba` function should take a `prior`
    keyword argument.
    """

    @wraps(log_probs_conditional)
    def wrapper(
        prompts: Sequence[str], completions: Sequence[str], *args, **kwargs
    ) -> npt.NDArray[np.floating]:
        ## Before hitting any APIs ($$), let's check the prior
        prior = kwargs.get("prior", None)
        _check.prior(prior)
        if prior is not None and len(completions) != len(prior):
            raise ValueError(
                "completions and prior are different lengths: "
                f"{len(completions)}, {len(prior)}."
            )

        log_probs_completions = log_probs_conditional(
            prompts, completions, *args, **kwargs
        )
        likelihoods = agg_log_probs(log_probs_completions)
        ## If there's only 1 completion, normalizing will cause the probability to
        ## trivially be 1! So let's not normalize in that case, and hope the user knows
        ## what they're doing
        normalize = len(completions) > 1
        return posterior_prob(likelihoods, axis=1, prior=prior, normalize=normalize)

    return wrapper


def _predict_proba_examples(log_probs_conditional_examples):
    """
    Decorator which converts a `log_probs_conditional_examples` function call into a
    `predict_proba_examples` call.
    """

    @wraps(log_probs_conditional_examples)
    def wrapper(
        examples, *args, **kwargs
    ) -> Union[list[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
        log_probs_completions = log_probs_conditional_examples(
            examples, *args, **kwargs
        )
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
                    check_prior=False,  ## already checked during example construction
                )
                for likelihoods_ex, example, normalize_ex in zip(
                    likelihoods, examples, normalize
                )
            ]
        ## Vectorize!
        if all([example.prior is None for example in examples]):
            prior = None
        else:
            ## For coding simplicity, just supply a prior which is non-None *everywhere*
            ## It's the same shape as likelihoods
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
            check_prior=False,  ## already checked during example construction
        )

    return wrapper


def _predict(predict_proba_func):
    """
    Decorator which converts a `predict_proba` function call into a `predict` call.
    """

    @wraps(predict_proba_func)
    def wrapper(
        prompts: Sequence[str], completions: Sequence[str], *args, **kwargs
    ) -> list[str]:
        pred_probs: npt.NDArray = predict_proba_func(
            prompts, completions, *args, **kwargs
        )
        pred_class_idxs = pred_probs.argmax(axis=1)
        return [completions[pred_class_idx] for pred_class_idx in pred_class_idxs]

    return wrapper


def _predict_examples(predict_proba_examples_func):
    """
    Decorator which converts a `predict_proba_examples` function call into a
    `predict_examples` call.
    """

    @wraps(predict_proba_examples_func)
    def wrapper(examples, *args, **kwargs) -> list[str]:
        pred_probs: Union[
            list[list[float]], npt.NDArray[np.floating]
        ] = predict_proba_examples_func(examples, *args, **kwargs)
        ## If it's an array, we can call .argmax, which is faster
        try:
            pred_class_idxs = pred_probs.argmax(axis=1)
        except AttributeError:
            pred_class_idxs = [
                np.argmax(example_pred_probs) for example_pred_probs in pred_probs
            ]
        return [
            example.completions[pred_class_idx]
            for example, pred_class_idx in zip(examples, pred_class_idxs)
        ]

    return wrapper

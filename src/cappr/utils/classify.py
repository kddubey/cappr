"""
Transform completion token log-probabilites into a probability distribution over
completions.
"""
from __future__ import annotations
from functools import wraps
from inspect import getmodule
from typing import Callable, Optional, Sequence, Union
import warnings

import numpy as np
import numpy.typing as npt

from cappr.utils import _check


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
    # At this point, we've verified that the number of completions is constant. (numpy
    # will automatically check if the number of tokens is constant later. For
    # demonstration purposes, say that the number of tokens is also constant.)
    # Say, e.g., we have 2 completions, ['a b', 'c d e'], and 2 prompts.
    # Then log_probs looks like:
    # [ [ [a1, b1],      (token log-probs for completion 1 | prompt 1)
    #     [c1, d1, e1]], (token log-probs for completion 2 | prompt 1)
    #   [ [a2, b2],      (token log-probs for completion 1 | prompt 2)
    #     [c2, d2, e2]]  (token log-probs for completion 2 | prompt 2)
    # ]
    # We can re-shape this "jagged" list as a list of (non-jagged) arrays:
    # [ array([[a1, b1]],
    #         [[a2, b2]]),
    #   array([[c1, d1, e1]],
    #         [[c2, d2, e2]])
    # ]
    num_completions_per_prompt = list(num_completions_per_prompt_set)[0]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            category=np.VisibleDeprecationWarning,
            message="Creating an ndarray from ragged nested sequences",
        )
        # Intentionally raise an error if there are a non-constant # of tokens, because
        # vectorization is not possible in this case. In older versions of numpy, this
        # error is just a warning.
        try:
            array_list = [
                np.array(
                    [
                        log_probs_completions[completion_idx]
                        for log_probs_completions in log_probs
                    ]
                )
                for completion_idx in range(num_completions_per_prompt)
            ]
        except:
            raise ValueError(
                "log_probs has a constant # of completions, but there are a "
                "non-constant # of tokens."
            )
    # Now apply the vectorized function to each array in the list
    likelihoods: npt.NDArray[np.floating] = np.exp(
        [func(array, axis=1) for array in array_list]
    )
    # likelihoods looks like:
    # array([[likelihood_a1b1,   likelihood_a2b2  ],
    #        [likelihood_c1d1e1, likelihood_c2d2e2]
    #       ])
    # Transpose it to satisfy likelihoods[i][j] = exp(func(log_probs[i][j]))
    return likelihoods.T


def _agg_log_probs(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> list[list[float]]:
    """
    Aggregate using a slow, nested list comprehension.
    """
    return [
        [
            np.exp(func(log_probs_completion))
            for log_probs_completion in log_probs_completions
        ]
        for log_probs_completions in log_probs
    ]


def _is_sequence(object) -> bool:
    # Should catch most objects we care about: lists, tuples, arrays, tensors. No sets.
    try:
        len(object)
        object[0]
    except:
        return False
    else:
        return True


def _sequence_depth(sequence) -> int:
    """
    Like `len(np.shape(sequence))` but `sequence` can be any object.
    """
    if _is_sequence(sequence):
        return 1 + max(_sequence_depth(item) for item in sequence)
    else:
        return 0


def agg_log_probs(
    log_probs: Union[Sequence[Sequence[float]], Sequence[Sequence[Sequence[float]]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> Union[list[float], list[list[float]]]:
    """
    Aggregate token log-probabilities along the last dimension into probabilities.

    Parameters
    ----------
    log_probs : Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]
        nested sequences where token log-probabilities are in the last dimension. A 2-D
        sequence corresponds to inputting a single prompt string or
        :class:`cappr.Example` object. A 3-D sequence corresponds to inputting multiple
        prompt strings or :class:`cappr.Example` objects
    func : Callable[[Sequence[float]], float], optional
        function which aggregates a sequence of token log-probabilities into a single
        log-probability. If the function is vectorized, it must take an ``axis``
        argument. By default, `numpy.mean`

    Returns
    -------
    probs: list[float] | list[list[float]]
        If `log_probs` is 2-D, then `probs` is a list of probabilities where::

            probs[j] = exp(func(log_probs[j]))

        If `log_probs` is 3-D, then `probs` is a list of list of probabilities where::

            probs[i][j] = exp(func(log_probs[i][j]))
    """

    depth = _sequence_depth(log_probs[0]) + 1  # check the first element is sufficient
    if depth not in {2, 3}:
        raise ValueError(
            f"log_probs is expected to be 2-D or 3-D. Got {depth} dimensions."
        )
    if depth == 2:  # make it 3-D for no computational cost, some software eng benefit
        log_probs = [log_probs]

    try:
        likelihoods = _agg_log_probs_from_constant_completions(log_probs, func)
    except (
        ValueError,  # log_probs is jagged
        TypeError,  # func doesn't take an axis argument
    ):
        likelihoods = _agg_log_probs(log_probs, func)

    if depth == 2:  # it's a single prompt
        return likelihoods[0]
    return likelihoods


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
        1-D or 2-D array of probabilities of data given a hypothesis
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
        array of probabilities of a hypothesis given data. Its shape is the same as
        `likelihood.shape`

    Raises
    ------
    ValueError
        if `normalize` is a sequence whose length is different than that of
        `likelihoods`
    """
    # Input checks and preprocessing
    likelihoods = np.array(likelihoods)  # it should not be jagged/inhomogenous
    if not isinstance(normalize, (Sequence, np.ndarray)):
        # For code simplicity, just repeat it
        # If likelihoods is 1-D, there's only a single probability distr to normalize
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

    # Apply Bayes' rule, w/ optional normalization per row
    if prior is None:
        posteriors_unnorm = likelihoods
    else:
        posteriors_unnorm = likelihoods * prior
    marginals = posteriors_unnorm.sum(axis=axis, keepdims=True)
    marginals[~normalize] = 1  # denominator of 1 <=> no normalization
    return posteriors_unnorm / marginals


def _wrap_call_unwrap(
    type_indicating_singleness: type,
    input,
    log_probs_conditional_func: Callable,
    *args,
    **kwargs,
):
    """
    Handles single inputs for a `log_probs_conditional_func` call which only takes
    multiple inputs.
    """
    is_single_input = isinstance(input, type_indicating_singleness)
    if is_single_input:
        input = [input]  # wrap
    log_probs_completions = log_probs_conditional_func(input, *args, **kwargs)  # call
    if is_single_input:
        return log_probs_completions[0]  # unwrap
    return log_probs_completions


def _log_probs_conditional(log_probs_conditional):
    """
    Decorator which does basic input checking, and allows for `prompts` to be a single
    string for a `log_probs_conditional` function.
    """

    @wraps(log_probs_conditional)
    def wrapper(
        prompts: Union[str, Sequence[str]], completions: Sequence[str], *args, **kwargs
    ) -> list[list[list[float]]]:
        # TODO: check prompts is a str or non-empty sequence of strings
        # TODO: check completions is a non-empty sequence of strings
        return _wrap_call_unwrap(
            str, prompts, log_probs_conditional, completions, *args, **kwargs
        )

    return wrapper


def _log_probs_conditional_examples(log_probs_conditional_examples):
    """
    Decorator which does basic input checking, and allows for `examples` to be a single
    `Example` for a `log_probs_conditional_examples` function.
    """

    from cappr import Example

    @wraps(log_probs_conditional_examples)
    def wrapper(
        examples: Union[Example, Sequence[Example]], *args, **kwargs
    ) -> list[list[list[float]]]:
        # TODO: check examples is not an empty sequence
        return _wrap_call_unwrap(
            Example, examples, log_probs_conditional_examples, *args, **kwargs
        )

    return wrapper


def _discount(
    token_logprobs_func,
    completions: Sequence[str],
    log_probs_completions: Union[list[list[float]], list[list[list[float]]]],
    is_single_input: bool,
    *model_args,  # rest are kwarg-only, and must come from the og function's kwargs
    discount_completions: float,
    log_marginal_probs_completions: Optional[Sequence[Sequence[float]]] = None,
    **kwargs,
) -> list[list[list[float]]]:
    """
    Highly experimental feature: discount completion given prompt probabilities by
    completion probabilities. Useful when particular completions are getting
    over-predicted. Currently isn't used by `_examples` functions.
    """
    if not discount_completions:
        return log_probs_completions

    # log Pr(completion token i | completion token :i) for each completion
    if log_marginal_probs_completions is None:
        log_marginal_probs_completions = token_logprobs_func(
            completions, *model_args, **kwargs
        )
    for x in log_marginal_probs_completions:
        if x[0] is None:
            x[0] = 0  # no discount for the first token
    # pre-multiply by the discount amount
    log_marginal_probs_completions_discounted = [
        discount_completions * np.array(log_marginal_probs_completion)
        for log_marginal_probs_completion in log_marginal_probs_completions
    ]
    if is_single_input:
        return [
            np.array(log_probs_completions[completion_idx])
            + log_marginal_probs_completions_discounted[completion_idx]
            for completion_idx in range(len(completions))
        ]
    return [
        [
            np.array(log_probs_prompt_completions[completion_idx])
            + log_marginal_probs_completions_discounted[completion_idx]
            for completion_idx in range(len(completions))
        ]
        for log_probs_prompt_completions in log_probs_completions
    ]


def _predict_proba(log_probs_conditional):
    """
    Decorator which converts a `log_probs_condtional` function call into a
    `predict_proba` call.
    """

    @wraps(log_probs_conditional)
    def wrapper(
        prompts: Union[str, Sequence[str]], completions: Sequence[str], *args, **kwargs
    ) -> npt.NDArray[np.floating]:
        # Check inputs before making expensive model calls
        # Check the prior
        prior = kwargs.get("prior", None)
        _check.prior(prior)
        if prior is not None and len(completions) != len(prior):
            raise ValueError(
                "completions and prior are different lengths: "
                f"{len(completions)}, {len(prior)}."
            )

        # Check normalization
        normalize = kwargs.get("normalize", True)
        _check.normalize(completions, normalize)

        # Check inputs for discount feature
        discount_completions = kwargs.get("discount_completions", 0)
        log_marginal_probs_completions = kwargs.get(
            "log_marginal_probs_completions", None
        )
        if not discount_completions and (log_marginal_probs_completions is not None):
            raise ValueError(
                "log_marginal_probs_completions is set, but they will not be used "
                "because discount_completions was not set."
            )

        # Do the expensive computation
        log_probs_completions = log_probs_conditional(
            prompts, completions, *args, **kwargs
        )

        # Maybe apply discount
        is_single_input = isinstance(prompts, str)
        if discount_completions:
            log_probs_completions = _discount(
                getattr(getmodule(log_probs_conditional), "token_logprobs"),
                completions,
                log_probs_completions,
                is_single_input,
                *args,
                **kwargs,
            )

        # Aggregate probs
        likelihoods = agg_log_probs(log_probs_completions)
        axis = 0 if is_single_input else 1
        return posterior_prob(likelihoods, axis=axis, prior=prior, normalize=normalize)

    return wrapper


def _predict_proba_examples(log_probs_conditional_examples):
    """
    Decorator which converts a `log_probs_conditional_examples` function call into a
    `predict_proba_examples` call.
    """

    from cappr import Example  # done locally to avoid circular import lol

    @wraps(log_probs_conditional_examples)
    def wrapper(
        examples: Union[Example, Sequence[Example]], *args, **kwargs
    ) -> Union[npt.NDArray[np.floating], list[npt.NDArray[np.floating]]]:
        log_probs_completions = log_probs_conditional_examples(
            examples, *args, **kwargs
        )
        likelihoods = agg_log_probs(log_probs_completions)
        if isinstance(examples, Example):
            return posterior_prob(
                likelihoods,
                axis=0,
                prior=examples.prior,
                normalize=examples.normalize,
                check_prior=False,  # already checked during example construction
            )

        # Determine whether vectorization is possible for the posterior probability calc
        num_completions_per_prompt = [len(example.completions) for example in examples]
        normalize = [example.normalize for example in examples]
        num_completions_per_prompt_set = set(num_completions_per_prompt)
        if len(num_completions_per_prompt_set) != 1:
            # Can't be easily vectorized :-(
            return [
                posterior_prob(
                    likelihoods_ex,
                    axis=0,
                    prior=example.prior,
                    normalize=normalize_ex,
                    check_prior=False,  # already checked during example construction
                )
                for likelihoods_ex, example, normalize_ex in zip(
                    likelihoods, examples, normalize
                )
            ]
        # Vectorize!
        if all([example.prior is None for example in examples]):
            prior = None
        else:
            # For coding simplicity, just supply a prior which is non-None *everywhere*
            # It's the same shape as likelihoods
            num_completions_per_prompt = list(num_completions_per_prompt_set)[0]
            uniform_prior = [
                1 / num_completions_per_prompt
            ] * num_completions_per_prompt
            prior = np.array([example.prior or uniform_prior for example in examples])
        # prior cannot be jagged b/c every example has the same # of completions
        return posterior_prob(
            likelihoods,
            axis=1,
            prior=prior,
            normalize=normalize,
            check_prior=False,  # already checked during example construction
        )

    return wrapper


def _predict(predict_proba_func):
    """
    Decorator which converts a `predict_proba` function call into a `predict` call.
    """

    @wraps(predict_proba_func)
    def wrapper(
        prompts: Union[str, Sequence[str]], completions: Sequence[str], *args, **kwargs
    ) -> list[str]:
        if len(completions) == 1:
            raise ValueError(
                "completions only has one completion. predict will trivially return "
                "back this completion. Perhaps you meant to call predict_proba instead "
                "of predict."
            )
        pred_probs: npt.NDArray = predict_proba_func(
            prompts, completions, *args, **kwargs
        )
        num_dimensions = pred_probs.ndim
        if isinstance(prompts, str):
            # User convenience: prompts was a single string, so pred_probs is 1-D
            assert num_dimensions == 1
            return completions[pred_probs.argmax()]
        assert num_dimensions == 2
        pred_class_idxs = pred_probs.argmax(axis=1)
        return [completions[pred_class_idx] for pred_class_idx in pred_class_idxs]

    return wrapper


def _predict_examples(predict_proba_examples_func):
    """
    Decorator which converts a `predict_proba_examples` function call into a
    `predict_examples` call.
    """

    from cappr import Example  # done locally to avoid circular import lol

    @wraps(predict_proba_examples_func)
    def wrapper(
        examples: Union[Example, Sequence[Example]], *args, **kwargs
    ) -> list[str]:
        pred_probs: Union[
            npt.NDArray[np.floating], list[npt.NDArray[np.floating]]
        ] = predict_proba_examples_func(examples, *args, **kwargs)
        if isinstance(examples, Example):
            # User convenience: examples is a singleton
            assert pred_probs.ndim == 1  # double check
            pred_class_idx = pred_probs.argmax()
            return examples.completions[pred_class_idx]
        try:
            # If it's an array, we can call .argmax on the whole thing, which is faster
            pred_class_idxs = pred_probs.argmax(axis=1)
        except (
            AttributeError,  # no argmax attr
            TypeError,  # no axis kwarg
        ):
            pred_class_idxs = [
                example_pred_probs.argmax() for example_pred_probs in pred_probs
            ]
        return [
            example.completions[pred_class_idx]
            for example, pred_class_idx in zip(examples, pred_class_idxs)
        ]

    return wrapper

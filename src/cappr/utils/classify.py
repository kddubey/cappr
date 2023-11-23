"""
Transform completion token log-probabilites into a probability distribution over
completions
"""
from __future__ import annotations
from contextlib import contextmanager
from functools import wraps
from inspect import getmodule
from typing import Callable, Sequence
import warnings

import numpy as np
import numpy.typing as npt

from cappr.utils import _check


@contextmanager
def _setattr(obj, name: str, value):
    """
    In this context, `obj.name` is set to `value`.
    """
    hasattr_ = hasattr(obj, name)
    if hasattr_:
        value_prev = getattr(obj, name)
    try:
        if hasattr_:
            setattr(obj, name, value)
        yield
    finally:
        if hasattr_:
            setattr(obj, name, value_prev)


########################################################################################
############################## Aggregate log probabilities #############################
########################################################################################


def _agg_log_probs_vectorized(
    log_probs: Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[Sequence[float]]], Sequence[float]] = np.mean,
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
    # Say, e.g., we have 2 completions, ['a b', 'c d e'], and 2 prompts
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
        # If there are a non-constant # of tokens, vectorization is not possible. In
        # older versions of numpy, this error is just a warning. So catch the warning
        # and raise it as an error
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
        except (np.VisibleDeprecationWarning, ValueError):
            raise ValueError(
                "log_probs has a constant # of completions, but there are a "
                "non-constant # of tokens. Vectorization is not possible."
            )
    # Now apply the vectorized function to each array in the list
    likelihoods: npt.NDArray[np.floating] = np.exp(
        [func(array, axis=1) for array in array_list]
    )
    # likelihoods looks like:
    # array([[likelihood_a1b1,   likelihood_a2b2  ],
    #        [likelihood_c1d1e1, likelihood_c2d2e2]])
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


def _ndim(sequence) -> int:
    """
    Like `array.ndim` or `len(np.shape(sequence))` but `sequence` can be more jagged. It
    can't be nested inconsistently though, since we'll just check the first element.
    """

    def _is_sliceable(object) -> bool:
        try:
            object[0:0]
        except (TypeError, IndexError, ValueError):
            return False
        else:
            return True

    ndim = 0
    while _is_sliceable(sequence):
        ndim += 1
        sequence = sequence[0]
    return ndim


def agg_log_probs(
    log_probs: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]],
    func: Callable[[Sequence[float]], float] = np.mean,
) -> npt.NDArray[np.floating] | list[float] | list[list[float]]:
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
    probs: npt.NDArray[np.floating] | list[float] | list[list[float]]
        If `log_probs` is 2-D, then `probs` is an array or list of probabilities where::

            probs[j] = exp(func(log_probs[j]))

        If `log_probs` is 3-D, then `probs` is an array or a list of list of
        probabilities where::

            probs[i][j] = exp(func(log_probs[i][j]))

    Raises
    ------
    ValueError
        if `log_probs` is not 2-D or 3-D
    """
    # 1. Determine the dimensionality of log_probs. The aggregation computation assumes
    #    a 3-D input, so we'll wrap it if it's 2-D
    ndim = _ndim(log_probs)
    if ndim not in {2, 3}:
        raise ValueError(f"log_probs must be 2-D or 3-D. Got {ndim} dimensions.")
    log_probs = [log_probs] if ndim == 2 else log_probs

    # 2. Run the aggregation computation, vectorizing if possible
    try:
        likelihoods = _agg_log_probs_vectorized(log_probs, func)
    except (
        ValueError,  # log_probs is jagged
        TypeError,  # func doesn't take an axis argument
    ):
        likelihoods = _agg_log_probs(log_probs, func)

    # 3. If we wrapped it, unwrap it for user convenience
    return likelihoods[0] if ndim == 2 else likelihoods


def posterior_prob(
    likelihoods: npt.ArrayLike[float],
    axis: int,
    prior: Sequence[float] | None = None,
    normalize: bool | Sequence[bool] = True,
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
    prior : Sequence[float] | None, optional
        a probability distribution over the `axis` of `likelihoods`. By default, a
        uniform prior is used
    normalize : bool | Sequence[bool], optional
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
        prior = _check.prior(prior, expected_length=likelihoods.shape[axis])

    # Apply Bayes' rule, w/ optional normalization per row
    if prior is None:
        posteriors_unnorm = likelihoods
    else:
        posteriors_unnorm = likelihoods * prior
    marginals = posteriors_unnorm.sum(axis=axis, keepdims=True)
    marginals[~normalize] = 1  # denominator of 1 <=> no normalization
    return posteriors_unnorm / marginals


########################################################################################
############################ Decorators for classify modules ###########################
########################################################################################


def _wrap_call_unwrap(
    type_indicating_singleness: type, func: Callable, input, *args, **kwargs
):
    """
    Handles single inputs for a `func` call which only takes multiple inputs and returns
    multiple outputs.
    """
    is_single_input = isinstance(input, type_indicating_singleness)
    input = [input] if is_single_input else input
    output = func(input, *args, **kwargs)
    return output[0] if is_single_input else output


def _token_logprobs(token_log_probs):
    """
    Decorator which does input checking, and allows for `texts` to be a single string
    for a `token_log_probs` function.
    """

    @wraps(token_log_probs)
    def wrapper(
        texts: str | Sequence[str], *args, **kwargs
    ) -> list[float] | list[list[float]]:
        _check.nonempty_and_ordered(texts, variable_name="texts")
        if not isinstance(texts, str):
            texts = list(texts)  # 0-index
        _check.end_of_prompt(kwargs.get("end_of_prompt", " "))
        return _wrap_call_unwrap(str, token_log_probs, texts, *args, **kwargs)

    return wrapper


def _log_probs_conditional(log_probs_conditional):
    """
    Decorator which does input checking, and allows for `prompts` to be a single string
    for a `log_probs_conditional` function.
    """

    @wraps(log_probs_conditional)
    def wrapper(
        prompts: str | Sequence[str], completions: Sequence[str], *args, **kwargs
    ) -> list[list[float]] | list[list[list[float]]]:
        _check.nonempty_and_ordered(prompts, variable_name="prompts")
        _check.completions(completions)
        _check.end_of_prompt(kwargs.get("end_of_prompt", " "))
        return _wrap_call_unwrap(
            str, log_probs_conditional, prompts, completions, *args, **kwargs
        )

    return wrapper


def _log_probs_conditional_examples(log_probs_conditional_examples):
    """
    Decorator which does input checking, and allows for `examples` to be a single
    `Example` for a `log_probs_conditional_examples` function.
    """

    from cappr import Example

    @wraps(log_probs_conditional_examples)
    def wrapper(
        examples: Example | Sequence[Example], *args, **kwargs
    ) -> list[list[float]] | list[list[list[float]]]:
        if not isinstance(examples, Example):
            _check.nonempty_and_ordered(examples, variable_name="examples")
        return _wrap_call_unwrap(
            Example, log_probs_conditional_examples, examples, *args, **kwargs
        )

    return wrapper


def _discount(
    token_logprobs_func,
    completions: Sequence[str],
    log_probs_completions: list[list[float]] | list[list[list[float]]],
    is_single_input: bool,
    *model_args,  # rest are kwarg-only, and must come from the og function's kwargs
    discount_completions: float,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    **kwargs,
) -> list[npt.NDArray[np.floating]] | list[list[npt.NDArray[np.floating]]]:
    """
    Highly experimental feature: discount completion given prompt probabilities by
    completion probabilities. Useful when particular completions are getting
    over-predicted. Currently isn't used by `_examples` functions.
    """
    # log Pr(completion token i | completion tokens :i) for each completion
    if log_marg_probs_completions is None:
        log_marg_probs_completions = token_logprobs_func(
            completions, *model_args, **kwargs
        )
    for x in log_marg_probs_completions:
        if x[0] is None:
            x[0] = 0  # no discount for the first token

    # pre-multiply by the discount amount
    log_marg_probs_completions_discounted = [
        discount_completions * np.array(log_marginal_probs_completion)
        for log_marginal_probs_completion in log_marg_probs_completions
    ]

    # add discount
    if is_single_input:
        return [
            np.array(log_probs_completions[completion_idx])
            + log_marg_probs_completions_discounted[completion_idx]
            for completion_idx in range(len(completions))
        ]
    return [
        [
            np.array(log_probs_prompt_completions[completion_idx])
            + log_marg_probs_completions_discounted[completion_idx]
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
        prompts: str | Sequence[str], completions: Sequence[str], *args, **kwargs
    ) -> npt.NDArray[np.floating]:
        # Check inputs before making expensive model calls
        # Check the prior
        prior = kwargs.get("prior", None)
        prior = _check.prior(prior, expected_length=len(completions))

        # Check normalization
        normalize = kwargs.get("normalize", True)
        _check.normalize(normalize, completions)

        # Check inputs for discount feature
        discount_completions = kwargs.get("discount_completions", 0)
        log_marg_probs_completions = kwargs.get("log_marg_probs_completions", None)
        if not discount_completions and (log_marg_probs_completions is not None):
            raise TypeError(
                "log_marg_probs_completions is set, but they will not be used "
                "because discount_completions was not set."
            )

        # Do the expensive model calls
        log_probs_completions = log_probs_conditional(
            prompts, completions, *args, **kwargs
        )

        is_single_input = isinstance(prompts, str)

        # Maybe apply discount
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

    from cappr import Example

    @wraps(log_probs_conditional_examples)
    def wrapper(
        examples: Example | Sequence[Example], *args, **kwargs
    ) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
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
            num_completions_per_prompt = list(num_completions_per_prompt_set)[0]
            uniform_prior = [
                1 / num_completions_per_prompt
            ] * num_completions_per_prompt
            prior = np.array(
                [
                    uniform_prior if example.prior is None else example.prior
                    for example in examples
                ]
            )  # same shape as likelihoods: (len(examples), num_completions_per_prompt)
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
        prompts: str | Sequence[str], completions: Sequence[str], *args, **kwargs
    ) -> str | list[str]:
        if len(completions) == 1:
            raise ValueError(
                "completions only has one completion. predict will trivially return "
                "back this completion. Perhaps you meant to call predict_proba with "
                "normalize=False instead of predict."
            )
        pred_probs: npt.NDArray = predict_proba_func(
            prompts, completions, *args, **kwargs
        )
        # We need completions to support 0-indexed __getitem__
        completions = list(completions)
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

    from cappr import Example

    @wraps(predict_proba_examples_func)
    def wrapper(
        examples: Example | Sequence[Example], *args, **kwargs
    ) -> str | list[str]:
        pred_probs: npt.NDArray[np.floating] | list[
            npt.NDArray[np.floating]
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

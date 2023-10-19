"""
Shared input checks. These check conditions that would cause silent, difficult-to-debug,
or expensive errors.

For example:
    - inputting an unordered iterable will cause an output array of predicted
      probabilities to be meaningless
    - inputting an empty object can cause an obscure index error in some downstream
      model/tokenization functions
    - inputting an incorrectly structured prior will cause all of the model's compute to
      be a waste.
"""
from __future__ import annotations
from typing import Literal, Sequence

import numpy as np


def _is_reversible(object) -> bool:
    # Returns True for:
    # - list, tuple, dict keys, dict values
    # - numpy array, torch Tensor
    # - pandas and polars Series
    # - str (but other places in this package filter those out before getting here)
    # Returns False for:
    # - set
    # reversed(object) is often a generator, so checking this is often cheap.
    try:
        reversed(object)
    except TypeError:
        return False
    else:
        return True


def ordered(object: Sequence, variable_name: str):
    """
    Raises a `TypeError` is `object` is not a sequence.
    """
    # This check isn't perfect, but it works well enough.
    # I just want [x for x in object] to be meaningful and deterministic.
    # Note: isinstance(object, Sequence) is not what I want. Sequence requires that
    # object.__class__ additionally implements index and count. See:
    # https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes
    if not _is_reversible(object):
        raise TypeError(
            f"{variable_name} must be an ordered collection. Consider converting it to "
            "a list or tuple."
        )


def nonempty(object: Sequence, variable_name: str):
    """
    Raises a `ValueError` if `object` is empty.
    """
    if len(object) == 0:
        raise ValueError(f"{variable_name} must be non-empty.")


def nonempty_and_ordered(object: Sequence, variable_name: str):
    """
    Raises an error if `object` is not nonempty and ordered.
    """
    nonempty(object, variable_name)
    ordered(object, variable_name)


def completions(completions: Sequence[str]):
    """
    Raises an error if `completions` is not a nonempty, ordered, non-string, or if it
    contains an empty string.
    """
    nonempty_and_ordered(completions, variable_name="completions")
    if isinstance(completions, str):
        raise TypeError(
            "completions cannot be a string. It must be a sequence of strings. If you "
            "intend on inputting a single completion to estimate its probability, wrap "
            "it in a list or tuple and set normalize=False."
        )
    idxs_empty_completions = [
        i for i, completion in enumerate(completions) if not completion
    ]
    if len(idxs_empty_completions) == len(completions):
        raise ValueError(
            "All completions are empty. Expected all completions to be non-empty "
            "strings. Did you mean to use the token_logprobs function instead?"
        )
    elif idxs_empty_completions:
        raise ValueError(
            f"completions {idxs_empty_completions} are empty. Expected all completions "
            "to be non-empty strings."
        )


def end_of_prompt(end_of_prompt: Literal[" ", ""]):
    """
    Raises an error if `end_of_prompt` is not a whitespace or an empty string.
    """
    msg = 'end_of_prompt must be a whitespace " " or an empty string "".'
    if not isinstance(end_of_prompt, str):
        raise TypeError(msg)
    if end_of_prompt not in {" ", ""}:
        raise ValueError(msg)


def prior(prior: Sequence[float] | None, expected_length: int):
    """
    Raises an error if `prior is not None` and isn't a probability distribution over
    `expected_length` categories.
    """
    if prior is None:  # it's a uniform prior, no need to check anything
        return
    # Going to be stricter on the type here. We do the prior * likelihood computation
    # after making expensive model calls. Don't want that multiplication to fail b/c
    # that'd be a complete waste of model compute.
    if not isinstance(prior, (Sequence, np.ndarray)):
        raise TypeError("prior must be None, a Sequence, or a numpy array.")
    if len(np.shape(prior)) != 1:
        raise ValueError("prior must be 1-D.")
    prior_arr = np.array(prior, dtype=float)  # try casting to float
    if not np.all(prior_arr >= 0):
        raise ValueError("prior must contain probabilities between 0 and 1.")
    if not np.all(prior_arr <= 1):
        raise ValueError("prior must contain probabilities between 0 and 1.")
    if not np.isclose(prior_arr.sum(), 1):
        raise ValueError("prior must sum to 1.")
    if prior is not None and len(prior) != expected_length:
        raise ValueError(
            f"Expected prior to have length {expected_length} (the number of "
            f"completions), got {len(prior)}."
        )


def normalize(completions: Sequence[str], normalize: bool):
    """
    Raises a `ValueError` if `len(completions) == 1 and normalize`.
    """
    if len(completions) == 1 and normalize:
        raise ValueError(
            "Setting normalize=True when there's only 1 completion causes the "
            "probability to trivially be 1. Did you mean to set normalize=False, or "
            "did you mean to include more completions?"
        )

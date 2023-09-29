"""
Shared input checks. These check conditions that would cause silent and annoying
failures. For example, inputting an unordered iterable will cause outputs like predicted
probabilities to be meaningless.
"""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np


def _is_reversible(object) -> bool:
    # First, cheaply check if it implements __reversed__
    if hasattr(object, "__reversed__"):
        return True
    # Some objects don't have the attribute, but still can be reversible, like a tuple.
    # reversed(object) is often a generator, so checking this is often cheap.
    try:
        reversed(object)
    except TypeError:
        return False
    else:
        return True


def ordered(object: Sequence, variable_name: str):
    # This check isn't perfect, but it works well enough.
    if not _is_reversible(object):
        raise TypeError(
            f"{variable_name} must be an ordered collection. Consider converting it to "
            "a list or tuple."
        )


def nonempty(object: Sequence, variable_name: str):
    if len(object) == 0:
        raise ValueError(f"{variable_name} must be non-empty.")


def nonempty_and_ordered(object: Sequence, variable_name: str):
    nonempty(object, variable_name)
    ordered(object, variable_name)


def completions(completions: Sequence[str]):
    """
    Raise an error if `completions` is not a nonempty, ordered, non-string.
    """
    nonempty_and_ordered(completions, variable_name="completions")
    if isinstance(completions, str):
        raise TypeError(
            "completions cannot be a string. It must be a sequence of strings. If you "
            "intend on inputting a single completion to estimate its probability, wrap "
            "it in a list or tuple and set normalize=False."
        )


def prior(prior: Optional[Sequence[float]], expected_length: int):
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
        raise TypeError("prior must be None, a Sequence, or numpy array.")
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

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


from cappr.utils import _check


@dataclass(frozen=True)
class Example:
    """
    Represents a single prompt-completion classification example.

    This data structure is useful when different classification examples may belong to
    a different set of classes.

    Parameters
    ----------
    prompt : str
        string, which, e.g., contains the text to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after the
        prompt
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the `prompt`. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "

    Raises
    ------
    TypeError
        if `prompt` is not a string
    TypeError
        if `completions` is not a sequence of strings
    TypeError
        if `prior` is not None, or it isn't a sequence
    ValueError
        if `prior` is a sequence but isn't 1-D
    ValueError
        if `prior` is a sequence but doesn't sum to 1
    ValueError
        if `prior` is a sequence but `completions` and `prior` are different lengths
    """

    prompt: str
    completions: Sequence[str]
    prior: Optional[Sequence[float]] = None
    end_of_prompt: str = " "

    def __post_init__(self):
        ## Check inputs here so that fxns of Example don't need to check
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string.")
        if isinstance(self.completions, str) or not isinstance(
            self.completions, Sequence
        ):
            raise TypeError("completions must be a Sequence of strings.")
        _check.prior(self.prior)
        if self.prior is not None and len(self.completions) != len(self.prior):
            raise ValueError(
                "completions and prior are different lengths: "
                f"{len(self.completions)}, {len(self.prior)}."
            )

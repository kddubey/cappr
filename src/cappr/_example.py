from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence


from cappr.utils import _check


@dataclass(frozen=True)
class Example:
    """
    Represents a single prompt-completion classification example.

    This data structure is only useful if different classification examples may belong
    to a different set of classes, and you want to run the model in batches.

    Parameters
    ----------
    prompt : str
        string, which, e.g., contains the text to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after the
        prompt
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the `prompt`. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    normalize : bool | None, optional
        whether or not to normalize completion-after-prompt probabilities into a
        probability distribution over completions. Set this to `False` if you'd like the
        raw completion-after-prompt probability, or you're solving a multi-label
        prediction problem. By default, True

    Raises
    ------
    TypeError
        if `prompt` is not a string
    ValueError
        if `prompt` is empty
    TypeError
        if `completions` is not a sequence
    ValueError
        if `completions` is empty, or contains an empty string
    TypeError
        if `end_of_prompt` is not a string
    ValueError
        if `end_of_prompt` is not a whitespace or empty
    TypeError
        if `prior` is not None, or it isn't a sequence or numpy array
    ValueError
        if `prior` is not a 1-D probability distribution over `completions`
    ValueError
        if `normalize` is True but there's only one completion in `completions`
    """

    prompt: str
    completions: Sequence[str]
    prior: Sequence[float] | None = None
    end_of_prompt: Literal[" ", ""] = " "
    normalize: bool = True

    def __post_init__(self):
        # Check inputs here so that functions of Example don't need to check
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string.")
        _check.nonempty(self.prompt, variable_name="prompt")
        _check.completions(self.completions)
        # If completions is a pandas Series, __getitem__ and __contains__ are on the
        # Series index, not its values. To avoid such issues, just convert completions
        # to a tuple. Re-setting an attribute in a frozen object requires this call:
        object.__setattr__(self, "completions", tuple(self.completions))
        _check.end_of_prompt(self.end_of_prompt)
        _check.prior(self.prior, expected_length=len(self.completions))
        _check.normalize(self.completions, self.normalize)

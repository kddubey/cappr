from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence

from cappr.utils import _check


@dataclass(frozen=True)
class Example:
    """
    Represents a single prompt-completion task.

    This data structure is only useful if different prompts correspond to a different
    set of possible choices/completions, and you want to run the model in batches for
    greater throughput.

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
        prior = _check.prior(self.prior, expected_length=len(self.completions))
        object.__setattr__(self, "prior", prior)
        _check.normalize(self.normalize, self.completions)

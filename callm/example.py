from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


from callm.utils import check


@dataclass(frozen=True)
class Example:
    """
    Represents a single test example for a prompt-completion text classification task.
    This data structure is useful when different examples from the same dataset may
    belong to different classes.
    This applies to, e.g., [COPA](https://people.ict.usc.edu/~gordon/copa.html).

    `prompt`: cointains the text to classify, perhaps with instructions

    `completions`: possible completions/answers to the `prompt`

    `prior`: (optional) a probability distribution over `completions`.

    `end_of_prompt`: (default: `' '`) the string used to join the `prompt` and each
    completion.
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
        check.prior(self.prior)
        if self.prior is not None and len(self.completions) != len(self.prior):
            raise ValueError(
                "completions and prior are different lengths: "
                f"{len(self.completions)}, {len(self.prior)}."
            )

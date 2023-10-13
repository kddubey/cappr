from __future__ import annotations
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from cappr import Example


class classify_module(Protocol):
    """
    Protocol for all `classify` modules.
    """

    def token_logprobs(self, texts: Sequence[str], Any) -> list[list[float]]:
        ...

    def log_probs_conditional(
        self, prompts: str | Sequence[str], completions: Sequence[str], Any
    ) -> list[list[float]] | list[list[list[float]]]:
        ...

    def log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], Any
    ) -> list[list[float]] | list[list[list[float]]]:
        ...

    def predict_proba(
        self, prompts: str | Sequence[str], completions: Sequence[str], Any
    ) -> npt.NDArray[np.floating]:
        ...

    def predict_proba_examples(
        self, examples: Example | Sequence[Example], Any
    ) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
        ...

    def predict(
        self, prompts: str | Sequence[str], completions: Sequence[str], Any
    ) -> str | list[str]:
        ...

    def predict_examples(
        self, examples: Example | Sequence[Example], Any
    ) -> str | list[str]:
        ...

from __future__ import annotations
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from cappr import Example


class classify_module(Protocol):
    """
    Protocol for all `classify` modules.
    """

    def token_logprobs(self, texts: str | Sequence[str], Any) -> list[list[float]]:
        """
        For each text, log Pr(token[i] | tokens[:i]) for each token
        """

    def log_probs_conditional(
        self, prompts: str | Sequence[str], completions: Sequence[str], Any
    ) -> list[list[float]] | list[list[list[float]]]:
        """
        list[i][j][k] = log Pr(
            completions[j][token k] | prompts[i] + completions[j][tokens :k]
        )
        """

    def log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], Any
    ) -> list[list[float]] | list[list[list[float]]]:
        """
        list[i][j][k] = log Pr(
            examples[i].completions[j][token k]
            | examples[i].prompt + examples[i].completions[j][tokens :k]
        )
        """

    def predict_proba(
        self, prompts: str | Sequence[str], completions: Sequence[str], Any
    ) -> npt.NDArray[np.floating]:
        """
        array[i, j] = Pr(completions[j] | prompts[i])
        """

    def predict_proba_examples(
        self, examples: Example | Sequence[Example], Any
    ) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
        """
        list[i][j] = Pr(examples[i].completions[j] | examples[i].prompt)
        """

    def predict(
        self, prompts: str | Sequence[str], completions: Sequence[str], Any
    ) -> str | list[str]:
        """
        list[i] = argmax_j Pr(completions[j] | prompts[i])
        """

    def predict_examples(
        self, examples: Example | Sequence[Example], Any
    ) -> str | list[str]:
        """
        list[i] = argmax_j Pr(examples[i].completions[j] | examples[i].prompt)
        """

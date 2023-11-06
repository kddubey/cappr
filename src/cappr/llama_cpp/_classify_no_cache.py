"""
Mirror. For testing purposes only. Is strictly slower.
"""
from __future__ import annotations
from functools import partial
from typing import Literal, Sequence

from llama_cpp import Llama
import numpy as np
import numpy.typing as npt

from cappr.utils import _no_cache, classify
from cappr import Example
from cappr.llama_cpp.classify import token_logprobs
from cappr.llama_cpp import _utils


def _tokenize(model: Llama, texts: Sequence[str]) -> list[list[int]]:
    return [model.tokenize(text.encode("utf-8"), add_bos=False) for text in texts]


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    **kwargs,
) -> list[list[list[float]]]:
    end_of_prompt_for_slicing = (
        end_of_prompt if _utils.does_tokenizer_need_prepended_space(model) else ""
    )
    return _no_cache.log_probs_conditional(
        token_logprobs,
        partial(_tokenize, model),
        prompts,
        completions,
        model,
        end_of_prompt=end_of_prompt,
        end_of_prompt_for_slicing=end_of_prompt_for_slicing,
        add_bos=True,
    )


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model: Llama,
) -> list[list[float]] | list[list[list[float]]]:
    should_end_of_prompt_be_empty = not _utils.does_tokenizer_need_prepended_space(
        model
    )
    return _no_cache.log_probs_conditional_examples(
        token_logprobs,
        partial(_tokenize, model),
        examples,
        model,
        should_end_of_prompt_be_empty=should_end_of_prompt_be_empty,
        add_bos=True,
    )


@classify._predict_proba
def predict_proba(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    prior: Sequence[float] | None = None,
    normalize: bool = True,
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
) -> npt.NDArray[np.floating]:
    return log_probs_conditional(**locals())


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Example | Sequence[Example],
    model: Llama,
) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
    return log_probs_conditional_examples(**locals())


@classify._predict
def predict(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    prior: Sequence[float] | None = None,
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
) -> str | list[str]:
    return predict_proba(**locals())


@classify._predict_examples
def predict_examples(
    examples: Example | Sequence[Example],
    model: Llama,
) -> str | list[str]:
    return predict_proba_examples(**locals())

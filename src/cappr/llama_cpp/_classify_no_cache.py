"""
Mirror. For testing purposes only. Should be strictly slower.
"""
from __future__ import annotations
from typing import Literal, Sequence

from llama_cpp import Llama
import numpy as np
import numpy.typing as npt

from cappr.utils import _batch, classify
from cappr import Example
from cappr.llama_cpp.classify import token_logprobs
from cappr.llama_cpp import _utils


def _slice_completions(
    completions: Sequence[str],
    end_of_prompt: Literal[" ", ""],
    log_probs: Sequence[Sequence[float]],
    model: Llama,
) -> list[list[float]]:
    """
    Returns a list `log_probs_completions` where `log_probs_completions[i]` is a list of
    conditional log-probablities for each token in `completions[i]`, extracted by
    slicing `log_probs[i]`.
    """
    if len(completions) != len(log_probs):
        raise ValueError(
            "Different number of completions and log_probs: "
            f"{len(completions)}, {len(log_probs)}."
        )  # pragma: no cover
    if not _utils.does_tokenizer_need_prepended_space(model):
        end_of_prompt = ""
    completions = [end_of_prompt + completion for completion in completions]
    completion_lengths = []
    for completion in completions:
        input_ids = model.tokenize(completion.encode("utf-8"), add_bos=False)
        completion_lengths.append(len(input_ids))
    return [
        log_probs_text[-num_completion_tokens:]
        for num_completion_tokens, log_probs_text in zip(completion_lengths, log_probs)
    ]


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    **kwargs,
) -> list[list[list[float]]]:
    # Flat list of prompts and their completions. Will post-process
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    log_probs = token_logprobs(texts, model, end_of_prompt="", add_bos=True)
    # Since log_probs is a flat list, we'll need to batch them by the size and order of
    # completions to fulfill the spec.
    return [
        _slice_completions(completions, end_of_prompt, log_probs_batch, model)
        for log_probs_batch in _batch.constant(log_probs, size=len(completions))
    ]


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model: Llama,
) -> list[list[float]] | list[list[list[float]]]:
    # Little weird. I want my IDE to know that examples is always a Sequence[Example]
    # b/c of the decorator.
    examples: Sequence[Example] = examples
    # Flat list of prompts and their completions. Will post-process
    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    log_probs_all = token_logprobs(texts, model=model, end_of_prompt="", add_bos=True)
    # Flatten completions in same order as examples were flattened
    should_end_of_prompt_be_empty = not _utils.does_tokenizer_need_prepended_space(
        model
    )
    completions_all = []
    for example in examples:
        end_of_prompt = "" if should_end_of_prompt_be_empty else example.end_of_prompt
        for completion in example.completions:
            completions_all.append(end_of_prompt + completion)
    log_probs_completions_all = _slice_completions(
        completions_all, "", log_probs_all, model
    )
    # Batch by completions to fulfill the spec
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(
        _batch.variable(log_probs_completions_all, sizes=num_completions_per_prompt)
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

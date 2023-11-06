"""
Utilities for implementations which don't cache.
"""
from __future__ import annotations
from typing import Callable, Literal, Sequence

from cappr.utils import _batch


def _slice_completions(
    completions: Sequence[str],
    end_of_prompt: str,
    log_probs: Sequence[Sequence[float]],
    tokenize: Callable[[Sequence[str]], list[list[int]]],
) -> list[list[float]]:
    """
    Returns a list `log_probs_completions` where `log_probs_completions[i]` is a list of
    conditional log-probablities for each token in `end_of_prompt + completions[i]`,
    extracted by slicing `log_probs[i]`.
    """
    if len(completions) != len(log_probs):
        raise ValueError(
            "Different number of completions and log_probs: "
            f"{len(completions)}, {len(log_probs)}."
        )  # pragma: no cover
    completions = [end_of_prompt + completion for completion in completions]
    completion_lengths = [len(tokens) for tokens in tokenize(completions)]
    return [
        log_probs_text[-num_completion_tokens:]
        for num_completion_tokens, log_probs_text in zip(completion_lengths, log_probs)
    ]


def log_probs_conditional(
    token_logprobs: Callable[[Sequence[str]], list[list[float]]],
    tokenize: Callable[[list[str]], list[list[int]]],
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *token_logprobs_args,
    end_of_prompt: Literal[" ", ""] = " ",
    end_of_prompt_for_slicing: Literal[" ", ""] = " ",
    **token_logprobs_kwargs,
):
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    log_probs = token_logprobs(
        texts,
        *token_logprobs_args,
        end_of_prompt="",
        **token_logprobs_kwargs,
    )
    # Since log_probs is a flat list, we'll need to batch them by the size and order of
    # completions to fulfill the spec
    return [
        _slice_completions(
            completions, end_of_prompt_for_slicing, log_probs_batch, tokenize
        )
        for log_probs_batch in _batch.constant(log_probs, size=len(completions))
    ]


def log_probs_conditional_examples(
    token_logprobs: Callable[[Sequence[str]], list[list[float]]],
    tokenize: Callable[[Sequence[str]], list[list[int]]],
    examples,
    *token_logprobs_args,
    should_end_of_prompt_be_empty: bool,
    **token_logprobs_kwargs,
):
    from cappr import Example

    # Little weird. I want my IDE to know that examples is always a Sequence[Example]
    examples: Sequence[Example] = examples

    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    log_probs_all = token_logprobs(
        texts, *token_logprobs_args, end_of_prompt="", **token_logprobs_kwargs
    )
    # Slice out completion tokens
    num_completions_per_prompt = []
    completions_all = []
    for example in examples:
        num_completions_per_prompt.append(len(example.completions))
        end_of_prompt = "" if should_end_of_prompt_be_empty else example.end_of_prompt
        for completion in example.completions:
            completions_all.append(end_of_prompt + completion)
    log_probs_completions_all = _slice_completions(
        completions_all, end_of_prompt="", log_probs=log_probs_all, tokenize=tokenize
    )
    # Batch by completions to fulfill the spec
    return list(
        _batch.variable(log_probs_completions_all, sizes=num_completions_per_prompt)
    )

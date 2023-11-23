"""
Utilities for implementations which don't cache
"""
from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, cast, Literal, Sequence

from cappr.utils import _batch, _check


@lru_cache()
def _does_tokenizer_need_prepended_space(
    tokenize: Callable[[Sequence[str]], list[list[int]]], bos_token_id: int | None
):
    tokenize_single_text = lambda text: tokenize([text])[0]
    return _check.does_tokenizer_need_prepended_space(
        tokenize_single_text, bos_token_id
    )


def _slice_completions(
    completions: Sequence[str],
    end_of_prompt: str,
    log_probs: Sequence[Sequence[float]],
    tokenize: Callable[[Sequence[str]], list[list[int]]],
    bos_token_id: int | None,
) -> list[list[float]]:
    """
    Slice the completion's tokens from each list of log-probabilities in `log_probs`.
    """
    if len(completions) != len(log_probs):
        raise ValueError(
            f"Different numbers of completions and log_probs: {len(completions)}, "
            f"{len(log_probs)}, likely due to an issue with the token_logprobs function"
        )  # pragma: no cover
    if not _does_tokenizer_need_prepended_space(tokenize, bos_token_id):
        end_of_prompt = ""
    completions = [end_of_prompt + completion for completion in completions]
    completion_lengths = [len(tokens) for tokens in tokenize(completions)]
    return [
        log_probs_text[-num_completion_tokens:]
        for num_completion_tokens, log_probs_text in zip(completion_lengths, log_probs)
    ]


def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    end_of_prompt: Literal[" ", ""],
    token_logprobs: Callable[[Sequence[str], Any], list[list[float]]],
    tokenize_completions: Callable[[list[str]], list[list[int]]],
    bos_token_id: int | None,
    *token_logprobs_args,
    **token_logprobs_kwargs,
) -> list[list[list[float]]]:
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    log_probs_texts = token_logprobs(
        texts,
        *token_logprobs_args,
        end_of_prompt="",
        **token_logprobs_kwargs,
    )
    # log_probs_texts is a 2-D list. Batch it by the size and order of completions to
    # fulfill the spec
    return [
        _slice_completions(
            completions, end_of_prompt, log_probs, tokenize_completions, bos_token_id
        )
        for log_probs in _batch.constant(log_probs_texts, size=len(completions))
    ]


def log_probs_conditional_examples(
    examples,
    token_logprobs: Callable[[Sequence[str], Any], list[list[float]]],
    tokenize_completions: Callable[[Sequence[str]], list[list[int]]],
    bos_token_id: int | None,
    *token_logprobs_args,
    **token_logprobs_kwargs,
) -> list[list[list[float]]]:
    from cappr import Example

    # examples is always a Sequence[Example] b/c of the decorator
    examples = cast(Sequence[Example], examples)

    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    log_probs_texts = token_logprobs(
        texts, *token_logprobs_args, end_of_prompt="", **token_logprobs_kwargs
    )
    should_end_of_prompt_be_empty = not _does_tokenizer_need_prepended_space(
        tokenize_completions, bos_token_id
    )
    end_of_prompts = [
        "" if should_end_of_prompt_be_empty else example.end_of_prompt
        for example in examples
    ]
    completions = [
        end_of_prompt + completion
        for end_of_prompt, example in zip(end_of_prompts, examples)
        for completion in example.completions
    ]
    end_of_prompt = ""  # we already added it in completions
    log_probs_completions = _slice_completions(
        completions, end_of_prompt, log_probs_texts, tokenize_completions, bos_token_id
    )
    # log_probs_completions is a 2-D list. Batch it by the size and order of completions
    # to fulfill the spec
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(
        _batch.variable(log_probs_completions, sizes=num_completions_per_prompt)
    )

"""
Perform prompt-completion classification using a `transformers.AutoModelForCausalLM`.

This module is a slow mirror of `classify`. It **does not** precompute attention block
keys and values for prompts. It's only used for testing and benchmarking purposes.
"""
from __future__ import annotations
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from cappr._utils import batch, classify
from cappr import Example
from cappr import huggingface as hf


def _keys_values_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
):
    if not tokenizer.padding_side == "right":
        raise ValueError("Gotta use right padding to ensure position IDs are correct.")
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(num_completions_per_prompt, Sequence):
        if not len(prompts) == len(num_completions_per_prompt):
            raise ValueError(
                "If num_completions_per_prompt is a Sequence, then it must be the same "
                f"length as prompts. Got lengths {len(num_completions_per_prompt)}, "
                f"{len(prompts)}."
            )
    if isinstance(num_completions_per_prompt, int):
        ## For code simplicity, just repeat it
        num_completions_per_prompt = [num_completions_per_prompt] * len(prompts)
    prompts_repeated = [
        prompt
        for prompt, num_repeats in zip(prompts, num_completions_per_prompt)
        for _ in range(num_repeats)
    ]
    # fmt: off
    encodings: BatchEncoding = (tokenizer(prompts_repeated, return_tensors="pt",
                                          padding=True)
                                .to(hf._utils.DEVICE))
    # fmt: on
    with torch.no_grad():
        out = model(**encodings)

    offsets: torch.Tensor = encodings.attention_mask.sum(dim=1)

    ## Need (next-token) logits from prompts, i.e., last non-pad prompt token, since
    ## that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    last_nonpad_token_logits: torch.Tensor = out.logits.take_along_dim(
        _last_nonpad_token_idxs, dim=1
    )

    return out.past_key_values, encodings, offsets, last_nonpad_token_logits


def _logits_texts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
) -> tuple[torch.Tensor, BatchEncoding]:
    encodings = tokenizer(texts, return_tensors="pt", padding=True).to(hf._utils.DEVICE)
    with torch.no_grad():
        out = model(**encodings)
    return out.logits, encodings


def _prompts_offsets(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
) -> torch.Tensor:
    if not isinstance(num_completions_per_prompt, int) and not isinstance(
        num_completions_per_prompt, torch.Tensor
    ):
        num_completions_per_prompt = torch.tensor(num_completions_per_prompt)
    return (
        tokenizer(prompts, return_tensors="pt", padding=True)
        .attention_mask.repeat_interleave(num_completions_per_prompt, dim=0)
        .sum(dim=1)
        .to(hf._utils.DEVICE)
    )


def _logits_completions_given_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: str = " ",
):
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError("completions must be a Sequence of strings.")
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    logits, encodings = _logits_texts(model, tokenizer, texts)
    ## Need these indices to slice completion tokens
    encodings["offsets"] = _prompts_offsets(
        tokenizer, prompts, num_completions_per_prompt=len(completions)
    )
    return logits, encodings


def _logits_completions_given_prompts_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: Sequence[classify.Example],
):
    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    logits, encodings = _logits_texts(model, tokenizer, texts)
    ## Need these indices to slice completion tokens
    prompts = [example.prompt for example in examples]
    num_completions_per_prompt = [len(example.completions) for example in examples]
    encodings["offsets"] = _prompts_offsets(
        tokenizer, prompts, num_completions_per_prompt=num_completions_per_prompt
    )
    return logits, encodings


def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor]
) -> list[list[float]]:
    log_probs = hf._utils.logits_to_log_probs(
        logits, encodings["input_ids"], input_ids_start_idx=1, logits_end_idx=-1
    )
    last_idx_non_pad = encodings["attention_mask"].sum(dim=1)
    ## i.e., # of tokens per text
    return [
        log_probs_prompt_completion[completion_start:completion_end].tolist()
        for log_probs_prompt_completion, completion_start, completion_end in zip(
            log_probs, encodings["offsets"] - 1, last_idx_non_pad - 1
        )
    ]


def log_probs_conditional(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, AutoTokenizer] = None,
    end_of_prompt: str = " ",
    batch_size: int = 32,
) -> list[list[list[float]]]:
    model, tokenizer = hf._utils.load_model_and_tokenizer(
        model=model, model_and_tokenizer=model_and_tokenizer
    )

    @batch.flatten
    @batch.batchify(batchable_arg="prompts", progress_bar_desc="log-probs (slow)")
    def log_probs_completions_batch(prompts, batch_size=batch_size):
        logits, encodings = _logits_completions_given_prompts(
            model, tokenizer, prompts, completions, end_of_prompt=end_of_prompt
        )
        return _logits_to_log_probs_completions(logits, encodings)

    log_probs_completions = log_probs_completions_batch(prompts)
    return list(batch.constant(log_probs_completions, size=len(completions)))


def log_probs_conditional_examples(
    examples: Sequence[Example],
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, AutoTokenizer] = None,
    batch_size: int = 32,
) -> list[list[list[float]]]:
    model, tokenizer = hf._utils.load_model_and_tokenizer(
        model=model, model_and_tokenizer=model_and_tokenizer
    )

    @batch.flatten
    @batch.batchify(batchable_arg="examples", progress_bar_desc="log-probs (slow)")
    def log_probs_completions_batch(examples, batch_size=batch_size):
        logits, encodings = _logits_completions_given_prompts_examples(
            model, tokenizer, examples
        )
        return _logits_to_log_probs_completions(logits, encodings)

    log_probs_completions = log_probs_completions_batch(examples)
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(batch.variable(log_probs_completions, sizes=num_completions_per_prompt))


@classify._predict_proba
def predict_proba(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, AutoTokenizer] = None,
    end_of_prompt: str = " ",
    batch_size: int = 32,
) -> npt.NDArray[np.floating]:
    return log_probs_conditional(
        prompts,
        completions,
        model=model,
        model_and_tokenizer=model_and_tokenizer,
        end_of_prompt=end_of_prompt,
        batch_size=batch_size,
    )


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Sequence[Example],
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, AutoTokenizer] = None,
    batch_size: int = 32,
) -> Union[list[list[float]], npt.NDArray[np.floating]]:
    return log_probs_conditional_examples(
        examples,
        model=model,
        model_and_tokenizer=model_and_tokenizer,
        batch_size=batch_size,
    )


@classify._predict
def predict(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, AutoTokenizer] = None,
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    batch_size: int = 32,
) -> list[str]:
    return predict_proba(
        prompts,
        completions,
        model=model,
        model_and_tokenizer=model_and_tokenizer,
        prior=prior,
        end_of_prompt=end_of_prompt,
        batch_size=batch_size,
    )


@classify._predict_examples
def predict_examples(
    examples: Sequence[Example],
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, AutoTokenizer] = None,
    batch_size: int = 32,
) -> list[str]:
    return predict_proba_examples(
        examples,
        model=model,
        model_and_tokenizer=model_and_tokenizer,
        batch_size=batch_size,
    )

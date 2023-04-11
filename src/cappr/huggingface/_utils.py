"""
YouTils
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def logits_to_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    input_ids_start_idx: int,
    logits_end_idx: int,
):
    """
    TODO: docstring fix
    Returns a tensor `log_probs` with shape

        `(logits.shape[0], logits.shape[1]-1)`

    where `log_probs[i,j]` is the log-probability of token

        `input_ids[i,j]`

    given its previous tokens

        `input_ids[i,:j]`

    for `j in range(input_ids_start_idx, input_ids.shape[1])`.

    `logits[i,j]` is assumed to be an unnormalized distribution (over tokens in
    the vocab) given tokens `input_ids[i,:j]`.
    """
    ## logits.shape is    (# texts, max # tokens in texts, vocab size)
    log_probs = F.log_softmax(logits, dim=2)

    ## Only keep the log-prob from the vocab dimension whose index is is the
    ## next token's input ID.
    ## input_ids.shape is (# texts, max # tokens in texts)
    return (
        log_probs[:, :logits_end_idx, :]
        .take_along_dim(input_ids[:, input_ids_start_idx:, None], dim=2)
        .squeeze(-1)
    )


def load_model_and_tokenizer(
    model: str = None,
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    TODO: docstring
    """
    if (model is not None and model_and_tokenizer is not None) or (
        model is None and model_and_tokenizer is None
    ):
        raise ValueError(
            "One of model and model_and_tokenizer must be None, and the other not None."
        )
    from_str = model is not None
    if from_str:
        model_: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
    else:
        model_, tokenizer = model_and_tokenizer
    ## Prepare model
    model_.eval()
    if from_str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_.to(device)
    ## Prepare tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model_, tokenizer

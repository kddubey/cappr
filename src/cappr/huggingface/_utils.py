"""
YouTils
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def logits_to_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    input_ids_start_idx: int,
    logits_end_idx: int,
):
    """
    TODO: docstring
    """
    ## logits.shape is (# texts, max # tokens in texts, vocab size)
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
    model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizer],
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Returns back `model_and_tokenizer` where the model is in eval mode and the
    tokenizer's padding settings are correctly configured.
    """
    model_, tokenizer = model_and_tokenizer
    ## Prepare model
    model_.eval()
    ## Prepare tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model_, tokenizer

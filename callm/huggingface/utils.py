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


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
## TODO: don't do this. Ok for now b/c I know I'll only use a single GPU


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
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    TODO: docstring
    """
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return model, tokenizer

"""
YouTils
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    BatchEncoding,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def set_up_model_and_tokenizer(
    model_and_tokenizer: tuple[PreTrainedModel, PreTrainedTokenizer],
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Returns back `model_and_tokenizer` where the model is in eval mode and the
    tokenizer's padding settings are correctly configured. This function is quick and
    idempotent, so you can apply it whenever.

    TODO: consider changing this to an (internally used) context manager so that we
    don't modify the model and tokenizer / the user doesn't have to reset attributes
    """
    model, tokenizer = model_and_tokenizer
    # Prepare model
    model.eval()
    # Prepare tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    try:
        tokenizer.add_eos_token = False
        # We don't want the prompt or completion to end w/ EOS, so this should always
        # be False
    except AttributeError:
        pass
    return model, tokenizer


def logits_texts(
    texts: Sequence[str],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
) -> tuple[torch.Tensor, BatchEncoding]:
    # TODO: auto-batch? consider adding a batch_size kwarg, and decorating the func like
    # token_logprobs
    encodings = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model(**encodings)
    if getattr(tokenizer, "add_bos_token", False):
        # Drop the first bos token after we're done encoding so that the shape is
        # consistent w/ other tokenizers
        logits = out.logits[:, 1:, :]
        encodings = BatchEncoding(
            {key: value[:, 1:] for key, value in encodings.items()}
        )
    else:
        logits = out.logits
    return logits, encodings


def logits_to_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    input_ids_start_idx: int = None,
    logits_end_idx: int = None,
):
    """
    TODO: docstring
    """
    # logits.shape is (# texts, max # tokens in texts, vocab size)
    log_probs = F.log_softmax(logits, dim=2)

    # Only keep the log-prob from the vocab dimension whose index is is the
    # next token's input ID.
    # input_ids.shape is (# texts, max # tokens in texts)
    return (
        log_probs[:, :logits_end_idx, :]
        .take_along_dim(input_ids[:, input_ids_start_idx:, None], dim=2)
        .squeeze(-1)
    )


def does_tokenizer_prepend_space_to_first_token(
    tokenizer: PreTrainedTokenizer,
) -> bool:
    return not isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast))

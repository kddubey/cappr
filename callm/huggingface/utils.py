"""
Some janky stuff
"""
from __future__ import annotations
from collections import defaultdict
from functools import wraps
from typing import Collection, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
## TODO: don't do this. Ok for now b/c I know I'll only use a single GPU


def _cat_logits_batches(
    logits_batches: Sequence[torch.Tensor], pad_value=float("-inf")
):
    ## logits_batches[i].shape is
    ## (batchsize, max # completion tokens in batch, vocabsize)
    num_tokens = max(logits_batch.shape[1] for logits_batch in logits_batches)
    ## Pad on the right side of the token dimension, then concatenate.
    ## For some reason, the padding dimensions need to be supplied in reverse
    ## order of the input tensor's dimensions.
    # fmt: off
    return torch.cat([F.pad(logits_batch,
                            (0, 0,  ## left, right for vocab dim 
                             0, num_tokens - logits_batch.shape[1],
                             0, 0), ## left, right for batch dim
                            value=pad_value)
                      for logits_batch in logits_batches])
    # fmt: on


def _cat_encodings_batches(
    encodings_batches: Sequence[Mapping[str, torch.Tensor]],
    default_pad_value=0,
    dont_pad: Collection[str] = ("offsets",),
    **name_to_value,
):
    num_tokens = max(
        encodings_batch["input_ids"].shape[1] for encodings_batch in encodings_batches
    )
    encodings = defaultdict(list)
    for encodings_batch in encodings_batches:
        for name, tensor in encodings_batch.items():
            if name in dont_pad:
                pad = (0, 0)
            else:
                pad = (0, num_tokens - tensor.shape[1])  ## only right-padding
            value = name_to_value.get(name, default_pad_value)
            encodings[name].append(F.pad(tensor, pad, value=value))
    return BatchEncoding(
        {name: torch.cat(tensors) for name, tensors in encodings.items()}
    )


def cat_logits_encodings(batchified_func):
    """
    Decorates a `batchify`'d function which returns a logits tensor and a
    `BatchEncoding`. Pads and concatenates batches of logits and encodings into a single
    logits tensor and a single `BatchEncoding`.
    """

    @wraps(batchified_func)
    def wrapper(*args, **kwargs):
        outputs, tokenizer = batchified_func(*args, **kwargs)
        if tokenizer.padding_side != "right":
            raise ValueError("Gotta use right padding.")
        logits_batches, encodings_batches = tuple(zip(*outputs))
        logits = _cat_logits_batches(logits_batches)
        encodings = _cat_encodings_batches(
            encodings_batches, completions_input_ids=tokenizer.pad_token_id
        )
        return logits, encodings

    return wrapper


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
    return model, tokenizer

"""
YouTils
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Sequence, TypeVar

import torch
import torch.nn.functional as F
from transformers import (
    BatchEncoding,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# AutoModelForCausalLM is not actually a superclass for pretrained causal/autoregressive
# LMs. Its from_pretrained method is a loading utility which always returns a
# PreTrainedModel. Let's create a type for them to make the documentation clearer.
PreTrainedModelForCausalLM = TypeVar(
    "PreTrainedModelForCausalLM", bound=PreTrainedModel
)
"A model loaded via `transformers.AutoModelForCausalLM.from_pretrained`"


@contextmanager
def set_up_model_and_tokenizer(
    model_and_tokenizer: tuple[PreTrainedModelForCausalLM, PreTrainedTokenizerBase]
):
    """
    In this context, internal attributes of the model and tokenizer are set to enable
    correct, batched inference. Namely:
      - the model is set in eval mode
      - the tokenizer pads on the right
      - the tokenizer does not add an EOS token.
    """
    model, tokenizer = model_and_tokenizer

    # Grab attributes - model
    model_is_train = model.training
    # Grab attributes - tokenizer
    tokenizer_pad_token_id = tokenizer.pad_token_id
    tokenizer_padding_side = tokenizer.padding_side
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer_add_eos_token = tokenizer.add_eos_token

    # Set attributes - model
    # 1. Just ensure that the model is in eval mode. CAPPr only makes sense as an
    #    inference computation.
    model.eval()

    # Set attributes - tokenizer
    # Note: PreTrainedTokenizerBase is smart about setting auxiliary attributes, e.g.,
    # it updates tokenizer.special_tokens_map after setting tokenizer.pad_token_id.
    # 1. Set the pad token (if it's not set) so that batch inference is possible. These
    #    get masked out. Keep in mind that you need to be careful about setting position
    #    IDs correctly.
    if tokenizer_pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # 2. Set the padding side to right. Left-padding would alter position IDs for
    #    non-pad tokens, which makes things a bit more confusing.
    tokenizer.padding_side = "right"
    # 3. Don't add an end-of-sentence token. We'll never need it for the CAPPr scheme.
    #    Keeping it would throw off the classify module (which caches).
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False

    yield

    # Reset attributes - model
    model.train(model_is_train)
    # Reset attributes - tokenizer
    if tokenizer_pad_token_id is None:
        tokenizer.pad_token_id = tokenizer_pad_token_id
    tokenizer.padding_side = tokenizer_padding_side
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = tokenizer_add_eos_token


@contextmanager
def disable_add_bos_token(tokenizer: PreTrainedTokenizerBase):
    if hasattr(tokenizer, "add_bos_token"):
        add_bos_token: bool = tokenizer.add_bos_token
        tokenizer.add_bos_token = False

    yield

    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = add_bos_token


def logits_texts(
    texts: Sequence[str],
    model: PreTrainedModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
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
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    return not isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast))

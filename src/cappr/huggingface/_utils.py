"""
YouTils
"""

from __future__ import annotations
from contextlib import contextmanager, ExitStack, nullcontext
from functools import lru_cache
from typing import Mapping, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutput

from cappr.utils import _check
from cappr.utils.classify import _setattr


BatchEncodingPT = Mapping[str, torch.Tensor]
"""
Type of the output of `tokenizer(texts, return_tensors="pt", ...)`
"""
# transformers.BatchEncoding doesn't annotate values as tensors b/c tokenizers can
# return other objects. In this package, tokenizers will always return PyTorch tensors


ModelForCausalLM = PreTrainedModel
"""
A pretrained model with the same inference interface as a model loaded via
`transformers.AutoModelForCausalLM.from_pretrained`
"""


########################################################################################
# Set up the model and tokenizer to enable correct, batched inference. Use context
# managers b/c a user's settings shouldn't be modified when CAPPr is done. It's
# conceivable that someone uses CAPPr as part of a larger system where the model and
# tokenizer needs to be configured differently
########################################################################################


@contextmanager
def _eval_mode(model: ModelForCausalLM):
    """
    In this context, the model is set in eval mode.
    """
    is_training = model.training
    try:
        model.eval()
        yield
    finally:
        model.train(is_training)


@contextmanager
def _no_grad(model: ModelForCausalLM):  # model given to keep interface the same
    """
    In this context, gradients are not computed.
    """
    with torch.no_grad():
        yield


# Some models don't perfectly implement the HF model call interface. In particular,
# they're missing the return_dict and use_cache kwargs. They're instead in the model
# config. I see that as a more extensible design anyway.


@contextmanager
def _return_dict(model: ModelForCausalLM):
    """
    In this context, the model returns a dataclass when it's called.
    """
    with (
        _setattr(model.config, "return_dict", True)
        if hasattr(model, "config")
        else nullcontext()
    ):  # null b/c just try model(...).logits when needed
        yield


@contextmanager
def _use_cache(model: ModelForCausalLM):
    """
    In this context, the model output includes a `past_key_values` attribute.
    """
    with (
        _setattr(model.config, "use_cache", True)
        if hasattr(model, "config")
        else nullcontext()
    ):  # null b/c just try model(...).past_key_values when needed
        yield


@contextmanager
def _pad_on_right(tokenizer: PreTrainedTokenizerBase):
    """
    In this context, the pad token is set (if it's not set already) to the EOS token so
    that batch inference is possible. These get masked out. The padding side is set to
    right. Left-padding would alter position IDs for non-pad tokens, which makes things
    a bit more confusing.
    """
    pad_token_id = tokenizer.pad_token_id
    padding_side = tokenizer.padding_side
    try:
        if pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            # PreTrainedTokenizerBase is smart about setting auxiliary attributes, e.g.,
            # tokenizer.special_tokens_map is updated after tokenizer.pad_token_id = ...
        tokenizer.padding_side = "right"
        yield
    finally:
        tokenizer.pad_token_id = pad_token_id
        tokenizer.padding_side = padding_side


@contextmanager
def dont_add_eos_token(tokenizer: PreTrainedTokenizerBase):
    """
    In this context, don't add an end-of-sentence token.
    """
    with _setattr(tokenizer, "add_eos_token", False):
        yield


@contextmanager
def _combine_context_managers(context_managers, obj):
    with ExitStack() as stack:
        [
            stack.enter_context(context_manager(obj))
            for context_manager in context_managers
        ]
        yield


@contextmanager
def set_up_model(
    model: ModelForCausalLM,
    context_managers=(_eval_mode, _no_grad, _return_dict, _use_cache),
):
    with _combine_context_managers(context_managers, model):
        yield


@contextmanager
def set_up_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    context_managers=(_pad_on_right, dont_add_eos_token),
):
    with _combine_context_managers(context_managers, tokenizer):
        yield


@contextmanager
def dont_add_bos_token(tokenizer: PreTrainedTokenizerBase):
    """
    In this context, don't add a beginning-of-sentence token.
    """
    with _setattr(tokenizer, "add_bos_token", False):
        yield


########################################################################################
####################### Handle SentencePiece vs BPE tokenization #######################
########################################################################################


@lru_cache()
def does_tokenizer_need_prepended_space(
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    with dont_add_eos_token(tokenizer):
        tokenize = lambda text: tokenizer(text)["input_ids"]
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        return _check.does_tokenizer_need_prepended_space(tokenize, bos_token_id)


########################################################################################
##################################### Logits stuff #####################################
########################################################################################


def _batched_model_call(
    batch_size: int,
    model: ModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> CausalLMOutput:
    input_ids = torch.split(input_ids, batch_size)
    attention_mask = torch.split(attention_mask, batch_size)
    outs: list[CausalLMOutput] = []
    num_batches = len(input_ids)
    with set_up_model(model):
        for batch_idx in range(num_batches):
            out: CausalLMOutput = model(
                input_ids=input_ids[batch_idx],
                attention_mask=attention_mask[batch_idx],
            )
            outs.append(out)
    logits = torch.cat([out.logits for out in outs], dim=0)
    return CausalLMOutput(logits=logits)


def logits_texts(
    texts: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    drop_bos_token: bool = True,
    batch_size: int | None = None,
) -> tuple[torch.Tensor, BatchEncodingPT]:
    """
    Basically::

        return model(**tokenizer(texts)).logits, tokenizer(texts)

    The kwargs are set for CAPPr's convenience.
    """
    model, tokenizer = model_and_tokenizer
    with set_up_tokenizer(tokenizer):
        encodings: BatchEncodingPT = tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(model.device)
    if batch_size is not None:
        out = _batched_model_call(batch_size, model, **encodings)
    else:
        with set_up_model(model):
            out: CausalLMOutput = model(**encodings)
    if drop_bos_token and getattr(tokenizer, "add_bos_token", False):
        # Drop the first/bos token after we're done encoding so that the shape is
        # consistent w/ other tokenizers. For CAPPr, we'll never be interested in
        # Pr(token | <bos>). We're only interested in completion tokens
        logits = out.logits[:, 1:, :]
        encodings = {key: value[:, 1:] for key, value in encodings.items()}
    else:
        logits = out.logits
    return logits, encodings


def logits_to_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    input_ids_start_idx: int | None = None,
    logits_end_idx: int | None = None,
):
    """
    Log-softmax and then slice out input IDs to get token log-probabilities.
    """
    # logits.shape is (# texts, max # tokens in texts, vocab size)
    log_probs = logits.log_softmax(dim=-1)

    # Only keep the log-prob from the vocab dimension whose index is is the next token's
    # input ID
    # input_ids.shape is (# texts, max # tokens in texts)
    return (
        log_probs[:, :logits_end_idx, :]
        .take_along_dim(input_ids[:, input_ids_start_idx:, None], dim=-1)
        .squeeze(-1)
    )

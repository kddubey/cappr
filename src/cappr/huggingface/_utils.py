"""
YouTils
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack, nullcontext
from typing import Collection, Mapping, Sequence, TypeVar

import torch
from transformers import (
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutput


BatchEncoding = TypeVar("BatchEncoding", bound=Mapping[str, torch.Tensor])
"""
Type of the output of `tokenizer(texts, return_tensors="pt", ...)`.
"""
# transformers.BatchEncoding doesn't annotate values as tensors b/c tokenizers can
# return other objects. In this package, tokenizers will always return PyTorch tensors.


ModelForCausalLM = TypeVar("ModelForCausalLM", bound=PreTrainedModel)
"""
A pretrained model with the same inference interface as a model loaded via
`transformers.AutoModelForCausalLM.from_pretrained`.
"""


def does_tokenizer_prepend_space_to_first_token(
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """
    Why is this? Good question. Run and read this code::

        from transformers import AutoTokenizer

        model_name = "Maykeye/TinyLLama-v0"
        # After running all of the code below on Llama, try the following
        # model (with BPE tokenization) instead:
        # model_name = "gpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Run on the whole input text
        print(tokenizer(["Answer: A"])["input_ids"])
        # [[1, 19775, 31871, 308]]

        # Run on a partitioned version
        print(tokenizer(["Answer:", " A"])["input_ids"])
        # [[1, 19775, 31871], [1, 31822, 308]]
        # The input IDs are NOT the same as the whole input for Llama.
        # They are the same for GPT-2.

        # Run on a split but fixed version, removing the space before "A"
        print(tokenizer(["Answer:", "A"])["input_ids"])
        # [[1, 19775, 31871], [1, 308]]
        # Besides the <bos> token (id 1), the input IDs are now the same
        # as running on the whole input text for Llama.
        # For GPT-2, they're different.
    """
    # TODO: should somehow check if it's not a SentencePiece tokenizer / if it's a BPE
    # tokenizer? We should be able to try running the tokenizer on something and seeing
    # what happens. That'd also let us get rid of the crude isinstance check
    return not isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast))


########################################################################################
# Set up the model and tokenizer using context managers, because we shouldn't modify a
# user's settings when CAPPr is done. It's conceivable that someone uses CAPPr as part
# of a larger system where the model and tokenizer needs to be configured differently.
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
    In this context, gradients are not computed. This saves memory.
    """
    with torch.no_grad():
        yield


@contextmanager
def _return_dict(model: ModelForCausalLM):
    """
    In this context, the model returns a dataclass when it's called. (B/c of previous
    versions of `transformers`, the attribute is called `return_dict`.)
    """
    hasattr_return_dict = hasattr(model, "config") and hasattr(
        model.config, "return_dict"
    )
    try:
        if hasattr_return_dict:
            return_dict = model.config.return_dict
            model.config.return_dict = True
        yield
    finally:
        if hasattr_return_dict:
            model.config.return_dict = return_dict


@contextmanager
def _use_cache(model: ModelForCausalLM):
    """
    In this context, the model output includes a `past_key_values` attribute.
    """
    hasattr_use_cache = hasattr(model, "config") and hasattr(model.config, "use_cache")
    if hasattr_use_cache:
        use_cache: bool = getattr(model.config, "use_cache")
    try:
        if hasattr_use_cache:
            setattr(model.config, "use_cache", True)
        yield
    finally:
        if hasattr_use_cache:
            setattr(model.config, "use_cache", use_cache)


_DEFAULT_CONTEXT_MANAGERS_MODEL = (
    _eval_mode,
    _no_grad,
    _return_dict,
    _use_cache,
)
"""
Default model settings:
- set the model in eval mode
- don't compute gradients
- return output as a dataclass instead of a tuple
- return past_key_values if possible.
"""


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
            # Note: a PreTrainedTokenizerBase tokenizer is smart about setting auxiliary
            # attributes, e.g., it updates tokenizer.special_tokens_map after setting
            # tokenizer.pad_token_id
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
    has_attr_add_eos_token = hasattr(tokenizer, "add_eos_token")
    if has_attr_add_eos_token:
        add_eos_token: bool = getattr(tokenizer, "add_eos_token")
    try:
        if has_attr_add_eos_token:
            setattr(tokenizer, "add_eos_token", False)
        yield
    finally:
        if has_attr_add_eos_token:
            setattr(tokenizer, "add_eos_token", add_eos_token)


_DEFAULT_CONTEXT_MANAGERS_TOKENIZER = (
    _pad_on_right,
    dont_add_eos_token,
)
"""
Default tokenizer settings:
- pad on right
- don't add EOS token.
"""


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
    model: ModelForCausalLM, context_managers=_DEFAULT_CONTEXT_MANAGERS_MODEL
):
    with _combine_context_managers(context_managers, model):
        yield


@contextmanager
def set_up_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    context_managers=_DEFAULT_CONTEXT_MANAGERS_TOKENIZER,
):
    with _combine_context_managers(context_managers, tokenizer):
        yield


@contextmanager
def set_up_model_and_tokenizer(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    context_managers_model: Collection = _DEFAULT_CONTEXT_MANAGERS_MODEL,
    context_managers_tokenizer: Collection = _DEFAULT_CONTEXT_MANAGERS_TOKENIZER,
):
    """
    In this context, internal attributes of the model and tokenizer are set to enable
    correct, batched inference.
    """
    with set_up_model(model, context_managers_model), set_up_tokenizer(
        tokenizer, context_managers_tokenizer
    ):
        yield


@contextmanager
def dont_add_bos_token(tokenizer: PreTrainedTokenizerBase):
    """
    In this context, don't add a beginning-of-sentence token.
    """
    hasattr_add_bos_token = hasattr(tokenizer, "add_bos_token")
    if hasattr_add_bos_token:
        add_bos_token: bool = getattr(tokenizer, "add_bos_token")
    try:
        if hasattr_add_bos_token:
            setattr(tokenizer, "add_bos_token", False)
        yield
    finally:
        if hasattr_add_bos_token:
            setattr(tokenizer, "add_bos_token", add_bos_token)


########################################################################################
##################################### Logits stuff #####################################
########################################################################################
def drop_first_token(
    logits: torch.Tensor, encodings: BatchEncoding
) -> tuple[torch.Tensor, BatchEncoding]:
    logits = logits[:, 1:, :]
    encodings: BatchEncoding = {key: value[:, 1:] for key, value in encodings.items()}
    return logits, encodings


def logits_texts(
    texts: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    padding: bool | None = None,
    drop_bos_token: bool = True,
    do_not_add_eos_token: bool = True,
) -> tuple[torch.Tensor, BatchEncoding]:
    """
    Basically::

        return model(**tokenizer(texts)).logits, tokenizer(texts)

    The kwargs are set for CAPPr's convenience.
    """
    model, tokenizer = model_and_tokenizer
    # TODO: auto-batch?
    if padding is None:
        padding = getattr(tokenizer, "pad_token_id", None) is not None
    with dont_add_eos_token(tokenizer) if do_not_add_eos_token else nullcontext():
        encodings: BatchEncoding = tokenizer(
            texts, return_tensors="pt", padding=padding
        ).to(model.device)
    with set_up_model(model):
        out: CausalLMOutput = model(**encodings)
    if drop_bos_token and getattr(tokenizer, "add_bos_token", False):
        # Drop the first bos token after we're done encoding so that the shape is
        # consistent w/ other tokenizers. For CAPPr, we'll never be interested in
        # Pr(token | <bos>). We're only interested in completion tokens.
        logits, encodings = drop_first_token(out.logits, encodings)
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
    # input ID.
    # input_ids.shape is (# texts, max # tokens in texts)
    return (
        log_probs[:, :logits_end_idx, :]
        .take_along_dim(input_ids[:, input_ids_start_idx:, None], dim=-1)
        .squeeze(-1)
    )

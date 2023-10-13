"""
YouTils
"""
from __future__ import annotations
from contextlib import contextmanager
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
The output of a `tokenizer(texts, return_tensors="pt")` call.
"""
# transformers.BatchEncoding doesn't annotate values as tensors b/c tokenizers can
# return other objects. In this package, tokenizers will always return PyTorch tensors.


ModelForCausalLM = TypeVar("ModelForCausalLM", bound=PreTrainedModel)
"""
A pretrained model with the same interface as a model loaded via
`transformers.AutoModelForCausalLM.from_pretrained`.
"""


def does_tokenizer_prepend_space_to_first_token(
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """
    Why is this? Good question. Run and read this code::

        from transformers import AutoTokenizer

        model_name = "Maykeye/TinyLLama-v0"
        # After running the code below on Llama, try the following model
        # (with BPE tokenization) instead
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
    # tokenizer? TBH I'm not sure how to generalize this properly.
    return not isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast))


########################################################################################
############################## Set up model and tokenizer ##############################
########################################################################################
class _model_eval_mode:
    """
    Set the model in eval mode. CAPPr only makes sense as an inference computation.
    """

    def __init__(self, model: ModelForCausalLM):
        self.model = model
        self.training = model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, *args):
        self.model.train(self.training)


class _tokenizer_pad:
    """
    Set the pad token (if it's not set) so that batch inference is possible. These get
    masked out. Keep in mind that you need to be careful about setting position IDs
    correctly.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __enter__(self):
        # Note: a PreTrainedTokenizerBase tokenizer is smart about setting auxiliary
        # attributes, e.g., it updates tokenizer.special_tokens_map after setting
        # tokenizer.pad_token_id.
        if self.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __exit__(self, *args):
        self.tokenizer.pad_token_id = self.pad_token_id


class _tokenizer_pad_on_right:
    """
    Set the padding side to right. Left-padding would alter position IDs for non-pad
    tokens, which makes things a bit more confusing.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.padding_side = tokenizer.padding_side

    def __enter__(self):
        self.tokenizer.padding_side = "right"

    def __exit__(self, *args):
        self.tokenizer.padding_side = self.padding_side


class _tokenizer_dont_add_eos_token:
    """
    Don't add an end-of-sentence token. We'll never need it for the CAPPr scheme.
    Adding them would throw off :mod:`cappr.huggingface.classify`
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.tokenizer_has_attr_add_eos_token = hasattr(tokenizer, "add_eos_token")
        if self.tokenizer_has_attr_add_eos_token:
            self.add_eos_token: bool = tokenizer.add_eos_token

    def __enter__(self):
        if self.tokenizer_has_attr_add_eos_token:
            self.tokenizer.add_eos_token = False

    def __exit__(self, *args):
        if self.tokenizer_has_attr_add_eos_token:
            self.tokenizer.add_eos_token = self.add_eos_token


_DEFAULT_CONTEXTS_MODEL = (_model_eval_mode,)
"Model settings: set the model in eval mode."


_DEFAULT_CONTEXTS_TOKENIZER = (
    _tokenizer_pad,
    _tokenizer_pad_on_right,
    _tokenizer_dont_add_eos_token,
)
"Tokenizer settings: pad on right, don't add EOS token."


@contextmanager
def set_up_model_and_tokenizer(
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    contexts_tokenizer: Collection = _DEFAULT_CONTEXTS_MODEL,
    contexts_model: Collection = _DEFAULT_CONTEXTS_TOKENIZER,
):
    """
    In this context, internal attributes of the model and tokenizer are set to enable
    correct, batched inference.

    Usage::

        with set_up_model_and_tokenizer(model_and_tokenizer):
            model, tokenizer = model_and_tokenizer
            # your model/tokenization code
    """
    model, tokenizer = model_and_tokenizer

    init_contexts_model = [context(model) for context in contexts_tokenizer]
    init_contexts_tokenizer = [context(tokenizer) for context in contexts_model]
    init_contexts = init_contexts_model + init_contexts_tokenizer
    for init_context in init_contexts:
        init_context.__enter__()

    yield

    for init_context in init_contexts:
        init_context.__exit__()


@contextmanager
def disable_add_bos_token(tokenizer: PreTrainedTokenizerBase):
    if hasattr(tokenizer, "add_bos_token"):
        add_bos_token: bool = tokenizer.add_bos_token
        tokenizer.add_bos_token = False

    yield

    if hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = add_bos_token


########################################################################################
##################################### Logits stuff #####################################
########################################################################################
def logits_texts(
    texts: Sequence[str],
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, BatchEncoding]:
    """
    Basically `model(**tokenizer(texts)), tokenizer(texts)`.
    """
    # TODO: auto-batch? consider adding a batch_size kwarg, and decorating the func like
    # token_logprobs
    encodings: BatchEncoding = tokenizer(texts, return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.no_grad():
        out: CausalLMOutput = model(**encodings)
    if getattr(tokenizer, "add_bos_token", False):
        # Drop the first bos token after we're done encoding so that the shape is
        # consistent w/ other tokenizers
        logits = out.logits[:, 1:, :]
        encodings: BatchEncoding = {
            key: value[:, 1:] for key, value in encodings.items()
        }
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

"""
Perform prompt-completion classification using a model which can be loaded via

- ``transformers.AutoModelForCausalLM.from_pretrained`` or
- ``auto_gptq.AutoGPTQForCausalLM.from_quantized``.

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)

In the implementation, attention block keys and values for prompts are automatically
cached and shared across completions.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import cast, Literal, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

from cappr.utils import _batch, _check, classify
from cappr import Example
from cappr import huggingface as hf
from cappr.huggingface._utils import BatchEncodingPT, ModelForCausalLM


@classify._token_logprobs
@_batch.flatten
@_batch.batchify(batchable_arg="texts", progress_bar_desc="log-probs")
def token_logprobs(
    texts: str | Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    add_bos: bool = False,
    batch_size: int = 16,
    **kwargs,
) -> list[float] | list[list[float]]:
    """
    For each text, compute each token's log-probability conditional on all previous
    tokens in the text.

    Parameters
    ----------
    texts : str | Sequence[str]
        input text(s)
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    end_of_prompt : Literal[' ', ''], optional
        This string gets added to the beginning of each text. It's important to set this
        if you're using the discount feature. Otherwise, set it to "". By default " "
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    add_bos : bool, optional
        whether or not to add a beginning-of-sentence token to each text in `texts` if
        the tokenizer has a beginning-of-sentence token, by default False
    batch_size : int, optional
        the maximum number of `texts` that the model will process in parallel, by
        default 16

    Returns
    -------
    log_probs : list[float] | list[list[float]]

        If `texts` is a string, then a 1-D list is returned: `log_probs[token_idx]` is
        the log-probability of the token at `token_idx` of `texts` conditional on all
        previous tokens in `texts`.

        If `texts` is a sequence of strings, then a 2-D list is returned:
        `log_probs[text_idx][token_idx]` is the log-probability of the token at
        `token_idx` of `texts[text_idx]` conditional on all previous tokens in
        `texts[text_idx]`.

    Warning
    -------
    Set `end_of_prompt="", add_bos=True` unless you're using the discount feature.

    Note
    ----
    For each text, the first token's log-probability is always ``None`` because no
    autoregressive LM directly estimates the marginal probability of a token.

    Raises
    ------
    TypeError
        if `texts` is not a sequence
    ValueError
        if `texts` is empty
    """
    model, tokenizer = model_and_tokenizer
    if not hf._utils.does_tokenizer_need_prepended_space(tokenizer):
        end_of_prompt = ""
    texts = [end_of_prompt + text for text in texts]

    with hf._utils.dont_add_bos_token(tokenizer) if not add_bos else nullcontext():
        logits, encodings = hf._utils.logits_texts(
            texts, (model, tokenizer), drop_bos_token=not add_bos
        )
    log_probs_texts = hf._utils.logits_to_log_probs(
        logits=logits,
        input_ids=encodings["input_ids"],
        input_ids_start_idx=1,  # this token's log-prob is in the prev token's logit
        logits_end_idx=-1,
    )

    # Remove pad token logprobs
    num_non_pad_tokens = encodings["attention_mask"].sum(dim=1)
    log_probs = []
    first_token_log_prob = [None]
    for log_probs_text, n in zip(log_probs_texts, num_non_pad_tokens):
        # Slice off the right side b/c the tokenizer was set up to do right-padding
        log_probs.append(first_token_log_prob + log_probs_text[: (n - 1)].tolist())
    return log_probs


########################################################################################
########################## Attention past_key_values utilities #########################
########################################################################################


_PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor]]
"""
The `past_key_values` input to a HuggingFace `transformers` model's forward pass. It's a
2-D tuple of 4-D tensors.

Index items:

(
    # attention blocks = 12 for gpt2,

    2 = (attention keys, attention values), (or 6 for LongLlama)

    batch size = # input texts,

    # heads = 12 for gpt2,

    max # tokens in batch,

    key/value hidden dimension = 64 for gpt2
)
"""


# past_key_values must be a tuple for model calls. So we have to do slightly costly
# transfers from Python to CUDA. I don't think there's anything we can do about that


def _past_key_values_tuple_to_tensor(past_key_values: _PastKeyValues) -> torch.Tensor:
    if past_key_values is None:
        raise TypeError(
            "past_key_values is None. Can your model be configured to output it? If "
            "not, please use cappr.huggingface.classify_no_cache"
        )  # pragma: no cover
    return torch.stack([torch.stack(block) for block in past_key_values], dim=0)


def _past_key_values_tensor_to_tuple(past_key_values: torch.Tensor) -> _PastKeyValues:
    return tuple(
        tuple(block[i] for i in range(len(block))) for block in past_key_values
    )


def _past_key_values_get(
    past_key_values: _PastKeyValues,
    batch_idxs: Sequence[int],
) -> _PastKeyValues:
    return _past_key_values_tensor_to_tuple(
        _past_key_values_tuple_to_tensor(past_key_values)[:, :, batch_idxs, ...]
    )


########################################################################################
############################# KV caching + batch inference #############################
########################################################################################


@dataclass
class _CAPPr:
    model: ModelForCausalLM
    logits_all: bool = True
    _past: tuple[BatchEncodingPT, CausalLMOutputWithPast] | None = None
    batch_idxs: torch.Tensor | None = None
    update_cache: bool = False
    _is_cache_cleared: bool = False

    @property
    def past(self):
        return self._past

    @past.setter
    def past(self, new_past: tuple[BatchEncodingPT, CausalLMOutputWithPast] | None):
        if new_past is None:
            self._is_cache_cleared = True
        self._past = new_past


class _CacheClearedError(Exception):
    """Raise to prevent a user from using a cached model outside of the context"""


class _ModelWithCache:
    def __init__(
        self,
        model: ModelForCausalLM,
        encodings_to_cache: BatchEncodingPT,
        past: tuple[BatchEncodingPT, CausalLMOutputWithPast] | None = None,
        logits_all: bool = True,
    ):
        self._cappr = _CAPPr(model, logits_all, past)
        """
        Contains data which controls the cache
        """
        # This data is in one place to minimize pollution of the inputted model's
        # namespace. This object should be treated like a ModelForCausalLM by the user
        self._cappr.update_cache = True
        _ = self.forward(**encodings_to_cache)
        del _
        self._cappr.update_cache = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> CausalLMOutputWithPast:
        if self._cappr._is_cache_cleared:
            raise _CacheClearedError(
                "This model is no longer usable. It was used in a context where "
                "clear_cache_on_exit=True. If you instead meant to retain the cache "
                "for future use, then use: `cached_model_and_tokenizer = "
                "cache_model(model_and_tokenizer, prefixes)`"
            )

        encodings = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self._cappr.past is None:
            with hf._utils.set_up_model(self._cappr.model):
                out: CausalLMOutputWithPast = self._cappr.model(**encodings)
            self._cappr.past = encodings, out
            return out

        encodings_past, out_past = self._cappr.past

        if self._cappr.batch_idxs is None:
            # Attempt to broadcast the past with the present
            if encodings_past["input_ids"].shape[0] == 1 and input_ids.shape[0] > 1:
                # Assume the past of present input[i] is past[0] (the only past)
                batch_idxs = torch.zeros(
                    input_ids.shape[0], device=self._cappr.model.device, dtype=torch.int
                )
            elif encodings_past["input_ids"].shape[0] == input_ids.shape[0]:
                # Assume the past of present input[i] is past[i]
                batch_idxs = torch.arange(
                    input_ids.shape[0], device=self._cappr.model.device
                )
            else:
                raise ValueError(
                    f'Could not broadcast {encodings_past["input_ids"].shape[0]} '
                    f"cached inputs with {input_ids.shape[0]} new inputs."
                )
        else:
            # Externally set to a tensor. Bad design but whatever
            batch_idxs: torch.Tensor = self._cappr.batch_idxs

        if not torch.equal(
            batch_idxs,
            torch.arange(
                encodings_past["input_ids"].shape[0], device=self._cappr.model.device
            ),
        ):
            # Must extract past by converting the tuple to a tensor and back
            past_key_values = _past_key_values_get(out_past.past_key_values, batch_idxs)
        else:
            past_key_values = out_past.past_key_values

        attention_mask_past = encodings_past["attention_mask"][batch_idxs]
        attention_mask_past_cat_present = torch.cat(
            [attention_mask_past, encodings["attention_mask"]], dim=1
        )
        # Set position_ids to what they'd be had we fed prompt + completion together
        # Some model implementations don't do this, so gotta do it manually
        offsets = attention_mask_past.sum(dim=1)  # numbers of nonpad tokens in past
        position_ids_present = (
            torch.arange(input_ids.shape[1], device=self._cappr.model.device)
            + offsets[:, None]
        )

        with hf._utils.set_up_model(self._cappr.model):
            out: CausalLMOutputWithPast = self._cappr.model(
                input_ids=input_ids,
                attention_mask=attention_mask_past_cat_present,
                position_ids=position_ids_present,
                past_key_values=past_key_values,
            )

        # past_key_values is already concatenated in out.past_key_values
        if self._cappr.logits_all:
            out.logits = torch.cat([out_past.logits[batch_idxs], out.logits], dim=1)
        if self._cappr.update_cache:
            # Concatenate encodings for future model calls
            input_ids_past = encodings_past["input_ids"][batch_idxs]
            encodings = {
                "input_ids": torch.cat([input_ids_past, input_ids], dim=1),
                "attention_mask": attention_mask_past_cat_present,
            }
            encodings = cast(BatchEncodingPT, encodings)
            self._cappr.past = encodings, out
        return out

    def __call__(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> CausalLMOutputWithPast:
        return self.forward(input_ids, attention_mask)

    def __getattr__(self, __name: str):
        return getattr(self._cappr.model, __name)

    def __repr__(self) -> str:
        return repr(self._cappr.model)


def cache_model(
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    prefixes: str | Sequence[str],
    logits_all: bool = True,
) -> tuple[ModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Caches the model so that every future computation with it starts with `prefixes`. As
    a result, computations with this model are faster.

    Use this function instead of the context manager :func:`cache` to keep the cache for
    future computations, including those outside of a context.

    Parameters
    ----------
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    prefixes : str | Sequence[str]
        prefix(es) for all strings that will be processed in this context, e.g., a
        string containing shared prompt instructions, or a string containing
        instructions and exemplars for few-shot prompting
    logits_all : bool, optional
        whether or not to have the cached model include logits for all tokens (including
        the past). By default, past token logits are included

    Returns
    -------
    tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        cached model and the (unmodified) tokenizer

    Note
    ----
    If you're inputting the cached model and tokenizer to a function in this module,
    e.g., :func:`predict`, `prefixes` and future strings are assumed to be separated by
    a whitespace. Otherwise, ensure that any strings that are processed by the tokenizer
    start correctly. Furthermore, if applicable, set ``tokenizer.add_bos_token = False``
    for future computations.

    Example
    -------
    Usage with :func:`predict_proba`::

        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import cache_model, predict_proba

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model_and_tokenizer = (model, tokenizer)

        # Create data
        prompt_prefix = '''Instructions: complete the sequence.
        Here are examples:
        A, B, C => D
        1, 2, 3 => 4

        Complete this sequence:'''

        prompts = ["a, b, c =>", "X, Y =>"]
        completions = ["d", "Z", "Hi"]

        # Cache
        cached_model_and_tokenizer = cache_model(
            model_and_tokenizer, prompt_prefix
        )

        # Compute
        pred_probs = predict_proba(
            prompts, completions, cached_model_and_tokenizer
        )

        # The above computation is equivalent to this one:
        prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
        pred_probs_wo_cache = predict_proba(
            prompts_full, completions, model_and_tokenizer
        )
        assert np.allclose(pred_probs, pred_probs_wo_cache, atol=1e-5)

        print(pred_probs.round(1))
        # [[1. 0. 0.]
        #  [0. 1. 0.]]
    """
    _check.nonempty_and_ordered(prefixes, variable_name="prefixes")
    prefixes = [prefixes] if isinstance(prefixes, str) else list(prefixes)
    model, tokenizer = model_and_tokenizer
    try:
        past = getattr(getattr(model, "_cappr"), "past")
    except AttributeError:
        past = None

    with hf._utils.set_up_tokenizer(tokenizer):
        with nullcontext() if past is None else hf._utils.dont_add_bos_token(tokenizer):
            encodings_prefixes: BatchEncodingPT = tokenizer(
                prefixes, return_tensors="pt", padding=True
            ).to(model.device)

    model_for_causal_lm = (
        model._cappr.model if isinstance(model, _ModelWithCache) else model
    )
    model_with_cache = _ModelWithCache(
        model_for_causal_lm,
        encodings_to_cache=encodings_prefixes,
        past=past,
        logits_all=logits_all,
    )
    return model_with_cache, tokenizer  # return tokenizer for consistent interface


@contextmanager
def cache(
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    prefixes: str | Sequence[str],
    clear_cache_on_exit: bool = True,
    logits_all: bool = True,
):
    """
    In this context, every prompt processed by `model_and_tokenizer` starts with a fixed
    prefix. As a result, computations in this context are faster.

    Parameters
    ----------
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    prefixes : str | Sequence[str]
        prefix(es) for all strings that will be processed in this context, e.g., a
        string containing shared prompt instructions, or a string containing
        instructions and exemplars for few-shot prompting
    clear_cache_on_exit : bool, optional
        whether or not to clear the cache and render the returned model and tokenizer
        unusable when we exit the context. This is important because it saves memory,
        and makes code more explicit about the model's state. By default, True
    logits_all : bool, optional
        whether or not to have the cached model include logits for all tokens (including
        the past). By default, past token logits are included

    Note
    ----
    If you're inputting the cached model and tokenizer to a function in this module,
    e.g., :func:`predict`, `prefixes` and future strings are assumed to be separated by
    a whitespace. Otherwise, ensure that any strings that are processed by the tokenizer
    start correctly.

    Example
    -------
    Usage with :func:`predict_proba`::

        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import cache, predict_proba

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model_and_tokenizer = (model, tokenizer)

        # Create data
        prompt_prefix = '''Instructions: complete the sequence.
        Here are examples:
        A, B, C => D
        1, 2, 3 => 4

        Complete this sequence:'''

        prompts = ["a, b, c =>", "X, Y =>"]
        completions = ["d", "Z", "Hi"]

        # Compute
        with cache(
            model_and_tokenizer, prompt_prefix
        ) as cached_model_and_tokenizer:
            # prompt_prefix and each prompt are separated by a whitespace
            pred_probs = predict_proba(
                prompts, completions, cached_model_and_tokenizer
            )

        # The above computation is equivalent to this one:
        prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
        pred_probs_wo_cache = predict_proba(
            prompts_full, completions, model_and_tokenizer
        )
        assert np.allclose(pred_probs, pred_probs_wo_cache, atol=1e-5)

        print(pred_probs.round(1))
        # [[1. 0. 0.]
        #  [0. 1. 0.]]

    Here's a more complicated example, which might help in explaining usage::

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import cache
        from cappr.huggingface._utils import (
            does_tokenizer_need_prepended_space,
            logits_texts,
        )

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model_and_tokenizer = (model, tokenizer)

        # Assume that all strings will be separated by a whitespace
        delim = " "
        if not does_tokenizer_need_prepended_space(tokenizer):
            # for SentencePiece tokenizers like Llama's
            delim = ""

        logits = lambda *args, **kwargs: logits_texts(*args, **kwargs)[0]
        '''
        Returns next-token logits for each token in an inputted text.
        '''

        with cache(model_and_tokenizer, "a") as cached_a:
            with cache(cached_a, delim + "b c") as cached_a_b_c:
                with cache(cached_a_b_c, delim + "d") as cached_a_b_c_d:
                    logits1 = logits([delim + "e f"], cached_a_b_c_d)
                    logits2 = logits([delim + "x"], cached_a_b_c_d)
                logits3 = logits([delim + "1 2 3"], cached_a_b_c)
            logits4 = logits([delim + "b c d"], cached_a)

        logits_correct = lambda texts, **kwargs: logits(
            texts, model_and_tokenizer, drop_bos_token=False
        )

        atol = 1e-4
        assert torch.allclose(logits1, logits_correct(["a b c d e f"]), atol=atol)
        assert torch.allclose(logits2, logits_correct(["a b c d x"]), atol=atol)
        assert torch.allclose(logits3, logits_correct(["a b c 1 2 3"]), atol=atol)
        assert torch.allclose(logits4, logits_correct(["a b c d"]), atol=atol)
    """
    try:
        past = getattr(getattr(model_and_tokenizer[0], "_cappr"), "past")
    except AttributeError:
        past = None
    model_with_cache, tokenizer = cache_model(
        model_and_tokenizer, prefixes, logits_all=logits_all
    )
    model_with_cache = cast(_ModelWithCache, model_with_cache)
    # Now that model_with_cache has a past, we should never add a BOS token
    with hf._utils.dont_add_bos_token(tokenizer), hf._utils.set_up_tokenizer(tokenizer):
        yield model_with_cache, tokenizer

    if clear_cache_on_exit:
        # model_with_cache._cappr.past contains a lot of data—logits, past_key_values,
        # hidden_states (usually taking up GPU RAM)—that should be cleared when we exit
        # the context
        model_with_cache._cappr.past = None
    else:
        model_with_cache._cappr.past = past


########################################################################################
############################### Logits from cached model ###############################
########################################################################################


def _last_nonpad_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor):
    last_nonpad_token_idxs = (
        logits.shape[1] - (attention_mask.flip(dims=[1]) == 1).max(dim=1).indices - 1
    )
    return logits.take_along_dim(last_nonpad_token_idxs[:, None, None], dim=1)


def _blessed_helper(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    num_completions_per_prompt: int | Sequence[int],
    completions_repeats: int,
    batch_size_completions: int | None = None,
) -> tuple[torch.Tensor, BatchEncodingPT]:
    if not isinstance(num_completions_per_prompt, int):
        num_completions_per_prompt: torch.Tensor = torch.tensor(
            num_completions_per_prompt, device=model.device
        )
    batch_size_completions = batch_size_completions or len(completions)
    prompts_idxs: tuple[torch.Tensor] = (
        torch.arange(len(prompts), device=model.device)
        .repeat_interleave(num_completions_per_prompt)
        .split(batch_size_completions)
    )
    # Prepare completions data
    # Completions shouldn't start w/ a bos token <s> b/c we need to mimic sending the
    # prompt + completion together. For example, if 'a b' is the prompt and 'c' is the
    # completion, the encoding should correspond to '<s> a b c' not '<s> a b <s> c'
    with hf._utils.set_up_tokenizer(tokenizer), hf._utils.dont_add_bos_token(tokenizer):
        # This computation is repeated for constant completions, but it doesn't matter
        completions_encoding: BatchEncodingPT = tokenizer(
            completions, return_tensors="pt", padding=True
        ).to(model.device)
    completions_encoding = cast(BatchEncodingPT, completions_encoding)
    # Repeat then batch completions. The other way around corrupts the order
    completions_input_ids = torch.split(
        completions_encoding["input_ids"].repeat(completions_repeats, 1),
        batch_size_completions,
    )
    completions_attention_mask = torch.split(
        completions_encoding["attention_mask"].repeat(completions_repeats, 1),
        batch_size_completions,
    )
    num_batches = len(completions_input_ids)

    # TODO: put this in the context manager? Little weird
    if not hf._utils.does_tokenizer_need_prepended_space(tokenizer):
        start_of_prompt = ""
    else:
        in_cache_context = isinstance(model, _ModelWithCache)
        start_of_prompt = " " if in_cache_context else ""
    prompts = [start_of_prompt + prompt for prompt in prompts]

    # Run model, caching prompts
    with cache((model, tokenizer), prompts, logits_all=False) as cached_prompts:
        cached_prompts_model, _ = cached_prompts
        prompts_encodings, prompts_out = cached_prompts_model._cappr.past
        prompts_last_nonpad_token_logits = _last_nonpad_token_logits(
            prompts_out.logits, prompts_encodings["attention_mask"]
        )
        # fmt: off
        _are_completions_constant = (
            isinstance(num_completions_per_prompt, int) and
            completions_repeats == len(prompts)
        )
        # fmt: on
        if (
            completions_encoding["input_ids"].shape[1] == 1
            and _are_completions_constant
        ):
            # Single-token optimization: if every completion is a single token, we don't
            # need to repeat stuff or run the model on any of the completions data
            # Currently, this optimization is only done for constant completions, i.e.,
            # not _examples
            # Note that completions_encoding["input_ids"].shape[1] == logits.shape[1]
            return prompts_last_nonpad_token_logits, completions_encoding

        completions_logits = []
        _batch_idxs = cached_prompts_model._cappr.batch_idxs
        for batch_idx in range(num_batches):
            cached_prompts_model._cappr.batch_idxs = prompts_idxs[batch_idx]
            out: CausalLMOutput = cached_prompts_model(
                input_ids=completions_input_ids[batch_idx],
                attention_mask=completions_attention_mask[batch_idx],
            )
            # B/c logits_all=False, out.logits only has completion token logits
            completions_logits.append(out.logits)
        cached_prompts_model._cappr.batch_idxs = _batch_idxs
        offsets = prompts_encodings["attention_mask"].sum(dim=1)

    # Drop the next-token logits for the last completion token. They're not useful for
    # CAPPr. Moreover, dropping ensures completions_logits.shape[:2] ==
    # completions_encoding["input_ids"].shape, as one expects. Just keep in mind that
    # `logits` are shifted behind
    completions_logits = torch.cat(completions_logits, dim=0)[:, :-1, :]
    prompts_last_nonpad_token_logits = (
        prompts_last_nonpad_token_logits.repeat_interleave(
            num_completions_per_prompt, dim=0
        )
    )
    completions_logits = torch.cat(
        [prompts_last_nonpad_token_logits, completions_logits], dim=1
    )
    # You may need the offsets to be able to ignore pad tokens
    completions_encoding = {
        "input_ids": torch.cat(completions_input_ids),
        "attention_mask": torch.cat(completions_attention_mask),
        "offsets": offsets.repeat_interleave(num_completions_per_prompt, dim=0),
    }
    if getattr(tokenizer, "add_bos_token", False):
        # Drop the first <s> token after we're done encoding so that the shape is
        # consistent w/ other tokenizers
        completions_encoding["offsets"] = completions_encoding["offsets"] - 1
    return completions_logits, completions_encoding


def _logits_completions_given_prompts(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: Literal[" ", ""] = " ",
    batch_size_completions: int | None = None,
) -> tuple[torch.Tensor, BatchEncodingPT]:
    if not hf._utils.does_tokenizer_need_prepended_space(tokenizer):
        end_of_prompt = ""
    completions = [end_of_prompt + completion for completion in completions]
    return _blessed_helper(
        model,
        tokenizer,
        prompts,
        completions,
        num_completions_per_prompt=len(completions),
        completions_repeats=len(prompts),
        batch_size_completions=batch_size_completions,
    )


def _logits_completions_given_prompts_examples(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
    batch_size_completions: int | None = None,
) -> tuple[torch.Tensor, BatchEncodingPT]:
    should_end_of_prompt_be_empty = not hf._utils.does_tokenizer_need_prepended_space(
        tokenizer
    )
    prompts = [example.prompt for example in examples]
    end_of_prompts = [
        "" if should_end_of_prompt_be_empty else example.end_of_prompt
        for example in examples
    ]
    completions = [
        end_of_prompt + completion
        for end_of_prompt, example in zip(end_of_prompts, examples)
        for completion in example.completions
    ]
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return _blessed_helper(
        model,
        tokenizer,
        prompts,
        completions,
        num_completions_per_prompt=num_completions_per_prompt,
        completions_repeats=1,
        batch_size_completions=batch_size_completions,
    )


########################################################################################
################################## Logits to log-probs #################################
########################################################################################


def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor], from_examples: bool
) -> list[list[float]]:
    if (not from_examples) and logits.shape[1] == 1:
        # Single-token optimization: all of the completions are always a single token
        # Slice out their tokens from the prompts' last non-pad token logits. Currently,
        # this optimization is only done for constant completions, i.e., not _examples
        completions_input_ids: torch.Tensor = (
            encodings["input_ids"]
            .repeat_interleave(logits.shape[0], dim=1)  # the number of prompts
            .T
        )
        log_probs = hf._utils.logits_to_log_probs(logits, completions_input_ids)
        # Need to reshape them to the expected shape
        return log_probs.flatten()[:, None].tolist()

    # There are some completions with multiple tokens
    log_probs = hf._utils.logits_to_log_probs(logits, encodings["input_ids"])
    last_idx_non_pad = encodings["attention_mask"].sum(dim=1)
    # i.e., # of tokens per completion
    return [
        log_probs_prompt_completion[:completion_end].tolist()
        for log_probs_prompt_completion, completion_end in zip(
            log_probs, last_idx_non_pad
        )
    ]


########################################################################################
##################################### Implementation ###################################
########################################################################################


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
    **kwargs,
) -> list[list[float]] | list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt and previous
    completion tokens.

    Parameters
    ----------
    prompts : str | Sequence[str]
        string(s), where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    batch_size : int, optional
        the maximum number of `prompts` that the model will process in parallel, by
        default 2
    batch_size_completions : int, optional
        the maximum number of `completions` that the model will process in parallel. By
        default, all completions are processed in parallel

    Returns
    -------
    log_probs_completions : list[list[float]] | list[list[list[float]]]

        If `prompts` is a string, then a 2-D list is returned:
        `log_probs_completions[completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompt + end_of_prompt` and previous completion tokens.

        If `prompts` is a sequence of strings, then a 3-D list is returned:
        `log_probs_completions[prompt_idx][completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompts[prompt_idx] + end_of_prompt` and previous completion
        tokens.

    Note
    ----
    To efficiently aggregate `log_probs_completions`, use
    :func:`cappr.utils.classify.agg_log_probs`.

    Example
    -------
    Here we'll use single characters (which are single tokens) to more clearly
    demonstrate what this function does::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import log_probs_conditional

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Create data
        prompts = ["x y", "a b c"]
        completions = ["z", "d e"]

        # Compute
        log_probs_completions = log_probs_conditional(
            prompts, completions, model_and_tokenizer=(model, tokenizer)
        )

        # Outputs (rounded) next to their symbolic representation

        print(log_probs_completions[0])
        # [[-4.5],        [[log Pr(z | x, y)],
        #  [-5.6, -3.2]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        print(log_probs_completions[1])
        # [[-9.7],        [[log Pr(z | a, b, c)],
        #  [-0.2, -0.03]]  [log Pr(d | a, b, c), log Pr(e | a, b, c, d)]]
    """

    @_batch.flatten
    @_batch.batchify(batchable_arg="prompts", progress_bar_desc="conditional log-probs")
    def log_probs_completions_batch(
        prompts, show_progress_bar=show_progress_bar, batch_size=batch_size
    ):
        logits, encodings = _logits_completions_given_prompts(
            *model_and_tokenizer,
            prompts,
            completions,
            end_of_prompt=end_of_prompt,
            batch_size_completions=batch_size_completions,
        )
        return _logits_to_log_probs_completions(logits, encodings, from_examples=False)

    log_probs_completions = log_probs_completions_batch(prompts)
    return list(_batch.constant(log_probs_completions, size=len(completions)))


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    show_progress_bar: bool | None = None,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
) -> list[list[float]] | list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt and previous
    completion tokens.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    batch_size : int, optional
        the maximum number of `examples` that the model will process in parallel, by
        default 2
    batch_size_completions : int, optional
        the maximum number of `completions` that the model will process in parallel. By
        default, all completions are processed in parallel

    Returns
    -------
    log_probs_completions : list[list[float]] | list[list[list[float]]]

        If `examples` is a :class:`cappr.Example`, then a 2-D list is returned:
        `log_probs_completions[completion_idx][completion_token_idx]` is the
        log-probability of the completion token in
        `example.completions[completion_idx]`, conditional on `example.prompt +
        example.end_of_prompt` and previous completion tokens.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a 3-D list
        is returned:
        `log_probs_completions[example_idx][completion_idx][completion_token_idx]` is
        the log-probability of the completion token in
        `examples[example_idx].completions[completion_idx]`, conditional on
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt` and
        previous completion tokens.

    Note
    ----
    To aggregate `log_probs_completions`, use
    :func:`cappr.utils.classify.agg_log_probs`.

    Note
    ----
    The attribute :attr:`cappr.Example.prior` is unused.

    Example
    -------
    Here we'll use single characters (which are single tokens) to more clearly
    demonstrate what this function does::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr import Example
        from cappr.huggingface.classify import log_probs_conditional_examples

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Create examples
        examples = [
            Example(prompt="x y", completions=("z", "d e")),
            Example(prompt="a b c", completions=("1 2",), normalize=False),
        ]

        # Compute
        log_probs_completions = log_probs_conditional_examples(
            examples, model_and_tokenizer=(model, tokenizer)
        )

        # Outputs (rounded) next to their symbolic representation

        print(log_probs_completions[0])  # corresponds to examples[0]
        # [[-4.5],        [[log Pr(z | x, y)],
        #  [-5.6, -3.2]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        print(log_probs_completions[1])  # corresponds to examples[1]
        # [[-5.0, -1.7]]  [[log Pr(1 | a, b, c)], log Pr(2 | a, b, c, 1)]]
    """
    # examples is always a Sequence[Example] b/c of the decorator
    examples = cast(Sequence[Example], examples)

    @_batch.flatten
    @_batch.batchify(
        batchable_arg="examples", progress_bar_desc="conditional log-probs"
    )
    def log_probs_completions_batch(
        examples, show_progress_bar=show_progress_bar, batch_size=batch_size
    ):
        logits, encodings = _logits_completions_given_prompts_examples(
            *model_and_tokenizer,
            examples,
            batch_size_completions=batch_size_completions,
        )
        return _logits_to_log_probs_completions(logits, encodings, from_examples=True)

    log_probs_completions = log_probs_completions_batch(examples)
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(
        _batch.variable(log_probs_completions, sizes=num_completions_per_prompt)
    )


@classify._predict_proba
def predict_proba(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    prior: Sequence[float] | None = None,
    end_of_prompt: Literal[" ", ""] = " ",
    normalize: bool = True,
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    show_progress_bar: bool | None = None,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
) -> npt.NDArray[np.floating]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    prompts : str | Sequence[str]
        string(s), where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    normalize : bool, optional
        whether or not to normalize completion-after-prompt probabilities into a
        probability distribution over completions. Set this to `False` if you'd like the
        raw completion-after-prompt probability, or you're solving a multi-label
        prediction problem. By default, True
    discount_completions : float, optional
        experimental feature: set it (e.g., 1.0 may work well) if a completion is
        consistently getting too high predicted probabilities. You could instead fudge
        the `prior`, but this hyperparameter may be easier to tune than the `prior`. By
        default 0.0
    log_marg_probs_completions : Sequence[Sequence[float]] | None, optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Pre-compute them by passing `completions`, `model`, and
        `end_of_prompt` to :func:`token_logprobs`. By default, if `not
        discount_completions`, they are (re-)computed
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    batch_size : int, optional
        the maximum number of `prompts` that the model will process in parallel, by
        default 2
    batch_size_completions : int, optional
        the maximum number of `completions` that the model will process in parallel. By
        default, all completions are processed in parallel

    Returns
    -------
    pred_probs : npt.NDArray[np.floating]

        If `prompts` is a string, then an array with shape `len(completions),` is
        returned: `pred_probs[completion_idx]` is the model's estimate of the
        probability that `completions[completion_idx]` comes after `prompt +
        end_of_prompt`.

        If `prompts` is a sequence of strings, then an array with shape `(len(prompts),
        len(completions))` is returned: `pred_probs[prompt_idx, completion_idx]` is the
        model's estimate of the probability that `completions[completion_idx]` comes
        after `prompts[prompt_idx] + end_of_prompt`.

    Note
    ----
    In this function, the set of possible completions which could follow each prompt is
    the same for every prompt. If instead, each prompt could be followed by a
    *different* set of completions, then construct a sequence of :class:`cappr.Example`
    objects and pass them to :func:`predict_proba_examples`.

    Example
    -------
    Let's have GPT-2 (small) predict where stuff is in the kitchen. This example also
    conveys that it's not the greatest model out there::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import predict_proba

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Define a classification task
        prompts = ["The tacos are cooking", "Ice cream is"]
        class_names = ("on the stove", "in the freezer", "in the fridge")
        prior = (1 / 5, 2 / 5, 2 / 5)

        pred_probs = predict_proba(
            prompts,
            completions=class_names,
            model_and_tokenizer=(model, tokenizer),
            prior=prior,
        )
        pred_probs_rounded = pred_probs.round(1)  # just for cleaner output

        # predicted probability that tacos cook on the stove
        print(pred_probs_rounded[0, 0])
        # 0.4

        # predicted probability that ice cream is in the freezer
        print(pred_probs_rounded[1, 1])
        # 0.5

        # predicted probability that ice cream is in the fridge
        print(pred_probs_rounded[1, 2])
        # 0.4
    """
    return log_probs_conditional(**locals())


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Example | Sequence[Example],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    show_progress_bar: bool | None = None,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    batch_size : int, optional
        the maximum number of `examples` that the model will process in parallel, by
        default 2
    batch_size_completions : int, optional
        the maximum number of `completions` that the model will process in parallel. By
        default, all completions are processed in parallel

    Returns
    -------
    pred_probs : npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]

        If `examples` is an :class:`cappr.Example`, then an array with shape
        `(len(example.completions),)` is returned: `pred_probs[completion_idx]` is the
        model's estimate of the probability that `example.completions[completion_idx]`
        comes after `example.prompt + example.end_of_prompt`.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a list with
        length `len(examples)` is returned: `pred_probs[example_idx][completion_idx]` is
        the model's estimate of the probability that
        `examples[example_idx].completions[completion_idx]` comes after
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`. If the
        number of completions per example is a constant `k`, then an array with shape
        `(len(examples), k)` is returned instead of a list of 1-D arrays.

    Example
    -------
    GPT-2 (small) doing media trivia::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr import Example
        from cappr.huggingface.classify import predict_proba_examples

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Create examples
        examples = [
            Example(
                prompt="Jodie Foster played",
                completions=("Clarice Starling", "Trinity in The Matrix"),
            ),
            Example(
                prompt="Batman, from Batman: The Animated Series, was played by",
                completions=("Pete Holmes", "Kevin Conroy", "Spongebob!"),
                prior=(1 / 3, 2 / 3, 0),
            ),
        ]

        pred_probs = predict_proba_examples(
            examples, model_and_tokenizer=(model, tokenizer)
        )

        # predicted probability that Jodie Foster played Clarice Starling, not Trinity
        print(pred_probs[0][0].round(2))
        # 0.7

        # predicted probability that Batman was played by Kevin Conroy
        print(pred_probs[1][1].round(2))
        # 0.97
    """
    return log_probs_conditional_examples(**locals())


@classify._predict
def predict(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    prior: Sequence[float] | None = None,
    end_of_prompt: Literal[" ", ""] = " ",
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    show_progress_bar: bool | None = None,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
) -> str | list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    prompts : str | Sequence[str]
        string(s), where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    discount_completions : float, optional
        experimental feature: set it to >0.0 (e.g., 1.0 may work well) if a completion
        is consistently getting over-predicted. You could instead fudge the `prior`, but
        this hyperparameter may be easier to tune than the `prior`. By default 0.0
    log_marg_probs_completions : Sequence[Sequence[float]] | None, optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Pre-compute them by passing `completions`, `model`, and
        `end_of_prompt` to :func:`token_logprobs`. By default, if `not
        discount_completions`, they are (re-)computed
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    batch_size : int, optional
        the maximum number of `prompts` that the model will process in parallel, by
        default 2
    batch_size_completions : int, optional
        the maximum number of `completions` that the model will process in parallel. By
        default, all completions are processed in parallel

    Returns
    -------
    preds : str | list[str]

        If `prompts` is a string, then the completion from `completions` which is
        predicted to most likely follow `prompt + end_of_prompt` is returned.

        If `prompts` is a sequence of strings, then a list with length `len(prompts)` is
        returned. `preds[prompt_idx]` is the completion in `completions` which is
        predicted to follow `prompts[prompt_idx] + end_of_prompt`.

    Note
    ----
    In this function, the set of possible completions which could follow each prompt is
    the same for every prompt. If instead, each prompt could be followed by a
    *different* set of completions, then construct a sequence of :class:`cappr.Example`
    objects and pass them to :func:`predict_examples`.

    Example
    -------
    Let's have GPT-2 (small) predict where stuff is in the kitchen::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import predict

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Define a classification task
        prompts = ["The tacos are cooking", "Ice cream is"]
        class_names = ("on the stove", "in the freezer", "in the fridge")
        prior = (1 / 5, 2 / 5, 2 / 5)

        preds = predict(
            prompts,
            completions=class_names,
            model_and_tokenizer=(model, tokenizer),
            prior=prior,
        )
        print(preds)
        # ['on the stove',
        #  'in the freezer']
    """
    return predict_proba(**locals())


@classify._predict_examples
def predict_examples(
    examples: Example | Sequence[Example],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    show_progress_bar: bool | None = None,
    batch_size: int = 2,
    batch_size_completions: int | None = None,
) -> str | list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        a model and its tokenizer
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    batch_size : int, optional
        the maximum number of `examples` that the model will process in parallel, by
        default 2
    batch_size_completions : int, optional
        the maximum number of `completions` that the model will process in parallel. By
        default, all completions are processed in parallel

    Returns
    -------
    preds : str | list[str]

        If `examples` is an :class:`cappr.Example`, then the completion from
        `example.completions` which is predicted to most likely follow `example.prompt +
        example.end_of_prompt` is returned.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a list with
        length `len(examples)` is returned: `preds[example_idx]` is the completion in
        `examples[example_idx].completions` which is predicted to most likely follow
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.

    Example
    -------
    GPT-2 (small) doing media trivia::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr import Example
        from cappr.huggingface.classify import predict_examples

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Create examples
        examples = [
            Example(
                prompt="Jodie Foster played",
                completions=("Clarice Starling", "Trinity in The Matrix"),
            ),
            Example(
                prompt="Batman, from Batman: The Animated Series, was played by",
                completions=("Pete Holmes", "Kevin Conroy", "Spongebob!"),
                prior=(1 / 3, 2 / 3, 0),
            ),
        ]

        preds = predict_examples(
            examples, model_and_tokenizer=(model, tokenizer)
        )
        print(preds)
        # ['Clarice Starling',
        #  'Kevin Conroy']
    """
    return predict_proba_examples(**locals())

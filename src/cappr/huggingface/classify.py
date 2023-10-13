"""
Perform prompt-completion classification using a model which can be loaded via

- ``transformers.AutoModelForCausalLM.from_pretrained`` or
- ``auto_gptq.AutoGPTQForCausalLM.from_quantized``.

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)

In the implementation, attention block keys and values for prompts are cached and shared
across completions.
"""
from __future__ import annotations
from typing import Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from cappr.utils import _batch, _check, classify
from cappr import Example
from cappr import huggingface as hf
from cappr.huggingface._utils import BatchEncoding, ModelForCausalLM


@_batch.flatten
@_batch.batchify(batchable_arg="texts", progress_bar_desc="log-probs")
def token_logprobs(
    texts: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    batch_size: int = 32,
    **kwargs,
) -> list[list[float]]:
    """
    For each text, compute each token's log-probability conditional on all previous
    tokens in the text.

    Parameters
    ----------
    texts : Sequence[str]
        input texts
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        an instantiated model and its corresponding tokenizer
    end_of_prompt : Literal[' ', ''], optional
        This string gets added to the beginning of each text. It's important to set this
        if you're using the discount feature. Otherwise, set it to "". By default " "
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    batch_size : int, optional
        the maximum number of `texts` that the model will process in parallel, by
        default 32

    Returns
    -------
    log_probs : list[list[float]]
        `log_probs[text_idx][token_idx]` is the log-probability of the token at
        `token_idx` of `texts[text_idx]` conditional on all previous tokens in
        `texts[text_idx]`. If `texts[text_idx]` is a single token, then
        `log_probs[text_idx]` is `[None]`.

    Raises
    ------
    TypeError
        if `texts` is a string
    TypeError
        if `texts` is not a sequence
    ValueError
        if `texts` is empty
    """
    # Input checks
    if isinstance(texts, str):
        raise TypeError("texts cannot be a string. It must be a sequence of strings.")
    _check.nonempty_and_ordered(texts, variable_name="texts")
    _check.end_of_prompt(end_of_prompt)

    with hf._utils.set_up_model_and_tokenizer(model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        # bleh
        if (
            end_of_prompt == " "
            and not hf._utils.does_tokenizer_prepend_space_to_first_token(tokenizer)
        ):
            end_of_prompt = ""
        texts = [end_of_prompt + text for text in texts]

        # Batch inference
        logits, encodings = hf._utils.logits_texts(texts, model, tokenizer)
        # y is this^ type hint not working
        encodings: BatchEncoding = encodings

        # Convert next-token logits to this-token logprobs.
        # It's still wrong to set input_ids_start_idx=0 for tokenizers which add a bos
        # token (like SentencePiece for Llama). Pr(token | <s>) is not Pr(token).
        log_probs_texts = hf._utils.logits_to_log_probs(
            logits=logits,
            input_ids=encodings["input_ids"],
            input_ids_start_idx=1,  # this token's log-prob is in the prev token's logit
            logits_end_idx=-1,
        )

        # Remove pad token logprobs
        num_non_pad_tokens = encodings["attention_mask"].sum(dim=1)
        log_probs = []
        first_token_log_prob = [
            None
        ]  # no CausalLM estimates Pr(token), so call it None
        for log_probs_text, n in zip(log_probs_texts, num_non_pad_tokens):
            # we slice off the right side b/c the tokenizer was set up to do padding on
            # the right
            log_probs.append(first_token_log_prob + log_probs_text[: (n - 1)].tolist())
        return log_probs


def _keys_values_prompts(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    num_repeats_per_prompt: int | Sequence[int],
) -> tuple[
    tuple[tuple[torch.Tensor, torch.Tensor]], BatchEncoding, torch.Tensor, torch.Tensor
]:
    """
    Efficiently performs this procedure:

    1. Repeat-interleave `prompts[i]` `num_repeats_per_prompt[i]` times.

       Or, if `num_repeats_per_prompt` is an integer, repeat-interleave `prompts[i]`
       `num_repeats_per_prompt` times.

       For example, if there are 2 prompts and `num_repeats_per_prompt=(2,3)`, the
       repeated prompts look like::

           [prompts[0],
            prompts[0],
            prompts[1],
            prompts[1],
            prompts[1]]

    2. Apply `tokenizer` to the repeated prompts.

    3. Apply `model`.
    """
    # Input checks
    if not tokenizer.padding_side == "right":
        raise ValueError(
            "Gotta use right padding to ensure position IDs are correct. "
            "Run tokenizer.padding_side = 'right' if sensible."
        )
    if isinstance(prompts, str):
        raise TypeError("prompts must be a sequence of strings, not a string itself.")
    if not isinstance(num_repeats_per_prompt, int):
        if not len(prompts) == len(num_repeats_per_prompt):
            raise ValueError(
                "If num_repeats_per_prompt is a Sequence, then it must be the same "
                f"length as prompts. Got lengths {len(num_repeats_per_prompt)}, "
                f"{len(prompts)}."
            )

    # Need to determine whether we actually need to repeat the prompt's keys and values.
    # Running that repeat operation (despite not needing it) may be expensive b/c its
    # intermediate steps require allocating memory to hold big tensors. It'd be nice if
    # past_key_values were a tensor instead of a nested tuple. That way, we could just
    # use repeat_interleave and not worry.
    if isinstance(num_repeats_per_prompt, int):
        must_repeat: bool = num_repeats_per_prompt > 1
    else:
        if not isinstance(num_repeats_per_prompt, torch.Tensor):
            # Need to put num_repeats_per_prompt on the device
            num_repeats_per_prompt: torch.Tensor = torch.tensor(
                num_repeats_per_prompt, device=model.device
            )
        must_repeat: bool = (num_repeats_per_prompt > 1).any()

    # Batch inference prompts
    prompts = list(prompts)  # tokenizer requires list
    encodings: BatchEncoding = tokenizer(prompts, return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.no_grad():
        out: CausalLMOutputWithPast = model(**encodings)

    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]] = out.past_key_values
    if must_repeat:
        # We need to repeat each prompt's keys and values num_repeats_per_prompt times
        # For layer i, prompts_out.past_key_values[i] is a tuple (key, value),
        # Each w/ shape: (batch size=len(prompts),
        #                 number of attention heads=12 for gpt2,
        #                 max # tokens in batch=encodings["input_ids"].shape[-1],
        #                 key/value hidden dimension=64 for gpt2)
        past_key_values = (
            torch.stack([torch.stack(block) for block in past_key_values], dim=0)
            # The tuple is now a tensor w/ shape:
            # (# layers=12 for gpt2,
            #  2 for key and value,
            #  and then the rest as before)
            # Repeat along batch size dim so that it aligns downstream w/ completions
            .repeat_interleave(num_repeats_per_prompt, dim=2)
        )
        # Re-format this tensor to the nested tuple format we'd get if we passed
        # multiple copies of the prompt at the same time to the model
        past_key_values = tuple(
            [(layer[0], layer[1]) for layer in past_key_values]  # keys, values
        )

    # Repeat prompt encodings data
    encodings["attention_mask"] = encodings["attention_mask"].repeat_interleave(
        num_repeats_per_prompt, dim=0
    )
    encodings["input_ids"] = encodings["input_ids"].repeat_interleave(
        num_repeats_per_prompt, dim=0
    )

    # Need offsets so that position_ids for future tokens are set correctly
    offsets: torch.Tensor = encodings["attention_mask"].sum(dim=1)

    # Need (next-token) logits from prompts, i.e., last non-pad prompt token, since
    # that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    last_nonpad_token_logits: torch.Tensor = out.logits.repeat_interleave(
        num_repeats_per_prompt, dim=0
    ).take_along_dim(_last_nonpad_token_idxs, dim=1)

    return past_key_values, encodings, offsets, last_nonpad_token_logits


def _blessed_helper(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    num_completions_per_prompt: int | Sequence[int],
    completions_repeats: int,
) -> tuple[torch.Tensor, BatchEncoding]:
    """
    TODO: docstring
    """
    # Input checks
    if not tokenizer.padding_side == "right":
        raise ValueError(
            "Gotta use right padding to ensure position IDs are correct. "
            "Run tokenizer.padding_side = 'right' if sensible."
        )

    # Prepare completion data
    completions = list(completions)  # tokenizer requires list
    # For Llama (and probably others) we don't want the completions to start w/ a bos
    # token <s> b/c we need to mimic sending the prompt + completion together.
    # For example, if 'a b' is the prompt and 'c' is the completion, the encoding
    # should correspond to '<s> a b c' not '<s> a b <s> c'.
    with hf._utils.disable_add_bos_token(tokenizer):
        completions_encoding: BatchEncoding = tokenizer(
            completions, return_tensors="pt", padding=True
        ).to(model.device)

    # Single-token optimization: if every completion is a single token, we don't need to
    # repeat stuff or run the model on any of the completions data. Currently, this
    # optimization is only done for constant completions, i.e., not _examples.
    # fmt: off
    _are_completions_constant = (
        isinstance(num_completions_per_prompt, int) and
        completions_repeats == len(prompts)
    )
    # fmt: on
    if _are_completions_constant and completions_encoding["input_ids"].shape[1] == 1:
        # Note that completions_encoding["input_ids"].shape[1] == logits.shape[1]
        prompts_last_nonpad_token_logits = _keys_values_prompts(
            model, tokenizer, prompts, num_repeats_per_prompt=1
        )[-1]
        return prompts_last_nonpad_token_logits, completions_encoding

    # We need to repeat stuff
    completions_input_ids: torch.Tensor = completions_encoding["input_ids"].repeat(
        completions_repeats, 1
    )
    completions_attention_mask: torch.Tensor = completions_encoding[
        "attention_mask"
    ].repeat(completions_repeats, 1)

    # Prepare prompt data
    (
        past_key_values,
        prompts_encodings,
        offsets,
        prompts_last_nonpad_token_logits,
    ) = _keys_values_prompts(
        model, tokenizer, prompts, num_repeats_per_prompt=num_completions_per_prompt
    )

    # Set position_ids to what they were had we fed the prompt + completion together w/
    # right padding
    _num_completion_tokens = completions_encoding["input_ids"].shape[1]
    completions_position_ids = (
        torch.arange(_num_completion_tokens, device=model.device)
        + offsets[:, None]  # broadcast
    )
    # Need attention_mask to include the prompt since it prolly has padding
    attention_mask = torch.cat(
        (prompts_encodings["attention_mask"], completions_attention_mask), dim=1
    )

    # Everything should now be aligned 🤞 🙏
    with torch.no_grad():
        completions_out: CausalLMOutputWithPast = model(
            input_ids=completions_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=completions_position_ids,
        )
    # 😎

    # Let's drop the next-token logits for the last completion token b/c they're not
    # useful for our purposes. Moreover, dropping ensures
    # logits.shape[:2] == encodings['input_ids'].shape, as one expects.
    # Just keep in mind that `logits` are shifted behind.
    logits = torch.cat(
        [prompts_last_nonpad_token_logits, completions_out.logits[:, :-1, :]], dim=1
    )

    # You need to be able to ignore pad tokens, so you need the tokenization and offset
    # data as well
    encodings: BatchEncoding = {
        "input_ids": completions_input_ids,
        "attention_mask": completions_attention_mask,
        "offsets": offsets,
    }

    if getattr(tokenizer, "add_bos_token", False):
        # Drop the first <s> token after we're done encoding so that the shape is
        # consistent w/ other tokenizers.
        # Note: to modify a BatchEncoding value, you must use setitem, not setattr
        encodings["offsets"] = encodings["offsets"] - 1

    return logits, encodings


def _logits_completions_given_prompts(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: Literal[" ", ""] = " ",
):
    """
    TODO: convert docstring to numpy style

    If `texts` is

    ```python
    [prompt + end_of_prompt + completions
     for prompt in prompts
     for completion in completions]
    ```

    then this function returns

    1. `logits`: tensor with shape

        (`len(texts)`, max # tokens `completions`, `tokenizer.vocab_size`)

    where `logits[i,j]` are the `model`'s logits for token `j+1` of the completion in
    `texts[i]` given the prompt in `texts[i]`. This tensor includes logits for
    right-padded tokens. Use the `encodings["attention_mask"]` to ignore them before
    further processing.

    2. `encodings`: `BatchEncoding` containing the input IDs, attention mask,
    and position offsets.
    """
    if not hf._utils.does_tokenizer_prepend_space_to_first_token(tokenizer):
        end_of_prompt = ""
    completions = [end_of_prompt + completion for completion in completions]
    return _blessed_helper(
        model,
        tokenizer,
        prompts,
        completions,
        num_completions_per_prompt=len(completions),
        completions_repeats=len(prompts),
    )


def _logits_completions_given_prompts_examples(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
):
    """
    TODO: convert docstring to numpy style

    If `texts` is

    ```python
    [example.prompt + example.end_of_prompt + completion
     for example in examples
     for completion in example.completions]
    ```
    then this function returns

    1. `logits`: tensor with shape

        (`len(texts)`, max # tokens `completions`, `tokenizer.vocab_size`)

    where `logits[i,j]` are the `model`'s logits for token `j+1` of the completion in
    `texts[i]` given the prompt in `texts[i]`. This tensor includes logits for
    right-padded tokens. Use the `encodings["attention_mask"]` to ignore them before
    further processing.

    2. `encodings`: `BatchEncoding` containing the input IDs, attention mask,
    and position offsets.
    """
    should_end_of_prompt_be_empty = (
        not hf._utils.does_tokenizer_prepend_space_to_first_token(tokenizer)
    )
    prompts: list[str] = []
    num_completions_per_prompt: list[int] = []
    completions: list[str] = []
    for example in examples:
        prompts.append(example.prompt)
        num_completions_per_prompt.append(len(example.completions))
        end_of_prompt = "" if should_end_of_prompt_be_empty else example.end_of_prompt
        for completion in example.completions:
            completions.append(end_of_prompt + completion)
    return _blessed_helper(
        model,
        tokenizer,
        prompts,
        completions,
        num_completions_per_prompt=num_completions_per_prompt,
        completions_repeats=1,
    )


def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor], from_examples: bool
) -> list[list[float]]:
    """
    TODO: convert docstring to numpy style

    Returns a list `log_probs_completions` where `log_probs_completions[i][j]` is the
    log-probablity of *completion* token

        `encodings['input_ids'][i,j]`

    given its previous tokens

        `encodings['input_ids'][i,:j]`

    Pad tokens, i.e., tokens where `encodings['attention_mask'] == 0` are excluded.

    `logits[i,j]` is assumed to be an unnormalized distribution (over tokens in
    the vocab) given tokens `input_ids[i,:j]`.
    """
    if (not from_examples) and logits.shape[1] == 1:
        # Single-token optimization: all of the completions are always a single token.
        # So we just need to intelligently slice out their tokens from the prompts' last
        # non-pad token logits. Currently, this optimization is only done for constant
        # completions, i.e., not _examples.
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


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    batch_size: int = 32,
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
        an instantiated model and its corresponding tokenizer
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    batch_size : int, optional
        the maximum number of `prompts` that the model will process in parallel, by
        default 32

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
    Here we'll use single characters (which are of course single tokens) to more clearly
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
    with hf._utils.set_up_model_and_tokenizer(model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        @_batch.flatten
        @_batch.batchify(
            batchable_arg="prompts", progress_bar_desc="conditional log-probs"
        )
        def log_probs_completions_batch(
            prompts, show_progress_bar=show_progress_bar, batch_size=batch_size
        ):
            logits, encodings = _logits_completions_given_prompts(
                model, tokenizer, prompts, completions, end_of_prompt=end_of_prompt
            )
            return _logits_to_log_probs_completions(
                logits, encodings, from_examples=False
            )

        log_probs_completions = log_probs_completions_batch(prompts)
        return list(_batch.constant(log_probs_completions, size=len(completions)))


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    show_progress_bar: bool | None = None,
    batch_size: int = 32,
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
        an instantiated model and its corresponding tokenizer
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    batch_size : int, optional
        the maximum number of `examples` that the model will process in parallel, by
        default 32

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
    Here we'll use single characters (which are of course single tokens) to more clearly
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
    # Little weird. I want my IDE to know that examples is always a Sequence[Example]
    # b/c of the decorator.
    examples: Sequence[Example] = examples
    with hf._utils.set_up_model_and_tokenizer(model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

        @_batch.flatten
        @_batch.batchify(
            batchable_arg="examples", progress_bar_desc="conditional log-probs"
        )
        def log_probs_completions_batch(
            examples, show_progress_bar=show_progress_bar, batch_size=batch_size
        ):
            logits, encodings = _logits_completions_given_prompts_examples(
                model, tokenizer, examples
            )
            return _logits_to_log_probs_completions(
                logits, encodings, from_examples=True
            )

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
    batch_size: int = 32,
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
        an instantiated model and its corresponding tokenizer
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
        default 32

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
    batch_size: int = 32,
) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        an instantiated model and its corresponding tokenizer
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    batch_size : int, optional
        the maximum number of `examples` that the model will process in parallel, by
        default 32

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
    batch_size: int = 32,
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
        an instantiated model and its corresponding tokenizer
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
        default 32

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
    batch_size: int = 32,
) -> str | list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        an instantiated model and its corresponding tokenizer
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    batch_size : int, optional
        the maximum number of `examples` that the model will process in parallel, by
        default 32

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

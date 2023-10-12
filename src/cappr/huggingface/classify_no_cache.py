"""
Perform prompt-completion classification using a model which can be loaded via

- ``transformers.AutoModelForCausalLM.from_pretrained`` or
- ``auto_gptq.AutoGPTQForCausalLM.from_quantized`` or
- ``awq.AutoAWQForCausalLM.from_quantized``.

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)

This module is a mirror of :mod:`cappr.huggingface.classify`. The difference is that
this module **does not** precompute attention block keys and values for prompts.
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
        the maximum number of texts that the model will process in parallel, by default
        32

    Returns
    -------
    log_probs : list[list[float]]
        `log_probs[text_idx][token_idx]` is the log-probability of the token at
        `token_idx` of `texts[text_idx]` conditional on all previous tokens in
        `texts[text_idx]`. If `texts[text_idx]` is a single token, then
        `log_probs[text_idx]` is `[None]`.
    """
    return hf.classify.token_logprobs(
        texts,
        model_and_tokenizer,
        end_of_prompt=end_of_prompt,
        show_progress_bar=show_progress_bar,
        batch_size=batch_size,
    )


def _keys_values_prompts(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    num_completions_per_prompt: int | Sequence[int],
) -> tuple[
    tuple[tuple[torch.Tensor, torch.Tensor]], BatchEncoding, torch.Tensor, torch.Tensor
]:
    """
    Only for testing purposes.
    """
    if not tokenizer.padding_side == "right":
        raise ValueError("Gotta use right padding to ensure position IDs are correct.")
    if isinstance(prompts, str):
        raise TypeError("prompts must be a sequence of strings, not a string itself.")
    if not isinstance(num_completions_per_prompt, int):
        if not len(prompts) == len(num_completions_per_prompt):
            raise ValueError(
                "If num_completions_per_prompt is a Sequence, then it must be the same "
                f"length as prompts. Got lengths {len(num_completions_per_prompt)}, "
                f"{len(prompts)}."
            )
    if isinstance(num_completions_per_prompt, int):
        # For code simplicity, just repeat it
        num_completions_per_prompt = [num_completions_per_prompt] * len(prompts)
    prompts_repeated = [
        prompt
        for prompt, num_repeats in zip(prompts, num_completions_per_prompt)
        for _ in range(num_repeats)
    ]
    encodings = tokenizer(prompts_repeated, return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.no_grad():
        out: CausalLMOutputWithPast = model(**encodings)

    offsets: torch.Tensor = encodings["attention_mask"].sum(dim=1)

    # Need (next-token) logits from prompts, i.e., last non-pad prompt token, since
    # that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    last_nonpad_token_logits: torch.Tensor = out.logits.take_along_dim(
        _last_nonpad_token_idxs, dim=1
    )

    return out.past_key_values, encodings, offsets, last_nonpad_token_logits


def _prompts_offsets(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    num_completions_per_prompt: int | Sequence[int],
) -> torch.Tensor:
    if not isinstance(num_completions_per_prompt, int) and not isinstance(
        num_completions_per_prompt, torch.Tensor
    ):
        num_completions_per_prompt = torch.tensor(num_completions_per_prompt)
    prompts = list(prompts)  # tokenizer requires list
    offsets: torch.Tensor = (
        tokenizer(prompts, return_tensors="pt", padding=True)["attention_mask"]
        .sum(dim=1)
        .repeat_interleave(num_completions_per_prompt, dim=0)
    )
    if getattr(tokenizer, "add_bos_token", False):
        # Drop the first <s> token after we're done encoding so that the shape is
        # consistent w/ other tokenizers
        offsets -= 1
    return offsets


def _logits_completions_given_prompts(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: Literal[" ", ""] = " ",
):
    if isinstance(prompts, str):
        raise TypeError("prompts must be a sequence of strings, not a string itself.")
    _check.completions(completions)
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    logits, encodings = hf._utils.logits_texts(texts, model, tokenizer)
    # Need these indices to slice completion tokens
    encodings["offsets"] = _prompts_offsets(
        tokenizer, prompts, num_completions_per_prompt=len(completions)
    ).to(model.device)
    return logits, encodings


def _logits_completions_given_prompts_examples(
    model: ModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    examples: Sequence[Example],
):
    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    logits, encodings = hf._utils.logits_texts(texts, model, tokenizer)
    # Need these indices to slice completion tokens
    prompts = [example.prompt for example in examples]
    num_completions_per_prompt = [len(example.completions) for example in examples]
    encodings["offsets"] = _prompts_offsets(
        tokenizer, prompts, num_completions_per_prompt=num_completions_per_prompt
    )
    return logits, encodings


def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor]
) -> list[list[float]]:
    log_probs = hf._utils.logits_to_log_probs(
        logits, encodings["input_ids"], input_ids_start_idx=1, logits_end_idx=-1
    )
    last_idx_non_pad = encodings["attention_mask"].sum(dim=1)
    # i.e., # of tokens per text
    return [
        log_probs_prompt_completion[completion_start:completion_end].tolist()
        for log_probs_prompt_completion, completion_start, completion_end in zip(
            log_probs, encodings["offsets"] - 1, last_idx_non_pad - 1
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
        from cappr.huggingface.classify_no_cache import log_probs_conditional

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
            batchable_arg="prompts",
            progress_bar_desc="conditional log-probs (no cache)",
        )
        def log_probs_completions_batch(
            prompts, show_progress_bar=show_progress_bar, batch_size=batch_size
        ):
            logits, encodings = _logits_completions_given_prompts(
                model, tokenizer, prompts, completions, end_of_prompt=end_of_prompt
            )
            return _logits_to_log_probs_completions(logits, encodings)

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
        from cappr.huggingface.classify_no_cache import log_probs_conditional_examples

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
            batchable_arg="examples",
            progress_bar_desc="conditional log-probs (no cache)",
        )
        def log_probs_completions_batch(
            examples, show_progress_bar=show_progress_bar, batch_size=batch_size
        ):
            logits, encodings = _logits_completions_given_prompts_examples(
                model, tokenizer, examples
            )
            return _logits_to_log_probs_completions(logits, encodings)

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
        from cappr.huggingface.classify_no_cache import predict_proba

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
        from cappr.huggingface.classify_no_cache import predict_proba_examples

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
        print(pred_probs[0][0])
        # 0.7

        # predicted probability that Batman was played by Kevin Conroy
        print(pred_probs[1][1])
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
        from cappr.huggingface.classify_no_cache import predict

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
        from cappr.huggingface.classify_no_cache import predict_examples

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

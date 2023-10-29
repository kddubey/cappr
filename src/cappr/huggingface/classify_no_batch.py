"""
Perform prompt-completion classification using a model which can be loaded via

- ``transformers.AutoModelForCausalLM.from_pretrained`` or
- ``auto_gptq.AutoGPTQForCausalLM.from_quantized``.

You probably just want the :func:`predict` function :-)

This module is a mirror of :mod:`cappr.huggingface.classify`. The difference is that
this module **does not** batch any inputs—every prompt-completion pair is processed one
at a time. As a result, memory usage is minimized.
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import torch
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from cappr.utils import _batch, classify
from cappr import Example
from cappr import huggingface as hf
from cappr.huggingface import classify_no_cache as hf_no_cache
from cappr.huggingface._utils import BatchEncoding, ModelForCausalLM


def token_logprobs(
    texts: str | Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""] = " ",
    drop_bos_token_log_prob: bool = True,
    show_progress_bar: bool | None = None,
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
        an instantiated model and its corresponding tokenizer
    end_of_prompt : Literal[' ', ''], optional
        This string gets added to the beginning of each text. It's important to set this
        if you're using the discount feature. Otherwise, set it to "". By default " "
    drop_bos_token_log_prob : bool, optional
        whether or not to include the tokenizer's beginning-of-sentence token
        log-probability in the output if the tokenizer adds this token. It's important
        to set this to `True` if you're using the discount feature By default, its
        log-probability is not included in the output
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts

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

    Note
    ----
    For each text, the first token's log-probability is always ``None``.

    Raises
    ------
    TypeError
        if `texts` is not a sequence
    ValueError
        if `texts` is empty
    """
    return hf.classify.token_logprobs(
        texts,
        model_and_tokenizer,
        end_of_prompt=end_of_prompt,
        drop_bos_token_log_prob=drop_bos_token_log_prob,
        show_progress_bar=show_progress_bar,
        batch_size=1,
    )


########################################################################################
#### KV caching as a context manager. One day this will be made simpler or obsolete. ###
########################################################################################


class _ModelWithCache:
    def __init__(
        self,
        model: ModelForCausalLM,
        encoding_to_cache: BatchEncoding,
        past: tuple[BatchEncoding, CausalLMOutputWithPast] | None = None,
    ):
        self._model = model
        self._cappr_past = past
        self._update_cache = True
        _ = self.forward(**encoding_to_cache)
        self._update_cache = False

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args, **kwargs
    ) -> CausalLMOutputWithPast:
        if not hasattr(self, "_cappr_past"):
            raise AttributeError(
                "This model is no longer usable. It was used in a temporary context "
                "where clear_cache_on_exit=True to save memory. If you meant to retain "
                "the cache for future use, then do: "
                "`with cache(..., clear_cache_on_exit=False)`"
            )

        encoding = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._cappr_past is None:
            with hf._utils.set_up_model(self._model):
                out: CausalLMOutputWithPast = self._model(**encoding)
            self._cappr_past = encoding, out
            return out

        encoding_past, out_past = self._cappr_past

        # Set position_ids to what they'd be had we fed prompt + completion together
        _num_completion_tokens = encoding["input_ids"].shape[1]
        position_ids = (
            torch.arange(_num_completion_tokens, device=self._model.device)
            + encoding_past["attention_mask"].sum(dim=1)[:, None]
        )
        attention_mask = torch.cat(
            (encoding_past["attention_mask"], encoding["attention_mask"]), dim=1
        )
        # Everything should now be aligned 🤞 🙏
        with hf._utils.set_up_model(self._model):
            out = self._model(
                input_ids=encoding["input_ids"],
                attention_mask=attention_mask,
                past_key_values=out_past.past_key_values,
                position_ids=position_ids,
            )

        # Concatenate encodings for future model calls
        encoding = {
            key: torch.cat((encoding_past[key], encoding[key]), dim=1)
            for key in encoding
        }
        # Concatenate logits to fulfill the spec
        out.logits = torch.cat((out_past.logits, out.logits), dim=1)
        if self._update_cache:
            self._cappr_past = encoding, out
        return out

    def __call__(self, *args, **kwargs) -> CausalLMOutputWithPast:
        return self.forward(*args, **kwargs)

    def __getattr__(self, __name: str):
        return getattr(self._model, __name)


@contextmanager
def cache(
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    prefix: str,
    clear_cache_on_exit: bool = True,
):
    """
    In this context, every prompt processed by `model_and_tokenizer` starts with
    `prefix`. As a result, computations in this context are faster.

    Parameters
    ----------
    model_and_tokenizer : tuple[ModelForCausalLM, PreTrainedTokenizerBase]
        an instantiated model and its corresponding tokenizer
    prefix : str
        prefix for all strings/prompts that will be processed in this context, e.g., a
        set of shared instructions, or exemplars for few-shot prompting
    clear_cache_on_exit : bool, optional
        whether or not to clear the cache and render the returned model and tokenizer
        unusable when we exit the context. This is important because it saves memory,
        and makes code more explicit about the model's state. By default, True

    Note
    ----
    Do **not**::

        with cache(model_and_tokenizer, "string") as model_and_tokenizer:
            # use model_and_tokenizer

        # The original, uncached model_and_tokenizer object has been
        # overwritten!
        # This is almost always not what you want. Name the returned model
        # and tokenizer something else:
        with cache(
            model_and_tokenizer, "string"
        ) as cached_model_and_tokenizer:
            # use cached_model_and_tokenizer

        # Now you can use model_and_tokenizer for computations outside of
        # the context. Its state is completely unchanged.

    Warning
    -------
    In this context, you must ensure that any strings that are processed by the
    tokenizer start correctly. Strings are assumed to be separated by `end_of_prompt` if
    you're calling a function in this module.

    Warning
    -------
    In this context, only un-batched computations are allowed.

    Example
    -------
    Usage with :func:`predict_proba`::

        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify_no_batch import (
            cache, predict_proba
        )

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
            pred_probs = predict_proba(
                prompts, completions, cached_model_and_tokenizer
            )

        # The above computation is equivalent to this one:
        prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
        pred_probs_wo_cache = predict_proba(
            prompts_full, completions, model_and_tokenizer
        )
        assert np.allclose(pred_probs, pred_probs_wo_cache)

        print(pred_probs.round(1))
        # [[1. 0. 0.]
        #  [0. 1. 0.]]

    Here's a more complicated example, which might help in explaining usage::

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify_no_batch import cache
        from cappr.huggingface._utils import (
            does_tokenizer_prepend_space_to_first_token,
            logits_texts,
        )

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model_and_tokenizer = (model, tokenizer)

        # Assume that all strings will be separated by a whitespace
        delim = " "
        if not does_tokenizer_prepend_space_to_first_token(tokenizer):
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
    model, tokenizer = model_and_tokenizer

    past = getattr(model, "_cappr_past", None)

    # Because we're implicitly concatenating strings, we should never add an EOS token
    with hf._utils.dont_add_eos_token(tokenizer):
        is_first = past is None
        with hf._utils.dont_add_bos_token(tokenizer) if not is_first else nullcontext():
            encoding: BatchEncoding = tokenizer([prefix], return_tensors="pt").to(
                model.device
            )

        model_for_causal_lm = (
            model._model if isinstance(model, _ModelWithCache) else model
        )
        model_with_cache = _ModelWithCache(model_for_causal_lm, encoding, past)

        # Now that we've started the cache, we should never add a BOS token
        with hf._utils.dont_add_bos_token(tokenizer):
            yield model_with_cache, tokenizer

    if clear_cache_on_exit:
        # model_with_cache._cappr_past contains a ton of data—logits, past_key_values,
        # hidden_states (usually taking up GPU RAM)—that should be cleared when we exit
        # the context
        delattr(model_with_cache, "_cappr_past")
    else:
        model_with_cache._cappr_past = past


def _log_probs_conditional_prompt(
    prompt: str,
    completions: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""],
):
    model, tokenizer = model_and_tokenizer
    in_caching_context = isinstance(model, _ModelWithCache)
    start_of_prompt = end_of_prompt if in_caching_context else ""
    does_tokenizer_add_bos_token = getattr(tokenizer, "add_bos_token", False)
    log_probs_completions = []
    with cache(model_and_tokenizer, start_of_prompt + prompt) as cached:
        encoding_prompt = cached[0]._cappr_past[0]
        offset = torch.tensor(
            [encoding_prompt["input_ids"].shape[1]], device=model.device
        )
        for completion in completions:
            logits, encodings = hf._utils.logits_texts(
                [end_of_prompt + completion], cached
            )
            encodings = {
                key: torch.cat((encoding_prompt[key], encodings[key]), dim=1)
                for key in encodings
            }
            if in_caching_context and does_tokenizer_add_bos_token:
                logits, encodings = hf._utils.drop_first_token(logits, encodings)
            encodings["offsets"] = offset
            log_probs_completions.append(
                hf_no_cache._logits_to_log_probs_completions(logits, encodings)[0]
            )
    return log_probs_completions


########################################################################################
############################### Classification functions ###############################
########################################################################################


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
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
        from cappr.huggingface.classify_no_batch import log_probs_conditional

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
    if not hf._utils.does_tokenizer_prepend_space_to_first_token(
        model_and_tokenizer[1]
    ):
        end_of_prompt = ""
    return [
        _log_probs_conditional_prompt(
            prompt, completions, model_and_tokenizer, end_of_prompt
        )
        for prompt in _batch.ProgressBar(
            prompts, show_progress_bar=show_progress_bar, desc="conditional log-probs"
        )
    ]


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model_and_tokenizer: tuple[ModelForCausalLM, PreTrainedTokenizerBase],
    show_progress_bar: bool | None = None,
    **kwargs,
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
        from cappr.huggingface.classify_no_batch import log_probs_conditional_examples

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
    should_end_of_prompt_be_empty = (
        not hf._utils.does_tokenizer_prepend_space_to_first_token(
            model_and_tokenizer[1]
        )
    )
    return [
        _log_probs_conditional_prompt(
            example.prompt,
            example.completions,
            model_and_tokenizer,
            end_of_prompt=(
                "" if should_end_of_prompt_be_empty else example.end_of_prompt
            ),
        )
        for example in _batch.ProgressBar(
            examples, show_progress_bar=show_progress_bar, desc="conditional log-probs"
        )
    ]


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
        from cappr.huggingface.classify_no_batch import predict_proba

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
        from cappr.huggingface.classify_no_batch import predict_proba_examples
        )

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
        from cappr.huggingface.classify_no_batch import predict

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
        from cappr.huggingface.classify_no_batch import predict_examples

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

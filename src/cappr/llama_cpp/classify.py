"""
Perform prompt-completion classification using a model which can be loaded via
``llama_cpp.Llama``.

You probably just want the :func:`predict` function :-)

The examples below use a 6 MB model to quickly demonstrate functionality. To download
it, first install ``huggingface-hub`` if you don't have it already::

    pip install huggingface-hub

And then download `the model
<https://huggingface.co/aladar/TinyLLama-v0-GGUF/blob/main/TinyLLama-v0.Q8_0.gguf>`_ (to
your current working directory)::

    huggingface-cli download \\
    aladar/TinyLLama-v0-GGUF \\
    TinyLLama-v0.Q8_0.gguf \\
    --local-dir . \\
    --local-dir-use-symlinks False
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import cast, Literal, Sequence

from llama_cpp import Llama
import numpy as np
import numpy.typing as npt

from cappr.utils import classify
from cappr.utils._batch import ProgressBar
from cappr import Example
from cappr.llama_cpp import _utils


@classify._token_logprobs
def token_logprobs(
    texts: str | Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    add_bos: bool = False,
    **kwargs,
) -> list[float] | list[list[float]]:
    """
    For each text, compute each token's log-probability conditional on all previous
    tokens in the text.

    Parameters
    ----------
    texts : str | Sequence[str]
        input text(s)
    model : Llama
        a Llama CPP model
    end_of_prompt : Literal[' ', ''], optional
        This string gets added to the beginning of each text. It's important to set this
        if you're using the discount feature. Otherwise, set it to "". By default " "
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    add_bos : bool, optional
        whether or not to add a beginning-of-sentence token to each text in `texts` if
        the tokenizer has a beginning-of-sentence token, by default False

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
    if not _utils.does_tokenizer_need_prepended_space(model):
        end_of_prompt = ""
    texts = [end_of_prompt + text for text in texts]
    # Loop through completions, b/c llama cpp currently doesn't support batch inference
    # Note: we could instead run logits_to_log_probs over a batch to save a bit of time,
    # but that'd cost more memory
    log_probs = []
    first_token_log_prob = [None]
    with _utils.set_up_model(model):
        for text in ProgressBar(
            texts, show_progress_bar=show_progress_bar, desc="marginal log-probs"
        ):
            input_ids = model.tokenize(text.encode(), add_bos=add_bos)
            model.reset()  # clear the model's KV cache and logits
            model.eval(input_ids)
            log_probs_text: list[float] = _utils.logits_to_log_probs(
                np.array(model.eval_logits),
                np.array(input_ids),
                input_ids_start_idx=1,  # this log-prob is in the prev logit
                logits_end_idx=-1,
            ).tolist()
            log_probs.append(first_token_log_prob + log_probs_text)
        model.reset()
    return log_probs


########################################################################################
###################################### KV caching ######################################
########################################################################################


def cache_model(model: Llama, prefix: str) -> Llama:
    """
    Caches the model so that every future computation with it starts with `prefix`. As a
    result, computations with this model are faster.

    Use this function instead of the context manager :func:`cache` to keep the cache for
    future computations, including those outside of a context.

    Parameters
    ----------
    model : Llama
        a Llama CPP model
    prefix : str
        prefix for all strings/prompts that will be processed in this context, e.g., a
        set of shared instructions, or exemplars for few-shot prompting

    Note
    ----
    When inputting this model to a function from this module, set the function's
    `reset_model=False`.

    Example
    -------
    Usage with :func:`predict_proba`::

        import numpy as np
        from llama_cpp import Llama
        from cappr.llama_cpp.classify import cache_model, predict_proba

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Create data
        prompt_prefix = "Once upon a time,"
        prompts = ["there was", "in a land far far, far away,"]
        completions = [
            "a llama in pajamas",
            "an alpaca in Havana",
        ]

        # Compute
        model = cache_model(model, prompt_prefix)
        # Always set reset_model=False
        pred_probs = predict_proba(
            prompts, completions, model, reset_model=False
        )

        # The above computation is equivalent to this one:
        prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
        pred_probs_wo_cache = predict_proba(
            prompts_full, completions, model, reset_model=True
        )
        assert np.allclose(pred_probs, pred_probs_wo_cache)
    """
    if prefix:
        # W/o the if condition, we'd eval on <bos> if there's no prefix and no cache
        input_ids_prefix = model.tokenize(prefix.encode(), add_bos=model.n_tokens == 0)
        with _utils.set_up_model(model):
            model.eval(input_ids_prefix)
    return model


@contextmanager
def cache(model: Llama, prefix: str, reset_model: bool = True):
    """
    In this context, every prompt processed by the `model` starts with `prefix + " "`.
    As a result, computations in this context are faster.

    Parameters
    ----------
    model : Llama
        a Llama CPP model
    prefix : str
        prefix for all strings/prompts that will be processed in this context, e.g., a
        set of shared instructions, or exemplars for few-shot prompting
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Note
    ----
    In this context, when using a function from this module, set their
    `reset_model=False`.

    Example
    -------
    Usage with :func:`predict_proba`::

        import numpy as np
        from llama_cpp import Llama
        from cappr.llama_cpp.classify import cache, predict_proba

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Create data
        prompt_prefix = "Once upon a time,"
        prompts = ["there was", "in a land far far, far away,"]
        completions = [
            "a llama in pajamas",
            "an alpaca in Havana",
        ]

        # Compute
        with cache(model, prompt_prefix):
            # Always set reset_model=False
            pred_probs = predict_proba(
                prompts, completions, model, reset_model=False
            )

        # The above computation is equivalent to this one:
        prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
        pred_probs_wo_cache = predict_proba(
            prompts_full, completions, model, reset_model=True
        )
        assert np.allclose(pred_probs, pred_probs_wo_cache)
    """
    if reset_model:
        model.reset()
    n_tokens = model.n_tokens
    try:
        yield cache_model(model, prefix)
    finally:
        model.n_tokens = n_tokens


########################################################################################
############################## Logprobs from cached model ##############################
########################################################################################


def _log_probs_conditional_prompt(
    prompt: str,
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""],
) -> list[list[float]]:
    # Prepend whitespaces if the tokenizer or context call for it
    # TODO: put this in the context manager? Little weird
    if not _utils.does_tokenizer_need_prepended_space(model):
        start_of_prompt = ""
        end_of_prompt = ""
    else:
        in_cache_context = model.n_tokens > 0
        start_of_prompt = " " if in_cache_context else ""
    prompt = start_of_prompt + prompt
    completions = [end_of_prompt + completion for completion in completions]

    # Cache the prompt's KVs and logits
    with cache(model, prompt, reset_model=False):
        num_tokens_prompt = model.n_tokens
        # Tokenize completions to determine whether or not we can do the single-token
        # optimization
        #
        # For Llama (and probably others) we don't want the completions to start w/ a
        # bos token <s> b/c we need to mimic sending the prompt + completion together
        # For example, if 'a b' is the prompt and 'c' is the completion, the encoding
        # should correspond to '<s> a b c' not '<s> a b <s> c'
        input_ids_completions = [
            model.tokenize(completion.encode(), add_bos=False)
            for completion in completions
        ]
        if all(
            len(input_ids_completion) == 1
            for input_ids_completion in input_ids_completions
        ):
            # Single-token optimization
            prompt_next_token_log_probs = _utils.log_softmax(
                np.array(model.eval_logits[-1])
            )
            return [
                [prompt_next_token_log_probs[input_ids_completion[0]]]
                for input_ids_completion in input_ids_completions
            ]
        # Loop through completions, b/c llama cpp currently doesn't support batch
        # inference
        log_probs_completions: list[list[float]] = []
        with _utils.set_up_model(model):
            for input_ids_completion in input_ids_completions:
                # Given the prompt, compute next-token logits for each completion token
                model.eval(input_ids_completion)
                # Logits -> log-probs. We need the prompt's last token's logits b/c it
                # contains the first completion token's log-prob. But we don't need the
                # last completion token's next-token logits ofc. Also, it's
                # num_tokens_prompt - 1 b/c of 0-indexing
                logits_completion = np.array(model.eval_logits)[
                    num_tokens_prompt - 1 : -1
                ]
                log_probs_completion: list[float] = _utils.logits_to_log_probs(
                    logits_completion, np.array(input_ids_completion)
                ).tolist()
                log_probs_completions.append(log_probs_completion)
                # Reset the model's KV cache to the prompt. Without this line, the cache
                # would include this completion's KVs, which is mega wrong
                model.n_tokens = num_tokens_prompt
        return log_probs_completions


########################################################################################
#################################### Implementation ####################################
########################################################################################


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    reset_model: bool = True,
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
    model : Llama
        a Llama CPP model
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Returns
    -------
    log_probs_completions : list[list[float]] | list[list[list[float]]]

        If `prompts` is a string, then a 2-D list is returned:
        `log_probs_completions[completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompt` and previous completion tokens.

        If `prompts` is a sequence of strings, then a 3-D list is returned:
        `log_probs_completions[prompt_idx][completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompts[prompt_idx]` and previous completion tokens.

    Note
    ----
    To efficiently aggregate `log_probs_completions`, use
    :func:`cappr.utils.classify.agg_log_probs`.

    Example
    -------
    Here we'll use single characters (which are single tokens) to more clearly
    demonstrate what this function does::

        from llama_cpp import Llama
        from cappr.llama_cpp.classify import log_probs_conditional

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Create data
        prompts = ["x y", "a b c"]
        completions = ["z", "d e"]

        # Compute
        log_probs_completions = log_probs_conditional(
            prompts, completions, model
        )

        # Outputs (rounded) next to their symbolic representation

        print(log_probs_completions[0])
        # [[-12.8],        [[log Pr(z | x, y)],
        #  [-10.8, -10.7]]  [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        print(log_probs_completions[1])
        # [[-9.5],        [[log Pr(z | a, b, c)],
        #  [-9.9, -10.0]]  [log Pr(d | a, b, c), log Pr(e | a, b, c, d)]]
    """
    if reset_model:
        model.reset()
    log_probs_completions = [
        _log_probs_conditional_prompt(prompt, completions, model, end_of_prompt)
        for prompt in ProgressBar(
            prompts, show_progress_bar=show_progress_bar, desc="conditional log-probs"
        )
    ]
    if reset_model:
        model.reset()
    return log_probs_completions


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model: Llama,
    show_progress_bar: bool | None = None,
    reset_model: bool = True,
) -> list[list[float]] | list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt and previous
    completion tokens.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model : Llama
        a Llama CPP model
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Returns
    -------
    log_probs_completions : list[list[float]] | list[list[list[float]]]

        If `examples` is a :class:`cappr.Example`, then a 2-D list is returned:
        `log_probs_completions[completion_idx][completion_token_idx]` is the
        log-probability of the completion token in
        `example.completions[completion_idx]`, conditional on `example.prompt` and
        previous completion tokens.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a 3-D list
        is returned:
        `log_probs_completions[example_idx][completion_idx][completion_token_idx]` is
        the log-probability of the completion token in
        `examples[example_idx].completions[completion_idx]`, conditional on
        `examples[example_idx].prompt` and previous completion tokens.

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

        from llama_cpp import Llama
        from cappr import Example
        from cappr.llama_cpp.classify import log_probs_conditional_examples

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Create examples
        examples = [
            Example(prompt="x y", completions=("z", "d e")),
            Example(prompt="a b c", completions=("d e",), normalize=False),
        ]

        # Compute
        log_probs_completions = log_probs_conditional_examples(
            examples, model
        )

        # Outputs (rounded) next to their symbolic representation

        print(log_probs_completions[0])  # corresponds to examples[0]
        # [[-12.8],        [[log Pr(z | x, y)],
        #  [-10.8, -10.7]]  [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        print(log_probs_completions[1])  # corresponds to examples[1]
        # [[-9.90, -10.0]] [[log Pr(d | a, b, c)], log Pr(e | a, b, c, d)]]
    """
    # examples is always a Sequence[Example] b/c of the decorator
    examples = cast(Sequence[Example], examples)
    if reset_model:
        model.reset()
    log_probs_completions = [
        _log_probs_conditional_prompt(
            example.prompt, example.completions, model, example.end_of_prompt
        )
        for example in ProgressBar(
            examples, show_progress_bar=show_progress_bar, desc="conditional log-probs"
        )
    ]
    if reset_model:
        model.reset()
    return log_probs_completions


@classify._predict_proba
def predict_proba(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    prior: Sequence[float] | None = None,
    normalize: bool = True,
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    show_progress_bar: bool | None = None,
    reset_model: bool = True,
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
    model : Llama
        a Llama CPP model
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
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
        discount_completions`. Pre-compute them by passing `completions` and `model` to
        :func:`token_logprobs`. By default, if `not discount_completions`, they are
        (re-)computed
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Returns
    -------
    pred_probs : npt.NDArray[np.floating]

        If `prompts` is a string, then an array with shape `len(completions),` is
        returned: `pred_probs[completion_idx]` is the model's estimate of the
        probability that `completions[completion_idx]` comes after `prompt`.

        If `prompts` is a sequence of strings, then an array with shape `(len(prompts),
        len(completions))` is returned: `pred_probs[prompt_idx, completion_idx]` is the
        model's estimate of the probability that `completions[completion_idx]` comes
        after `prompts[prompt_idx]`.

    Note
    ----
    In this function, the set of possible completions which could follow each prompt is
    the same for every prompt. If instead, each prompt could be followed by a
    *different* set of completions, then construct a sequence of :class:`cappr.Example`
    objects and pass them to :func:`predict_proba_examples`.

    Example
    -------
    Let's have our little Llama predict some story beginnings::

        from llama_cpp import Llama
        from cappr.llama_cpp.classify import predict_proba

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Define a classification task
        prompts = ["In a hole in", "Once upon"]
        completions = ("a time", "the ground")

        # Compute
        pred_probs = predict_proba(prompts, completions, model)

        pred_probs_rounded = pred_probs.round(2)  # just for cleaner output

        # predicted probability that the ending for the clause
        # "In a hole in" is "the ground"
        print(pred_probs_rounded[0, 1]) # 0.98

        # predicted probability that the ending for the clause
        # "Once upon" is "a time"
        print(pred_probs_rounded[1, 0]) # 1.0
    """
    return log_probs_conditional(**locals())


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Example | Sequence[Example],
    model: Llama,
    show_progress_bar: bool | None = None,
    reset_model: bool = True,
) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model : Llama
        a Llama CPP model
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Returns
    -------
    pred_probs : npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]

        If `examples` is an :class:`cappr.Example`, then an array with shape
        `(len(example.completions),)` is returned: `pred_probs[completion_idx]` is the
        model's estimate of the probability that `example.completions[completion_idx]`
        comes after `example.prompt`.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a list with
        length `len(examples)` is returned: `pred_probs[example_idx][completion_idx]` is
        the model's estimate of the probability that
        `examples[example_idx].completions[completion_idx]` comes after
        `examples[example_idx].prompt`. If the number of completions per example is a
        constant `k`, then an array with shape `(len(examples), k)` is returned instead
        of a list of 1-D arrays.

    Example
    -------
    Some story analysis::

        from llama_cpp import Llama
        from cappr import Example
        from cappr.llama_cpp.classify import predict_proba_examples

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Create examples
        examples = [
            Example(
                prompt="Story: I enjoyed pizza with my buddies.\\nMoral:",
                completions=("make friends", "food is yummy", "absolutely nothing"),
                prior=(2 / 5, 2 / 5, 1 / 5),
            ),
            Example(
                prompt="The child rescued the animal. The child is a",
                completions=("hero", "villain"),
            ),
        ]

        # Compute
        pred_probs = predict_proba_examples(examples, model)

        # predicted probability that the moral of the 1st story is that food is yummy
        print(pred_probs[0][1].round(2))
        # 0.72

        # predicted probability that the hero of the 2nd story is the child
        print(pred_probs[1][0].round(2))
        # 0.95
    """
    return log_probs_conditional_examples(**locals())


@classify._predict
def predict(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: Llama,
    end_of_prompt: Literal[" ", ""] = " ",
    prior: Sequence[float] | None = None,
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    show_progress_bar: bool | None = None,
    reset_model: bool = True,
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
    model : Llama
        a Llama CPP model
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    discount_completions : float, optional
        experimental feature: set it to >0.0 (e.g., 1.0 may work well) if a completion
        is consistently getting over-predicted. You could instead fudge the `prior`, but
        this hyperparameter may be easier to tune than the `prior`. By default 0.0
    log_marg_probs_completions : Sequence[Sequence[float]] | None, optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Pre-compute them by passing `completions` and `model` to
        :func:`token_logprobs`. By default, if `not discount_completions`, they are
        (re-)computed
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompts
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Returns
    -------
    preds : str | list[str]

        If `prompts` is a string, then the completion from `completions` which is
        predicted to most likely follow `prompt` is returned.

        If `prompts` is a sequence of strings, then a list with length `len(prompts)` is
        returned. `preds[prompt_idx]` is the completion in `completions` which is
        predicted to follow `prompts[prompt_idx]`.

    Note
    ----
    In this function, the set of possible completions which could follow each prompt is
    the same for every prompt. If instead, each prompt could be followed by a
    *different* set of completions, then construct a sequence of :class:`cappr.Example`
    objects and pass them to :func:`predict_examples`.

    Example
    -------
    Let's have our little Llama predict some story beginnings::

        from llama_cpp import Llama
        from cappr.llama_cpp.classify import predict

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Define a classification task
        prompts = ["In a hole in", "Once upon"]
        completions = ("a time", "the ground")

        # Compute
        preds = predict(prompts, completions, model)

        # Predicted ending for the first clause: "In a hole in"
        print(preds[0])
        # the ground

        # Predicted ending for the first clause: "Once upon"
        print(preds[1])
        # a time
    """
    return predict_proba(**locals())


@classify._predict_examples
def predict_examples(
    examples: Example | Sequence[Example],
    model: Llama,
    show_progress_bar: bool | None = None,
    reset_model: bool = True,
) -> str | list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model : Llama
        a Llama CPP model
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 `examples`
    reset_model : bool, optional
        whether or not to reset the model's KV cache and logits. Set this to False when
        you're in a :func:`cache` context. By default, True

    Returns
    -------
    preds : str | list[str]

        If `examples` is an :class:`cappr.Example`, then the completion from
        `example.completions` which is predicted to most likely follow `example.prompt`
        is returned.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a list with
        length `len(examples)` is returned: `preds[example_idx]` is the completion in
        `examples[example_idx].completions` which is predicted to most likely follow
        `examples[example_idx].prompt`.

    Example
    -------
    Some story analysis::

        from llama_cpp import Llama
        from cappr import Example
        from cappr.llama_cpp.classify import predict_examples

        # Load model
        # The top of this page has instructions to download this model
        model_path = "./TinyLLama-v0.Q8_0.gguf"
        model = Llama(model_path, verbose=False)

        # Create examples
        examples = [
            Example(
                prompt="Story: I enjoyed pizza with my buddies.\\nMoral:",
                completions=("make friends", "food is yummy", "absolutely nothing"),
                prior=(2 / 5, 2 / 5, 1 / 5),
            ),
            Example(
                prompt="The child rescued the animal. The child is a",
                completions=("hero", "villain"),
            ),
        ]

        # Compute
        preds = predict_examples(examples, model)

        # the moral of the 1st story
        print(preds[0])
        # food is yummy

        # the character of the 2nd story
        print(preds[1])
        # hero

    """
    return predict_proba_examples(**locals())

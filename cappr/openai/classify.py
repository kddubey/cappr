"""
Perform prompt-completion classification using models from OpenAI's text
completion API.
"""
from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import tiktoken

from cappr.utils import batch, classify
from cappr import Example
from cappr import openai


def _token_logprobs(
    texts: Sequence[str], model: openai.api.Model, ask_if_ok: bool = False
) -> list[list[float]]:
    """
    TODO: convert docstring to numpy style

    Returns a list `log_probs` where `log_probs[i]` is the value of
    `'log_probs' -> 'token_logprobs'` (from the OpenAI Completion endpoint) for
    `texts[i]` using `model`.
    """
    choices = openai.api.gpt_complete(
        texts,
        ask_if_ok=ask_if_ok,
        model=model,
        ## rest must be hard-coded
        max_tokens=0,
        logprobs=1,
        echo=True,
    )
    return [choice["logprobs"]["token_logprobs"] for choice in choices]


def _slice_completions(
    completions: Sequence[str],
    log_probs: Sequence[Sequence[float]],
    model: openai.api.Model,
) -> list[list[float]]:
    """
    TODO: convert docstring to numpy style

    Returns a list `log_probs_completions` where `log_probs_completions[i]` is a list of
    conditional log-probablities for each token in `completions[i]`, extracted by
    slicing `log_probs[i]`.
    """
    if len(completions) != len(log_probs):
        raise ValueError(
            "Different number of completions and log_probs: "
            f"{len(completions)}, {len(log_probs)}."
        )
    tokenizer = tiktoken.encoding_for_model(model)
    completion_lengths = [len(tokens) for tokens in tokenizer.encode_batch(completions)]
    return [
        log_probs_text[-num_completion_tokens:]
        for num_completion_tokens, log_probs_text in zip(completion_lengths, log_probs)
    ]


def log_probs_conditional(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    end_of_prompt: str = " ",
    ask_if_ok: bool = False,
):
    """
    Log-probabilities of each completion token conditional on a prompt.

    Parameters
    ----------
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint: https://platform.openai.com/docs/models/model-endpoint-compatibility
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls, by default
        False

    Returns
    -------
    log_probs_completions : list[list[list[float]]]
        `log_probs_completions[prompt_idx][completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on conditional on `prompts[prompt_idx] + end_of_prompt` and previous
        completion tokens.
    """
    ## str / non-Sequence[str] inputs silently, wastefully, and irreparably fail
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError("completions must be a Sequence of strings.")
    ## Flat list of prompts and their completions. Will post-process
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    log_probs = _token_logprobs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Since log_probs is a flat list, we'll need to batch them by the size and order of
    ## completions to fulfill the spec.
    return [
        _slice_completions(completions, log_probs_batch, model)
        for log_probs_batch in batch.constant(log_probs, size=len(completions))
    ]


def log_probs_conditional_examples(
    examples: Sequence[Example], model: openai.api.Model, ask_if_ok: bool = False
) -> list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on a prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint: https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls, by default
        False

    Returns
    -------
    log_probs_completions : list[list[list[float]]]
        `log_probs_completions[example_idx][completion_idx][completion_token_idx]` is
        the log-probability of the completion token in
        `examples[example_idx].completions[completion_idx]`, conditional on
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt` and
        previous completion tokens.
    """
    ## Flat list of prompts and their completions. Will post-process
    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    log_probs_all = _token_logprobs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Flatten completions in same order as examples were flattened
    completions_all = [
        completion for example in examples for completion in example.completions
    ]
    log_probs_completions_all = _slice_completions(
        completions_all, log_probs_all, model
    )
    ## Batch by completions to fulfill the spec
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(
        batch.variable(log_probs_completions_all, sizes=num_completions_per_prompt)
    )


@classify._predict_proba
def predict_proba(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    ask_if_ok: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Predict probabilities of each completion coming after each prompt.

    Here, the set of possible completions which could follow each prompt is the same for
    every prompt. If instead, each prompt could be followed by a *different* set of
    completions, then construct a sequence of `cappr.example.Example` objects and pass
    them to `cappr.openai.classify.predict_proba_examples`.

    Parameters
    ----------
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint: https://platform.openai.com/docs/models/model-endpoint-compatibility
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls, by default
        False

    Returns
    -------
    pred_probs : npt.NDArray[np.floating]
        Array with shape `(len(prompts), len(completions))`.
        `pred_probs[prompt_idx, completion_idx]` is the `model`'s estimate of the
        probability that `completions[completion_idx]` comes after
        `prompts[prompt_idx] + end_of_prompt`.
    """
    return log_probs_conditional(
        prompts,
        completions,
        model,
        end_of_prompt=end_of_prompt,
        ask_if_ok=ask_if_ok,
    )


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Sequence[Example], model: openai.api.Model, ask_if_ok: bool = False
) -> Union[list[list[float]], npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint: https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls, by default
        False

    Returns
    -------
    pred_probs : list[list[float]] | npt.NDArray[np.floating]
        `pred_probs[example_idx][completion_idx]` is the model's estimate of the
        probability that `examples[example_idx].completions[completion_idx]` comes after
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.

        If the number of completions per example is a constant `k`, then an array with
        shape `(len(examples), k)` is returned instead of a nested/2-D list.
    """
    return log_probs_conditional_examples(examples, model, ask_if_ok=ask_if_ok)


@classify._predict
def predict(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    ask_if_ok: bool = False,
) -> list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Here, the set of possible completions which could follow each prompt is the same for
    every prompt. If instead, each prompt could be followed by a *different* set of
    completions, then construct a sequence of `cappr.example.Example` objects and pass
    them to `cappr.openai.classify.predict_proba_examples`.

    Parameters
    ----------
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint: https://platform.openai.com/docs/models/model-endpoint-compatibility
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls, by default
        False

    Returns
    -------
    preds : list[str]
        List with length `len(prompts)`.
        `preds[prompt_idx]` is the completion in `completions` which is predicted to
        follow `prompts[prompt_idx] + end_of_prompt`.
    """
    return predict_proba(
        prompts,
        completions,
        model,
        prior=prior,
        end_of_prompt=end_of_prompt,
        ask_if_ok=ask_if_ok,
    )


@classify._predict_examples
def predict_examples(
    examples: Sequence[Example], model: openai.api.Model, ask_if_ok: bool = False
) -> list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint: https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls, by default
        False

    Returns
    -------
    preds : list[str]
        List with length `len(examples)`.
        `preds[example_idx]` is the completion in `examples[example_idx].completions`
        which is predicted to follow
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.
    """
    return predict_proba_examples(examples, model, ask_if_ok=ask_if_ok)

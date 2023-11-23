"""
Helpers functions which are compatible with the Python OpenAI API (before and after
v1.0.0)
"""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Callable, Literal, Sequence
import warnings

import openai

try:
    from openai import OpenAI
except ImportError:
    OpenAI = type("OpenAI", (object,), {})
    _IS_OPENAI_AFTER_V1 = False
    _RETRY_ERRORS = (
        openai.error.ServiceUnavailableError,
        openai.error.RateLimitError,
    )
else:
    _IS_OPENAI_AFTER_V1 = True
    _RETRY_ERRORS = (openai.InternalServerError, openai.RateLimitError)
import tiktoken

from cappr.utils import _batch, _check


logger = logging.getLogger(__name__)


Model = Literal[  # Literal so that docs and IDEs easily parse it
    "babbage-002",
    "davinci-002",
    # TODO: deprecate all of these on 01/04/2024
    # https://platform.openai.com/docs/deprecations/instructgpt-models
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-002",
    "text-davinci-003",
]
"""
These are /v1/completions models where `echo=True, logprobs=1` can be passed. On
October 5, 2023, OpenAI deprecated this combination of arguments for the
``gpt-3.5-turbo-instruct`` model. So it cannot supported by CAPPr.
"""


@dataclass(frozen=True)
class _DollarCostPer1kTokens:
    """
    Represents the cost of processing prompt and completion text.

    Parameters
    ----------
    prompt : float | None
        dollar cost for inputting 1k prompt tokens. `None` means cost is unknown.
    completion: float | None
        dollar cost for generating 1k completion tokens. `None` means cost is unknown.
    """

    prompt: float | None
    completion: float | None


# https://openai.com/api/pricing/
# TODO: figure out how to get this automatically from openai, if possible
_MODEL_TO_COST_PER_1K_TOKENS = {
    # /v1/chat/completions
    "gpt-4": _DollarCostPer1kTokens(prompt=0.03, completion=0.06),
    "gpt-4-0613": _DollarCostPer1kTokens(prompt=0.03, completion=0.06),
    "gpt-4-32k": _DollarCostPer1kTokens(prompt=0.06, completion=0.12),
    "gpt-4-32k-0613": _DollarCostPer1kTokens(prompt=0.06, completion=0.12),
    "gpt-3.5-turbo": _DollarCostPer1kTokens(prompt=0.0015, completion=0.002),
    "gpt-3.5-turbo-0613": _DollarCostPer1kTokens(prompt=0.0015, completion=0.002),
    "gpt-3.5-turbo-16k": _DollarCostPer1kTokens(prompt=0.003, completion=0.004),
    "gpt-3.5-turbo-16k-0613": _DollarCostPer1kTokens(prompt=0.003, completion=0.004),
    # /v1/completions
    "babbage-002": _DollarCostPer1kTokens(prompt=0.0004, completion=0.0004),
    "davinci-002": _DollarCostPer1kTokens(prompt=0.002, completion=0.002),
    # Deprecated echo=True, logprobs=1 on 10/05/2023
    "gpt-3.5-turbo-instruct": _DollarCostPer1kTokens(prompt=0.0015, completion=0.002),
    # Will be deprecated completely on 01/04/2024
    "text-ada-001": _DollarCostPer1kTokens(prompt=0.0004, completion=0.0004),
    "text-babbage-001": _DollarCostPer1kTokens(prompt=0.0005, completion=0.0005),
    "text-curie-001": _DollarCostPer1kTokens(prompt=0.002, completion=0.002),
    "text-davinci-002": _DollarCostPer1kTokens(prompt=0.02, completion=0.02),
    "text-davinci-003": _DollarCostPer1kTokens(prompt=0.02, completion=0.02),
}


class _UserCanceled(Exception):
    pass


def _openai_api_call_is_ok(
    texts: list[str],
    model: Model,
    max_tokens: int = 0,
    cost_per_1k_tokens_prompt: float | None = None,
    cost_per_1k_tokens_completion: float | None = None,
) -> tuple[int, int, int | str]:
    """
    After displaying the cost (usually an upper bound) of hitting the OpenAI API
    text completion endpoint, prompt the user to manually input ``y`` or ``n`` to
    indicate whether the program can proceed.

    Parameters
    ----------
    texts : list[str]
        texts or prompts inputted to the `model` OpenAI API call
    model : Model
        name of the OpenAI API text completion model
    max_tokens : int, optional
        maximum number of tokens to generate, by default 0
    cost_per_1k_tokens_prompt : float | None, optional
        OpenAI API dollar cost for processing 1k prompt tokens. If unset,
        `cappr.openai.api._MODEL_TO_COST_PER_1K_TOKENS[model]["prompt"]` is used. If
        it's still unknown, the cost will be displayed as `unknown`. By default, the
        cost is unknown
    cost_per_1k_tokens_completion : float | None, optional
        OpenAI API dollar cost for processing 1k completion tokens. If unset,
        `cappr.openai.api._MODEL_TO_COST_PER_1K_TOKENS[model]["completion"]` is used. If
        it's still unknown, the cost will be displayed as `unknown`. By default, the
        cost is unknown

    Returns
    -------
    tuple[int, int, int | str]
        - number of prompt tokens, i.e., the number of tokens in `texts`
        - upper bound on number of completion tokens, i.e., `len(texts) * max_tokens`
        - if the `model`'s costs are known, the dollar cost of having it process
        `texts`. Else, a string which is a URL to OpenAI's pricing page.

    Raises
    ------
    _UserCanceled
        if the user inputs ``n`` when prompted to give the go-ahead
    """
    texts = list(texts)
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:  # that's fine, we just need an approximation
        tokenizer = tiktoken.get_encoding("gpt2")

    _dollar_cost_unknown = _DollarCostPer1kTokens(prompt=None, completion=None)

    # prompts
    num_tokens_prompts = sum(len(tokens) for tokens in tokenizer.encode_batch(texts))
    cost_per_1k_tokens_prompt = (
        cost_per_1k_tokens_prompt
        or _MODEL_TO_COST_PER_1K_TOKENS.get(model, _dollar_cost_unknown).prompt
    )
    if cost_per_1k_tokens_prompt is None:
        cost = None
    else:
        cost = num_tokens_prompts * cost_per_1k_tokens_prompt / 1_000

    # completions
    num_tokens_completions = len(texts) * max_tokens  # upper bound
    cost_per_1k_tokens_completion = (
        cost_per_1k_tokens_completion
        or _MODEL_TO_COST_PER_1K_TOKENS.get(model, _dollar_cost_unknown).completion
    )
    if cost is not None and cost_per_1k_tokens_completion is not None:
        cost += num_tokens_completions * cost_per_1k_tokens_completion / 1_000
        cost = round(cost, 2)
    else:
        cost = "unknown (see https://openai.com/api/pricing/)"

    num_tokens_total = num_tokens_prompts + num_tokens_completions
    output = None
    while output not in {"y", "n"}:
        output = input(
            f"This API call will cost about ${cost} (≤{num_tokens_total:_} tokens). "
            "Proceed? (y/n): "
        )
    if output == "n":
        raise _UserCanceled("No API requests will be submitted.")
    return num_tokens_prompts, num_tokens_completions, cost  # for testing purposes


def openai_method_retry(
    openai_method: Callable,
    max_num_tries: int = 5,
    sleep_sec: float = 10,
    retry_errors: tuple = _RETRY_ERRORS,
    **openai_method_kwargs,
):
    """
    Wrapper around OpenAI API calls which automatically retries and sleeps if requests
    fail. Logs at level INFO when requests fail, and level ERROR if an exception is
    raised.

    Parameters
    ----------
    openai_method : Callable
        a function or method whose inputs are `openai_method_kwargs`
    max_num_tries : int, optional
        maximum number of times to retry the request before raising the exception, by
        default 5
    sleep_sec : float, optional
        number of seconds to sleep before re-submitting the request, by default 10
    retry_errors : tuple[Exception], optional
        if one of these exceptions is raised by the request, then retry, else the
        exception is immediately raised. By default, these are the
        ServiceUnavailableError/InternalServerError and RateLimitError.

    Returns
    -------
    Any
        `openai_method(**openai_method_kwargs)`

    Raises
    ------
    Exception
        if `max_num_tries` is exceeded or an exception not in `retry_errors` is raised
    """
    num_tries = 0
    while num_tries < max_num_tries:
        try:
            return openai_method(**openai_method_kwargs)
        except retry_errors as e:
            num_tries += 1
            logger.info(f"openai error: {e}")
            logger.info(f"Try {num_tries}. Sleeping for {sleep_sec} sec.")
            exception = e  # allow it to be referenced later
            time.sleep(sleep_sec)
    logger.error(f"Max retries exceeded. openai error: {exception}")
    raise exception


@contextmanager
def _set_openai_api_key(api_key: str | None = None):
    """
    In this context, the OpenAI module attribute ``openai.api_key`` is set to whatever
    is closest in scope.
    """
    api_key_from_module = openai.api_key
    try:
        # Priority is kinda in increasing order of scope
        openai.api_key = api_key or api_key_from_module or os.getenv("OPENAI_API_KEY")
        # The openai module will raise a good error (when a request is submitted) if
        # this module attr is None in the end
        yield
    finally:
        openai.api_key = api_key_from_module


def _to_dict(response) -> dict[str, Any]:
    if not hasattr(response, "model_dump"):
        # It's a dict b/c we're on openai < v1.0.0
        return response
    else:
        # It's a pydantic model b/c we're on openai >= v1.0.0
        dump_nested_dicts = getattr(response, "model_dump")
        # When the model is dumped, it gives a Pydantic serializer warning for each
        # token: Expected `int` but got `float`
        # https://github.com/openai/openai-python/issues/744
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Pydantic serializer warnings",
            )
            response = dump_nested_dicts()
        return response


def gpt_complete(
    texts: Sequence[str],
    model: Model,
    client: OpenAI | None = None,
    show_progress_bar: bool | None = None,
    progress_bar_desc: str = "log-probs",
    ask_if_ok: bool = False,
    api_key: str | None = None,
    max_tokens: int = 0,
    **openai_completion_kwargs,
) -> list[dict[str, Any]]:
    """
    Wrapper around the OpenAI text completion endpoint which automatically batches
    texts for greater efficiency, retries requests that fail, and displays a progress
    bar.

    OpenAI API text completion reference:
    https://platform.openai.com/docs/api-reference/completions

    Warning
    -------
    By default, no tokens will be generated/sampled.

    Parameters
    ----------
    texts : Sequence[str]
        these are passed as the `prompt` argument in a text completion request
    model : Model
        which text completion model to use
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    progress_bar_desc: str, optional
        description of the progress bar if shown, by default ``'log-probs'``
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``
    max_tokens : int, optional
        maximum number of tokens to generate, by default 0
    **openai_completion_kwargs
        other arguments passed to the text completion endpoint, e.g., `logprobs=1`

    Returns
    -------
    list[dict[str, Any]]
        list with the same length as `texts`. Each element is the ``choices`` mapping
    """
    _check.ordered(texts, variable_name="texts")
    if not _IS_OPENAI_AFTER_V1:
        openai_method = openai.Completion.create
    else:
        openai_method = (
            openai.completions.create if client is None else client.completions.create
        )
    with _set_openai_api_key(api_key) if client is None else nullcontext():
        _batch_size = 20  # max that the API can currently handle
        # Passing in a string will silently but majorly fail. Handle it
        texts = [texts] if isinstance(texts, str) else texts
        if ask_if_ok:
            _ = _openai_api_call_is_ok(texts, model, max_tokens=max_tokens)
        choices: list[dict[str, Any]] = []
        with _batch.ProgressBar(
            total=len(texts),
            show_progress_bar=show_progress_bar,
            desc=progress_bar_desc,
        ) as progress_bar:
            for texts_batch in _batch.constant(texts, _batch_size):
                response = openai_method_retry(
                    openai_method,
                    prompt=texts_batch,
                    model=model,
                    max_tokens=max_tokens,
                    **openai_completion_kwargs,
                )
                response = _to_dict(response)
                choices.extend(response["choices"])
                progress_bar.update(len(texts_batch))
        return choices


def gpt_chat_complete(
    texts: Sequence[str],
    model: str = "gpt-3.5-turbo",
    client: OpenAI | None = None,
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
    system_msg: str = "You are an assistant which classifies text.",
    max_tokens: int = 5,
    temperature: float = 0,
    **openai_chat_kwargs,
) -> list[dict[str, Any]]:
    """
    Wrapper around the OpenAI chat completion endpoint which sends `texts` 1-by-1 as
    individual chat messages in a chat. It retries requests that fail and displays a
    progress bar.

    OpenAI API chat completion reference:
    https://platform.openai.com/docs/api-reference/chat

    Warning
    -------
    By default, the `system_msg` asks ChatGPT to perform text classification. And the
    `temperature` is set to 0.

    Parameters
    ----------
    texts : Sequence[str]
        texts which are passed in one by one immediately after the system content as
        ``{"role": "user", "content": text}``
    model : str, optional
        one of the chat model names, by default "gpt-3.5-turbo"
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``
    system_msg : str, optional
        text which is passed in 1-by-1 immediately before every piece of user content in
        `texts` as ``{"role": "system", "content": system_msg}``. By default ``"You are
        an assistant which classifies text."``
    max_tokens : int, optional
        maximum number of tokens to generate, by default 5
    temperature : float, optional
        probability re-scaler to apply before sampling, by default 0
    **openai_chat_kwargs
        other arguments passed to the chat completion endpoint

    Returns
    -------
    list[dict[str, Any]]
        list with the same length as `texts`. Each element is the ``choices`` mapping
    """
    _check.ordered(texts, variable_name="texts")
    if not _IS_OPENAI_AFTER_V1:
        openai_method = openai.ChatCompletion.create
    else:
        openai_method = (
            openai.chat.completions.create
            if client is None
            else client.chat.completions.create
        )
    # TODO: batch, if possible
    with _set_openai_api_key(api_key) if client is None else nullcontext():
        # Passing in a string will silently but majorly fail. Handle it
        texts = [texts] if isinstance(texts, str) else texts
        if ask_if_ok:
            _ = _openai_api_call_is_ok(texts, model, max_tokens=max_tokens)
        choices: list[dict[str, Any]] = []
        for text in _batch.ProgressBar(
            texts, show_progress_bar=show_progress_bar, desc="Completing chats"
        ):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ]
            response = openai_method_retry(
                openai_method,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **openai_chat_kwargs,
            )
            response = _to_dict(response)
            choices.extend(response["choices"])
        return choices

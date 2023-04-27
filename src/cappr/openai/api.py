"""
Helpers for interacting with the OpenAI API.
"""
from __future__ import annotations
import logging
import os
import time
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, get_args

import openai
import tiktoken
from tqdm.auto import tqdm

from cappr.utils import _batch


logger = logging.getLogger(__name__)


openai.api_key = os.getenv("OPENAI_API_KEY")


end_of_prompt = "\n\n###\n\n"
## https://platform.openai.com/docs/guides/fine-tuning/data-formatting


Model = Literal[
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-002",
    "text-davinci-003",
]
## https://platform.openai.com/docs/models/model-endpoint-compatibility
_costs = [0.0004, 0.0005, 0.002, 0.02, 0.02]
## https://openai.com/api/pricing/
## TODO: figure out how to get this automatically from openai, if possible
model_to_cost_per_1k: dict[Model, float] = dict(zip(get_args(Model), _costs))


def openai_method_retry(
    openai_method: Callable,
    max_num_tries: int = 5,
    sleep_sec: float = 10,
    retry_errors: tuple = (
        openai.error.ServiceUnavailableError,
        openai.error.RateLimitError,
    ),
    **openai_method_kwargs,
) -> Any:
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
        exception is immediately raised.
        By default
        ``(openai.error.ServiceUnavailableError, openai.error.RateLimitError)``

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
            time.sleep(sleep_sec)
            exception = e  ## allow it to be referenced later
    logger.error(f"Max retries exceeded. openai error: {exception}")
    raise exception


class _UserCanceled(Exception):
    pass


def _openai_api_call_is_ok(
    model: Model,
    texts: list[str],
    max_tokens: int = 0,
    cost_per_1k_tokens: Optional[float] = None,
):
    """
    After displaying the cost (usually an upper bound) of hitting the OpenAI API
    text completion endpoint, prompt the user to manually input ``y`` or ``n`` to
    indicate whether the program can proceed.

    Parameters
    ----------
    model : Model
        name of the OpenAI API text completion model
    texts : list[str]
        texts or prompts inputted to the `model` OpenAI API call
    max_tokens : int, optional
        maximum number of tokens to generate, by default 0
    cost_per_1k_tokens : Optional[float], optional
        OpenAI API dollar cost for processing 1k tokens. If unset,
        `cappr.openai.api.model_to_cost_per_1k[model]` is used. If it's still unknown,
        the cost will be displayed as `unknown`. By default None

    Raises
    ------
    _UserCanceled
        if the user inputs ``n`` when prompted to give the go-ahead
    """
    texts = list(texts)
    try:
        tokenizer = tiktoken.encoding_for_model(model)
    except KeyError:  ## that's fine, we just need an approximation
        tokenizer = tiktoken.get_encoding("gpt2")
    _num_tokens_prompts = sum(len(tokens) for tokens in tokenizer.encode_batch(texts))
    _num_tokens_completions = len(texts) * max_tokens  ## upper bound ofc
    num_tokens = _num_tokens_prompts + _num_tokens_completions
    cost_per_1k_tokens = cost_per_1k_tokens or model_to_cost_per_1k.get(model)
    if cost_per_1k_tokens is None:
        cost = "unknown (see https://openai.com/api/pricing/)"
    else:
        cost = round(num_tokens * cost_per_1k_tokens / 1_000, 2)
    output = None
    while output not in {"y", "n"}:
        output = input(
            f"This API call will cost you about ${cost} "
            f"({num_tokens:_} tokens). Proceed? (y/n): "
        )
    if output == "n":
        raise _UserCanceled("No API requests will be submitted.")


def gpt_complete(
    texts: Sequence[str],
    model: Model,
    ask_if_ok: bool = False,
    progress_bar_desc: str = "log-probs",
    max_tokens: int = 0,
    **openai_completion_kwargs,
) -> list[Mapping[str, Any]]:
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
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False
    progress_bar_desc: str, optional
        description of the progress bar that's displayed, by default ``'log-probs'``
    max_tokens : int, optional
        maximum number of tokens to generate, by default 0
    **openai_completion_kwargs
        other arguments passed to the text completion endpoint, e.g., `logprobs=1`

    Returns
    -------
    list[Mapping[str, Any]]
        list with the same length as `texts`. Each element is the ``choices`` mapping
        which the OpenAI text completion endpoint returns.
    """
    _batch_size = 32  ## max that the API can currently handle
    if isinstance(texts, str):
        ## Passing in a string will silently but majorly fail. Handle it
        texts = [texts]
    if ask_if_ok:
        _openai_api_call_is_ok(model, texts, max_tokens=max_tokens)
    choices = []
    with tqdm(total=len(texts), desc=progress_bar_desc) as progress_bar:
        for texts_batch in _batch.constant(texts, _batch_size):
            response = openai_method_retry(
                openai.Completion.create,
                prompt=texts_batch,
                model=model,
                max_tokens=max_tokens,
                **openai_completion_kwargs,
            )
            choices.extend(response["choices"])
            progress_bar.update(len(texts_batch))
    return choices


def gpt_chat_complete(
    texts: Sequence[str],
    model: str = "gpt-3.5-turbo",
    ask_if_ok: bool = False,
    system_msg: str = ("You are an assistant which classifies text."),
    max_tokens: int = 5,
    temperature: float = 0,
    **openai_chat_kwargs,
) -> list[Mapping[str, Any]]:
    """
    Wrapper around the OpenAI chat completion endpoint which performs text
    classification 1-by-1. It retries requests that fail and displays a progress bar.

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
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False
    system_msg : str, optional
        text which is passed in 1-by-1 immediately before every piece of
        user content in `texts` as ``{"role": "system", "content": system_msg}``. By
        default ``"You are an assistant which classifies text."``
    max_tokens : int, optional
        maximum number of tokens to generate, by default 5
    temperature : float, optional
        probability re-scaler to apply before sampling, by default 0
    **openai_chat_kwargs
        other arguments passed to the chat completion endpoint

    Returns
    -------
    list[Mapping[str, Any]]
        (flat) list of the `choices` mappings which the chat completion endpoint
        returns. More specifically, it's a list of
        ``openai.openai_object.OpenAIObject``
    """
    ## TODO: batch, if possible
    if isinstance(texts, str):
        texts = [texts]
    if ask_if_ok:
        _openai_api_call_is_ok(
            model=model, texts=texts, max_tokens=max_tokens, cost_per_1k_tokens=0.002
        )
    choices = []
    for text in tqdm(texts, total=len(texts), desc="Completing chats"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ]
        response = openai_method_retry(
            openai.ChatCompletion.create,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **openai_chat_kwargs,
        )
        choices.extend(response["choices"])
    return choices

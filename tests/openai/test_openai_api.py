"""
Unit tests `cappr.openai.api`. Currently pretty barebones.
"""

from __future__ import annotations
import os

from httpx import Request, Response
import openai
import pytest

from cappr.openai import api


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oobydoobydoo")


def test_openai_method_retry():
    def openai_method(**kwargs):
        return kwargs["waddup"]

    assert api.openai_method_retry(openai_method, waddup="nada") == "nada"

    def openai_method():
        raise openai.InternalServerError(
            "ought to be caught",
            response=Response(500, request=Request("dummy", "dummy")),
            body=None,
        )

    with pytest.raises(openai.InternalServerError):
        api.openai_method_retry(openai_method, sleep_sec=0)


def test__openai_api_call_is_ok(monkeypatch: pytest.MonkeyPatch):
    # Inputs
    texts = ["cherry", "coke"]  # tokenized: [[331, 5515], [1030, 441]]
    model = "some-bpe-tokenizer"  # will be gpt-2
    max_tokens = 3  # number of completion tokens per prompt
    cost_per_1k_tokens_prompt = 1  # make it somewhat big to avoid numerical issues
    cost_per_1k_tokens_completion = 2

    # Expected outputs
    num_tokens_prompts_expected = 4
    num_tokens_completions_expected = len(texts) * max_tokens
    cost_expected = round(
        (
            (num_tokens_prompts_expected * cost_per_1k_tokens_prompt)
            + (num_tokens_completions_expected * cost_per_1k_tokens_completion)
        )
        / 1_000,
        2,
    )

    # Mimic the user seeing the prompt and entering y for yes, submit the requests
    monkeypatch.setattr("builtins.input", lambda _: "y")
    (
        num_tokens_prompts_observed,
        num_tokens_completions_observed,
        cost_observed,
    ) = api._openai_api_call_is_ok(
        texts,
        model,
        max_tokens=max_tokens,
        cost_per_1k_tokens_prompt=cost_per_1k_tokens_prompt,
        cost_per_1k_tokens_completion=cost_per_1k_tokens_completion,
    )
    assert num_tokens_completions_observed == num_tokens_completions_expected
    assert num_tokens_prompts_observed == num_tokens_prompts_expected
    assert cost_observed == cost_expected

    # Mimic the user seeing the prompt and entering n, don't submit the requests
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with pytest.raises(api._UserCanceled):
        api._openai_api_call_is_ok(texts, model)


@pytest.mark.parametrize("user_inputted_api_key", ("sk-oobydoo", None))
@pytest.mark.parametrize("openai_module_api_key", ("sk-allywag", None))
@pytest.mark.parametrize("environmt_var_api_key", ("sk-widward", None))
def test__set_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
    user_inputted_api_key: str | None,
    openai_module_api_key: str | None,
    environmt_var_api_key: str | None,
):
    monkeypatch.setattr("openai.api_key", openai_module_api_key)
    if environmt_var_api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", environmt_var_api_key)

    with api._set_openai_api_key(user_inputted_api_key):
        # if user_inputted_api_key is None, default to openai_module_api_key. if that's
        # None, default to the environment variable.
        expected_openai_api_key = (
            user_inputted_api_key
            or openai_module_api_key
            or os.getenv("OPENAI_API_KEY")
        )
        assert openai.api_key == expected_openai_api_key

    assert openai.api_key == openai_module_api_key


def test_gpt_chat_complete(monkeypatch: pytest.MonkeyPatch):
    completion_expected = "heyteam howsitgoin"

    def mocked(openai_method, messages: list[dict[str, str]], **kwargs):
        return openai.types.completion.Completion(
            id="dummy",
            choices=[
                openai.types.CompletionChoice(
                    finish_reason="length",
                    index=0,
                    logprobs=None,
                    text=completion_expected,
                )
            ],
            created=0,
            model="dummy",
            object="text_completion",
        )

    monkeypatch.setattr("cappr.openai.api.openai_method_retry", mocked)

    prompts = ["hey there", "hi", "hmm"]
    choices = api.gpt_chat_complete(prompts, model="o_o hi")
    completions = [choice["text"] for choice in choices]
    assert len(prompts) == len(completions)
    for completion_observed in completions:
        assert completion_observed == completion_expected

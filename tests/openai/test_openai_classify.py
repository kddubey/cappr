"""
Unit and integration tests for tests `cappr.openai.classify`.
"""
from __future__ import annotations
import os
import sys
from typing import Sequence

import numpy as np
import openai
import pytest
import tiktoken

from cappr import Example
from cappr.openai import classify

# sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _base


########################################################################################
###################################### Fixtures ########################################
########################################################################################


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oobydoobydoo")


@pytest.fixture(autouse=True)
def patch_openai_method_retry(monkeypatch: pytest.MonkeyPatch):
    # During testing, there's never going to be a case where we want to actually hit
    # an OpenAI endpoint!
    def _log_probs(texts: list[str]) -> list[list[float]]:
        """
        Returns a list `log_probs` where `log_probs[i]` is `-range(1, size+1))` where
        `size` is the number of tokens in `texts[i]`.
        """
        tokenizer = tiktoken.get_encoding("gpt2")
        return [
            -np.array(range(1, len(tokens) + 1))
            for tokens in tokenizer.encode_batch(texts)
        ]

    def mocked(openai_method, prompt: list[str], **kwargs):
        token_logprobs_batch = _log_probs(prompt)
        return openai.types.completion.Completion(
            id="dummy",
            choices=[
                openai.types.CompletionChoice(
                    logprobs=openai.types.completion_choice.Logprobs(
                        token_logprobs=list(token_logprobs)
                    ),
                    finish_reason="length",
                    index=0,
                    text="",
                )
                for token_logprobs in token_logprobs_batch
            ],
            created=0,
            model="dummy",
            object="text_completion",
        )

    monkeypatch.setattr("cappr.openai.api.openai_method_retry", mocked)


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "tiktoken.encoding_for_model", lambda _: tiktoken.get_encoding("gpt2")
    )


@pytest.fixture(scope="module")
def model():
    """
    This name is intentionally *not* an OpenAI API model. That's to prevent un-mocked
    API calls from going through.
    """
    return "🦖 ☄️ 💥"


########################################################################################
#################################### One-off tests #####################################
########################################################################################


def test_token_logprobs(model):
    texts = ["a b c", "d e"]
    log_probs = classify.token_logprobs(texts, model)
    assert log_probs == [[-1, -2, -3], [-1, -2]]  # cuz the API is mocked w/ range

    texts = "a b c"
    log_probs = classify.token_logprobs(texts, model)
    assert log_probs == [-1, -2, -3]


########################################################################################
####################################### Tests ##########################################
########################################################################################


class Modules:
    @property
    def module(self):
        return classify

    @property
    def module_correct(self):
        return None


class TestPromptsCompletions(Modules, _base.TestPromptsCompletions):
    def test_log_probs_conditional(self, prompts, completions, model):
        super().test_log_probs_conditional(prompts, completions, model)

    def test_predict_proba(
        self,
        prompts,
        completions,
        model,
        prior,
        discount_completions,
        normalize,
    ):
        super().test_predict_proba(
            prompts,
            completions,
            model,
            prior=prior,
            discount_completions=discount_completions,
            normalize=normalize,
        )

    def test_predict(self, prompts, completions, model):
        super().test_predict(prompts, completions, model)


class TestExamples(Modules, _base.TestExamples):
    def test_log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], model
    ):
        super().test_log_probs_conditional_examples(examples, model)

    def test_predict_proba_examples(self, examples: Example | Sequence[Example], model):
        super().test_predict_proba_examples(examples, model)

    def test_predict_examples(self, examples: Example | Sequence[Example], model):
        super().test_predict_examples(examples, model)

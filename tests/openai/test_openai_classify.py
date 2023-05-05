"""
Unit tests `cappr.openai.classify`.
"""
from __future__ import annotations
import os
import sys

import pytest
import tiktoken

from cappr import Example as Ex
from cappr.openai import classify

## sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _test


@pytest.fixture(autouse=True)
def patch_openai_method_retry(monkeypatch):
    ## During testing, there's never going to be a case where we want to actually hit
    ## an OpenAI endpoint!
    def _log_probs(texts: list[str]) -> list[list[float]]:
        """
        Returns a list `log_probs` where `log_probs[i]` is `list(range(size))` where
        `size` is the number of tokens in `texts[i]`.
        """
        tokenizer = tiktoken.get_encoding("gpt2")
        return [list(range(len(tokens))) for tokens in tokenizer.encode_batch(texts)]

    ## Note that the the text completion endpoint uses a kwarg named prompt, but that's
    ## actually set to prompt + completion in the CAPPr scheme. We input that to get
    ## log-probs of completion tokens given prompt
    def mocked(openai_method, prompt: list[str], **kwargs):
        ## Technically, we should return a openai.openai_object.OpenAIObject
        ## For now, just gonna return the minimum dict required
        token_logprobs_batch = _log_probs(prompt)
        return {
            "choices": [
                {"logprobs": {"token_logprobs": list(token_logprobs)}}
                for token_logprobs in token_logprobs_batch
            ]
        }

    monkeypatch.setattr("cappr.openai.api.openai_method_retry", mocked)


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch):
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


@pytest.fixture(scope="module")
def prompts():
    return [
        "Fill in the blank. Have an __ day!",
        "i before e except after",
        "Popular steak sauce:",
        "The English alphabet: a, b,",
    ]


@pytest.fixture(scope="module")
def completions():
    return ["A1", "c"]


@pytest.fixture(scope="module")
def examples():
    ## Let's make these ragged (different # completions per prompt), since
    ## that's a use case for an Example
    return [
        Ex(prompt="lotta", completions=("media", "food"), prior=(1 / 2, 1 / 2)),
        Ex(
            prompt=("🎶The best time to wear a striped sweater is all the"),
            completions=("time🎶",),
        ),
        Ex(
            prompt="machine",
            completions=("-washed", " learnt", " 🤖"),
            prior=(1 / 6, 2 / 3, 1 / 6),
            end_of_prompt="",
        ),
    ]


def test_token_logprobs(model):
    texts = ["a b c", "d e"]
    log_probs = classify.token_logprobs(texts, model)
    assert log_probs == [[0, 1, 2], [0, 1]]


def test__slice_completions(completions, model):
    log_probs = [[0, 1, 2], [0, 1]]
    log_probs_completions = classify._slice_completions(
        completions, "", log_probs, model
    )
    assert log_probs_completions == [[1, 2], [1]]


def test_log_probs_conditional(prompts, completions, model):
    log_probs_conditional = classify.log_probs_conditional(prompts, completions, model)
    expected = [[[10, 11], [10]], [[5, 6], [5]], [[5, 6], [5]], [[8, 9], [8]]]
    assert log_probs_conditional == expected


def test_log_probs_conditional_examples(examples, model):
    log_probs_conditional = classify.log_probs_conditional_examples(examples, model)
    expected = [[[2], [2]], [[14, 15, 16, 17]], [[1, 2], [1], [1, 2, 3]]]
    assert log_probs_conditional == expected


def test_predict_proba(prompts, completions, model):
    _test.predict_proba(classify.predict_proba, prompts, completions, model)

    ## test discount_completions > 0.0
    _test.predict_proba(
        classify.predict_proba, prompts, completions, model, discount_completions=1.0
    )

    ## test bad prior input. TODO: standardize for other inputs
    prior = [1 / len(completions)] * len(completions)
    prior_bad = prior + [0]
    expected_error_msg = (
        "completions and prior are different lengths: "
        f"{len(completions)}, {len(prior_bad)}."
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        _test.predict_proba(
            classify.predict_proba, prompts, completions, model, prior=prior_bad
        )

    prior_bad = prior[:-1] + [0]
    expected_error_msg = "prior must sum to 1."
    with pytest.raises(ValueError, match=expected_error_msg):
        _test.predict_proba(
            classify.predict_proba, prompts, completions, model, prior=prior_bad
        )


def test_predict_proba_examples(examples: list[Ex], model):
    _test.predict_proba_examples(classify.predict_proba_examples, examples, model)


def test_predict(prompts, completions, model):
    _test.predict(classify.predict, prompts, completions, model)

    ## test discount_completions > 0.0
    _test.predict(
        classify.predict, prompts, completions, model, discount_completions=1.0
    )


def test_predict_examples(examples: list[Ex], model):
    _test.predict_examples(classify.predict_examples, examples, model)

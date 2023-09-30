"""
Unit and integration tests for tests `cappr.openai.classify`.
TODO: this should instead be structured like test_huggingface_classify.py!
"""
from __future__ import annotations
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest
import tiktoken

from cappr import Example as Ex
from cappr.openai import classify

# sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _test


@pytest.fixture(autouse=True)
def patch_openai_method_retry(monkeypatch):
    # During testing, there's never going to be a case where we want to actually hit
    # an OpenAI endpoint!
    def _log_probs(texts: list[str]) -> list[list[float]]:
        """
        Returns a list `log_probs` where `log_probs[i]` is `list(range(size))` where
        `size` is the number of tokens in `texts[i]`.
        """
        tokenizer = tiktoken.get_encoding("gpt2")
        return [list(range(len(tokens))) for tokens in tokenizer.encode_batch(texts)]

    # Note that the the text completion endpoint uses a kwarg named prompt, but that's
    # actually set to prompt + completion in the CAPPr scheme. We input that to get
    # log-probs of completion tokens given prompt
    def mocked(openai_method, prompt: list[str], **kwargs):
        # Technically, we should return a openai.openai_object.OpenAIObject
        # For now, just gonna return the minimum dict required
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
    # Let's make these ragged (different # completions per prompt), since
    # that's a use case for an Example
    return [
        Ex(prompt="lotta", completions=("media", "food"), prior=(1 / 2, 1 / 2)),
        Ex(
            prompt=("🎶The best time to wear a striped sweater is all the"),
            completions=("time🎶",),
            normalize=False,
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
    assert log_probs == [[0, 1, 2], [0, 1]]  # cuz the API is mocked w/ range(len())


def test__slice_completions(completions, model):
    log_probs = [[0, 1, 2], [0, 1]]
    log_probs_completions = classify._slice_completions(
        completions, end_of_prompt="", log_probs=log_probs, model=model
    )
    assert log_probs_completions == [[1, 2], [1]]


def test_log_probs_conditional(prompts, completions, model):
    expected = [[[10, 11], [10]], [[5, 6], [5]], [[5, 6], [5]], [[8, 9], [8]]]

    # Test plain input
    _log_probs_conditional = classify.log_probs_conditional(prompts, completions, model)
    assert _log_probs_conditional == expected

    # Test that you can input a pandas Series w/ an arbitrary index
    _prompts_series = pd.Series(
        prompts, index=np.random.choice(len(prompts), size=len(prompts))
    )
    _completions_series = pd.Series(
        completions, index=np.random.choice(len(completions), size=len(completions))
    )
    _log_probs_conditional = classify.log_probs_conditional(
        _prompts_series, _completions_series, model
    )
    assert _log_probs_conditional == expected

    # Test bad prompts - empty
    with pytest.raises(ValueError, match="prompts must be non-empty."):
        classify.log_probs_conditional([], completions, model)

    # Test bad prompts - non-ordered
    with pytest.raises(TypeError, match="prompts must be an ordered collection."):
        classify.log_probs_conditional(set(prompts), completions, model)

    # Test bad completions - empty
    with pytest.raises(ValueError, match="completions must be non-empty."):
        classify.log_probs_conditional(prompts, [], model)

    # Test bad completions - non-ordered
    with pytest.raises(TypeError, match="completions must be an ordered collection."):
        classify.log_probs_conditional(prompts, set(completions), model)

    # Test bad completions - string
    with pytest.raises(TypeError, match="completions cannot be a string."):
        classify.log_probs_conditional(prompts, completions[0], model)


def test_log_probs_conditional_examples(examples, model):
    log_probs_conditional = classify.log_probs_conditional_examples(examples, model)
    expected = [[[2], [2]], [[14, 15, 16, 17]], [[1, 2], [1], [1, 2, 3]]]
    assert log_probs_conditional == expected

    # Test bad examples - non-ordered
    with pytest.raises(TypeError, match="examples must be an ordered collection."):
        classify.log_probs_conditional_examples(set(examples), model)

    # Test bad examples - empty
    with pytest.raises(ValueError, match="examples must be non-empty."):
        classify.log_probs_conditional_examples([], model)


def test_predict_proba(prompts, completions, model):
    _test.predict_proba(classify.predict_proba, prompts, completions, model)

    # Test prior
    prior = [1 / len(completions)] * len(completions)
    _test.predict_proba(
        classify.predict_proba, prompts, completions, model, prior=prior
    )

    # Test discount_completions > 0.0
    _test.predict_proba(
        classify.predict_proba, prompts, completions, model, discount_completions=1.0
    )

    # Test discount_completions > 0.0 with prior
    _test.predict_proba(
        classify.predict_proba,
        prompts,
        completions,
        model,
        prior=prior,
        discount_completions=1.0,
    )

    # Test bad prior input - wrong size
    prior_bad = prior + [0]
    expected_error_msg = re.escape(
        f"Expected prior to have length {len(completions)} (the number of "
        f"completions), got {len(prior_bad)}."
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        classify.predict_proba(prompts, completions, model, prior=prior_bad)

    # Test bad prior input - not a probability distr
    prior_bad = prior[:-1] + [0]
    expected_error_msg = "prior must sum to 1."
    with pytest.raises(ValueError, match=expected_error_msg):
        classify.predict_proba(prompts, completions, model, prior=prior_bad)

    # Test bad normalize input
    expected_error_msg = "Setting normalize=True when there's only 1 completion"
    with pytest.raises(ValueError, match=expected_error_msg):
        classify.predict_proba(prompts, completions[:1], model, normalize=True)


def test_predict_proba_examples(examples: list[Ex], model):
    _test.predict_proba_examples(classify.predict_proba_examples, examples, model)


def test_predict(prompts, completions, model):
    _test.predict(classify.predict, prompts, completions, model)

    # Test that you can input a pandas Series w/ an arbitrary index
    _prompts_series = pd.Series(
        prompts, index=np.random.choice(len(prompts), size=len(prompts))
    )
    _completions_series = pd.Series(
        completions, index=np.random.choice(len(completions), size=len(completions))
    )
    _test.predict(classify.predict, _prompts_series, _completions_series, model)

    # test discount_completions > 0.0
    _test.predict(
        classify.predict, prompts, completions, model, discount_completions=1.0
    )


def test_predict_examples(examples: list[Ex], model):
    _test.predict_examples(classify.predict_examples, examples, model)

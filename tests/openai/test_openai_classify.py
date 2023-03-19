'''
Unit tests `callm.openai.classify`.
'''
from __future__ import annotations

import pytest
import tiktoken

from callm.example import Example as Ex
from callm.openai import classify


def _log_probs(texts: list[str]) -> list[list[float]]:
    '''
    Returns a list `log_probs` where `log_probs[i]` is `list(range(size))`
    where `size` is the number of tokens in `texts[i]`.
    '''
    tokenizer = tiktoken.get_encoding('gpt2')
    return [list(range(len(tokens)))
            for tokens in tokenizer.encode_batch(texts)]


@pytest.fixture(autouse=True)
def patch_openai_method_retry(monkeypatch):
    ## During testing, there's never going to be a case where we want to
    ## actually hit an OpenAI endpoint
    def mocked(openai_method, prompt: list[str], **kwargs):
        ## Technically, we should return a openai.openai_object.OpenAIObject
        ## For now, just gonna return the minimum dict required
        token_logprobs_batch = _log_probs(prompt)
        return {'choices': [{'logprobs': {'token_logprobs': list(token_logprobs)}}
                            for token_logprobs in token_logprobs_batch]}

    monkeypatch.setattr('callm.openai.api.openai_method_retry', mocked)


@pytest.fixture(autouse=True)
def patch_tokenizer(monkeypatch):
    monkeypatch.setattr('tiktoken.encoding_for_model',
                        lambda _: tiktoken.get_encoding('gpt2'))


@pytest.fixture(scope='module')
def model():
    '''
    This name is intentionally *not* an OpenAI API model. That's to prevent
    un-mocked API calls from going through.
    '''
    return '🦖 ☄️ 💥'


@pytest.fixture(scope='module')
def prompts():
    return ['Fill in the blank. Have an __ day!',
            'i before e except after',
            'Popular steak sauce:',
            'The English alphabet: a, b,']


@pytest.fixture(scope='module')
def completions():
    return ['A1', 'c']


@pytest.fixture(scope='module')
def examples():
    ## Let's make these ragged (different # completions per prompt), since
    ## that's a use case for an Example
    return [Ex(prompt='lotta',
               completions=('media', 'food'),
               prior=(1/2, 1/2)),
            Ex(prompt=('🎶The best time to wear a striped sweater '
                       'is all the'),
               completions=('time🎶',)),
            Ex(prompt='machine',
               completions=('-washed', ' learnt', ' 🤖'),
               prior=(1/6, 2/3, 1/6),
               end_of_prompt='')]


def test_token_logprobs(model):
    texts = ['a b c', 'd e']
    log_probs = classify.token_logprobs(texts, model)
    assert log_probs == [[0, 1, 2], [0, 1]]


def test_slice_completions(completions, model):
    log_probs = [[0, 1, 2], [0, 1]]
    log_probs_completions = classify.slice_completions(completions,
                                                       log_probs, model)
    assert log_probs_completions == [[1, 2], [1]]


def test_log_probs_conditional(prompts, completions, model):
    log_probs_conditional = classify.log_probs_conditional(prompts, completions,
                                                           model)
    expected = [[[10, 11], [10]],
                [[5, 6],   [5]],
                [[5, 6],   [5]],
                [[8, 9],   [8]]]
    assert log_probs_conditional == expected


def test_log_probs_conditional_examples(examples, model):
    log_probs_conditional = classify.log_probs_conditional_examples(examples,
                                                                    model)
    expected = [[[2], [2]],
                [[14, 15, 16, 17]],
                [[1, 2], [1], [1, 2, 3]]]
    assert log_probs_conditional == expected


def test_predict_proba(monkeypatch, prompts, completions, model):
    def mock_log_probs_conditional(prompts, completions, model, **kwargs):
        return [_log_probs(completions) for _ in prompts]
    monkeypatch.setattr('callm.openai.classify.log_probs_conditional',
                        mock_log_probs_conditional)
    pred_probs = classify.predict_proba(prompts, completions, model)
    assert pred_probs.shape == (len(prompts), len(completions))


def test_predict_proba_examples(monkeypatch, examples, model):
    def mock_log_probs_conditional_examples(examples: list[classify.Example],
                                            model, **kwargs):
        return [_log_probs(example.completions) for example in examples]
    monkeypatch.setattr('callm.openai.classify.log_probs_conditional_examples',
                        mock_log_probs_conditional_examples)
    pred_probs = classify.predict_proba_examples(examples, model)
    assert len(pred_probs) == len(examples)
    for pred_prob_example, example in zip(pred_probs, examples):
        assert len(pred_prob_example) == len(example.completions)

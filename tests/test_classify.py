'''
Unit tests `lm_classification.classify`.
'''
from __future__ import annotations

import numpy as np
import pytest
import tiktoken

from lm_classification import classify


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

    monkeypatch.setattr('lm_classification.utils.api.openai_method_retry',
                        mocked)


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
    ## Let's make these ragged (different # completions for each prompt), since
    ## that's the use case for a classify.Example
    return [classify.Example(prompt='lotta',
                             completions=('media',
                                          'food'),
                             prior=(1/2, 1/2)),
            classify.Example(prompt='🎶The best time to wear a striped sweater '
                                    'is all the',
                             completions=('time🎶',)),
            classify.Example(prompt='machine',
                             completions=('-washed',
                                          ' learnt',
                                          ' 🤖'),
                             prior=(1/6, 2/3, 1/6),
                             end_of_prompt='')]


def test_gpt_log_probs(model):
    texts = ['a b c', 'd e']
    log_probs = classify.gpt_log_probs(texts, model)
    assert log_probs == [[0, 1, 2], [0, 1]]


def test_log_probs_completions(completions, model):
    log_probs = [[0, 1, 2], [0, 1]]
    log_probs_completions = classify.log_probs_completions(completions,
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


def test_agg_log_probs():
    log_probs = [[[2,   2],   [1]],
                 [[1/2, 1/2], [4]]]
    log_probs_agg = classify.agg_log_probs(log_probs, func=sum)
    assert np.allclose(log_probs_agg, np.exp([[4,1], [1,4]]))


@pytest.mark.parametrize('likelihoods', ([[4,1], [1,4]],))
@pytest.mark.parametrize('prior', (None, [1/2, 1/2], [1/3, 2/3]))
@pytest.mark.parametrize('normalize', (True, False))
def test_posterior_prob_2d(likelihoods, prior, normalize):
    posteriors = classify.posterior_prob(likelihoods, axis=1, prior=prior,
                                         normalize=normalize)
    if prior == [1/2, 1/2]: ## hard-coded b/c idk how to engineer tests
        if normalize:
            assert np.all(np.isclose(posteriors, [[4/5, 1/5], [1/5, 4/5]]))
        else:
            assert np.all(posteriors == np.array(likelihoods)/2)
    elif prior is None:
        if normalize:
            assert np.all(np.isclose(posteriors, [[4/5, 1/5], [1/5, 4/5]]))
        else:
            assert np.all(posteriors == likelihoods)
    elif prior == [1/3, 2/3]:
        if normalize:
            assert np.all(np.isclose(posteriors, [[2/3, 1/3], [1/9, 8/9]]))
        else:
            assert np.all(np.isclose(posteriors, [[4/3, 2/3], [1/3, 8/3]]))
    else:
        raise ValueError('nooo')


@pytest.mark.parametrize('likelihoods', (np.array([4,1]),))
@pytest.mark.parametrize('prior', (None, [1/2, 1/2], [1/3, 2/3]))
@pytest.mark.parametrize('normalize', (True, False))
def test_posterior_prob_1d(likelihoods, prior, normalize):
    posteriors = classify.posterior_prob(likelihoods, axis=0, prior=prior,
                                         normalize=normalize)
    if prior == [1/2, 1/2]: ## hard-coded b/c idk how to engineer tests
        if normalize:
            assert np.all(np.isclose(posteriors, [4/5, 1/5]))
        else:
            assert np.all(posteriors == np.array(likelihoods)/2)
    elif prior is None:
        if normalize:
            assert np.all(np.isclose(posteriors, [4/5, 1/5]))
        else:
            assert np.all(posteriors == likelihoods)
    elif prior == [1/3, 2/3]:
        if normalize:
            assert np.all(np.isclose(posteriors, [2/3, 1/3]))
        else:
            assert np.all(np.isclose(posteriors, [4/3, 2/3]))
    else:
        raise ValueError('nooo')


def test_predict_proba(monkeypatch, prompts, completions, model):
    def mock_log_probs_conditional(prompts, completions, model, **kwargs):
        return [_log_probs(completions) for _ in prompts]
    monkeypatch.setattr('lm_classification.classify.log_probs_conditional',
                        mock_log_probs_conditional)
    pred_probs = classify.predict_proba(prompts, completions, model)
    assert pred_probs.shape == (len(prompts), len(completions))


def test_predict_proba_examples(monkeypatch, examples, model):
    def mock_log_probs_conditional_examples(examples: list[classify.Example],
                                            model, **kwargs):
        return [_log_probs(example.completions) for example in examples]
    monkeypatch.setattr('lm_classification'
                        '.classify.log_probs_conditional_examples',
                        mock_log_probs_conditional_examples)
    pred_probs = classify.predict_proba_examples(examples, model)
    assert len(pred_probs) == len(examples)
    for pred_prob_example, example in zip(pred_probs, examples):
        assert len(pred_prob_example) == len(example.completions)

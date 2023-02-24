'''
Unit tests `lm_classification.classify`.
'''
import pytest

import numpy as np

from lm_classification import classify
from lm_classification.utils import gpt2_tokenizer


def _log_probs(texts: list[str]) -> list[list[float]]:
    '''
    Returns a list `log_probs` where `log_probs[i]` is a list of random
    log-probabilities whose length is the number of tokens in `texts[i]`.
    '''
    sizes = [len(tokens) for tokens in gpt2_tokenizer(texts)['input_ids']]
    return [list(np.log(np.random.uniform(size=size)))
            for size in sizes]


def mock_openai_method_retry(openai_method, prompt, **kwargs):
    ## Technically, we should return a openai.openai_object.OpenAIObject
    ## For now, just gonna return the minimum dict required
    token_logprobs_batch = _log_probs(prompt)
    return {'choices': [{'logprobs': {'token_logprobs': list(token_logprobs)}}
                        for token_logprobs in token_logprobs_batch]}


@pytest.mark.parametrize('texts', (['a b', 'c'],))
def test_gpt3_log_probs(mocker, texts):
    ## We ofc shouldn't actually hit any endpoints during testing
    mocker.patch('lm_classification.utils.openai_method_retry',
                 mock_openai_method_retry)
    log_probs = classify.gpt3_log_probs(texts)
    ## Since the endpoint is mocked out, the only thing left to test is the
    ## the list-extend loop. Namely, check the overall size, and types
    assert len(log_probs) == len(texts)
    assert isinstance(log_probs, list)
    for log_probs_text in log_probs:
        assert isinstance(log_probs_text, list)
        for log_prob in log_probs_text:
            assert isinstance(log_prob, float)


@pytest.fixture(scope='module')
def completions():
    '''
    For convenience, these strings have the property that their lengths are the
    same as the number of GPT2 tokens.
    '''
    return ['A1', 'c']


@pytest.mark.parametrize('log_probs', ([list(range(10)), list(range(10))],))
def test_log_probs_completions(completions, log_probs):
    log_probs_completions = classify.log_probs_completions(completions,
                                                           log_probs)
    assert log_probs_completions == [[8,9], [9]]


@pytest.fixture(scope='module')
def prompts():
    return ["since day one I've been", 'i before e except after']


def mock_gpt3_log_probs(texts, **kwargs):
    return _log_probs(texts)


def mock_log_probs_completions(completions, log_probs):
    return _log_probs(completions)


def test_log_probs_conditional(mocker, prompts, completions):
    mocker.patch('lm_classification.classify.gpt3_log_probs',
                 mock_gpt3_log_probs)
    mocker.patch('lm_classification.classify.log_probs_completions',
                 mock_log_probs_completions)
    log_probs_conditional = classify.log_probs_conditional(prompts, completions)
    assert len(log_probs_conditional) == len(prompts)
    for log_probs_prompt in log_probs_conditional:
        assert len(log_probs_prompt) == len(completions)
        for log_probs_flat, completion in zip(log_probs_prompt, completions):
            assert len(log_probs_flat) == len(completion) ## see completions()


@pytest.fixture(scope='module')
def examples(prompts, completions):
    return [classify.Example(prompt, completions) for prompt in prompts]


def test_log_probs_conditional_examples(mocker,
                                        examples: list[classify.Example]):
    mocker.patch('lm_classification.classify.gpt3_log_probs',
                 mock_gpt3_log_probs)
    mocker.patch('lm_classification.classify.log_probs_completions',
                 mock_log_probs_completions)
    log_probs_conditional = classify.log_probs_conditional_examples(examples)
    assert len(log_probs_conditional) == len(examples)
    for log_probs_prompt, example in zip(log_probs_conditional, examples):
        completions = example.completions
        assert len(log_probs_prompt) == len(completions)
        for log_probs_flat, completion in zip(log_probs_prompt, completions):
            assert len(log_probs_flat) == len(completion) ## see completions()


@pytest.mark.parametrize('log_probs', ([[[2,2], [1]], [[1/2, 1/2], [4]]],))
def test_agg_log_probs(mocker, log_probs):
    mocker.patch('numpy.exp', lambda x: x)
    log_probs_agg = classify.agg_log_probs(log_probs, func=sum)
    assert log_probs_agg == [[4,1], [1,4]]


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


@pytest.mark.parametrize('likelihoods', ([4,1],))
@pytest.mark.parametrize('prior', (None, [1/2, 1/2], [1/3, 2/3]))
@pytest.mark.parametrize('normalize', (True, False))
def test_posterior_prob_1d(likelihoods: np.ndarray, prior, normalize):
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


def mock_log_probs_conditional(prompts, completions, **kwargs):
    return [_log_probs(completions) for _ in prompts]


def test_predict_proba(mocker, prompts, completions):
    mocker.patch('lm_classification.classify.log_probs_conditional',
                 mock_log_probs_conditional)
    pred_probs = classify.predict_proba(prompts, completions)
    ## As a unit test, only the shape needs to be tested
    assert pred_probs.shape == (len(prompts), len(completions))


def mock_log_probs_conditional_examples(examples: list[classify.Example],
                                        **kwargs):
    return [_log_probs(example.completions) for example in examples]


def test_predict_proba_examples(mocker, examples: list[classify.Example]):
    mocker.patch('lm_classification.classify.log_probs_conditional_examples',
                 mock_log_probs_conditional_examples)
    pred_probs = classify.predict_proba_examples(examples)
    ## As a unit test, only the shape needs to be tested
    assert len(pred_probs) == len(examples)
    for pred_prob_example, example in zip(pred_probs, examples):
        assert len(pred_prob_example) == len(example.completions)

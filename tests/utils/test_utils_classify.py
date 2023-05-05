"""
Unit tests `cappr.utils.classify`.
"""
from __future__ import annotations

import numpy as np
import pytest

from cappr.utils import classify


def test_agg_log_probs():
    log_probs = [
        [[0, 1], [2, 3, 4]],
        [[5], [6, 7, 8], [9, 10]],
    ]
    log_probs_agg = classify.agg_log_probs(log_probs, func=sum)
    assert len(log_probs_agg) == len(log_probs)
    assert np.allclose(log_probs_agg[0], np.exp([0 + 1, 2 + 3 + 4]))
    assert np.allclose(log_probs_agg[1], np.exp([5, 6 + 7 + 8, 9 + 10]))


def test__agg_log_probs_from_constant_completions():
    log_probs = [[[2, 2], [1]], [[1 / 2, 1 / 2], [4]]]
    log_probs_agg = classify._agg_log_probs_from_constant_completions(
        log_probs, func=np.sum
    )
    assert np.allclose(log_probs_agg, np.exp([[4, 1], [1, 4]]))


@pytest.mark.parametrize("likelihoods", ([[4, 1], [1, 4]],))
@pytest.mark.parametrize("prior", (None, [1 / 2, 1 / 2], [1 / 3, 2 / 3]))
@pytest.mark.parametrize("normalize", (True, False))
def test_posterior_prob_2d(likelihoods, prior, normalize):
    ## TODO: clean this up
    posteriors = classify.posterior_prob(
        likelihoods, axis=1, prior=prior, normalize=normalize
    )
    if prior == [1 / 2, 1 / 2]:
        if normalize:
            assert np.all(np.isclose(posteriors, [[4 / 5, 1 / 5], [1 / 5, 4 / 5]]))
        else:
            assert np.all(posteriors == np.array(likelihoods) / 2)
    elif prior is None:
        if normalize:
            assert np.all(np.isclose(posteriors, [[4 / 5, 1 / 5], [1 / 5, 4 / 5]]))
        else:
            assert np.all(posteriors == likelihoods)
    elif prior == [1 / 3, 2 / 3]:
        if normalize:
            assert np.all(np.isclose(posteriors, [[2 / 3, 1 / 3], [1 / 9, 8 / 9]]))
        else:
            assert np.all(np.isclose(posteriors, [[4 / 3, 2 / 3], [1 / 3, 8 / 3]]))
    else:
        raise ValueError("nooo")


def test_posterior_prob_2d_or_mixed_prior():
    likelihoods = [[1, 2], [3, 4]]
    prior = [[1 / 3, 2 / 3], [3 / 4, 1 / 4]]

    normalize = [True, False]
    posteriors = classify.posterior_prob(
        likelihoods, axis=1, prior=prior, normalize=normalize, check_prior=False
    )
    expected_output = np.array(
        [np.array([1 / 3, 4 / 3]) / (1 / 3 + 4 / 3), np.array([9 / 4, 4 / 4])]
    )
    assert np.allclose(posteriors, expected_output)

    normalize = True
    posteriors = classify.posterior_prob(
        likelihoods, axis=1, prior=prior, normalize=normalize, check_prior=False
    )
    expected_output = np.array(
        [
            np.array([1 / 3, 4 / 3]) / (1 / 3 + 4 / 3),
            np.array([9 / 4, 4 / 4]) / (9 / 4 + 4 / 4),
        ]
    )
    assert np.allclose(posteriors, expected_output)


@pytest.mark.parametrize("likelihoods", (np.array([4, 1]),))
@pytest.mark.parametrize("prior", (None, [1 / 2, 1 / 2], [1 / 3, 2 / 3]))
@pytest.mark.parametrize("normalize", (True, False))
def test_posterior_prob_1d(likelihoods, prior, normalize):
    posteriors = classify.posterior_prob(
        likelihoods, axis=0, prior=prior, normalize=normalize
    )
    if prior == [1 / 2, 1 / 2]:  ## hard-coded b/c idk how to engineer tests
        if normalize:
            assert np.all(np.isclose(posteriors, [4 / 5, 1 / 5]))
        else:
            assert np.all(posteriors == np.array(likelihoods) / 2)
    elif prior is None:
        if normalize:
            assert np.all(np.isclose(posteriors, [4 / 5, 1 / 5]))
        else:
            assert np.all(posteriors == likelihoods)
    elif prior == [1 / 3, 2 / 3]:
        if normalize:
            assert np.all(np.isclose(posteriors, [2 / 3, 1 / 3]))
        else:
            assert np.all(np.isclose(posteriors, [4 / 3, 2 / 3]))
    else:
        raise ValueError("nooo")

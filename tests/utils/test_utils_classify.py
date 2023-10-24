"""
Unit tests `cappr.utils.classify`.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pytest

from cappr.utils import classify


def test___agg_log_probs_vectorized():
    log_probs = [[[2, 2], [1]], [[1 / 2, 1 / 2], [4]]]
    log_probs_agg = classify._agg_log_probs_vectorized(log_probs, func=np.sum)
    assert np.allclose(log_probs_agg, np.exp([[4, 1], [1, 4]]))


@pytest.mark.parametrize(
    "sequence_and_depth_expected",
    (
        (0, 0),
        ([0, 1], 1),
        ([[0, 1], [2, 3]], 2),
        ([[[0, 1], [2]], [[4, 5], [6]]], 3),  # jagged. np.shape would raise an error
    ),
)
def test__ndim(sequence_and_depth_expected: tuple[Any, int]):
    sequence, depth_expected = sequence_and_depth_expected
    ndim_observed = classify._ndim(sequence)
    assert isinstance(ndim_observed, int)
    assert ndim_observed == depth_expected


def test_agg_log_probs():
    log_probs = [
        [[0, 1], [2, 3, 4]],
        [[5], [6, 7, 8], [9, 10]],
    ]
    log_probs_agg = classify.agg_log_probs(log_probs, func=sum)
    assert len(log_probs_agg) == len(log_probs)
    assert np.allclose(log_probs_agg[0], np.exp([0 + 1, 2 + 3 + 4]))
    assert np.allclose(log_probs_agg[1], np.exp([5, 6 + 7 + 8, 9 + 10]))

    # Test bad input
    with pytest.raises(
        ValueError, match="log_probs must be 2-D or 3-D. Got 4 dimensions."
    ):
        _ = classify.agg_log_probs([[[[0, 1]]]])


@pytest.mark.parametrize("likelihoods", ([[4, 1], [1, 4]],))
@pytest.mark.parametrize("prior", (None, [1 / 2, 1 / 2], [1 / 3, 2 / 3]))
@pytest.mark.parametrize("normalize", (True, False))
def test_posterior_prob_2d(likelihoods, prior, normalize):
    # TODO: clean this up
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

    # Test bad input
    expected_error_msg = (
        "If normalize is a Sequence, it must have the same length as likelihoods."
    )
    with pytest.raises(ValueError, match=expected_error_msg):
        _ = classify.posterior_prob(likelihoods, axis=1, normalize=[True, True, False])


@pytest.mark.parametrize("likelihoods", (np.array([4, 1]),))
@pytest.mark.parametrize("prior", (None, [1 / 2, 1 / 2], [1 / 3, 2 / 3]))
@pytest.mark.parametrize("normalize", (True, False))
def test_posterior_prob_1d(likelihoods, prior, normalize):
    posteriors = classify.posterior_prob(
        likelihoods, axis=0, prior=prior, normalize=normalize
    )
    if prior == [1 / 2, 1 / 2]:  # hard-coded b/c idk how to engineer tests
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

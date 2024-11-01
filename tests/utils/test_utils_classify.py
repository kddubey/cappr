"""
Unit tests `cappr.utils.classify`.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pytest

from cappr.utils import classify


def test___agg_log_probs_vectorized():
    # There are 2 prompts, each associated with 3 completions. The completions have 2,
    # 1, and 2 tokens, respectively
    log_probs = [
        [[2, 2], [1], [3, 2]],
        [[1 / 2, 1 / 2], [4], [3, 0]],
    ]
    log_probs_agg_expected = [
        [2 + 2, 1, 3 + 2],
        [1 / 2 + 1 / 2, 4, 3 + 0],
    ]
    log_probs_agg = classify._agg_log_probs_vectorized(log_probs, func=np.sum)
    assert np.allclose(log_probs_agg, log_probs_agg_expected)

    # Test the default _avg_then_exp func
    log_probs_agg = classify._agg_log_probs_vectorized(log_probs)
    log_probs_agg_expected = classify._agg_log_probs(log_probs)
    # There are a constant number of completions, so the output should be an array
    assert log_probs_agg.shape == log_probs_agg_expected.shape
    assert np.allclose(log_probs_agg, log_probs_agg_expected)


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
    # There are 2 prompts. The first prompt is associated with 2 completions, with 2 and
    # 3 tokens each. The second prompt is associated with 3 completions, with 1, 3, and
    # 2 tokens each
    log_probs = [
        [[0, 1], [2, 3, 4]],
        [[5], [6, 7, 8], [9, 10]],
    ]
    log_probs_agg_expected = [
        [0 + 1, 2 + 3 + 4],
        [5, 6 + 7 + 8, 9 + 10],
    ]
    log_probs_agg = classify.agg_log_probs(log_probs, func=sum)
    assert len(log_probs_agg) == len(log_probs)
    for prompt_idx in range(len(log_probs)):
        assert np.allclose(
            log_probs_agg[prompt_idx], log_probs_agg_expected[prompt_idx]
        )

    # There's 1 prompt with 2 completions, with 2 and 3 tokens each.
    log_probs_2d = [
        [0, 1],
        [2, 3, 4],
    ]
    log_probs_2d_agg_expected = np.array([0 + 1, 2 + 3 + 4])
    log_probs_2d_agg = classify.agg_log_probs(log_probs_2d, func=sum)
    assert len(log_probs_2d_agg) == len(log_probs_2d)
    # There are a constant number of completions, so the output should be an array
    assert log_probs_2d_agg.shape == log_probs_2d_agg_expected.shape
    assert np.allclose(log_probs_2d_agg, log_probs_2d_agg_expected)

    # Test bad input
    with pytest.raises(
        ValueError, match="log_probs must be 2-D or 3-D. Got 4 dimensions."
    ):
        _ = classify.agg_log_probs([[[[0, 1]]]])


@pytest.mark.parametrize(
    "likelihoods, prior, normalize, expected",
    [
        # Case 1: Prior = None
        ([[4, 1], [1, 4]], None, True, [[4 / 5, 1 / 5], [1 / 5, 4 / 5]]),
        ([[4, 1], [1, 4]], None, False, [[4, 1], [1, 4]]),
        # Case 2: Prior = [1 / 2, 1 / 2]
        ([[4, 1], [1, 4]], [1 / 2, 1 / 2], True, [[4 / 5, 1 / 5], [1 / 5, 4 / 5]]),
        ([[4, 1], [1, 4]], [1 / 2, 1 / 2], False, [[2, 0.5], [0.5, 2]]),
        # Case 3: Prior = [1 / 3, 2 / 3]
        ([[4, 1], [1, 4]], [1 / 3, 2 / 3], True, [[2 / 3, 1 / 3], [1 / 9, 8 / 9]]),
        ([[4, 1], [1, 4]], [1 / 3, 2 / 3], False, [[4 / 3, 2 / 3], [1 / 3, 8 / 3]]),
    ],
)
def test_posterior_prob_2d(likelihoods, prior, normalize, expected):
    posteriors = classify.posterior_prob(
        likelihoods, axis=1, prior=prior, normalize=normalize
    )
    assert np.allclose(posteriors, expected)


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


@pytest.mark.parametrize(
    "likelihoods, prior, normalize, expected",
    [
        (np.array([4, 1]), None, True, [4 / 5, 1 / 5]),
        (np.array([4, 1]), None, False, [4, 1]),
        (np.array([4, 1]), [1 / 2, 1 / 2], True, [4 / 5, 1 / 5]),
        (np.array([4, 1]), [1 / 2, 1 / 2], False, [2, 0.5]),
        (np.array([4, 1]), [1 / 3, 2 / 3], True, [2 / 3, 1 / 3]),
        (np.array([4, 1]), [1 / 3, 2 / 3], False, [4 / 3, 2 / 3]),
    ],
)
def test_posterior_prob_1d_(likelihoods, prior, normalize, expected):
    posteriors = classify.posterior_prob(
        likelihoods, axis=0, prior=prior, normalize=normalize
    )
    assert np.allclose(posteriors, expected)

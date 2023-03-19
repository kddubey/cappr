'''
Unit tests `callm.utils.classify`.
'''
from __future__ import annotations

import numpy as np
import pytest

from callm.utils import classify


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

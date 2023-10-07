"""
Utilz
"""
from __future__ import annotations

import numpy as np


def _log_sum_exp(array: np.ndarray, dim: int = -1) -> np.ndarray:
    # adapted from scipy:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
    array_maxs: np.ndarray = np.amax(array, axis=dim, keepdims=True)
    array_maxs[~np.isfinite(array_maxs)] = 0
    summed = np.sum(np.exp(array - array_maxs), axis=dim, keepdims=True)
    out = np.log(summed)
    out += array_maxs
    return out


def log_softmax(array: np.ndarray, dim: int = -1) -> np.ndarray:
    """
    Log-softmax `array` over its `dim` axis, by default on the last one.
    """
    return array - _log_sum_exp(array, dim=dim)


def logits_to_log_probs(
    logits: np.ndarray,
    input_ids: np.ndarray,
    input_ids_start_idx: int | None = None,
    logits_end_idx: int | None = None,
) -> np.ndarray:
    """
    Log-softmax and then slice out input IDs to get token log-probabilities. Currently
    assumes the inputs correspond to a single text, not a batch of texts.
    """
    # logits.shape is (# tokens in text, vocab size)
    log_probs: np.ndarray = log_softmax(logits)

    # Only keep the log-prob from the vocab dimension whose index is is the next token's
    # input ID.
    # input_ids.shape is (# tokens in text,)
    return np.take_along_axis(
        log_probs[:logits_end_idx, :], input_ids[input_ids_start_idx:, None], axis=1
    ).squeeze(-1)

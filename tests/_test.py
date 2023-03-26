"""
Helper functions to test that predict_proba and predict function outputs have the
correct form, but not necessarily the correct content.
"""
from __future__ import annotations
from typing import Any, Callable, Sequence, Union

import numpy as np
import numpy.typing as npt

from cappr import Example


def predict_proba(
    predict_proba_func: Callable[
        [Sequence[str], Sequence[str], Any], npt.NDArray[np.floating]
    ],
    prompts: Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    pred_probs = predict_proba_func(prompts, completions, *args, **kwargs)
    assert pred_probs.shape == (len(prompts), len(completions))
    if pred_probs.shape[1] > 1:  ## we don't normalize if there's one completion
        assert np.allclose(pred_probs.sum(axis=1), 1)


def predict_proba_examples(
    predict_proba_examples_func: Callable[
        [Sequence[Example], Any], Union[list[list[float]], npt.NDArray[np.floating]]
    ],
    examples: Sequence[Example],
    *args,
    **kwargs,
):
    pred_probs = predict_proba_examples_func(examples, *args, **kwargs)
    assert len(pred_probs) == len(examples)
    for pred_prob_example, example in zip(pred_probs, examples):
        assert len(pred_prob_example) == len(example.completions)
        if len(pred_prob_example) > 1:  ## we don't normalize if there's one completion
            assert np.isclose(sum(pred_prob_example), 1)


def predict(
    predict_func: Callable[[Sequence[str], Sequence[str], Any], list[str]],
    prompts: Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    preds = predict_func(prompts, completions, *args, **kwargs)
    assert len(preds) == len(prompts)
    assert all([pred in completions for pred in preds])


def predict_examples(
    predict_examples_func: Callable[[Sequence[Example], Any], list[str]],
    examples: list[Example],
    *args,
    **kwargs,
):
    preds = predict_examples_func(examples, *args, **kwargs)
    assert len(preds) == len(examples)
    for pred, example in zip(preds, examples):
        assert pred in example.completions

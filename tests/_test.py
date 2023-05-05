"""
Helper functions to test that `predict_proba` and `predict` function outputs have the
correct form.
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
    """
    Tests that `predict_proba_func(prompts, completions, *args, **kwargs)` returns a
    numpy array with the correct shape, and that each row is a probability distribution.
    """
    pred_probs = predict_proba_func(prompts, completions, *args, **kwargs)
    assert isinstance(pred_probs, np.ndarray)
    assert pred_probs.shape == (len(prompts), len(completions))
    assert np.all(pred_probs >= 0)
    if pred_probs.shape[1] == 1:
        ## We don't normalize if there's one completion. It's almost definitely the case
        ## that the predicted probability is < 1 by a decent amount.
        assert not any(np.isclose(pred_probs[:, 0], 1))
    else:
        assert np.allclose(pred_probs.sum(axis=1), 1)


def predict_proba_examples(
    predict_proba_examples_func: Callable[
        [Sequence[Example], Any], Union[list[list[float]], npt.NDArray[np.floating]]
    ],
    examples: Sequence[Example],
    *args,
    **kwargs,
):
    """
    Tests that `predict_proba_examples_func(examples, *args, **kwargs)` returns a
    list with the correct shape, and that each element is a probability distribution.
    """
    pred_probs = predict_proba_examples_func(examples, *args, **kwargs)
    assert len(pred_probs) == len(examples)
    for pred_prob_example, example in zip(pred_probs, examples):
        assert len(pred_prob_example) == len(example.completions)
        assert np.all(np.array(pred_prob_example) >= 0)
        if len(pred_prob_example) == 1:
            ## We don't normalize if there's one completion. It's almost definitely the
            ## case that the predicted probability is < 1 by a decent amount.
            assert not any(np.isclose(pred_prob_example, 1))
        else:
            assert np.isclose(sum(pred_prob_example), 1)


def predict(
    predict_func: Callable[[Sequence[str], Sequence[str], Any], list[str]],
    prompts: Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that `predict_func(prompts, completions, *args, **kwargs)` returns an array
    with length `len(prompts)`, and that each element is in `completions`.
    """
    preds = predict_func(prompts, completions, *args, **kwargs)
    assert len(preds) == len(prompts)
    assert all([pred in completions for pred in preds])


def predict_examples(
    predict_examples_func: Callable[[Sequence[Example], Any], list[str]],
    examples: list[Example],
    *args,
    **kwargs,
):
    """
    Tests that `predict_examples_func(examples, *args, **kwargs)` returns an array
    with length `len(examples)`, and that each element is one of the example's
    completions.
    """
    preds = predict_examples_func(examples, *args, **kwargs)
    assert len(preds) == len(examples)
    for pred, example in zip(preds, examples):
        assert pred in example.completions

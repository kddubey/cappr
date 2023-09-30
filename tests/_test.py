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
        [Union[str, Sequence[str]], Sequence[str], Any], npt.NDArray[np.floating]
    ],
    prompts: Union[str, Sequence[str]],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that `predict_proba_func(prompts, completions, *args, **kwargs)` returns a
    numpy array with the correct shape, and that each row is a probability distribution
    if `normalize=True`.
    """
    pred_probs = predict_proba_func(prompts, completions, *args, **kwargs)
    assert isinstance(pred_probs, np.ndarray)
    if isinstance(prompts, str):
        assert pred_probs.shape == (len(completions),)
        completions_axis = 0
    else:
        assert pred_probs.shape == (len(prompts), len(completions))
        completions_axis = 1
    assert np.all(pred_probs >= 0)
    assert np.all(pred_probs <= 1)

    if pred_probs.shape[-1] == 1:
        if isinstance(prompts, str):
            p = pred_probs
        else:
            p = pred_probs[:, 0]
        # We don't normalize if there's one completion. It's almost definitely the case
        # that the predicted probability is < 1 by a decent amount.
        assert not any(np.isclose(p, 1))
    else:
        prob_sums = pred_probs.sum(axis=completions_axis)
        are_close_to_1 = np.isclose(prob_sums, 1)
        if kwargs.get("normalize", True):
            assert np.all(are_close_to_1)
        else:
            assert not np.any(are_close_to_1)


def predict_proba_examples(
    predict_proba_examples_func: Callable[
        [Union[Example, Sequence[Example]], Any],
        Union[npt.NDArray[np.floating], list[npt.NDArray[np.floating]]],
    ],
    examples: Union[Example, Sequence[Example]],
    *args,
    **kwargs,
):
    """
    Tests that `predict_proba_examples_func(examples, *args, **kwargs)` returns a
    list with the correct shape, and that each element is a probability distribution.
    """

    def test_single_example(
        pred_prob_example: npt.NDArray[np.floating], example: Example
    ):
        assert pred_prob_example.shape == (len(example.completions),)
        assert np.all(pred_prob_example >= 0)
        if len(example.completions) > 1:
            # Testing artifact: for mocked log-prob API endpoints, we currently return
            # integers, not log-probs. This fact combined with the fact that we obv
            # don't normalize when there's 1 completion causes the test "probability"
            # output to be > 1. So let's only run this test for examples w/ more than 1
            # completion
            assert np.all(pred_prob_example <= 1)
        if len(example.completions) == 1:
            # We don't normalize if there's one completion. It's almost definitely the
            # case that the predicted probability is < 1 by a decent amount.
            assert not any(np.isclose(pred_prob_example, 1))
        else:
            prob_sum = pred_prob_example.sum()
            is_close_to_1 = np.isclose(prob_sum, 1)
            if example.normalize:
                assert is_close_to_1
            else:
                assert not is_close_to_1

    pred_probs = predict_proba_examples_func(examples, *args, **kwargs)
    if isinstance(examples, Example):
        test_single_example(pred_probs, examples)
    else:
        assert len(pred_probs) == len(examples)
        for pred_prob_example, example in zip(pred_probs, examples):
            test_single_example(pred_prob_example, example)


def predict(
    predict_func: Callable[
        [Union[str, Sequence[str]], Sequence[str], Any], Union[str, list[str]]
    ],
    prompts: Union[str, Sequence[str]],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that `predict_func(prompts, completions, *args, **kwargs)` returns an array
    with length `len(prompts)`, and that each element is in `completions`.
    """
    preds = predict_func(prompts, completions, *args, **kwargs)
    # If completions is a pandas Series, then __contains__ checks the Series index, not
    # the values! Just convert to tuple.
    completions = tuple(completions)
    if isinstance(prompts, str):
        assert isinstance(preds, str)
        assert preds in completions
    else:
        assert len(preds) == len(prompts)
        assert all([pred in completions for pred in preds])


def predict_examples(
    predict_examples_func: Callable[
        [Union[Example, Sequence[Example]], Any], Union[str, list[str]]
    ],
    examples: Union[Example, Sequence[Example]],
    *args,
    **kwargs,
):
    """
    Tests that `predict_examples_func(examples, *args, **kwargs)` returns an array
    with length `len(examples)`, and that each element is one of the example's
    completions.
    """
    preds = predict_examples_func(examples, *args, **kwargs)
    if isinstance(examples, Example):
        assert isinstance(preds, str)
        assert preds in examples.completions
    else:
        assert len(preds) == len(examples)
        for pred, example in zip(preds, examples):
            assert pred in example.completions

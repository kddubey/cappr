"""
Helper functions to test that a `classify` module's:
  - outputs have the correct shape and form (content is NOT checked)
  - inputs are correctly checked.
"""
from __future__ import annotations
import re
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from cappr import Example
from cappr.utils.classify import _ndim

from _protocol import classify_module


def _test_log_probs_conditional(
    log_probs_completions: list[list[float]] | list[list[list[float]]],
    expected_len: int,
    num_completions_per_prompt: int | None,
):
    # TODO: test that lengths of inner-most lists match # tokens in corresponding
    # completion
    if num_completions_per_prompt is None:
        # There's only one prompt and it was fed by itself, not in a sequence.
        assert len(log_probs_completions) == expected_len
    else:
        assert len(log_probs_completions) == expected_len
        # Test lengths before zipping
        assert len(log_probs_completions) == len(num_completions_per_prompt)
        for log_probs, num_completions in zip(
            log_probs_completions, num_completions_per_prompt
        ):
            assert len(log_probs) == num_completions


def log_probs_conditional(
    classify: classify_module,
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that `classify.log_probs_conditional(prompts, completions, *args, **kwargs)`
    returns a list with the right depth and length.
    """
    log_probs_completions = classify.log_probs_conditional(
        prompts, completions, *args, **kwargs
    )
    if isinstance(prompts, str):
        assert _ndim(log_probs_completions) == 2
        expected_len = len(completions)
        num_completions_per_prompt = None
    else:
        assert _ndim(log_probs_completions) == 3
        expected_len = len(prompts)
        num_completions_per_prompt = [len(completions)] * len(prompts)
    _test_log_probs_conditional(
        log_probs_completions, expected_len, num_completions_per_prompt
    )


def log_probs_conditional_examples(
    classify: classify_module,
    examples: Example | Sequence[Example],
    *args,
    **kwargs,
):
    """
    Tests that
    `classify.log_probs_conditional_examples(prompts, completions, *args, **kwargs)`
    returns a list with the right depth and length.
    """
    log_probs_completions = classify.log_probs_conditional_examples(
        examples, *args, **kwargs
    )
    if isinstance(examples, Example):
        assert _ndim(log_probs_completions) == 2
        expected_len = len(examples.completions)
        num_completions_per_prompt = None
    else:
        assert _ndim(log_probs_completions) == 3
        expected_len = len(examples)
        num_completions_per_prompt = [len(example.completions) for example in examples]
    _test_log_probs_conditional(
        log_probs_completions, expected_len, num_completions_per_prompt
    )


def predict_proba(
    classify: classify_module,
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that `classify.predict_proba(prompts, completions, *args, **kwargs)` returns a
    numpy array with the correct shape, and that each row is a probability distribution
    if `normalize=True`.
    """
    pred_probs = classify.predict_proba(prompts, completions, *args, **kwargs)
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

    # Test bad inputs
    if kwargs.get("prior", None) is not None:
        prior_good: list[float] = list(kwargs["prior"])

        # Wrong length
        prior_bad = prior_good + [0]
        expected_error_msg = re.escape(
            f"Expected prior to have length {len(completions)} (the number of "
            f"completions), got {len(prior_bad)}."
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            _kwargs = {**kwargs, "prior": prior_bad}
            classify.predict_proba(prompts, completions, *args, **_kwargs)

        # Not a probability distr
        prior_bad = prior_good[:-1] + [0]
        expected_error_msg = "prior must sum to 1."
        with pytest.raises(ValueError, match=expected_error_msg):
            _kwargs = {**kwargs, "prior": prior_bad}
            classify.predict_proba(prompts, completions, *args, **_kwargs)

    expected_error_msg = "Setting normalize=True when there's only 1 completion"
    with pytest.raises(ValueError, match=expected_error_msg):
        _kwargs = {**kwargs, "prior": None, "normalize": True}
        classify.predict_proba(prompts, completions[:1], *args, **_kwargs)


def predict_proba_examples(
    classify: classify_module, examples: Example | Sequence[Example], *args, **kwargs
):
    """
    Tests that `classify.predict_proba_examples(examples, *args, **kwargs)` returns a
    list with the correct shape, and that each element is a probability distribution.
    """

    def test_single_example(
        pred_prob_example: npt.NDArray[np.floating], example: Example
    ):
        assert pred_prob_example.shape == (len(example.completions),)
        assert np.all(pred_prob_example >= 0)
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

    pred_probs = classify.predict_proba_examples(examples, *args, **kwargs)
    if isinstance(examples, Example):
        test_single_example(pred_probs, examples)
    else:
        assert len(pred_probs) == len(examples)
        for pred_prob_example, example in zip(pred_probs, examples):
            test_single_example(pred_prob_example, example)
    # Don't need to test bad inputs b/c they're tested in tests/test_Example.py


def predict(
    classify: classify_module,
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that `classify.predict(prompts, completions, *args, **kwargs)` returns an
    array with length `len(prompts)`, and that each element is in `completions`.
    """
    preds = classify.predict(prompts, completions, *args, **kwargs)
    # If completions is a pandas Series, then __contains__ checks the Series index, not
    # the values! Just convert to tuple.
    completions = tuple(completions)
    if isinstance(prompts, str):
        assert isinstance(preds, str)
        assert preds in completions
    else:
        assert len(preds) == len(prompts)
        assert all([pred in completions for pred in preds])

    # Test bad inputs
    with pytest.raises(ValueError, match="prompts must be non-empty."):
        classify.log_probs_conditional([], completions, *args, **kwargs)

    with pytest.raises(TypeError, match="prompts must be an ordered collection."):
        classify.log_probs_conditional(set(prompts), completions, *args, **kwargs)

    if "end_of_prompt" in kwargs:
        with pytest.raises(
            ValueError,
            match='end_of_prompt must be a whitespace " " or an empty string "".',
        ):
            _kwargs = {**kwargs, "end_of_prompt": ": "}
            classify.log_probs_conditional(prompts, completions, *args, **_kwargs)

    with pytest.raises(ValueError, match="completions must be non-empty."):
        classify.log_probs_conditional(prompts, [], *args, **kwargs)

    with pytest.raises(TypeError, match="completions must be an ordered collection."):
        classify.log_probs_conditional(prompts, set(completions), *args, **kwargs)

    with pytest.raises(TypeError, match="completions cannot be a string."):
        classify.log_probs_conditional(prompts, completions[0], *args, **kwargs)


def predict_examples(
    classify: classify_module, examples: Example | Sequence[Example], *args, **kwargs
):
    """
    Tests that `classify.predict_examples(examples, *args, **kwargs)` returns an array
    with length `len(examples)`, and that each element is one of the example's
    completions.
    """
    preds = classify.predict_examples(examples, *args, **kwargs)
    if isinstance(examples, Example):
        assert isinstance(preds, str)
        assert preds in examples.completions
    else:
        assert len(preds) == len(examples)
        for pred, example in zip(preds, examples):
            assert pred in example.completions

    # Test bad inputs
    with pytest.raises(ValueError, match="examples must be non-empty."):
        classify.log_probs_conditional_examples([], *args, **kwargs)

    is_many_examples = not isinstance(examples, Example)

    def is_hashable(objects):
        try:
            set(objects)
        except TypeError:
            return False
        else:
            return True

    if is_many_examples and is_hashable(examples):
        # if example.completions is a list, or a prior is an array, then the example
        # isn't hashable => set(examples) isn't possible in the first place
        with pytest.raises(TypeError, match="examples must be an ordered collection."):
            classify.log_probs_conditional_examples(set(examples), *args, **kwargs)

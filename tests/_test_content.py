"""
Helper functions to test that outputs of `classify` modules have the correct content,
assuming a reference/correct implementation exists.
"""
from __future__ import annotations
from typing import Sequence

import torch

from cappr import Example
from _protocol import classify_module


atol = 1e-4  # TODO: fixture?


# Note: throughout this module, we'll cast to torch float16 tensors to be consistent
# with the way numerical closeness is defined for model-dependent outputs.
dtype = torch.float16


def _test_log_probs_conditional(
    log_probs_completions1: list[list[float]] | list[list[list[float]]],
    log_probs_completions2: list[list[float]] | list[list[list[float]]],
    is_single_input: bool,
):
    """
    Helper
    """

    def test_single_input(log_probs1, log_probs2):
        assert len(log_probs1) == len(log_probs2)
        for log_probs_tokens1, log_probs_tokens2 in zip(log_probs1, log_probs2):
            assert torch.allclose(
                torch.tensor(log_probs_tokens1).to(dtype),
                torch.tensor(log_probs_tokens2).to(dtype),
                atol=atol,
            )

    if is_single_input:
        test_single_input(log_probs_completions1, log_probs_completions2)
    else:
        assert len(log_probs_completions1) == len(log_probs_completions2)
        for log_probs1, log_probs2 in zip(
            log_probs_completions1, log_probs_completions2
        ):
            test_single_input(log_probs1, log_probs2)


def log_probs_conditional(
    classify1: classify_module,
    classify2: classify_module,
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that the conditional token log-probabilities are numerically close.
    """
    is_single_input = isinstance(prompts, str)
    log_probs_completions1 = classify1.log_probs_conditional(
        prompts, completions, *args, **kwargs
    )
    log_probs_completions2 = classify2.log_probs_conditional(
        prompts, completions, *args, **kwargs
    )
    _test_log_probs_conditional(
        log_probs_completions1, log_probs_completions2, is_single_input
    )


def log_probs_conditional_examples(
    classify1: classify_module,
    classify2: classify_module,
    examples: Example | Sequence[Example],
    *args,
    **kwargs,
):
    """
    Tests that the conditional token log-probabilities are numerically close.
    """
    is_single_input = isinstance(examples, str)
    log_probs_completions1 = classify1.log_probs_conditional_examples(
        examples, *args, **kwargs
    )
    log_probs_completions2 = classify2.log_probs_conditional_examples(
        examples, *args, **kwargs
    )
    _test_log_probs_conditional(
        log_probs_completions1, log_probs_completions2, is_single_input
    )


def predict_proba(
    classify1: classify_module,
    classify2: classify_module,
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that predicted probabilities are numerically close.
    """
    pred_probs1 = classify1.predict_proba(prompts, completions, *args, **kwargs)
    pred_probs2 = classify2.predict_proba(prompts, completions, *args, **kwargs)
    assert torch.allclose(
        torch.tensor(pred_probs1).to(dtype),
        torch.tensor(pred_probs2).to(dtype),
        atol=atol,
    )


def predict_proba_examples(
    classify1: classify_module,
    classify2: classify_module,
    examples: Example | Sequence[Example],
    *args,
    **kwargs,
):
    """
    Tests that predicted probabilities are numerically close.
    """
    pred_probs1 = classify1.predict_proba_examples(examples, *args, **kwargs)
    pred_probs2 = classify2.predict_proba_examples(examples, *args, **kwargs)
    for pred_probs1_ex, pred_probs2_ex in zip(pred_probs1, pred_probs2):
        assert torch.allclose(
            torch.tensor(pred_probs1_ex).to(dtype),
            torch.tensor(pred_probs2_ex).to(dtype),
            atol=atol,
        )


def predict(
    classify1: classify_module,
    classify2: classify_module,
    prompts: str | Sequence[str],
    completions: Sequence[str],
    *args,
    **kwargs,
):
    """
    Tests that predictions are identical.
    """
    preds1 = classify1.predict(prompts, completions, *args, **kwargs)
    preds2 = classify2.predict(prompts, completions, *args, **kwargs)
    assert preds1 == preds2


def predict_examples(
    classify1: classify_module,
    classify2: classify_module,
    examples: Example | Sequence[Example],
    *args,
    **kwargs,
):
    """
    Tests that predictions are identical.
    """
    preds1 = classify1.predict_examples(examples, *args, **kwargs)
    preds2 = classify2.predict_examples(examples, *args, **kwargs)
    assert preds1 == preds2

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


def _as_tensor(data: Sequence) -> torch.Tensor:
    # Note: throughout this module, we'll cast to torch float16 tensors to be consistent
    # with the way numerical closeness is defined for model-dependent outputs.
    dtype = torch.float16
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    return torch.tensor(data).to(dtype)


def token_logprobs(
    log_probs_texts_observed: Sequence[Sequence[float]],
    log_probs_texts_from_unbatched: Sequence[Sequence[Sequence[float]]],
    input_ids_from_unbatched: Sequence[Sequence[int]],
):
    # TODO: separate form and content checks, test number of tokens

    # The first logprob of every text must be None b/c no CausalLM estimates Pr(token)
    for log_probs_text_observed in log_probs_texts_observed:
        assert log_probs_text_observed[0] is None

    # Slice out log probs for the final expected result
    log_probs_texts_expected = []
    for _text_input_ids, _text_log_probs in zip(
        input_ids_from_unbatched, log_probs_texts_from_unbatched
    ):
        log_probs_texts_expected.append(
            [None]  # for the first token, no CausalLM estimates Pr(token)
            + [  # this token's data contains the next token's log-probability
                _text_log_probs[i, _text_input_ids[i + 1]]
                for i in range(0, len(_text_input_ids) - 1)
            ]
        )

    # Every log prob is correct, and sizes are correct
    assert len(log_probs_texts_observed) == len(log_probs_texts_expected)
    for log_probs_text_observed, log_probs_text_expected in zip(
        log_probs_texts_observed, log_probs_texts_expected
    ):
        assert len(log_probs_text_observed) == len(log_probs_text_expected)
        # skip the first token b/c its log prob is always None
        for log_prob_token_observed, log_prob_token_expected in zip(
            log_probs_text_observed[1:], log_probs_text_expected[1:]
        ):
            assert torch.isclose(
                _as_tensor(log_prob_token_observed),
                _as_tensor(log_prob_token_expected),
                atol=atol,
            )


def _test_log_probs_conditional(
    log_probs_completions1: list[list[float]] | list[list[list[float]]],
    log_probs_completions2: list[list[float]] | list[list[list[float]]],
    is_single_input: bool,
):
    def test_single_input(log_probs1: list[list[float]], log_probs2: list[list[float]]):
        assert len(log_probs1) == len(log_probs2)
        for log_probs_tokens1, log_probs_tokens2 in zip(log_probs1, log_probs2):
            assert torch.allclose(
                _as_tensor(log_probs_tokens1), _as_tensor(log_probs_tokens2), atol=atol
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
    log_probs_completions1 = classify1.log_probs_conditional(
        prompts, completions, *args, **kwargs
    )
    log_probs_completions2 = classify2.log_probs_conditional(
        prompts, completions, *args, **kwargs
    )
    _test_log_probs_conditional(
        log_probs_completions1,
        log_probs_completions2,
        is_single_input=isinstance(prompts, str),
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
    log_probs_completions1 = classify1.log_probs_conditional_examples(
        examples, *args, **kwargs
    )
    log_probs_completions2 = classify2.log_probs_conditional_examples(
        examples, *args, **kwargs
    )
    _test_log_probs_conditional(
        log_probs_completions1,
        log_probs_completions2,
        is_single_input=isinstance(examples, Example),
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
    assert torch.allclose(_as_tensor(pred_probs1), _as_tensor(pred_probs2), atol=atol)


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
            _as_tensor(pred_probs1_ex), _as_tensor(pred_probs2_ex), atol=atol
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

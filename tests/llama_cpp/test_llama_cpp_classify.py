"""
Unit and integration tests for `cappr.llama_cpp.classify`. Works by checking that its
functions' outputs are numerically close to those from
`cappr.llama_cpp._classify_no_cache`.

# TODO: factor out these tests so that they're shared across all backends.
"""
from __future__ import annotations
import os
import sys
from typing import Sequence

from llama_cpp import Llama
import numpy as np
import pandas as pd
import pytest
import torch

from cappr import Example as Ex
from cappr.llama_cpp import classify as fast
from cappr.llama_cpp import _classify_no_cache as slow
from cappr.llama_cpp._utils import log_softmax

# sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _test


_ABS_PATH_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


########################################################################################
###################################### Fixtures ########################################
########################################################################################


@pytest.fixture(
    scope="module",
    params=[
        "TinyLLama-v0.Q8_0.gguf",  # TODO: include a BPE-based model like GPT
    ],
)
def model(request: pytest.FixtureRequest) -> Llama:
    model_path = os.path.join(_ABS_PATH_THIS_DIR, "fixtures", "models", request.param)
    return Llama(model_path, logits_all=True, verbose=False)


@pytest.fixture(scope="module")
def atol() -> float:
    # Reading through some transformers tests, it looks like 1e-3 is considered
    # close-enough for hidden states. See, e.g.,
    # https://github.com/huggingface/transformers/blob/897a826d830e8b1e03eb482b165b5d88a7a08d5f/tests/models/gpt2/test_modeling_gpt2.py#L252
    return 1e-4


########################################################################################
#################################### One-off tests #####################################
########################################################################################


def test__check_model(model: Llama):
    model.context_params.logits_all = False
    with pytest.raises(TypeError):
        fast.token_logprobs(["not used"], model)
    with pytest.raises(TypeError):
        fast.predict_proba("not used", ["not used"], model, normalize=False)
    model.context_params.logits_all = True


@pytest.mark.parametrize("shape_and_dim", [((10,), 0), ((3, 10), 1)])
def test_log_softmax(shape_and_dim: tuple[int] | tuple[int, int], atol: float):
    shape, dim = shape_and_dim
    data: np.ndarray = np.random.randn(*shape)
    log_probs_observed = log_softmax(data, dim=dim)
    log_probs_expected = torch.tensor(data).log_softmax(dim=dim).numpy()
    assert np.allclose(log_probs_observed, log_probs_expected, atol=atol)


@pytest.mark.parametrize(
    "texts",
    (["a b", "c d e"], ["a fistful", "of tokens", "for a few", "tokens more"]),
)
def test_token_logprobs(texts: Sequence[str], model: Llama, atol: float):
    """
    Tests that the model's token log probabilities are correct by testing against a
    carefully, manually indexed result.
    """
    log_probs_texts_observed = fast.token_logprobs(texts, model, add_bos=True)

    # The first logprob of every text must be None b/c no CausalLM estimates Pr(token)
    for log_prob_observed in log_probs_texts_observed:
        assert log_prob_observed[0] is None

    # Gather un-batched un-sliced log probs for the expected result
    _texts_log_probs = []
    _texts_input_ids = []
    for text in texts:
        input_ids = model.tokenize(text.encode("utf-8"), add_bos=True)
        model.reset()
        model.eval(input_ids)
        _texts_log_probs.append(log_softmax(fast._check_logits(model.eval_logits)))
        _texts_input_ids.append(input_ids)
    model.reset()

    # Slice out log probs for the final expected result
    log_probs_texts_expected = []
    for _text_input_ids, _text_log_probs in zip(_texts_input_ids, _texts_log_probs):
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
            assert np.isclose(
                log_prob_token_observed, log_prob_token_expected, atol=atol
            )


########################################################################################
############################### Helpers for future tests ###############################
########################################################################################


def _test_log_probs(
    log_probs_completions_slow: list[list[float]] | list[list[list[float]]],
    log_probs_completions_fast: list[list[float]] | list[list[list[float]]],
    expected_len: int,
    num_completions_per_prompt: list[int] | None,
    atol,
):
    """
    Tests that the conditional token log-probabilities are the right shape and are
    numerically close.
    """

    def test_single_input(log_probs_slow, log_probs_fast, num_completions):
        assert len(log_probs_fast) == num_completions
        assert len(log_probs_slow) == num_completions
        zipped_inner = zip(log_probs_slow, log_probs_fast)
        for log_probs_tokens_slow, log_probs_tokens_fast in zipped_inner:
            # cast to tensor so that we are consistent w/ the way "numerical closeness"
            # is defined for model-dependent outputs
            assert np.allclose(
                np.array(log_probs_tokens_slow),
                np.array(log_probs_tokens_fast),
                atol=atol,
            )

    if num_completions_per_prompt is None:
        # there's only one prompt and it was fed by itself, not in a Sequence.
        test_single_input(
            log_probs_completions_slow, log_probs_completions_fast, expected_len
        )
    else:
        # Test lengths before zipping
        assert len(log_probs_completions_slow) == len(log_probs_completions_fast)
        assert len(log_probs_completions_fast) == expected_len
        assert len(log_probs_completions_slow) == len(num_completions_per_prompt)
        zipped_outer = zip(
            log_probs_completions_slow,
            log_probs_completions_fast,
            num_completions_per_prompt,
        )
        for log_probs_slow, log_probs_fast, num_completions in zipped_outer:
            test_single_input(log_probs_slow, log_probs_fast, num_completions)


def _function_kwargs(function_kwargs: dict, non_args: tuple[str] = ("self", "atol")):
    """
    This module tests two implementations of the same thing. This utility grabs their
    inputs so that arguments don't have to be repeated.
    """
    return {arg: value for arg, value in function_kwargs.items() if arg not in non_args}


########################################################################################
####################################### Tests ##########################################
########################################################################################


@pytest.mark.parametrize(
    "prompts",
    (
        ["a b c", "c"],
        "prompts can be a single string",
        pd.Series(["prompts can", "be a", "Series"], index=np.random.choice(3, size=3)),
    ),
)
@pytest.mark.parametrize(
    "completions",
    (
        ["d e f g", "1 2", "O"],
        ######################## Test Single-token optimization ########################
        ["d", "e", "f"],
        ########################### Test pandas Series input ###########################
        pd.Series(["completions as", "a", "Series"], index=np.random.choice(3, size=3)),
    ),
)
class TestPromptsCompletions:
    """
    Tests all model-dependent, non-`_examples` functions, sharing the same set of
    prompts and completions.
    """

    def test_log_probs_conditional(self, prompts, completions, model, atol):
        kwargs = _function_kwargs(locals())
        if isinstance(prompts, str):
            expected_len = len(completions)
            num_completions_per_prompt = None
        else:
            expected_len = len(prompts)
            num_completions_per_prompt = [len(completions)] * len(prompts)
        log_probs_completions_slow = slow.log_probs_conditional(**kwargs)
        log_probs_completions_fast = fast.log_probs_conditional(**kwargs)
        _test_log_probs(
            log_probs_completions_slow,
            log_probs_completions_fast,
            expected_len,
            num_completions_per_prompt,
            atol,
        )

    @pytest.mark.parametrize("discount_completions", (0.0, 1.0))
    @pytest.mark.parametrize("normalize", (True, False))
    def test_predict_proba(
        self,
        prompts,
        completions,
        model,
        normalize,
        discount_completions,
        atol,
    ):
        kwargs = _function_kwargs(locals())
        # Test form of output
        _test.predict_proba(fast.predict_proba, **kwargs)
        _test.predict_proba(slow.predict_proba, **kwargs)

        # Test that predictions match
        pred_probs_fast = fast.predict_proba(**kwargs)
        pred_probs_slow = slow.predict_proba(**kwargs)
        # cast to tensor so that we are consistent w/ the way "numerical closeness"
        # is defined for model-dependent outputs
        assert np.allclose(
            np.array(pred_probs_fast), np.array(pred_probs_slow), atol=atol
        )

    @pytest.mark.parametrize("discount_completions", (0.0, 1.0))
    def test_predict(
        self,
        prompts,
        completions,
        model,
        discount_completions,
    ):
        kwargs = _function_kwargs(locals())
        # Test form of output
        _test.predict(fast.predict, **kwargs)
        _test.predict(slow.predict, **kwargs)

        # Test that predictions match
        preds_fast = fast.predict(**kwargs)
        preds_slow = slow.predict(**kwargs)
        assert preds_fast == preds_slow


@pytest.mark.parametrize(
    "examples",
    (
        [
            Ex("a b c", ["d", "e f g"]),
            Ex("C", ["p G C p G", "D E F", "ya later alligator"]),
        ],
        ############################# Next set of examples #############################
        [
            Ex("chi", ["can", "ery"]),
            Ex("koyaa", ["nisqatsi"], normalize=False),
            Ex("hi hi", ["bye bye", "yo yo"]),
        ],
        ############## Test constant # completions, non-constant # tokens ##############
        [
            Ex("jelly", ["fin", "is"]),
            Ex("a great", ["thing.", "shout"]),
            Ex("out to", ["open", "source, yo."]),
        ],
        ############################ Test singleton example ############################
        Ex("lonesome", ["singleton", "example"]),
        ################### Test singleton example single completion ###################
        Ex("lonely", ["loner"], normalize=False),
    ),
)
class TestExamples:
    """
    Tests all model-dependent, `_examples` functions, sharing the same set of examples.
    """

    def test_log_probs_conditional_examples(self, examples: list[Ex], model, atol):
        kwargs = _function_kwargs(locals())
        if isinstance(examples, Ex):
            expected_len = len(examples.completions)
            num_completions_per_prompt = None
        else:
            expected_len = len(examples)
            num_completions_per_prompt = [
                len(example.completions) for example in examples
            ]
        log_probs_completions_slow = slow.log_probs_conditional_examples(**kwargs)
        log_probs_completions_fast = fast.log_probs_conditional_examples(**kwargs)
        _test_log_probs(
            log_probs_completions_slow,
            log_probs_completions_fast,
            expected_len,
            num_completions_per_prompt,
            atol,
        )

    def test_predict_proba_examples(self, examples, model, atol):
        kwargs = _function_kwargs(locals())
        # Test form of output
        _test.predict_proba_examples(fast.predict_proba_examples, **kwargs)
        _test.predict_proba_examples(slow.predict_proba_examples, **kwargs)

        # Test that predictions match
        pred_probs_fast = fast.predict_proba_examples(**kwargs)
        pred_probs_slow = slow.predict_proba_examples(**kwargs)
        for pred_probs_fast_ex, pred_probs_slow_ex in zip(
            pred_probs_fast, pred_probs_slow
        ):
            # cast to tensor so that we are consistent w/ the way "numerical closeness"
            # is defined for model-dependent outputs
            assert np.allclose(
                np.array(pred_probs_fast_ex),
                np.array(pred_probs_slow_ex),
                atol=atol,
            )

    def test_predict_examples(self, examples, model):
        kwargs = _function_kwargs(locals())
        if (
            isinstance(examples, Ex)
            and len(examples.completions) == 1
            and examples.normalize
        ):
            # not a valid input for this function. it's fine for predict_proba_examples
            return
        # Test form of output
        _test.predict_examples(fast.predict_examples, **kwargs)
        _test.predict_examples(slow.predict_examples, **kwargs)

        # Test that predictions match
        preds_fast = fast.predict_examples(**kwargs)
        preds_slow = slow.predict_examples(**kwargs)
        assert preds_fast == preds_slow

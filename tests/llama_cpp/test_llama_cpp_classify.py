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
import pytest
import torch

from cappr import Example
from cappr.llama_cpp import classify, _classify_no_cache
from cappr.llama_cpp._utils import log_softmax

# sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from _base import BaseTestPromptsCompletions, BaseTestExamples


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
        classify.token_logprobs(["not used"], model)
    with pytest.raises(TypeError):
        classify.predict_proba("not used", ["not used"], model, normalize=False)
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
    log_probs_texts_observed = classify.token_logprobs(texts, model, add_bos=True)

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
        _texts_log_probs.append(log_softmax(classify._check_logits(model.eval_logits)))
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
####################################### Tests ##########################################
########################################################################################


class Modules:
    @property
    def module_correct(self):
        return _classify_no_cache

    @property
    def modules_to_test(self):
        return (classify,)


class TestPromptsCompletions(Modules, BaseTestPromptsCompletions):
    def test_log_probs_conditional(self, prompts, completions, model):
        super().test_log_probs_conditional(prompts, completions, model)

    def test_predict_proba(
        self,
        prompts,
        completions,
        model,
        _use_prior,
        discount_completions,
        normalize,
    ):
        super().test_predict_proba(
            prompts,
            completions,
            model,
            _use_prior=_use_prior,
            discount_completions=discount_completions,
            normalize=normalize,
        )

    def test_predict(self, prompts, completions, model):
        super().test_predict(prompts, completions, model)


class TestExamples(Modules, BaseTestExamples):
    def test_log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], model
    ):
        super().test_log_probs_conditional_examples(examples, model)

    def test_predict_proba_examples(self, examples: Example | Sequence[Example], model):
        super().test_predict_proba_examples(examples, model)

    def test_predict_examples(self, examples: Example | Sequence[Example], model):
        super().test_predict_examples(examples, model)

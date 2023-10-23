"""
Unit and integration tests for `cappr.llama_cpp.classify`. Works by checking that its
functions' outputs are numerically close to those from
`cappr.llama_cpp._classify_no_cache`.
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
import _test_form
import _test_content


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
    model = Llama(model_path, logits_all=True, verbose=False)
    # Correctness should not be affected by a user's previous actions. Mimic that:
    model.eval(model.tokenize("a b c".encode("utf-8")))
    return model


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
def test_log_softmax(
    shape_and_dim: tuple[tuple[int] | tuple[int, int], int], atol: float
):
    shape, dim = shape_and_dim
    data: np.ndarray = np.random.randn(*shape)
    log_probs_observed = log_softmax(data, dim=dim)
    log_probs_expected = torch.tensor(data).log_softmax(dim=dim).numpy()
    assert np.allclose(log_probs_observed, log_probs_expected, atol=atol)


@pytest.mark.parametrize(
    "texts",
    (["a b", "c d e"], ["a fistful", "of tokens", "for a few", "tokens more"]),
)
def test_token_logprobs(texts: Sequence[str], model: Llama):
    """
    Tests that the model's token log probabilities are correct by testing against a
    carefully, manually indexed result.
    """
    log_probs_texts_observed = classify.token_logprobs(texts, model, add_bos=True)

    # Gather un-batched un-sliced log probs for the expected result
    log_probs_texts_from_unbatched = []
    input_ids_from_unbatched = []
    for text in texts:
        input_ids = model.tokenize(text.encode("utf-8"), add_bos=True)
        model.reset()
        model.eval(input_ids)
        log_probs_texts_from_unbatched.append(
            log_softmax(classify._check_logits(model.eval_logits))
        )
        input_ids_from_unbatched.append(input_ids)
    model.reset()

    _test_content.token_logprobs(
        log_probs_texts_observed,
        log_probs_texts_from_unbatched,
        input_ids_from_unbatched,
    )


def test_cache(model: Llama):
    prompt_prefix = "a b c"
    prompts = ["d", "d e"]
    completions = ["e f", "f g"]

    n_tokens = model.n_tokens
    with classify.cache(model, prompt_prefix):
        log_probs_completions = classify.log_probs_conditional(
            prompts, completions, model, reset_model=False
        )
    assert model.n_tokens == n_tokens
    _test_form._test_log_probs_conditional(
        log_probs_completions,
        expected_len=len(prompts),
        num_completions_per_prompt=[len(completions)] * len(prompts),
    )

    prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
    log_probs_completions_wo_cache = classify.log_probs_conditional(
        prompts_full, completions, model, reset_model=True
    )
    assert model.n_tokens == 0
    _test_content._test_log_probs_conditional(
        log_probs_completions, log_probs_completions_wo_cache, is_single_input=False
    )


def test_cache_examples(model: Llama):
    prompt_prefix = "a b c"
    _prompts = ["d", "d e"]
    completions = ["e f", "f g"]
    examples = [Example(prompt, completions) for prompt in _prompts]

    n_tokens = model.n_tokens
    with classify.cache(model, prompt_prefix):
        log_probs_completions = classify.log_probs_conditional_examples(
            examples, model, reset_model=False
        )
    assert model.n_tokens == n_tokens
    _test_form._test_log_probs_conditional(
        log_probs_completions,
        expected_len=len(examples),
        num_completions_per_prompt=[len(example.completions) for example in examples],
    )

    examples_full = [
        Example(prompt_prefix + " " + example.prompt, example.completions)
        for example in examples
    ]
    log_probs_completions_wo_cache = classify.log_probs_conditional_examples(
        examples_full, model, reset_model=True
    )
    assert model.n_tokens == 0
    _test_content._test_log_probs_conditional(
        log_probs_completions, log_probs_completions_wo_cache, is_single_input=False
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

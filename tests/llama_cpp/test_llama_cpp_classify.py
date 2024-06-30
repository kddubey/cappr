"""
Unit and integration tests for `cappr.llama_cpp.classify`. Works by checking that its
functions' outputs are numerically close to those from
`cappr.llama_cpp._classify_no_cache`.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import sys
from typing import Sequence

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import numpy as np
import pytest
import torch

from cappr import Example
from cappr.llama_cpp import _utils, classify, _classify_no_cache

# sys hack to import from parent
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _base
import _test_content


_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "models"
)


########################################################################################
###################################### Fixtures ########################################
########################################################################################


@dataclass(frozen=True)
class _HFHubModel:
    repo_id: str
    filename: str
    does_tokenizer_need_prepended_space: bool


@pytest.fixture(
    scope="module",
    params=[
        _HFHubModel(
            repo_id="aladar/tiny-random-BloomForCausalLM-GGUF",
            filename="tiny-random-BloomForCausalLM.gguf",
            does_tokenizer_need_prepended_space=True,  # BPE
        ),
        _HFHubModel(
            repo_id="aladar/tiny-random-LlamaForCausalLM-GGUF",
            filename="tiny-random-LlamaForCausalLM.gguf",
            does_tokenizer_need_prepended_space=False,  # SentencePiece
        ),
    ],
)
def model_and_does_tokenizer_need_prepended_space(
    request: pytest.FixtureRequest,
) -> tuple[Llama, bool]:
    hf_hub_model: _HFHubModel = request.param
    model_path = hf_hub_download(
        hf_hub_model.repo_id, hf_hub_model.filename, local_dir=_MODELS_DIR
    )
    model = Llama(model_path, verbose=False)
    return model, hf_hub_model.does_tokenizer_need_prepended_space


@pytest.fixture(scope="module")
def model(model_and_does_tokenizer_need_prepended_space: tuple[Llama, bool]) -> Llama:
    model, _ = model_and_does_tokenizer_need_prepended_space
    # Correctness should not be affected by a user's previous actions, which get cached
    model.eval(model.tokenize("a b c".encode()))
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


def test__does_tokenizer_need_prepended_space(
    model_and_does_tokenizer_need_prepended_space,
):
    """
    Explicitly test this b/c w/o it, headaches are possible.
    """
    (
        model,
        does_tokenizer_need_prepended_space_expected,
    ) = model_and_does_tokenizer_need_prepended_space
    assert (
        _utils.does_tokenizer_need_prepended_space(model)
        == does_tokenizer_need_prepended_space_expected
    )
    # If it's cached, let's double check that we still get the expected result
    assert (
        _utils.does_tokenizer_need_prepended_space(model)
        == does_tokenizer_need_prepended_space_expected
    )


def test_set_up_model(model: Llama):
    assert not model.context_params.logits_all
    with _utils.set_up_model(model):
        assert model.context_params.logits_all
    assert not model.context_params.logits_all


@pytest.mark.parametrize("shape_and_dim", [((10,), 0), ((3, 10), 1)])
def test_log_softmax(
    shape_and_dim: tuple[tuple[int] | tuple[int, int], int], atol: float
):
    shape, dim = shape_and_dim
    data: np.ndarray = np.random.randn(*shape)
    log_probs_observed = _utils.log_softmax(data, dim=dim)
    log_probs_expected = torch.tensor(data).log_softmax(dim=dim).numpy()
    assert np.allclose(log_probs_observed, log_probs_expected, atol=atol)


@pytest.mark.parametrize(
    "texts",
    (
        "lone string input",
        ["a b", "c d e"],
        ["a fistful", "of tokens", "for a few", "tokens more"],
    ),
)
def test_token_logprobs(texts: Sequence[str], model: Llama, end_of_prompt=""):
    """
    Tests that the model's token log probabilities are correct by testing against a
    carefully, manually indexed result.
    """
    log_probs_texts_observed = classify.token_logprobs(
        texts, model, add_bos=True, end_of_prompt=end_of_prompt
    )

    # Gather un-batched un-sliced log probs for the expected result
    is_str = isinstance(texts, str)
    texts = [texts] if is_str else texts
    if not _utils.does_tokenizer_need_prepended_space(model):
        end_of_prompt = ""
    log_probs_texts_from_unbatched = []
    input_ids_from_unbatched = []
    with _utils.set_up_model(model):
        for text in texts:
            input_ids = model.tokenize((end_of_prompt + text).encode(), add_bos=True)
            model.reset()
            model.eval(input_ids)
            log_probs_texts_from_unbatched.append(
                _utils.log_softmax(np.array(model.eval_logits))
            )
            input_ids_from_unbatched.append(input_ids)
    model.reset()

    log_probs_texts_observed = (
        [log_probs_texts_observed] if is_str else log_probs_texts_observed
    )
    _test_content.token_logprobs(
        log_probs_texts_observed,
        log_probs_texts_from_unbatched,
        input_ids_from_unbatched,
    )


########################################################################################
####################################### Tests ##########################################
########################################################################################


class Modules:
    @property
    def module(self):
        return classify

    @property
    def module_correct(self):
        return _classify_no_cache


class TestPromptsCompletions(Modules, _base.TestPromptsCompletions):
    def test_log_probs_conditional(self, prompts, completions, model):
        super().test_log_probs_conditional(prompts, completions, model)

    def test_predict_proba(
        self,
        prompts,
        completions,
        model,
        prior,
        discount_completions,
        normalize,
    ):
        super().test_predict_proba(
            prompts,
            completions,
            model,
            prior=prior,
            discount_completions=discount_completions,
            normalize=normalize,
        )

    def test_predict(self, prompts, completions, model):
        super().test_predict(prompts, completions, model)


class TestExamples(Modules, _base.TestExamples):
    def test_log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], model
    ):
        super().test_log_probs_conditional_examples(examples, model)

    def test_predict_proba_examples(self, examples: Example | Sequence[Example], model):
        super().test_predict_proba_examples(examples, model)

    def test_predict_examples(self, examples: Example | Sequence[Example], model):
        super().test_predict_examples(examples, model)


class TestCache(Modules, _base.TestCache):
    def _test_log_probs_conditional(self, *args, **kwargs):
        model: Llama = args[-1]
        model.reset()  # reset cache before evaluating correct module
        super()._test_log_probs_conditional(*args, **kwargs)

    def _test_log_probs_conditional_examples(self, *args, **kwargs):
        model: Llama = args[-1]
        model.reset()  # reset cache before evaluating correct module
        super()._test_log_probs_conditional_examples(*args, **kwargs)

    def test_cache(
        self, prompt_prefix: str, prompts: list[str], completions: list[str], model
    ):
        super().test_cache(
            prompt_prefix, prompts, completions, model, reset_model=False
        )

    def test_cache_model(
        self, prompt_prefix: str, prompts: list[str], completions: list[str], model
    ):
        super().test_cache_model(
            prompt_prefix, prompts, completions, model, reset_model=False
        )

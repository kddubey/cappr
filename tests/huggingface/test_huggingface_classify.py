"""
Unit and integration tests for `cappr.huggingface.classify`. Works by checking that its
functions' outputs are numerically close to those from
`cappr.huggingface.classify_no_cache`.
"""
from __future__ import annotations
import os
import sys
from typing import Sequence

import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
import huggingface_hub as hf_hub

from cappr import Example
from cappr.huggingface import classify_no_cache, classify, classify_no_batch
from cappr import huggingface as hf
from cappr.huggingface._utils import BatchEncoding, ModelForCausalLM

# sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from _base import BaseTestPromptsCompletions, BaseTestExamples
from _test_content import token_logprobs as _test_token_logprobs


########################################################################################
###################################### Fixtures ########################################
########################################################################################


@pytest.fixture(
    scope="module",
    params=[
        "hf-internal-testing/tiny-random-GPT2LMHeadModel",
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "Maykeye/TinyLLama-v0",
        "hf-internal-testing/tiny-random-MistralForCausalLM",
    ],
)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_name: str) -> ModelForCausalLM:
    model: ModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    contexts_model = [context(model) for context in hf._utils._DEFAULT_CONTEXTS_MODEL]
    for context in contexts_model:
        context.__enter__()
    return model


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    # hf-internal-testing/tiny-random-MistralForCausalLM's tokenizer_config.json has a
    # field, tokenizer_file, which is hard-coded to some specific machine
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        # tokenizer_file not found b/c it was hard-coded. Find it locally.
        local_path = hf_hub.try_to_load_from_cache(model_name, "tokenizer.json")
        tokenizer = AutoTokenizer.from_pretrained(model_name, tokenizer_file=local_path)
    return tokenizer


@pytest.fixture(scope="module")
def tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = _load_tokenizer(model_name)
    contexts_tokenizer = [
        context(tokenizer) for context in hf._utils._DEFAULT_CONTEXTS_TOKENIZER
    ]
    for context in contexts_tokenizer:
        context.__enter__()
    return tokenizer


@pytest.fixture(scope="module")
def model_and_tokenizer(
    model_name: str,
) -> tuple[ModelForCausalLM, PreTrainedTokenizerBase]:
    # This input is directly from a user, so we can't assume the model and tokenizer are
    # set up correctly. For testing, that means we shouldn't just return the fixtures:
    # return model, tokenizer
    # Instead, load them from scratch like a user would:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = _load_tokenizer(model_name)
    return model, tokenizer


@pytest.fixture(scope="module")
def atol() -> float:
    # Reading through some transformers tests, it looks like 1e-3 is considered
    # close-enough for hidden states. See, e.g.,
    # https://github.com/huggingface/transformers/blob/897a826d830e8b1e03eb482b165b5d88a7a08d5f/tests/models/gpt2/test_modeling_gpt2.py#L252
    return 1e-4


########################################################################################
#################################### One-off tests #####################################
########################################################################################


def test_set_up_model_and_tokenizer(model_and_tokenizer):
    """
    Tests that the context manager doesn't change any attributes of the model or
    tokenizer when we've exited the context. It's conceivable that someone uses CAPPr to
    evaluate their model on a downstream application during a train-and-validate loop.
    Or they're using CAPPr as part of a larger system where the tokenizer needs to be
    configured differently.
    """
    model, tokenizer = model_and_tokenizer
    # Not sure why type inference isn't catching these
    model: ModelForCausalLM
    tokenizer: PreTrainedTokenizerBase

    # Grab old attribute values.
    model_attribute_to_old_value = {
        "training": model.training,
        # we could keep recursing on children, but whatever
        **{i: module.training for i, module in enumerate(model.children())},
    }
    tokenizer_attributes = [
        "pad_token_id",
        "padding_side",
        "add_eos_token",
        "pad_token",
        "special_tokens_map",
    ]
    tokenizer_attribute_to_old_value = {
        attribute: getattr(tokenizer, attribute, None)
        # None is for tokenizers which don't have an add_eos_token attribute
        for attribute in tokenizer_attributes
    }

    # Enter the context
    with hf._utils.set_up_model_and_tokenizer(model_and_tokenizer):
        model, tokenizer = model_and_tokenizer

    # Exit the context. No attributes should have changed.
    assert model.training == model_attribute_to_old_value["training"]
    for i, module in enumerate(model.children()):
        assert module.training == model_attribute_to_old_value[i]

    for attribute, old_value in tokenizer_attribute_to_old_value.items():
        assert getattr(tokenizer, attribute, None) == old_value


@pytest.mark.parametrize("module", (classify_no_cache, classify, classify_no_batch))
@pytest.mark.parametrize(
    "texts",
    (["a b", "c d e"], ["a fistful", "of tokens", "for a few", "tokens more"]),
)
@pytest.mark.parametrize("batch_size", (2, 1))
def test_token_logprobs(
    module, texts, model_and_tokenizer, batch_size, atol, end_of_prompt=" "
):
    """
    Tests that the model's token log probabilities are correct by testing against an
    unbatched and carefully, manually indexed result.
    """
    module: classify = module  # just for the type hint
    log_probs_texts_observed = module.token_logprobs(
        texts, model_and_tokenizer, batch_size=batch_size
    )

    # Gather un-batched un-sliced log probs for the expected result
    # bleh
    if (
        end_of_prompt == " "
        and not hf._utils.does_tokenizer_prepend_space_to_first_token(
            model_and_tokenizer[1]
        )
    ):
        end_of_prompt = ""
    log_probs_texts_from_unbatched = []
    input_ids_from_unbatched = []
    for text in texts:
        with hf._utils.set_up_model_and_tokenizer(model_and_tokenizer):
            model, tokenizer = model_and_tokenizer
            _logits, _encoding = hf._utils.logits_texts(
                [end_of_prompt + text], model, tokenizer
            )
        # grab first index b/c we only gave it 1 text
        log_probs_texts_from_unbatched.append(_logits[0].log_softmax(dim=1))
        input_ids_from_unbatched.append(_encoding["input_ids"][0])

    _test_token_logprobs(
        log_probs_texts_observed,
        log_probs_texts_from_unbatched,
        input_ids_from_unbatched,
    )


@pytest.mark.parametrize("prompts", (["a b c", "c"],))
@pytest.mark.parametrize("num_completions_per_prompt", (2, (2, 3)))
def test__keys_values_prompts(
    model, tokenizer, prompts, num_completions_per_prompt, atol
):
    """
    Tests that the model's attention keys and values for the prompt are identical. If
    this test fails, all tests which call `_test_logits` will also fail.
    """
    _outputs_slow = classify_no_cache._keys_values_prompts(
        model, tokenizer, prompts, num_completions_per_prompt
    )
    _outputs_fast = classify._keys_values_prompts(
        model, tokenizer, prompts, num_completions_per_prompt
    )
    keys_vals_slow, encodings_slow, offsets_slow, logits_last_slow = _outputs_slow
    keys_vals_fast, encodings_fast, offsets_fast, logits_last_fast = _outputs_fast

    assert len(keys_vals_slow) == len(keys_vals_fast)  # same # attention blocks
    for (keys_slow, vals_slow), (keys_fast, vals_fast) in zip(
        keys_vals_slow, keys_vals_fast
    ):
        assert torch.allclose(keys_slow, keys_fast, atol=atol)
        assert torch.allclose(vals_slow, vals_fast, atol=atol)

    assert torch.equal(encodings_slow["input_ids"], encodings_fast["input_ids"])
    assert torch.equal(
        encodings_slow["attention_mask"], encodings_fast["attention_mask"]
    )
    assert torch.equal(offsets_slow, offsets_fast)

    assert torch.allclose(logits_last_slow, logits_last_fast, atol=atol)


########################################################################################
################################### Helpers for tests ##################################
########################################################################################


def _test_encodings(
    logits_slow: torch.Tensor,
    encodings_slow: BatchEncoding,
    logits_fast: torch.Tensor,
    encodings_fast: BatchEncoding,
):
    """
    Tests that all objects have the expected shape, and that the encodings `offsets` are
    identical.
    """
    if logits_slow.shape[0] > logits_fast.shape[0] and logits_fast.shape[1] == 1:
        # Single-token optimization: this test doesn't apply b/c the optimization
        # doesn't repeat any data, unlike what's done in the slow/no-cache module
        return

    def _test_shapes(logits: torch.Tensor, encodings: BatchEncoding):
        assert encodings["input_ids"].shape == logits.shape[:2]  # 3rd dim is vocab
        assert encodings["input_ids"].shape == encodings["attention_mask"].shape
        assert encodings["input_ids"].shape[0] == encodings["offsets"].shape[0]

    _test_shapes(logits_slow, encodings_slow)
    _test_shapes(logits_fast, encodings_fast)

    # Test offsets. These should be exactly the same b/c they're the number of
    # of non-pad tokens in each prompt
    assert torch.equal(encodings_slow["offsets"], encodings_fast["offsets"])


def _test_logits(
    logits_slow: torch.Tensor,
    encodings_slow: BatchEncoding,
    logits_fast: torch.Tensor,
    encodings_fast: BatchEncoding,
    atol,
):
    """
    Tests that logits have identical shape, and that their non-pad token logits are
    numerically close.
    """
    if logits_slow.shape[0] > logits_fast.shape[0] and logits_fast.shape[1] == 1:
        # Single-token optimization: we only need to compare the last nonpad token's
        # logits for each prompt.
        num_completions = int(logits_slow.shape[0] / logits_fast.shape[0])
        logits_fast = logits_fast.repeat_interleave(num_completions, dim=0)
        last_nonpad_token_idxs = (encodings_slow["offsets"] - 1)[:, None, None]
        logits_slow_last_nonpad_token = logits_slow.take_along_dim(
            last_nonpad_token_idxs, dim=1
        )
        assert (
            logits_fast.shape == logits_slow_last_nonpad_token.shape
        )  # allclose doesn't check this
        assert torch.allclose(logits_fast, logits_slow_last_nonpad_token, atol=atol)
        return

    # Middle dimension (for the # of tokens) is different b/c logits_slow includes
    # prompt and completion tokens, while logits_fast only includes completion tokens.
    assert logits_slow.shape[2] == logits_fast.shape[2]  # vocab size

    # Test logits at every *non-pad* token
    completion_token_idxs = [
        list(range(num_completion_tokens))
        for num_completion_tokens in encodings_fast["attention_mask"].sum(dim=1)
    ]
    for text_idx in range(logits_slow.shape[0]):
        offset = encodings_fast["offsets"][text_idx].item() - 1
        # number of non-pad prompt tokens - 1 (!) b/c in the fast version we
        # included the last non-pad prompt token
        for completion_token_idx in completion_token_idxs[text_idx]:
            assert torch.allclose(
                logits_fast[text_idx, completion_token_idx],
                logits_slow[text_idx, offset + completion_token_idx],
                atol=atol,
            )


########################################################################################
####################################### Tests ##########################################
########################################################################################


class Modules:
    @property
    def module_correct(self):
        return classify_no_cache

    @property
    def modules_to_test(self):
        return (classify, classify_no_batch)


class TestPromptsCompletions(Modules, BaseTestPromptsCompletions):
    def test__logits_completions_given_prompts(
        self, model, tokenizer, prompts, completions, atol
    ):
        # for this function, prompts can't be a single string
        if isinstance(prompts, str):
            return
        slow_out = classify_no_cache._logits_completions_given_prompts(
            model, tokenizer, prompts, completions
        )
        fast_out = classify._logits_completions_given_prompts(
            model, tokenizer, prompts, completions
        )
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out, atol)

    @pytest.mark.parametrize("batch_size", (2, 1))
    def test_log_probs_conditional(
        self, prompts, completions, model_and_tokenizer, batch_size
    ):
        super().test_log_probs_conditional(
            prompts, completions, model_and_tokenizer, batch_size=batch_size
        )

    def test_predict_proba(
        self,
        prompts,
        completions,
        model_and_tokenizer,
        _use_prior,
        discount_completions,
        normalize,
    ):
        super().test_predict_proba(
            prompts,
            completions,
            model_and_tokenizer,
            _use_prior=_use_prior,
            discount_completions=discount_completions,
            normalize=normalize,
        )

    def test_predict(self, prompts, completions, model_and_tokenizer):
        super().test_predict(prompts, completions, model_and_tokenizer)


class TestExamples(Modules, BaseTestExamples):
    def test__logits_completions_given_prompts_examples(
        self, model, tokenizer, examples, atol
    ):
        # for this helper function, examples can't be an Example
        if isinstance(examples, Example):
            return
        slow_out = classify_no_cache._logits_completions_given_prompts_examples(
            model, tokenizer, examples
        )
        fast_out = classify._logits_completions_given_prompts_examples(
            model, tokenizer, examples
        )
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out, atol)

    @pytest.mark.parametrize("batch_size", (2, 1))
    def test_log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], model_and_tokenizer, batch_size
    ):
        super().test_log_probs_conditional_examples(
            examples, model_and_tokenizer, batch_size=batch_size
        )

    def test_predict_proba_examples(
        self, examples: Example | Sequence[Example], model_and_tokenizer
    ):
        super().test_predict_proba_examples(examples, model_and_tokenizer)

    def test_predict_examples(
        self, examples: Example | Sequence[Example], model_and_tokenizer
    ):
        super().test_predict_examples(examples, model_and_tokenizer)

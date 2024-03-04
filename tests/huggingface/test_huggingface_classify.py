"""
Unit and integration tests for `cappr.huggingface.classify`. Works by checking that its
functions' outputs are numerically close to those from
`cappr.huggingface.classify_no_cache`.
"""

from __future__ import annotations
from contextlib import nullcontext
import os
import sys
from typing import Sequence

import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutput
import huggingface_hub as hf_hub

from cappr import Example
from cappr.huggingface import classify, classify_no_cache
from cappr import huggingface as hf
from cappr.huggingface._utils import BatchEncodingPT, ModelForCausalLM

# sys hack to import from parent
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _base
import _test_content
from _protocol import classify_module


########################################################################################
###################################### Fixtures ########################################
########################################################################################


@pytest.fixture(
    scope="module",
    params=[
        "hf-internal-testing/tiny-random-GPT2LMHeadModel",
        "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        # These models are reduntant. They're all transformers w/ BPE or SentencePiece
        # tokenization and a non-None eos_token
        # "hf-internal-testing/tiny-random-GPTJForCausalLM",
        # "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        # "hf-internal-testing/tiny-random-MistralForCausalLM",
    ],
)
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_name: str) -> ModelForCausalLM:
    model: ModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    # Set attributes to values that would break CAPPr, if not for the context managers
    model.train()  # LMs w/ dropout (GPT-2) will cause mismatched logits b/c random
    model.config.return_dict = False  # out.logits fails (not for transformers>=4.31)
    setattr(model.config, "use_cache", False)  # out.past_key_values fails
    return model


@pytest.fixture(scope="module")
def tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        # hf-internal-testing/tiny-random-MistralForCausalLM's tokenizer_config.json has
        # a field, tokenizer_file, which is hard-coded to some specific machine
        # Find it locally
        local_path = hf_hub.try_to_load_from_cache(model_name, "tokenizer.json")
        tokenizer = AutoTokenizer.from_pretrained(model_name, tokenizer_file=local_path)
    # Set attributes to values that would break CAPPr, if not for the context managers
    tokenizer.padding_side = "left"  # mismatched logits content b/c of position IDs
    if hasattr(tokenizer, "add_eos_token"):
        setattr(tokenizer, "add_eos_token", True)  # mismatched logits shape
    return tokenizer


@pytest.fixture(scope="module")
def model_and_tokenizer(
    model, tokenizer
) -> tuple[ModelForCausalLM, PreTrainedTokenizerBase]:
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


def test_set_up_model(model: ModelForCausalLM):
    # Grab old attribute values
    attribute_to_old_value = {
        "training": model.training,
        **{i: module.training for i, module in enumerate(model.children())},
    }
    config_attribute_to_old_value = {
        "return_dict": model.config.return_dict,
        "use_cache": getattr(model.config, "use_cache", None),
    }
    assert torch.is_grad_enabled()

    # Enter the context
    with hf._utils.set_up_model(model):
        assert not torch.is_grad_enabled()
        assert not model.training
        if hasattr(model, "config"):
            # If the model doesn't have these attributes, then we'll see the error loud
            # and clear later in the tests. So default to the expected values
            assert getattr(model.config, "return_dict", True)
            assert getattr(model.config, "use_cache", True)

    # Exit the context. No attributes should have changed
    assert torch.is_grad_enabled()
    assert model.training == attribute_to_old_value["training"]
    for i, module in enumerate(model.children()):
        assert module.training == attribute_to_old_value[i]

    for attribute, old_value in config_attribute_to_old_value.items():
        assert getattr(model.config, attribute, None) == old_value


def test_set_up_tokenizer(tokenizer: PreTrainedTokenizerBase):
    # Grab old attribute values
    attributes = [
        "padding_side",
        "pad_token",
        "pad_token_id",
        "add_eos_token",
        "special_tokens_map",
    ]
    attribute_to_old_value = {
        attribute: getattr(tokenizer, attribute, None)
        # None is for tokenizers which don't have an add_eos_token attribute
        for attribute in attributes
    }

    # Enter the context
    with hf._utils.set_up_tokenizer(tokenizer):
        assert tokenizer.padding_side == "right"
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token_id is not None
        assert not getattr(tokenizer, "add_eos_token", False)

    # Exit the context. No attributes should have changed
    for attribute, old_value in attribute_to_old_value.items():
        assert getattr(tokenizer, attribute, None) == old_value


@pytest.mark.parametrize("texts", (["a b c", "d e", "f g h i"], ["slow reverb"]))
@pytest.mark.parametrize("batch_size", (1, 2))
def test__batched_model_call(texts, model, tokenizer, batch_size, atol):
    with hf._utils.set_up_tokenizer(tokenizer):
        encodings: BatchEncodingPT = tokenizer(texts, return_tensors="pt", padding=True)
    with hf._utils.set_up_model(model):
        out_correct: CausalLMOutput = model(**encodings)
    out_batched: CausalLMOutput = hf._utils._batched_model_call(
        batch_size=batch_size, model=model, **encodings
    )
    assert torch.allclose(out_correct.logits, out_batched.logits, atol=atol)


@pytest.mark.parametrize("module", (classify, classify_no_cache))
@pytest.mark.parametrize(
    "texts",
    (
        "lone string input",
        ["a b", "c d e"],
        ["a fistful", "of tokens", "for a few", "tokens more"],
    ),
)
@pytest.mark.parametrize("batch_size", (1, 2))
def test_token_logprobs(
    module: classify_module,
    texts,
    model_and_tokenizer,
    batch_size,
    end_of_prompt=" ",
    add_bos=False,
):
    """
    Tests that the model's token log probabilities are correct by testing against an
    unbatched and carefully, manually indexed result.
    """
    log_probs_texts_observed = module.token_logprobs(
        texts, model_and_tokenizer, end_of_prompt=end_of_prompt, batch_size=batch_size
    )

    model, tokenizer = model_and_tokenizer
    # Gather un-batched un-sliced log probs for the expected result
    is_str = isinstance(texts, str)
    texts = [texts] if is_str else texts
    if not hf._utils.does_tokenizer_need_prepended_space(tokenizer):
        end_of_prompt = ""
    log_probs_texts_from_unbatched = []
    input_ids_from_unbatched = []
    with hf._utils.dont_add_bos_token(tokenizer) if not add_bos else nullcontext():
        for text in texts:
            _logits, _encoding = hf._utils.logits_texts(
                [end_of_prompt + text], (model, tokenizer)
            )
            # grab first index b/c we only gave it 1 text
            log_probs_texts_from_unbatched.append(_logits[0].log_softmax(dim=1))
            input_ids_from_unbatched.append(_encoding["input_ids"][0])

    log_probs_texts_observed = (
        [log_probs_texts_observed] if is_str else log_probs_texts_observed
    )
    _test_content.token_logprobs(
        log_probs_texts_observed,
        log_probs_texts_from_unbatched,
        input_ids_from_unbatched,
    )


def test_cache_nested(model_and_tokenizer, atol):
    delim = " "
    if not hf._utils.does_tokenizer_need_prepended_space(model_and_tokenizer[1]):
        # for SentencePiece tokenizers like Llama's
        delim = ""

    logits = lambda *args, **kwargs: hf._utils.logits_texts(*args, **kwargs)[0]
    """
    Returns next-token logits for each token in an inputted text.
    """

    with classify.cache(model_and_tokenizer, "a") as cached_a:
        with classify.cache(cached_a, delim + "b c") as cached_a_b_c:
            with classify.cache(cached_a_b_c, delim + "d") as cached_a_b_c_d:
                logits1 = logits([delim + "e f"], cached_a_b_c_d)
                logits2 = logits([delim + "x"], cached_a_b_c_d)
            logits3 = logits([delim + "1 2 3"], cached_a_b_c)
        logits4 = logits([delim + "b c d"], cached_a)

    logits_correct = lambda texts, **kwargs: logits(
        texts, model_and_tokenizer, drop_bos_token=False
    )

    assert torch.allclose(logits1, logits_correct(["a b c d e f"]), atol=atol)
    assert torch.allclose(logits2, logits_correct(["a b c d x"]), atol=atol)
    assert torch.allclose(logits3, logits_correct(["a b c 1 2 3"]), atol=atol)
    assert torch.allclose(logits4, logits_correct(["a b c d"]), atol=atol)

    # Test clear_cache_on_exit
    device = model_and_tokenizer[0].device
    with pytest.raises(
        classify._CacheClearedError, match="This model is no longer usable."
    ):
        cached_a[0](
            input_ids=torch.tensor([[1]], device=device),
            attention_mask=torch.tensor([[1]], device=device),
        )

    with classify.cache(
        model_and_tokenizer, "a", clear_cache_on_exit=False
    ) as cached_a:
        with classify.cache(
            cached_a, delim + "b c", clear_cache_on_exit=False
        ) as cached_a_b_c:
            _ = logits(["whatever"], cached_a_b_c)
    assert cached_a_b_c[0]._cappr.past is not None
    assert hasattr(cached_a[0]._cappr, "past")

    # Test repr
    assert repr(cached_a[0]) == repr(cached_a[0]._cappr.model)


########################################################################################
################################### Helpers for tests ##################################
########################################################################################


def _test_encodings(
    logits_slow: torch.Tensor,
    encodings_slow: BatchEncodingPT,
    logits_fast: torch.Tensor,
    encodings_fast: BatchEncodingPT,
):
    """
    Tests that all objects have the expected shape, and that the encodings `offsets` are
    identical.
    """
    if logits_slow.shape[0] > logits_fast.shape[0] and logits_fast.shape[1] == 1:
        # Single-token optimization: this test doesn't apply b/c the optimization
        # doesn't repeat any data, unlike what's done in the slow/no-cache module
        return

    def _test_shapes(logits: torch.Tensor, encodings: BatchEncodingPT):
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
    encodings_slow: BatchEncodingPT,
    logits_fast: torch.Tensor,
    encodings_fast: BatchEncodingPT,
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
    def module(self):
        return classify

    @property
    def module_correct(self):
        return classify_no_cache


class TestPromptsCompletions(Modules, _base.TestPromptsCompletions):
    def test__logits_completions_given_prompts(
        self, model, tokenizer, prompts, completions, atol
    ):
        # Test logits for better debuggability
        # For this helper function, prompts can't be a single string
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

    @pytest.mark.parametrize("batch_size", (1, 2))
    @pytest.mark.parametrize("batch_size_completions", (None, 1))
    def test_log_probs_conditional(
        self,
        prompts,
        completions,
        model_and_tokenizer,
        batch_size,
        batch_size_completions,
    ):
        super().test_log_probs_conditional(
            prompts,
            completions,
            model_and_tokenizer,
            batch_size=batch_size,
            batch_size_completions=batch_size_completions,
        )

    def test_predict_proba(
        self,
        prompts,
        completions,
        model_and_tokenizer,
        prior,
        discount_completions,
        normalize,
    ):
        super().test_predict_proba(
            prompts,
            completions,
            model_and_tokenizer,
            prior=prior,
            discount_completions=discount_completions,
            normalize=normalize,
        )

    def test_predict(self, prompts, completions, model_and_tokenizer):
        super().test_predict(prompts, completions, model_and_tokenizer)


class TestExamples(Modules, _base.TestExamples):
    def test__logits_completions_given_prompts_examples(
        self, model, tokenizer, examples, atol
    ):
        # Test logits for better debuggability
        # For this helper function, examples can't be an Example
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

    @pytest.mark.parametrize("batch_size", (1, 2))
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


@pytest.mark.parametrize("batch_size", (1, 2))
class TestCache(Modules, _base.TestCache):
    def test_cache(
        self,
        prompt_prefix: str,
        prompts: list[str],
        completions: list[str],
        model_and_tokenizer,
        batch_size,
    ):
        super().test_cache(
            prompt_prefix,
            prompts,
            completions,
            model_and_tokenizer,
            batch_size=batch_size,
        )

    def test_cache_model(
        self,
        prompt_prefix: str,
        prompts: list[str],
        completions: list[str],
        model_and_tokenizer,
        batch_size,
    ):
        super().test_cache_model(
            prompt_prefix,
            prompts,
            completions,
            model_and_tokenizer,
            batch_size=batch_size,
        )

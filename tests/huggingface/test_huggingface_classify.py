"""
Unit and integration tests for `cappr.huggingface.classify`. Works by checking that its
functions' outputs are numerically close to those from
`cappr.huggingface.classify_no_cache`, which is assumed to be correct (TODO: yeah I
really should test that).
"""
from __future__ import annotations
import os
import sys
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from cappr import Example as Ex
from cappr.huggingface import classify as fast
from cappr.huggingface import classify_no_cache as slow
from cappr import huggingface as hf
from cappr.huggingface._utils import PreTrainedModelForCausalLM

# sys hack to import from parent. If someone has a cleaner solution, lmk
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import _test


########################################################################################
###################################### Fixtures ########################################
########################################################################################


@pytest.fixture(
    scope="module",
    params=[
        "sshleifer/tiny-gpt2",
        "anton-l/gpt-j-tiny-random",  # this one is quite big and slow
        "Maykeye/TinyLLama-v0",
    ],
)
def model_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_name) -> PreTrainedModelForCausalLM:
    model: PreTrainedModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    # Set up the model as expected.
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer(model_name) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set up the tokenizer as expected.
    # These things are done in cappr.huggingface._utils.set_up_model_and_tokenizer.
    if tokenizer.pad_token_id is None:
        # allow padding -> allow batching
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    if hasattr(tokenizer, "add_eos_token"):
        tokenizer.add_eos_token = False
    return tokenizer


@pytest.fixture(scope="module")
def model_and_tokenizer(
    model_name,
) -> tuple[PreTrainedModelForCausalLM, PreTrainedTokenizerBase]:
    # This input is directly from a user, so we can't assume the model and tokenizer are
    # set up correctly. For testing, that means we shouldn't just do:
    # return model, tokenizer
    # Instead, load them from scratch:
    model: PreTrainedModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


@pytest.fixture(scope="module")
def atol():
    # Reading through some transformers tests, it looks like 1e-3 is considered
    # close-enough for hidden states. See, e.g.,
    # https://github.com/huggingface/transformers/blob/main/tests/models/gpt2/test_modeling_gpt2.py#L250
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
    model: PreTrainedModelForCausalLM
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
        pass

    # Exit the context. No attributes should have changed.
    assert model.training == model_attribute_to_old_value["training"]
    for i, module in enumerate(model.children()):
        assert module.training == model_attribute_to_old_value[i]

    for attribute, old_value in tokenizer_attribute_to_old_value.items():
        assert getattr(tokenizer, attribute, None) == old_value


@pytest.mark.parametrize(
    "texts", (["a", "fistful of", "tokens more", "the good, the bad, and the tokens."],)
)
@pytest.mark.parametrize("batch_size", (10, 3))
def test_token_logprobs(
    texts, model_and_tokenizer, batch_size, model, tokenizer, atol, end_of_prompt=" "
):
    """
    Tests that the model's token log probabilities are correct by testing against an
    unbatched and carefully/manually indexed result.
    """
    log_probs = fast.token_logprobs(texts, model_and_tokenizer, batch_size=batch_size)

    # bleh
    if (
        end_of_prompt == " "
        and not hf._utils.does_tokenizer_prepend_space_to_first_token(
            model_and_tokenizer[1]
        )
    ):
        end_of_prompt = ""
    texts = [end_of_prompt + text for text in texts]

    # Gather un-batched data to compare against as the expected result
    texts_log_probs = []
    texts_input_ids = []
    for text in texts:
        # the correct expected result requires that we use the model() and tokenizer()
        # fixtures b/c they're set up correctly
        logits, _encoding = hf._utils.logits_texts([text], model, tokenizer)
        # grab first index b/c we only gave it 1 text
        texts_log_probs.append(logits[0].log_softmax(dim=1))
        texts_input_ids.append(_encoding.input_ids[0])

    # The first logprob of every text must be None b/c no CausalLM estimates Pr(token)
    for log_prob in log_probs:
        assert log_prob[0] is None

    # The sizes are the same as the number of tokens
    assert len(log_probs) == len(texts)  # == len(texts_encodings) => zip is strict
    for log_prob, input_ids in zip(log_probs, texts_input_ids):
        assert len(log_prob) == len(input_ids)

    # Every log prob is correct
    for log_prob, input_ids, log_probs_expected in zip(
        log_probs, texts_input_ids, texts_log_probs
    ):
        for i in range(0, len(input_ids) - 1):
            log_prob_expected = log_probs_expected[i, input_ids[i + 1]]
            assert torch.isclose(
                torch.tensor(log_prob[i + 1]),
                log_prob_expected,
                atol=atol,
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
    _outputs_slow = slow._keys_values_prompts(
        model, tokenizer, prompts, num_completions_per_prompt
    )
    _outputs_fast = fast._keys_values_prompts(
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

    assert torch.equal(encodings_slow.input_ids, encodings_fast.input_ids)
    assert torch.equal(encodings_slow.attention_mask, encodings_fast.attention_mask)
    assert torch.equal(offsets_slow, offsets_fast)

    assert torch.allclose(logits_last_slow, logits_last_fast, atol=atol)


########################################################################################
############################### Helpers for future tests ###############################
########################################################################################


def _test_encodings(
    logits_slow: torch.Tensor,
    encodings_slow: Mapping[str, torch.Tensor],
    logits_fast: torch.Tensor,
    encodings_fast: Mapping[str, torch.Tensor],
):
    """
    Tests that all objects have the expected shape, and that the encodings `offsets` are
    identical.
    """
    if logits_slow.shape[0] > logits_fast.shape[0] and logits_fast.shape[1] == 1:
        # Single-token optimization: this test doesn't apply b/c the optimization
        # doesn't repeat any data, unlike what's done in the slow/no-cache module
        return

    def _test_shapes(logits, encodings):
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
    encodings_slow: Mapping[str, torch.Tensor],
    logits_fast: torch.Tensor,
    encodings_fast: Mapping[str, torch.Tensor],
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


def _test_log_probs(
    log_probs_completions_slow: Union[list[list[float]], list[list[list[float]]]],
    log_probs_completions_fast: Union[list[list[float]], list[list[list[float]]]],
    expected_len: int,
    num_completions_per_prompt: Optional[list[int]],
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
            assert torch.allclose(
                torch.tensor(log_probs_tokens_slow),
                torch.tensor(log_probs_tokens_fast),
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
@pytest.mark.parametrize("end_of_prompt", (" ",))  # TODO: expand
class TestPromptsCompletions:
    """
    Tests all model-dependent, non-`_examples` functions, sharing the same set of
    prompts and completions.
    """

    def test__logits_completions_given_prompts(
        self, model, tokenizer, prompts, completions, end_of_prompt, atol
    ):
        """
        Tests that encodings have the right shape and that logits are numerically close.
        If this test fails, all of the tests below will fail.
        """
        # for this function, prompts can't be a single string
        if isinstance(prompts, str):
            return
        kwargs = _function_kwargs(locals())
        slow_out = slow._logits_completions_given_prompts(**kwargs)
        fast_out = fast._logits_completions_given_prompts(**kwargs)
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out, atol)

    @pytest.mark.parametrize("batch_size", (2, 1))
    def test_log_probs_conditional(
        self, prompts, completions, model_and_tokenizer, end_of_prompt, batch_size, atol
    ):
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
        model_and_tokenizer,
        end_of_prompt,
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
        assert torch.allclose(
            torch.tensor(pred_probs_fast), torch.tensor(pred_probs_slow), atol=atol
        )

    @pytest.mark.parametrize("discount_completions", (0.0, 1.0))
    def test_predict(
        self,
        prompts,
        completions,
        model_and_tokenizer,
        end_of_prompt,
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

    def test__logits_completions_given_prompts_examples(
        self, model, tokenizer, examples, atol
    ):
        """
        Tests that encodings have the right shape and that logits are numerically close.
        If this test fails, all of the tests below will fail.
        """
        # for this helper function, examples can't be an Example
        if isinstance(examples, Ex):
            return
        kwargs = _function_kwargs(locals())
        slow_out = slow._logits_completions_given_prompts_examples(**kwargs)
        fast_out = fast._logits_completions_given_prompts_examples(**kwargs)
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out, atol)

    @pytest.mark.parametrize("batch_size", (2, 1))
    def test_log_probs_conditional_examples(
        self, examples: list[Ex], model_and_tokenizer, batch_size, atol
    ):
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

    def test_predict_proba_examples(self, examples, model_and_tokenizer, atol):
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
            assert torch.allclose(
                torch.tensor(pred_probs_fast_ex),
                torch.tensor(pred_probs_slow_ex),
                atol=atol,
            )

    def test_predict_examples(self, examples, model_and_tokenizer):
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

"""
Unit tests `cappr.huggingface.classify` by checking that its functions' outputs are
numerically close to those from `cappr.huggingface.classify_no_cache`, which is assumed
to be correct (TODO: yeah I really should test that).
"""
from __future__ import annotations
import os
import sys
from typing import Mapping

import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cappr import Example as Ex
from cappr.huggingface import classify as fast
from cappr.huggingface import classify_no_cache as slow
from cappr import huggingface as hf

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
        "anton-l/gpt-j-tiny-random",
        "Maykeye/TinyLLama-v0",
    ],
)
def model_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name)


@pytest.fixture(scope="module")
def tokenizer(model_name):
    # Set up the tokenizer as expected.
    # These things are done in cappr.huggingface._utils.set_up_model_and_tokenizer,
    # which is always applied to the user-inputted tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        # allow padding -> allow batching
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False
    return tokenizer


@pytest.fixture(scope="module")
def model_and_tokenizer(model, tokenizer):
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


@pytest.mark.parametrize(
    "texts", (["a", "fistful of", "tokens more", "the good, the bad, and the tokens."],)
)
@pytest.mark.parametrize("batch_size", (10, 3))
def test_token_logprobs(
    texts, model_and_tokenizer, batch_size, atol, end_of_prompt=" "
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
        logits, _encoding = hf._utils.logits_texts([text], *model_and_tokenizer)
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
#################################### Test helpers ######################################
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
    log_probs_completions_slow: list[list[list[float]]],
    log_probs_completions_fast: list[list[list[float]]],
    expected_len: int,
    num_completions_per_prompt: list[int],
    atol,
):
    """
    Tests that the conditional token log-probabilities are the right shape and are
    numerically close.
    """
    # Test lengths before zipping. Note transitivity
    assert len(log_probs_completions_slow) == len(log_probs_completions_fast)
    assert len(log_probs_completions_fast) == expected_len
    assert len(log_probs_completions_slow) == len(num_completions_per_prompt)
    zipped_outer = zip(
        log_probs_completions_slow,
        log_probs_completions_fast,
        num_completions_per_prompt,
    )
    for log_probs_slow, log_probs_fast, num_completions in zipped_outer:
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


########################################################################################
####################################### Tests ##########################################
########################################################################################


@pytest.mark.parametrize("prompts", (["a b c", "c"],))
@pytest.mark.parametrize(
    "completions",
    (
        ["d", "e f g h i"],
        ######################## Next set of completions to test #######################
        ["d e f", "1 2"],
        ######################## Test Single-token optimization ########################
        ["d", "e", "f"],
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
        slow_out = slow._logits_completions_given_prompts(
            model, tokenizer, prompts, completions, end_of_prompt=end_of_prompt
        )
        fast_out = fast._logits_completions_given_prompts(
            model, tokenizer, prompts, completions, end_of_prompt=end_of_prompt
        )
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out, atol)

    @pytest.mark.parametrize("batch_size", (2, 1))
    def test_log_probs_conditional(
        self, prompts, completions, model_and_tokenizer, end_of_prompt, batch_size, atol
    ):
        log_probs_completions_slow = slow.log_probs_conditional(
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            batch_size=batch_size,
        )
        log_probs_completions_fast = fast.log_probs_conditional(
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            batch_size=batch_size,
        )
        expected_len = len(prompts)
        num_completions_per_prompt = [len(completions)] * len(prompts)
        _test_log_probs(
            log_probs_completions_slow,
            log_probs_completions_fast,
            expected_len,
            num_completions_per_prompt,
            atol,
        )

    @pytest.mark.parametrize("discount_completions", (0.0, 1.0))
    def test_predict_proba(
        self,
        prompts,
        completions,
        model_and_tokenizer,
        end_of_prompt,
        discount_completions,
        atol,
    ):
        # Test form of output
        _test.predict_proba(
            fast.predict_proba,
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )
        _test.predict_proba(
            slow.predict_proba,
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )

        # Test that predictions match
        pred_probs_fast = fast.predict_proba(
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )
        pred_probs_slow = slow.predict_proba(
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )
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
        # Test form of output
        _test.predict(
            fast.predict,
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )
        _test.predict(
            slow.predict,
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )

        # Test that predictions match
        preds_fast = fast.predict(
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )
        preds_slow = slow.predict(
            prompts,
            completions,
            model_and_tokenizer=model_and_tokenizer,
            end_of_prompt=end_of_prompt,
            discount_completions=discount_completions,
        )
        assert preds_fast == preds_slow


@pytest.mark.parametrize(
    "examples",
    (
        [
            Ex("a b c", ["d", "e f g"]),
            Ex("C", ["p G C p G", "D E F", "ya later alligator"]),
        ],
        ########## Next set of examples ##########
        [
            Ex("chi", ["can", "ery"]),
            Ex("koyaa", ["nisqatsi"]),
            Ex("hi hi", ["bye bye", "yo yo"]),
        ],
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
        slow_out = slow._logits_completions_given_prompts_examples(
            model, tokenizer, examples
        )
        fast_out = fast._logits_completions_given_prompts_examples(
            model, tokenizer, examples
        )
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out, atol)

    @pytest.mark.parametrize("batch_size", (2, 1))
    def test_log_probs_conditional_examples(
        self, examples: list[Ex], model_and_tokenizer, batch_size, atol
    ):
        log_probs_completions_slow = slow.log_probs_conditional_examples(
            examples, model_and_tokenizer=model_and_tokenizer, batch_size=batch_size
        )
        log_probs_completions_fast = fast.log_probs_conditional_examples(
            examples, model_and_tokenizer=model_and_tokenizer, batch_size=batch_size
        )
        expected_len = len(examples)
        num_completions_per_prompt = [len(example.completions) for example in examples]
        _test_log_probs(
            log_probs_completions_slow,
            log_probs_completions_fast,
            expected_len,
            num_completions_per_prompt,
            atol,
        )

    def test_predict_proba_examples(self, examples, model_and_tokenizer, atol):
        # Test form of output
        _test.predict_proba_examples(
            fast.predict_proba_examples,
            examples,
            model_and_tokenizer=model_and_tokenizer,
        )
        _test.predict_proba_examples(
            slow.predict_proba_examples,
            examples,
            model_and_tokenizer=model_and_tokenizer,
        )

        # Test that predictions match
        pred_probs_fast = fast.predict_proba_examples(
            examples, model_and_tokenizer=model_and_tokenizer
        )
        pred_probs_slow = slow.predict_proba_examples(
            examples, model_and_tokenizer=model_and_tokenizer
        )
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
        # Test form of output
        _test.predict_examples(
            fast.predict_examples, examples, model_and_tokenizer=model_and_tokenizer
        )
        _test.predict_examples(
            slow.predict_examples, examples, model_and_tokenizer=model_and_tokenizer
        )

        # Test that predictions match
        preds_fast = fast.predict_examples(
            examples, model_and_tokenizer=model_and_tokenizer
        )
        preds_slow = slow.predict_examples(
            examples, model_and_tokenizer=model_and_tokenizer
        )
        assert preds_fast == preds_slow

'''
Unit tests functions in the `fast` module by comparing their outputs to
identically named functions in the `slow` module.

TODO: there isn't much rhyme or reason to the testing parameterizations.
'''
from __future__ import annotations
from typing import Mapping

import pytest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_classification.classify import Example as Ex

import fast
import slow


@pytest.fixture(scope='module')
def model_name():
    return 'sshleifer/tiny-gpt2' ## for testing this'll do fine


@pytest.fixture(scope='module')
def model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


@pytest.fixture(scope='module')
def tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        ## allow padding -> allow batching
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pytest.mark.parametrize('prompts', (['a b c', 'c'],))
@pytest.mark.parametrize('num_completions_per_prompt', (2, (2,3)))
def test__keys_values_prompts(model, tokenizer, prompts,
                              num_completions_per_prompt):
    _outputs_slow = slow._keys_values_prompts(model, tokenizer, prompts,
                                              num_completions_per_prompt)
    _outputs_fast = fast._keys_values_prompts(model, tokenizer, prompts,
                                              num_completions_per_prompt)
    keys_vals_slow, encodings_slow, offsets_slow, logits_last_slow = \
        _outputs_slow
    keys_vals_fast, encodings_fast, offsets_fast, logits_last_fast = \
        _outputs_fast

    assert len(keys_vals_slow) == len(keys_vals_fast) ## same # attention blocks
    for (keys_slow, vals_slow), (keys_fast, vals_fast) in zip(keys_vals_slow,
                                                              keys_vals_fast):
        assert torch.allclose(keys_slow, keys_fast)
        assert torch.allclose(vals_slow, vals_fast)

    assert torch.equal(encodings_slow.input_ids, encodings_fast.input_ids)
    assert torch.equal(encodings_slow.attention_mask,
                       encodings_fast.attention_mask)
    assert torch.equal(offsets_slow, offsets_fast)

    assert torch.allclose(logits_last_slow, logits_last_fast)


def _test_encodings(logits_slow: torch.Tensor,
                    encodings_slow: Mapping[str,torch.Tensor],
                    logits_fast: torch.Tensor,
                    encodings_fast: Mapping[str,torch.Tensor]):
    ## Test shapes
    def _test_shapes(logits, encodings):
        assert encodings['input_ids'].shape[:2] == logits.shape[:2]
        assert (encodings['input_ids'].shape[:2] ==
                encodings['attention_mask'].shape[:2])
        assert (encodings['input_ids'].shape[0] ==
                encodings['offsets'].shape[0])
    _test_shapes(logits_slow, encodings_slow)
    _test_shapes(logits_fast, encodings_fast)

    ## Test offsets. These should be exactly the same b/c they're the number of
    ## of non-pad tokens in each prompt
    assert torch.equal(encodings_slow['offsets'], encodings_fast['offsets'])


def _test_logits(logits_slow: torch.Tensor,
                 encodings_slow: Mapping[str,torch.Tensor],
                 logits_fast: torch.Tensor,
                 encodings_fast: Mapping[str,torch.Tensor]):
    ## Test shapes
    assert logits_slow.shape[0] == logits_fast.shape[0] ## batch size
    ## Middle dimension for the # of tokens is different (by design) b/c
    ## logits_slow includes prompt and completion tokens, while logits_fast only
    ## includes completion tokens.
    assert logits_slow.shape[2] == logits_fast.shape[2] ## vocab size

    ## Test logits at every *non-pad* token (automatic version)
    completion_token_idxs = [list(range(num_completion_tokens))
                             for num_completion_tokens
                             in encodings_fast['attention_mask'].sum(dim=1)]
    for text_idx in range(logits_slow.shape[0]):
        offset = encodings_fast['offsets'][text_idx].item() - 1
        ## number of non-pad prompt tokens - 1 (!) b/c in the fast version we
        ## included the last non-pad prompt token
        for completion_token_idx in completion_token_idxs[text_idx]:
            assert torch.allclose(logits_fast[text_idx,
                                              completion_token_idx],
                                  logits_slow[text_idx,
                                              offset + completion_token_idx])


def _test_log_probs(log_probs_completions_slow, log_probs_completions_fast,
                    expected_len, num_completions_per_prompt):
    assert len(log_probs_completions_slow) == len(log_probs_completions_fast)
    assert len(log_probs_completions_fast) == expected_len
    zipped_outer = zip(log_probs_completions_slow,
                       log_probs_completions_fast,
                       num_completions_per_prompt)
    for log_probs_slow, log_probs_fast, num_completions in zipped_outer:
        assert len(log_probs_fast) == num_completions
        assert len(log_probs_slow) == len(log_probs_fast)
        zipped_inner = zip(log_probs_slow, log_probs_fast)
        for log_probs_tokens_slow, log_probs_tokens_fast in zipped_inner:
            assert torch.allclose(torch.tensor(log_probs_tokens_slow),
                                  torch.tensor(log_probs_tokens_fast))


@pytest.mark.parametrize('prompts', (['a b c', 'c'],))
@pytest.mark.parametrize('completions', (['d', 'e f g h i'],
                                         ####### Next set of completions #######
                                         ['d', 'd e f']))
@pytest.mark.parametrize('end_of_prompt', (' ',)) ## TODO: We need to expand
@pytest.mark.parametrize('batch_size', (2, 1))
class TestPromptsCompletions:
    def test__logits_completions_given_prompts(self, model, tokenizer, prompts,
                                               completions, end_of_prompt,
                                               batch_size):
        slow_out = (slow._logits_completions_given_prompts(
                        model, tokenizer, prompts, completions,
                        end_of_prompt=end_of_prompt, batch_size=batch_size))
        fast_out = (fast._logits_completions_given_prompts(
                        model, tokenizer, prompts, completions,
                        end_of_prompt=end_of_prompt, batch_size=batch_size))
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out)

    def test_log_probs_conditional(self, prompts, completions, model_name,
                                   end_of_prompt, batch_size):
        log_probs_completions_slow = slow.log_probs_conditional(
                                        prompts, completions, model_name,
                                        end_of_prompt=end_of_prompt,
                                        batch_size=batch_size)
        log_probs_completions_fast = fast.log_probs_conditional(
                                        prompts, completions, model_name,
                                        end_of_prompt=end_of_prompt,
                                        batch_size=batch_size)
        expected_len = len(prompts)
        num_completions_per_prompt = [len(completions)] * len(prompts)
        _test_log_probs(log_probs_completions_slow, log_probs_completions_fast,
                        expected_len=expected_len,
                        num_completions_per_prompt=num_completions_per_prompt)


@pytest.mark.parametrize('examples', ([Ex('a b c', ['d', 'e f g']),
                                       Ex('c',     ['d', 'd e f', 'ya later'])],
                                      ########## Next set of examples ##########
                                      [Ex('chi',   ['can', 'ery']),
                                       Ex('koyaa', ['nisqatsi']),
                                       Ex('hi hi', ['bye bye', 'yo yo'])]))
@pytest.mark.parametrize('batch_size', (2, 1))
class TestExamples:
    def test__logits_completions_given_prompts_examples(self, model, tokenizer,
                                                        examples, batch_size):
        slow_out = slow._logits_completions_given_prompts_examples(
                        model, tokenizer, examples, batch_size=batch_size)
        fast_out = fast._logits_completions_given_prompts_examples(
                        model, tokenizer, examples, batch_size=batch_size)
        _test_encodings(*slow_out, *fast_out)
        _test_logits(*slow_out, *fast_out)

    def test_log_probs_conditional_examples(self, examples: list[Ex],
                                            model_name, batch_size):
        log_probs_completions_slow = slow.log_probs_conditional_examples(
                                        examples, model_name,
                                        batch_size=batch_size)
        log_probs_completions_fast = fast.log_probs_conditional_examples(
                                        examples, model_name,
                                        batch_size=batch_size)
        expected_len = len(examples)
        num_completions_per_prompt = [len(example.completions)
                                      for example in examples]
        _test_log_probs(log_probs_completions_slow, log_probs_completions_fast,
                        expected_len=expected_len,
                        num_completions_per_prompt=num_completions_per_prompt)

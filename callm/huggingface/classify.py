"""
Perform prompt-completion classification using a `transformers.AutoModelForCausalLM`.

This module is a fast mirror of `classify_slow`. It precomputes each attention block's
keys and values for prompts, enabling greater parallelization over completions.
"""
from __future__ import annotations
from typing import Mapping, Sequence, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from callm.utils import batch, classify, wrap
from callm.example import Example
from callm import huggingface as hf


@wrap.add_doc_before(hf.docstrings.KEYS_VALUES_PROMPTS)
def _keys_values_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
):
    if not tokenizer.padding_side == "right":
        raise ValueError(
            "Gotta use right padding to ensure position IDs are " "correct."
        )
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(num_completions_per_prompt, Sequence):
        if not len(prompts) == len(num_completions_per_prompt):
            raise ValueError(
                "If num_completions_per_prompt is a Sequence, then it must be the same "
                f"length as prompts. Got lengths {len(num_completions_per_prompt)}, "
                f"{len(prompts)}."
            )
    if not isinstance(num_completions_per_prompt, int) and not isinstance(
        num_completions_per_prompt, torch.Tensor
    ):
        num_completions_per_prompt = torch.tensor(num_completions_per_prompt)

    ## Batch inference prompts
    prompts = list(prompts)  ## 0-index in case it's a Series or something
    # fmt: off
    encodings = (tokenizer(prompts, return_tensors="pt", padding=True)
                 .to(hf.utils.DEVICE))
    # fmt: on
    with torch.no_grad():
        out = model(**encodings)

    ## We need to repeat each prompt's keys and values num_completions_per_prompt times
    ## For layer i, prompts_out.past_key_values[i] is a tuple (key, value),
    ## each w/ shape: (batch size=len(prompts),
    ##                 number of attention heads=12 for gpt2,
    ##                 encodings.input_ids.shape[-1],
    ##                 key/value hidden dimension=64 for gpt2)
    past_key_values = (
        torch.stack([torch.stack(block) for block in out.past_key_values], dim=0)
        ## The tuple is now a tensor w/ shape:
        ## (# layers=12 for gpt2,
        ##  2 (for key and value),
        ##  and then the rest as before)
        ## Repeat along batch size dim so that it aligns
        ## downstream w/ completions
        .repeat_interleave(num_completions_per_prompt, dim=2)
    )
    ## Re-format this tensor to the nested tuple format we'd get if we passed multiple
    ## copies of the prompt at the same time to the model
    past_key_values = tuple(
        [(layer[0], layer[1]) for layer in past_key_values]  ## keys, values
    )

    ## Repeat stuff
    # fmt: off
    encodings["attention_mask"] = (encodings
                                   .attention_mask
                                   .repeat_interleave(num_completions_per_prompt,
                                                      dim=0))
    encodings["input_ids"] = (encodings
                              .input_ids
                              .repeat_interleave(num_completions_per_prompt, dim=0))
    # fmt: on

    ## Need offsets so that position_ids for future tokens are set correctly
    offsets = encodings.attention_mask.sum(dim=1)

    ## Need (next-token) logits from prompts, i.e., last non-pad prompt token, since
    ## that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    # fmt: off
    last_nonpad_token_logits = (out
                                .logits
                                .repeat_interleave(num_completions_per_prompt, dim=0)
                                .take_along_dim(_last_nonpad_token_idxs, dim=1))
    # fmt: on

    return past_key_values, encodings, offsets, last_nonpad_token_logits


def _blessed_helper(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
    completions_repeats: int,
) -> tuple[torch.Tensor, BatchEncoding]:
    """
    TODO: docstring
    """
    if not tokenizer.padding_side == "right":
        raise ValueError("Gotta use right padding to ensure position IDs are correct.")
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError("completions must be a Sequence of strings.")

    ## Prepare prompt data
    (
        past_key_values,
        prompts_encodings,
        offsets,
        prompts_last_nonpad_token_logits,
    ) = _keys_values_prompts(model, tokenizer, prompts, num_completions_per_prompt)

    ## Prepare completion data
    completions = list(completions)  ## 0-index in case it's a Series or somethin
    # fmt: off
    completions_encoding = (tokenizer(completions, return_tensors="pt", padding=True)
                            .to(hf.utils.DEVICE))
    completions_input_ids = (completions_encoding
                             .input_ids
                             .repeat(completions_repeats, 1))
    completions_attention_mask = (completions_encoding
                                  .attention_mask
                                  .repeat(completions_repeats, 1))
    # fmt: on
    ## Set position_ids to what they were had we fed the prompt + completion together w/
    ## right-padding (right b/c GPT-2 uses absolute position ids)
    _num_completion_tokens = completions_encoding.input_ids.shape[1]
    completions_position_ids = torch.arange(_num_completion_tokens) + offsets[:, None]
    ## Need attention_mask to include the prompt since it prolly has padding
    attention_mask = torch.cat(
        (prompts_encodings["attention_mask"], completions_attention_mask), dim=1
    )

    ## Everything should now be aligned 🤞 🙏
    with torch.no_grad():
        completions_out = model(
            input_ids=completions_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=completions_position_ids,
        )

    ## You need to be able to ignore pad tokens, so need this data as well
    encodings = BatchEncoding(
        {
            "input_ids": completions_input_ids,
            "attention_mask": completions_attention_mask,
            "offsets": offsets,
        }
    )

    ## Let's drop the next-token logits for the last completion token b/c they're not
    ## useful for our purposes. Moreover, dropping ensures
    ## logits.shape[:2] == encodings['input_ids'].shape, as one expects.
    ## The user just needs to keep in mind that `logits` are shifted behind.
    logits = torch.cat(
        [prompts_last_nonpad_token_logits, completions_out.logits[:, :-1, :]], dim=1
    )

    return logits, encodings


@wrap.add_doc_before(
    hf.docstrings.LOGITS_COMPLETIONS_GIVEN_PROMPTS_OUTPUT.format(text="completion")
)
@wrap.add_doc_after(hf.docstrings.TEXTS_FROM_PROMPTS_COMPLETIONS)
@wrap.add_doc_after(hf.docstrings.BATCH_SIZE)
@hf.utils.cat_logits_encodings
@batch.batchify(
    batchable_arg="prompts", push_up_arg="tokenizer", progress_bar_desc="logits (fast)"
)
def _logits_completions_given_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: str = " ",
    batch_size: int = 32,
):
    if end_of_prompt != " ":
        raise ValueError("end_of_prompt must be ' ' for now. Sorry!")
    completions = [end_of_prompt + completion.lstrip() for completion in completions]
    ## TODO: figure out how to do this generally, not just for ' ' end_of_prompt
    return _blessed_helper(
        model,
        tokenizer,
        prompts,
        completions,
        num_completions_per_prompt=len(completions),
        completions_repeats=len(prompts),
    )


@wrap.add_doc_before(
    hf.docstrings.LOGITS_COMPLETIONS_GIVEN_PROMPTS_OUTPUT.format(text="completion")
)
@wrap.add_doc_after(hf.docstrings.TEXTS_FROM_EXAMPLES)
@wrap.add_doc_after(hf.docstrings.BATCH_SIZE)
@hf.utils.cat_logits_encodings
@batch.batchify(
    batchable_arg="examples", push_up_arg="tokenizer", progress_bar_desc="logits (fast)"
)
def _logits_completions_given_prompts_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: Sequence[Example],
    batch_size: int = 32,
):
    if any([example.end_of_prompt != " " for example in examples]):
        raise ValueError("Every example's end_of_prompt must be ' ' for now. Sorry!")
    prompts = [example.prompt for example in examples]
    completions = [
        example.end_of_prompt + completion.lstrip()
        for example in examples
        for completion in example.completions
    ]
    ## TODO: figure out how to do this generally, not just for ' ' end_of_prompt
    num_completions_per_prompt = [len(example.completions) for example in examples]
    completions_repeats = 1
    return _blessed_helper(
        model,
        tokenizer,
        prompts,
        completions,
        num_completions_per_prompt=num_completions_per_prompt,
        completions_repeats=completions_repeats,
    )


@wrap.add_doc_before(hf.docstrings.LOGITS_TO_LOG_PROBS_COMPLETIONS)
def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor]
) -> list[list[float]]:
    log_probs = hf.utils.logits_to_log_probs(
        logits, encodings["input_ids"], input_ids_start_idx=None, logits_end_idx=None
    )
    last_idx_non_pad = encodings["attention_mask"].sum(dim=1)
    ## i.e., # of tokens per completion
    return [
        log_probs_prompt_completion[:completion_end].tolist()
        for log_probs_prompt_completion, completion_end in zip(
            log_probs, last_idx_non_pad
        )
    ]


@wrap.add_doc_before(classify.docstrings.LOG_PROBS_CONDITIONAL)
@wrap.add_doc_after(hf.docstrings.BATCH_SIZE)
def log_probs_conditional(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: str,
    end_of_prompt: str = " ",
    batch_size: int = 32,
):
    model, tokenizer = hf.utils.load_model_and_tokenizer(model)
    logits, encodings = _logits_completions_given_prompts(
        model,
        tokenizer,
        prompts,
        completions,
        end_of_prompt=end_of_prompt,
        batch_size=batch_size,
    )
    log_probs_completions = _logits_to_log_probs_completions(logits, encodings)
    return list(batch.constant(log_probs_completions, size=len(completions)))


@wrap.add_doc_before(classify.docstrings.LOG_PROBS_CONDITIONAL_EXAMPLES)
@wrap.add_doc_after(hf.docstrings.BATCH_SIZE)
def log_probs_conditional_examples(
    examples: Sequence[Example], model: str, batch_size: int = 32
):
    model, tokenizer = hf.utils.load_model_and_tokenizer(model)
    logits, encodings = _logits_completions_given_prompts_examples(
        model, tokenizer, examples, batch_size=batch_size
    )
    log_probs_completions = _logits_to_log_probs_completions(logits, encodings)
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(batch.variable(log_probs_completions, sizes=num_completions_per_prompt))


@wrap.add_doc_after(hf.docstrings.BATCH_SIZE)
@classify.predict_proba
def predict_proba(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: str,
    end_of_prompt: str = " ",
    batch_size: int = 32,
):
    return log_probs_conditional(
        prompts, completions, model, end_of_prompt=end_of_prompt, batch_size=batch_size
    )


@wrap.add_doc_after(hf.docstrings.BATCH_SIZE)
@classify.predict_proba_examples
def predict_proba_examples(
    examples: Sequence[Example], model: str, batch_size: int = 32
):
    return log_probs_conditional_examples(examples, model, batch_size=batch_size)

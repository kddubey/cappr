"""
Perform prompt-completion classification using a ``transformers.AutoModelForCausalLM``.
Currently, only PyTorch models are supported.

In the implementation, attention block keys and values for prompts are cached and shared
across completions. See the computational performance `here`_.

.. _here: https://cappr.readthedocs.io/en/latest/6_computational_performance.html

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)

.. warning:: This module doesn't work with ``transformers.BartForCausalLM``, among
             others probably.
"""
from __future__ import annotations
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, BatchEncoding

from cappr.utils import _batch, classify
from cappr import Example
from cappr import huggingface as hf


def _keys_values_prompts(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    num_repeats_per_prompt: Union[int, Sequence[int]],
) -> tuple[
    tuple[torch.Tensor, torch.Tensor], BatchEncoding, torch.Tensor, torch.Tensor
]:
    """
    Efficiently performs this procedure:

    1. Repeat-interleave `prompts[i]` `num_repeats_per_prompt[i]` times.

       Or, if `num_repeats_per_prompt` is an integer, repeat-interleave `prompts[i]`
       `num_repeats_per_prompt` times.

       For example, if there are 2 prompts and `num_repeats_per_prompt=(2,3)`, the
       repeated prompts look like::

           [prompts[0],
            prompts[0],
            prompts[1],
            prompts[1],
            prompts[1]]

    2. Apply `tokenizer` to the repeated prompts.

    3. Apply `model`.

    Parameters
    ----------
    model : AutoModelForCausalLM
        an autoregressive transformer language model
    tokenizer : PreTrainedTokenizer
        the tokenizer corresponding to `model`
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    num_repeats_per_prompt : Union[int, Sequence[int]]
        the numer of times to repeat each prompt in `prompts`

    Returns
    -------
    past_key_values : tuple[torch.Tensor, torch.Tensor]
        for each attention block in `model`, the keys and values for each prompt in the
        repeated prompts
    encodings : BatchEncoding
        the tokenizer output for the repeated prompts
    offsets : torch.Tensor
        the number of (non-pad) tokens in each of the repeated prompts
    last_nonpad_token_logits : torch.Tensor
        next-token logits for the last non-pad token for each of the repeated prompts

    Raises
    ------
    ValueError
        if the `tokenizer` is not using right-padding
    TypeError
        if `prompts` is not a `Sequence`
    ValueError
        if `num_repeats_per_prompt` is a `Sequence` whose length is not the same as the
        length of `prompts`
    """
    if not tokenizer.padding_side == "right":
        raise ValueError(
            "Gotta use right padding to ensure position IDs are correct. "
            "Run tokenizer.padding_side = 'right' if sensible."
        )
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(num_repeats_per_prompt, Sequence):
        if not len(prompts) == len(num_repeats_per_prompt):
            raise ValueError(
                "If num_repeats_per_prompt is a Sequence, then it must be the same "
                f"length as prompts. Got lengths {len(num_repeats_per_prompt)}, "
                f"{len(prompts)}."
            )
    if not isinstance(num_repeats_per_prompt, int) and not isinstance(
        num_repeats_per_prompt, torch.Tensor
    ):
        num_repeats_per_prompt = torch.tensor(
            num_repeats_per_prompt, device=model.device
        )

    ## Batch inference prompts
    prompts = list(prompts)  ## 0-index in case it's a Series or something
    # fmt: off
    encodings: BatchEncoding = (tokenizer(prompts, return_tensors="pt", padding=True)
                                .to(model.device))
    # fmt: on
    with torch.no_grad():
        out = model(**encodings)

    ## We need to repeat each prompt's keys and values num_repeats_per_prompt times
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
        ## Repeat along batch size dim so that it aligns downstream w/ completions
        .repeat_interleave(num_repeats_per_prompt, dim=2)
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
                                   .repeat_interleave(num_repeats_per_prompt,
                                                      dim=0))
    encodings["input_ids"] = (encodings
                              .input_ids
                              .repeat_interleave(num_repeats_per_prompt, dim=0))
    # fmt: on

    ## Need offsets so that position_ids for future tokens are set correctly
    offsets: torch.Tensor = encodings.attention_mask.sum(dim=1)

    ## Need (next-token) logits from prompts, i.e., last non-pad prompt token, since
    ## that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    # fmt: off
    last_nonpad_token_logits: torch.Tensor = (out
                                              .logits
                                              .repeat_interleave(
                                                  num_repeats_per_prompt, dim=0)
                                              .take_along_dim(
                                                  _last_nonpad_token_idxs, dim=1))
    # fmt: on

    return past_key_values, encodings, offsets, last_nonpad_token_logits


def _blessed_helper(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
    completions_repeats: int,
) -> tuple[torch.Tensor, BatchEncoding]:
    """
    TODO: docstring
    """
    if not tokenizer.padding_side == "right":
        raise ValueError(
            "Gotta use right padding to ensure position IDs are correct. "
            "Run tokenizer.padding_side = 'right' if sensible."
        )
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
                            .to(model.device))
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
    completions_position_ids = (
        torch.arange(_num_completion_tokens, device=model.device) + offsets[:, None]
    )
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
    ## 😎

    ## You need to be able to ignore pad tokens, so you need this data as well
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
    ## Just keep in mind that `logits` are shifted behind.
    logits = torch.cat(
        [prompts_last_nonpad_token_logits, completions_out.logits[:, :-1, :]], dim=1
    )

    return logits, encodings


def _logits_completions_given_prompts(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: str = " ",
):
    """
    TODO: convert docstring to numpy style

    If `texts` is

    ```python
    [prompt + end_of_prompt + completions
     for prompt in prompts
     for completion in completions]
    ```

    then this function returns

    1. `logits`: tensor with shape

        (`len(texts)`, max # tokens `completions`, `tokenizer.vocab_size`)

    where `logits[i,j]` are the `model`'s logits for token `j+1` of the completion in
    `texts[i]` given the prompt in `texts[i]`. This tensor includes logits for
    right-padded tokens. Use the `encodings.attention_mask` to ignore them before
    further processing.

    2. `encodings`: `BatchEncoding` containing the input IDs, attention mask,
    and position offsets.
    """
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


def _logits_completions_given_prompts_examples(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    examples: Sequence[Example],
):
    """
    TODO: convert docstring to numpy style

    If `texts` is

    ```python
    [example.prompt + example.end_of_prompt + completion
     for example in examples
     for completion in example.completions]
    ```
    then this function returns

    1. `logits`: tensor with shape

        (`len(texts)`, max # tokens `completions`, `tokenizer.vocab_size`)

    where `logits[i,j]` are the `model`'s logits for token `j+1` of the completion in
    `texts[i]` given the prompt in `texts[i]`. This tensor includes logits for
    right-padded tokens. Use the `encodings.attention_mask` to ignore them before
    further processing.

    2. `encodings`: `BatchEncoding` containing the input IDs, attention mask,
    and position offsets.
    """
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


def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor]
) -> list[list[float]]:
    """
    TODO: convert docstring to numpy style

    Returns a list `log_probs_completions` where `log_probs_completions[i][j]` is the
    log-probablity of *completion* token

        `encodings['input_ids'][i,j]`

    given its previous tokens

        `encodings['input_ids'][i,:j]`

    Pad tokens, i.e., tokens where `encodings['attention_mask'] == 0` are excluded.

    `logits[i,j]` is assumed to be an unnormalized distribution (over tokens in
    the vocab) given tokens `input_ids[i,:j]`.
    """
    log_probs = hf._utils.logits_to_log_probs(
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


def log_probs_conditional(
    prompts: Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer],
    end_of_prompt: str = " ",
    batch_size: int = 32,
) -> list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt and previous
    completion tokens.

    Parameters
    ----------
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model_and_tokenizer : tuple[AutoModelForCausalLM, PreTrainedTokenizer]
        an instantiated model and its corresponding tokenizer
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    batch_size : int, optional
        the maximum number of inputs that the model will process in parallel, by default
        32

    Returns
    -------
    log_probs_completions : list[list[list[float]]]
        `log_probs_completions[prompt_idx][completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompts[prompt_idx] + end_of_prompt` and previous
        completion tokens.

    Note
    ----
    To efficiently aggregate `log_probs_completions`, use
    :func:`cappr.utils.classify.agg_log_probs`.

    Example
    -------
    Here we'll use single characters (which are of course single tokens) to more clearly
    demonstrate what this function does::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import log_probs_conditional

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Create data
        prompts = ['x y', 'a b c']
        completions = ['z', 'd e']

        # Compute
        log_probs_completions = log_probs_conditional(
                                    prompts,
                                    completions,
                                    model_and_tokenizer=(model, tokenizer)
                                )

        # Outputs (rounded) next to their symbolic representation

        log_probs_completions[0]
        # [[-4.5],        [[log Pr(z | x, y)],
        #  [-5.6, -3.2]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        log_probs_completions[1]
        # [[-9.7],        [[log Pr(z | a, b, c)],
        #  [-0.2, -0.03]]  [log Pr(d | a, b, c), log Pr(e | a, b, c, d)]]
    """
    model, tokenizer = hf._utils.load_model_and_tokenizer(model_and_tokenizer)

    @_batch.flatten
    @_batch.batchify(batchable_arg="prompts", progress_bar_desc="log-probs")
    def log_probs_completions_batch(prompts, batch_size=batch_size):
        logits, encodings = _logits_completions_given_prompts(
            model, tokenizer, prompts, completions, end_of_prompt=end_of_prompt
        )
        return _logits_to_log_probs_completions(logits, encodings)

    log_probs_completions = log_probs_completions_batch(prompts)
    return list(_batch.constant(log_probs_completions, size=len(completions)))


def log_probs_conditional_examples(
    examples: Sequence[Example],
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer],
    batch_size: int = 32,
) -> list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt and previous
    completion tokens.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[AutoModelForCausalLM, PreTrainedTokenizer]
        an instantiated model and its corresponding tokenizer
    batch_size : int, optional
        the maximum number of inputs that the model will process in parallel, by default
        32

    Returns
    -------
    log_probs_completions : list[list[list[float]]]
        `log_probs_completions[example_idx][completion_idx][completion_token_idx]` is
        the log-probability of the completion token in
        `examples[example_idx].completions[completion_idx]`, conditional on
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt` and
        previous completion tokens.

    Note
    ----
    To aggregate `log_probs_completions`, use
    :func:`cappr.utils.classify.agg_log_probs`.

    Note
    ----
    The attribute :attr:`cappr.Example.prior` is unused.

    Example
    -------
    Here we'll use single characters (which are of course single tokens) to more clearly
    demonstrate what this function does::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr import Example
        from cappr.huggingface.classify import log_probs_conditional_examples

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Create data
        examples = [Example(prompt='x y',   completions=('z', 'd e')),
                    Example(prompt='a b c', completions=('1 2',))]

        # Compute
        log_probs_completions = log_probs_conditional_examples(
                                    examples,
                                    model_and_tokenizer=(model, tokenizer)
                                )

        # Outputs (rounded) next to their symbolic representation

        log_probs_completions[0] # corresponds to examples[0]
        # [[-4.5],        [[log Pr(z | x, y)],
        #  [-5.6, -3.2]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        log_probs_completions[1] # corresponds to examples[1]
        # [[-5.0, -1.7]]  [[log Pr(1 | a, b, c)], log Pr(2 | a, b, c, 1)]]
    """
    model, tokenizer = hf._utils.load_model_and_tokenizer(model_and_tokenizer)

    @_batch.flatten
    @_batch.batchify(batchable_arg="examples", progress_bar_desc="log-probs")
    def log_probs_completions_batch(examples, batch_size=batch_size):
        logits, encodings = _logits_completions_given_prompts_examples(
            model, tokenizer, examples
        )
        return _logits_to_log_probs_completions(logits, encodings)

    log_probs_completions = log_probs_completions_batch(examples)
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(
        _batch.variable(log_probs_completions, sizes=num_completions_per_prompt)
    )


@classify._predict_proba
def predict_proba(
    prompts: Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer],
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    batch_size: int = 32,
) -> npt.NDArray[np.floating]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model_and_tokenizer : tuple[AutoModelForCausalLM, PreTrainedTokenizer]
        an instantiated model and its corresponding tokenizer
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    batch_size : int, optional
        the maximum number of inputs that the model will process in parallel, by default
        32

    Returns
    -------
    pred_probs : npt.NDArray[np.floating]
        Array with shape `(len(prompts), len(completions))`.
        `pred_probs[prompt_idx, completion_idx]` is the model's estimate of the
        probability that `completions[completion_idx]` comes after
        `prompts[prompt_idx] + end_of_prompt`.

    Note
    ----
    In this function, the set of possible completions which could follow each prompt is
    the same for every prompt. If instead, each prompt could be followed by a
    *different* set of completions, then construct a sequence of :class:`cappr.Example`
    objects and pass them to :func:`predict_proba_examples`.

    Example
    -------
    Let's have GPT-2 (small) predict where stuff is in the kitchen. This example also
    conveys that it's not the greatest model out there::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import predict_proba

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Define a classification task
        prompts = ['The tacos are cooking',
                   'Ice cream is']
        class_names = ('on the stove', 'in the freezer', 'in the fridge')
        prior       = (     1/5      ,       2/5       ,       2/5      )

        pred_probs = predict_proba(prompts,
                                   completions=class_names,
                                   model_and_tokenizer=(model, tokenizer),
                                   prior=prior)

        pred_probs = pred_probs.round(1) # just for cleaner output

        # predicted probability that tacos cook on the stove
        pred_probs[0,0]
        # 0.4

        # predicted probability that ice cream is in the freezer
        pred_probs[1,1]
        # 0.5

        # predicted probability that ice cream is in the fridge
        pred_probs[1,2]
        # 0.4
    """
    return log_probs_conditional(
        prompts,
        completions,
        model_and_tokenizer,
        end_of_prompt=end_of_prompt,
        batch_size=batch_size,
    )


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Sequence[Example],
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer],
    batch_size: int = 32,
) -> Union[list[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[AutoModelForCausalLM, PreTrainedTokenizer]
        an instantiated model and its corresponding tokenizer
    batch_size : int, optional
        the maximum number of inputs that the model will process in parallel, by default
        32

    Returns
    -------
    pred_probs : list[npt.NDArray[np.floating]] | npt.NDArray[np.floating]
        `pred_probs[example_idx][completion_idx]` is the model's estimate of the
        probability that `examples[example_idx].completions[completion_idx]` comes after
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.

        If the number of completions per example is a constant `k`, then an array with
        shape `(len(examples), k)` is returned instead of a list of 1-D arrays.

    Example
    -------
    GPT-2 (small) doing media trivia::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr import Example
        from cappr.huggingface.classify import predict_proba_examples

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Create data
        examples = [
            Example(prompt='Jodie Foster played',
                    completions=('Clarice Starling', 'Trinity in The Matrix')),
            Example(prompt='Batman, from Batman: The Animated Series, was played by',
                    completions=('Kevin Conroy', 'Pete Holmes', 'Spongebob!'),
                    prior=      (     2/3      ,      1/3     ,      0      ))
        ]

        pred_probs = predict_proba_examples(examples,
                                            model_and_tokenizer=(model, tokenizer))

        # predicted probability that Jodie Foster played Clarice Starling, not Trinity
        pred_probs[0][0]
        # 0.7

        # predicted probability that Batman was played by Kevin Conroy
        pred_probs[0][1]
        # 0.97
    """
    return log_probs_conditional_examples(
        examples,
        model_and_tokenizer,
        batch_size=batch_size,
    )


@classify._predict
def predict(
    prompts: Sequence[str],
    completions: Sequence[str],
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer],
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    batch_size: int = 32,
) -> list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    prompts : Sequence[str]
        strings, where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model_and_tokenizer : tuple[AutoModelForCausalLM, PreTrainedTokenizer]
        an instantiated model and its corresponding tokenizer
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    batch_size : int, optional
        the maximum number of inputs that the model will process in parallel, by default
        32

    Returns
    -------
    preds : list[str]
        List with length `len(prompts)`.
        `preds[prompt_idx]` is the completion in `completions` which is predicted to
        follow `prompts[prompt_idx] + end_of_prompt`.

    Note
    ----
    In this function, the set of possible completions which could follow each prompt is
    the same for every prompt. If instead, each prompt could be followed by a
    *different* set of completions, then construct a sequence of :class:`cappr.Example`
    objects and pass them to :func:`predict_examples`.

    Example
    -------
    Let's have GPT-2 (small) predict where stuff is in the kitchen::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr.huggingface.classify import predict

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Define a classification task
        prompts = ['The tacos are cooking', 'Ice cream is']
        class_names = ('on the stove', 'in the freezer', 'in the fridge')
        prior       = (     1/5      ,       2/5       ,       2/5      )

        preds = predict(prompts,
                        completions=class_names,
                        model_and_tokenizer=(model, tokenizer),
                        prior=prior)
        preds
        # ['on the stove',
        #  'in the freezer']
    """
    return predict_proba(
        prompts,
        completions,
        model_and_tokenizer,
        prior=prior,
        end_of_prompt=end_of_prompt,
        batch_size=batch_size,
    )


@classify._predict_examples
def predict_examples(
    examples: Sequence[Example],
    model_and_tokenizer: tuple[AutoModelForCausalLM, PreTrainedTokenizer],
    batch_size: int = 32,
) -> list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model_and_tokenizer : tuple[AutoModelForCausalLM, PreTrainedTokenizer]
        an instantiated model and its corresponding tokenizer
    batch_size : int, optional
        the maximum number of inputs that the model will process in parallel, by default
        32

    Returns
    -------
    preds : list[str]
        List with length `len(examples)`.
        `preds[example_idx]` is the completion in `examples[example_idx].completions`
        which is predicted to follow
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.

    Example
    -------
    GPT-2 (small) doing media trivia::

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from cappr import Example
        from cappr.huggingface.classify import predict_examples

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

        # Create data
        examples = [
            Example(prompt='Jodie Foster played',
                    completions=('Clarice Starling', 'Trinity in The Matrix')),
            Example(prompt='Batman, from Batman: The Animated Series, was played by',
                    completions=('Kevin Conroy', 'Pete Holmes', 'Spongebob!'),
                    prior=      (     2/3      ,      1/3     ,      0      ))
        ]

        preds = predict_examples(examples, model_and_tokenizer=(model, tokenizer))
        preds
        # ['Clarice Starling',
        #  'Kevin Conroy']
    """
    return predict_proba_examples(
        examples,
        model_and_tokenizer,
        batch_size=batch_size,
    )

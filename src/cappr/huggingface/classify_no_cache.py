"""
Perform prompt-completion classification using a ``transformers.AutoModelForCausalLM``.
Currently, only PyTorch models are supported.

This module is a mirror of :mod:`cappr.huggingface.classify` which **does not**
precompute attention block keys and values for prompts.

This module may happen to be compatible with a slightly broader class of
causal/autoregressive language models, as the model's forward method is only assumed
take input IDs and the attention mask.

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)
"""
from __future__ import annotations
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForCausalLM, BatchEncoding, PreTrainedTokenizer

from cappr.utils import _batch, classify
from cappr import Example
from cappr import huggingface as hf


def _keys_values_prompts(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
):
    """
    Performs this procedure:

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

    Note
    ----
    This function is only used to test
    :func:`cappr.huggingface.classify._keys_values_prompts`.

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
        raise ValueError("Gotta use right padding to ensure position IDs are correct.")
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(num_completions_per_prompt, Sequence):
        if not len(prompts) == len(num_completions_per_prompt):
            raise ValueError(
                "If num_completions_per_prompt is a Sequence, then it must be the same "
                f"length as prompts. Got lengths {len(num_completions_per_prompt)}, "
                f"{len(prompts)}."
            )
    if isinstance(num_completions_per_prompt, int):
        ## For code simplicity, just repeat it
        num_completions_per_prompt = [num_completions_per_prompt] * len(prompts)
    prompts_repeated = [
        prompt
        for prompt, num_repeats in zip(prompts, num_completions_per_prompt)
        for _ in range(num_repeats)
    ]
    # fmt: off
    encodings: BatchEncoding = (tokenizer(prompts_repeated, return_tensors="pt",
                                          padding=True)
                                .to(model.device))
    # fmt: on
    with torch.no_grad():
        out = model(**encodings)

    offsets: torch.Tensor = encodings.attention_mask.sum(dim=1)

    ## Need (next-token) logits from prompts, i.e., last non-pad prompt token, since
    ## that contains the first completion token's log-probability
    _last_nonpad_token_idxs = (offsets - 1)[:, None, None]
    last_nonpad_token_logits: torch.Tensor = out.logits.take_along_dim(
        _last_nonpad_token_idxs, dim=1
    )

    return out.past_key_values, encodings, offsets, last_nonpad_token_logits


def _logits_texts(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    texts: Sequence[str],
) -> tuple[torch.Tensor, BatchEncoding]:
    encodings = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model(**encodings)
    return out.logits, encodings


def _prompts_offsets(
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    num_completions_per_prompt: Union[int, Sequence[int]],
) -> torch.Tensor:
    if not isinstance(num_completions_per_prompt, int) and not isinstance(
        num_completions_per_prompt, torch.Tensor
    ):
        num_completions_per_prompt = torch.tensor(num_completions_per_prompt)
    return (
        tokenizer(prompts, return_tensors="pt", padding=True)
        .attention_mask.repeat_interleave(num_completions_per_prompt, dim=0)
        .sum(dim=1)
    )


def _logits_completions_given_prompts(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    end_of_prompt: str = " ",
):
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError("completions must be a Sequence of strings.")
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    logits, encodings = _logits_texts(model, tokenizer, texts)
    ## Need these indices to slice completion tokens
    encodings["offsets"] = _prompts_offsets(
        tokenizer, prompts, num_completions_per_prompt=len(completions)
    ).to(model.device)
    return logits, encodings


def _logits_completions_given_prompts_examples(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    examples: Sequence[Example],
):
    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    logits, encodings = _logits_texts(model, tokenizer, texts)
    ## Need these indices to slice completion tokens
    prompts = [example.prompt for example in examples]
    num_completions_per_prompt = [len(example.completions) for example in examples]
    encodings["offsets"] = _prompts_offsets(
        tokenizer, prompts, num_completions_per_prompt=num_completions_per_prompt
    )
    return logits, encodings


def _logits_to_log_probs_completions(
    logits: torch.Tensor, encodings: Mapping[str, torch.Tensor]
) -> list[list[float]]:
    log_probs = hf._utils.logits_to_log_probs(
        logits, encodings["input_ids"], input_ids_start_idx=1, logits_end_idx=-1
    )
    last_idx_non_pad = encodings["attention_mask"].sum(dim=1)
    ## i.e., # of tokens per text
    return [
        log_probs_prompt_completion[completion_start:completion_end].tolist()
        for log_probs_prompt_completion, completion_start, completion_end in zip(
            log_probs, encodings["offsets"] - 1, last_idx_non_pad - 1
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
        from cappr.huggingface.classify_no_cache import log_probs_conditional

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
    @_batch.batchify(batchable_arg="prompts", progress_bar_desc="log-probs (no cache)")
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
        from cappr.huggingface.classify_no_cache import log_probs_conditional_examples

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
    @_batch.batchify(batchable_arg="examples", progress_bar_desc="log-probs (no cache)")
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
        from cappr.huggingface.classify_no_cache import predict_proba

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
        from cappr.huggingface.classify_no_cache import predict_proba_examples

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
        from cappr.huggingface.classify_no_cache import predict

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
        from cappr.huggingface.classify_no_cache import predict_examples

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

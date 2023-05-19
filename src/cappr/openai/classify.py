"""
Perform prompt-completion classification using models from OpenAI's text
completion API.

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)
"""
from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import tiktoken

from cappr.utils import _batch, classify
from cappr import Example
from cappr import openai


def token_logprobs(
    texts: Sequence[str], model: openai.api.Model, ask_if_ok: bool = False
) -> list[list[float]]:
    """
    For each text, compute each token's log-probability conditional on all previous
    tokens in the text.

    Parameters
    ----------
    texts : Sequence[str]
        input texts
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

    Returns
    -------
    log_probs : list[list[float]]
        `log_probs[text_idx][token_idx]` is the log-probability of the token at
        `token_idx` of `texts[text_idx]` conditional on all previous tokens in
        `texts[text_idx]`. If `texts[text_idx]` is a single token, then
        `log_probs[text_idx]` is `[None]`.
    """
    ## Need to handle texts which are single tokens. Set their logprobs to [None]
    tokenizer = tiktoken.encoding_for_model(model)
    text_lengths = [len(tokens) for tokens in tokenizer.encode_batch(texts)]
    idxs_multiple_tokens = [i for i, length in enumerate(text_lengths) if length > 1]
    choices = openai.api.gpt_complete(
        texts=[texts[i] for i in idxs_multiple_tokens],
        ask_if_ok=ask_if_ok,
        model=model,
        ## rest must be hard-coded
        max_tokens=0,
        logprobs=1,
        echo=True,
    )
    ## Interleave
    log_probs_texts = [choice["logprobs"]["token_logprobs"] for choice in choices]
    log_probs = [[None]] * len(texts)
    for i, log_probs_text in zip(idxs_multiple_tokens, log_probs_texts):
        log_probs[i] = log_probs_text
    return log_probs


def _slice_completions(
    completions: Sequence[str],
    end_of_prompt: str,
    log_probs: Sequence[Sequence[float]],
    model: openai.api.Model,
) -> list[list[float]]:
    """
    TODO: convert docstring to numpy style, expose

    Returns a list `log_probs_completions` where `log_probs_completions[i]` is a list of
    conditional log-probablities for each token in `end_of_prompt + completions[i]`,
    extracted by slicing `log_probs[i]`.
    """
    if len(completions) != len(log_probs):
        raise ValueError(
            "Different number of completions and log_probs: "
            f"{len(completions)}, {len(log_probs)}."
        )
    tokenizer = tiktoken.encoding_for_model(model)
    completions = [end_of_prompt + completion for completion in completions]
    completion_lengths = [len(tokens) for tokens in tokenizer.encode_batch(completions)]
    return [
        log_probs_text[-num_completion_tokens:]
        for num_completion_tokens, log_probs_text in zip(completion_lengths, log_probs)
    ]


def log_probs_conditional(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    end_of_prompt: str = " ",
    ask_if_ok: bool = False,
):
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
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

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

        from cappr.openai.classify import log_probs_conditional

        # Create data
        prompts = ['x y', 'a b c']
        completions = ['z', 'd e']

        # Compute
        log_probs_completions = log_probs_conditional(
                                    prompts,
                                    completions,
                                    model='text-ada-001'
                                )

        # Outputs (rounded) next to their symbolic representation

        log_probs_completions[0]
        # [[-5.5],        [[log Pr(z | x, y)],
        #  [-8.2, -2.1]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        log_probs_completions[1]
        # [[-11.6],       [[log Pr(z | a, b, c)],
        #  [-0.3, -1.2]]   [log Pr(d | a, b, c), log Pr(e | a, b, c, d)]]
    """
    ## str / non-Sequence[str] inputs silently, wastefully, and irreparably fail
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a Sequence of strings.")
    if isinstance(completions, str) or not isinstance(completions, Sequence):
        raise TypeError("completions must be a Sequence of strings.")
    ## Flat list of prompts and their completions. Will post-process
    texts = [
        prompt + end_of_prompt + completion
        for prompt in prompts
        for completion in completions
    ]
    log_probs = token_logprobs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Since log_probs is a flat list, we'll need to batch them by the size and order of
    ## completions to fulfill the spec.
    return [
        _slice_completions(completions, end_of_prompt, log_probs_batch, model)
        for log_probs_batch in _batch.constant(log_probs, size=len(completions))
    ]


def log_probs_conditional_examples(
    examples: Sequence[Example], model: openai.api.Model, ask_if_ok: bool = False
) -> list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

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

        from cappr import Example
        from cappr.openai.classify import log_probs_conditional_examples

        # Create data
        examples = [Example(prompt='x y',   completions=('z', 'd e')),
                    Example(prompt='a b c', completions=('1 2',))]

        # Compute
        log_probs_completions = log_probs_conditional_examples(
                                    examples,
                                    model='text-ada-001'
                                )

        # Outputs (rounded) next to their symbolic representation

        log_probs_completions[0] # corresponds to examples[0]
        # [[-5.5],        [[log Pr(z | x, y)],
        #  [-8.2, -2.1]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        log_probs_completions[1] # corresponds to examples[1]
        # [[-11.2, -4.7]]  [[log Pr(1 | a, b, c)], log Pr(2 | a, b, c, 1)]]
    """
    ## Flat list of prompts and their completions. Will post-process
    texts = [
        example.prompt + example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    log_probs_all = token_logprobs(texts, model=model, ask_if_ok=ask_if_ok)
    ## Flatten completions in same order as examples were flattened
    completions_all = [
        example.end_of_prompt + completion
        for example in examples
        for completion in example.completions
    ]
    log_probs_completions_all = _slice_completions(
        completions_all, "", log_probs_all, model
    )
    ## Batch by completions to fulfill the spec
    num_completions_per_prompt = [len(example.completions) for example in examples]
    return list(
        _batch.variable(log_probs_completions_all, sizes=num_completions_per_prompt)
    )


@classify._predict_proba
def predict_proba(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    discount_completions: float = 0.0,
    log_marginal_probs_completions: Optional[Sequence[Sequence[float]]] = None,
    ask_if_ok: bool = False,
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
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    discount_completions : float, optional
        experimental feature: set it (e.g., 1.0 may work well) if a completion is
        consistently getting too high predicted probabilities. You could instead fudge
        the `prior`, but this hyperparameter may be easier to tune than the `prior`. By
        default 0.0
    log_marginal_probs_completions : Sequence[Sequence[float]] , optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Compute them by passing `completions` and `model` to
        :func:`cappr.openai.classify.token_logprobs`. By default, None
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

    Returns
    -------
    pred_probs : npt.NDArray[np.floating]
        Array with shape `(len(prompts), len(completions))`.
        `pred_probs[prompt_idx, completion_idx]` is the `model`'s estimate of the
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
    A more complicated business-y example—classify product reviews::

        from cappr.openai.classify import predict_proba


        # Define a classification task
        feedback_types = ('the product is too expensive',
                          'the product uses low quality materials',
                          'the product is difficult to use',
                          'the product is great')
        prior = (2/5, 1/5, 1/5, 1/5) # I already expect customers to say it's expensive


        # Write a prompt
        def prompt_func(product_review: str) -> str:
            return f'''
        This product review: {product_review}\n
        is best summarized as:'''


        # Supply the texts you wanna classify
        product_reviews = ["I can't figure out how to integrate it into my setup.",
                           "Yeah it's pricey, but it's definitely worth it."]
        prompts = [prompt_func(product_review) for product_review in product_reviews]


        pred_probs = predict_proba(prompts,
                                   completions=feedback_types,
                                   model='text-curie-001',
                                   prior=prior)

        pred_probs = pred_probs.round(1) # just for cleaner output

        # predicted probability that 1st product review says it's difficult to use
        pred_probs[0,2]
        # 0.9

        # predicted probability that 2nd product review says it's great
        pred_probs[1,3]
        # 0.7

        # predicted probability that 2nd product review says it's too expensive
        pred_probs[1,0]
        # 0.1
    """
    if not discount_completions and (log_marginal_probs_completions is not None):
        raise ValueError(
            "log_marginal_probs_completions is set, but they will not be used because "
            "discount_completions was not set."
        )
    log_probs_completions = log_probs_conditional(
        prompts,
        completions,
        model,
        end_of_prompt=end_of_prompt,
        ask_if_ok=ask_if_ok,
    )
    if not discount_completions:
        return log_probs_completions
    ## log Pr(completion token i | completion token :i) for each completion
    if log_marginal_probs_completions is None:
        log_marginal_probs_completions = token_logprobs(
            completions, model, ask_if_ok=ask_if_ok
        )
    for x in log_marginal_probs_completions:
        x[0] = 0  ## set it from None to 0, i.e., no discount for the first token
    ## pre-multiply by the discount amount
    log_marginal_probs_completions_discounted = [
        discount_completions * np.array(log_marginal_probs_completion)
        for log_marginal_probs_completion in log_marginal_probs_completions
    ]
    return [
        [
            np.array(log_probs_prompt_completions[completion_idx])
            + (np.array(log_marginal_probs_completions_discounted[completion_idx]))
            for completion_idx in range(len(completions))
        ]
        for log_probs_prompt_completions in log_probs_completions
    ]


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Sequence[Example], model: openai.api.Model, ask_if_ok: bool = False
) -> Union[list[npt.NDArray[np.floating]], npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

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
    Let's demo COPA https://people.ict.usc.edu/~gordon/copa.html::

        from cappr import Example
        from cappr.openai.classify import predict_proba_examples


        # Create data from the premises and alternatives
        examples = [Example(prompt='The man broke his toe because',
                            completions=('he got a hole in his sock.',
                                         'he dropped a hammer on his foot.')),
                    Example(prompt='I tipped the bottle, so',
                            completions=('the liquid in the bottle froze.',
                                         'the liquid in the bottle poured out.'))]


        pred_probs = predict_proba_examples(examples, model='text-curie-001')

        pred_probs = pred_probs.round(2) # just for cleaner output

        # predicted probability that 'he dropped a hammer on his foot' is the
        # alternative implied by the 1st premise: 'The man broke his toe'
        pred_probs[0,1]
        # 0.53

        # predicted probability that 'the liquid in the bottle poured out' is the
        # alternative implied by the 2nd premise: 'I tipped the bottle'
        pred_probs[1,1]
        # 0.75
    """
    return log_probs_conditional_examples(examples, model, ask_if_ok=ask_if_ok)


@classify._predict
def predict(
    prompts: Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    prior: Optional[Sequence[float]] = None,
    end_of_prompt: str = " ",
    discount_completions: float = 0.0,
    log_marginal_probs_completions: Optional[Sequence[Sequence[float]]] = None,
    ask_if_ok: bool = False,
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
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    prior : Sequence[float], optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : str, optional
        the string to tack on at the end of every prompt, by default " "
    discount_completions : float, optional
        experimental feature: set it to >0.0 (e.g., 1.0 may work well) if a completion
        is consistently getting over-predicted. You could instead fudge the `prior`, but
        this hyperparameter may be easier to tune than the `prior`. By default 0.0
    log_marginal_probs_completions : Sequence[Sequence[float]] , optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Compute them by passing `completions` and `model` to
        :func:`cappr.openai.classify.token_logprobs`. By default, None
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

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
    A more complicated business-y example—classify product reviews::

        from cappr.openai.classify import predict


        # Define a classification task
        feedback_types = ('the product is too expensive',
                          'the product uses low quality materials',
                          'the product is difficult to use',
                          'the product is great')
        prior = (2/5, 1/5, 1/5, 1/5) # I already expect customers to say it's expensive


        # Write a prompt
        def prompt_func(product_review: str) -> str:
            return f'''
        This product review: {product_review}\n
        is best summarized as:'''


        # Supply the texts you wanna classify
        product_reviews = ["I can't figure out how to integrate it into my setup.",
                           "Yeah it's pricey, but it's definitely worth it."]
        prompts = [prompt_func(product_review) for product_review in product_reviews]


        preds = predict(prompts,
                        completions=feedback_types,
                        model='text-curie-001',
                        prior=prior)
        preds
        # ['the product is difficult to use',
        #  'the product is great']
    """
    return predict_proba(
        prompts,
        completions,
        model,
        prior=prior,
        end_of_prompt=end_of_prompt,
        discount_completions=discount_completions,
        ask_if_ok=ask_if_ok,
    )


@classify._predict_examples
def predict_examples(
    examples: Sequence[Example], model: openai.api.Model, ask_if_ok: bool = False
) -> list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Sequence[Example]
        `Example` objects, where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default
        False

    Returns
    -------
    preds : list[str]
        List with length `len(examples)`.
        `preds[example_idx]` is the completion in `examples[example_idx].completions`
        which is predicted to follow
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.

    Example
    -------
    Let's demo COPA https://people.ict.usc.edu/~gordon/copa.html::

        from cappr import Example
        from cappr.openai.classify import predict_examples

        # Create data from the premises and alternatives
        examples = [Example(prompt='The man broke his toe because',
                            completions=('he got a hole in his sock.',
                                         'he dropped a hammer on his foot.')),
                    Example(prompt='I tipped the bottle, so',
                            completions=('the liquid in the bottle froze.',
                                         'the liquid in the bottle poured out.'))]

        preds = predict_examples(examples, model='text-curie-001')
        preds
        # ['he dropped a hammer on his foot',
        #  'the liquid in the bottle poured out']
    """
    return predict_proba_examples(examples, model, ask_if_ok=ask_if_ok)

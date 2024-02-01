"""
Perform prompt-completion classification using models from OpenAI's text completion API.
Note that only v1/completions models, excluding ``gpt-3.5-turbo-instruct``, are
compatible with CAPPr.

You probably just want the :func:`predict` or :func:`predict_examples` functions :-)
"""

from __future__ import annotations
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import tiktoken

from cappr.utils import _no_cache, classify
from cappr import Example
from cappr import openai
from cappr.openai.api import OpenAI


@classify._token_logprobs
def token_logprobs(
    texts: str | Sequence[str],
    model: openai.api.Model,
    client: OpenAI | None = None,
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
    **kwargs,
) -> list[float] | list[list[float]]:
    """
    For each text, compute each token's log-probability conditional on all previous
    tokens in the text.

    Parameters
    ----------
    texts : str | Sequence[str]
        input text(s)
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    end_of_prompt : Literal[' ', ''], optional
        This string gets added to the beginning of each text. It's important to set this
        if you're using the discount feature. Otherwise, set it to "". By default " "
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    log_probs : list[float] | list[list[float]]

        If `texts` is a string, then a 1-D list is returned: `log_probs[token_idx]` is
        the log-probability of the token at `token_idx` of `texts` conditional on all
        previous tokens in `texts`.

        If `texts` is a sequence of strings, then a 2-D list is returned:
        `log_probs[text_idx][token_idx]` is the log-probability of the token at
        `token_idx` of `texts[text_idx]` conditional on all previous tokens in
        `texts[text_idx]`.

    Warning
    -------
    Set `end_of_prompt=""` unless you're using the discount feature.

    Note
    ----
    For each text, the first token's log-probability is always ``None`` because no
    autoregressive LM directly estimates the marginal probability of a token.

    Raises
    ------
    TypeError
        if `texts` is not a sequence
    ValueError
        if `texts` is empty
    """
    # Need to handle texts which are single tokens. Set their logprobs to [None]
    tokenizer = tiktoken.encoding_for_model(model)
    num_tokens = [len(tokens) for tokens in tokenizer.encode_batch(texts)]
    idxs_multiple_tokens = [i for i, length in enumerate(num_tokens) if length > 1]
    choices = openai.api.gpt_complete(
        texts=[end_of_prompt + texts[i] for i in idxs_multiple_tokens],
        model=model,
        client=client,
        show_progress_bar=show_progress_bar,
        ask_if_ok=ask_if_ok,
        api_key=api_key,
        # rest must be hard-coded
        max_tokens=0,
        logprobs=1,
        echo=True,
    )
    # Interleave
    log_probs_texts = [choice["logprobs"]["token_logprobs"] for choice in choices]
    log_probs = [[None]] * len(texts)
    for i, log_probs_text in zip(idxs_multiple_tokens, log_probs_texts):
        log_probs[i] = log_probs_text
    return log_probs


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    client: OpenAI | None = None,
    end_of_prompt: Literal[" ", ""] = " ",
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
    **kwargs,
) -> list[list[float]] | list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt and previous
    completion tokens.

    Parameters
    ----------
    prompts : str | Sequence[str]
        string(s), where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompt-completion combinations
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    log_probs_completions : list[list[float]] | list[list[list[float]]]

        If `prompts` is a string, then a 2-D list is returned:
        `log_probs_completions[completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompt + end_of_prompt` and previous completion tokens.

        If `prompts` is a sequence of strings, then a 3-D list is returned:
        `log_probs_completions[prompt_idx][completion_idx][completion_token_idx]` is the
        log-probability of the completion token in `completions[completion_idx]`,
        conditional on `prompts[prompt_idx] + end_of_prompt` and previous completion
        tokens.

    Note
    ----
    To efficiently aggregate `log_probs_completions`, use
    :func:`cappr.utils.classify.agg_log_probs`.

    Example
    -------
    Here we'll use single characters (which are single tokens) to more clearly
    demonstrate what this function does::

        from cappr.openai.classify import log_probs_conditional

        # Create data
        prompts = ["x y", "a b c"]
        completions = ["z", "d e"]

        # Compute
        log_probs_completions = log_probs_conditional(
            prompts, completions, model="text-ada-001"
        )

        # Outputs (rounded) next to their symbolic representation

        print(log_probs_completions[0])
        # [[-5.5],        [[log Pr(z | x, y)],
        #  [-8.2, -2.1]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        print(log_probs_completions[1])
        # [[-11.6],       [[log Pr(z | a, b, c)],
        #  [-0.3, -1.2]]   [log Pr(d | a, b, c), log Pr(e | a, b, c, d)]]
    """
    bos_token_id = None  # no OpenAI BPE tokenizer adds a BOS token
    token_logprobs_kwargs = dict(
        client=client,
        show_progress_bar=show_progress_bar,
        ask_if_ok=ask_if_ok,
        api_key=api_key,
    )
    return _no_cache.log_probs_conditional(
        prompts,
        completions,
        end_of_prompt,
        token_logprobs,
        tiktoken.encoding_for_model(model).encode_batch,
        bos_token_id,
        model,
        **token_logprobs_kwargs,
    )


@classify._log_probs_conditional_examples
def log_probs_conditional_examples(
    examples: Example | Sequence[Example],
    model: openai.api.Model,
    client: OpenAI | None = None,
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
) -> list[list[float]] | list[list[list[float]]]:
    """
    Log-probabilities of each completion token conditional on each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompt-completion combinations
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    log_probs_completions : list[list[float]] | list[list[list[float]]]

        If `examples` is a :class:`cappr.Example`, then a 2-D list is returned:
        `log_probs_completions[completion_idx][completion_token_idx]` is the
        log-probability of the completion token in
        `example.completions[completion_idx]`, conditional on `example.prompt +
        example.end_of_prompt` and previous completion tokens.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a 3-D list
        is returned:
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
    Here we'll use single characters (which are single tokens) to more clearly
    demonstrate what this function does::

        from cappr import Example
        from cappr.openai.classify import log_probs_conditional_examples

        # Create data
        examples = [
            Example(prompt="x y", completions=("z", "d e")),
            Example(prompt="a b c", completions=("1 2",), normalize=False),
        ]

        # Compute
        log_probs_completions = log_probs_conditional_examples(
            examples, model="text-ada-001"
        )

        # Outputs (rounded) next to their symbolic representation

        print(log_probs_completions[0])  # corresponds to examples[0]
        # [[-5.5],        [[log Pr(z | x, y)],
        #  [-8.2, -2.1]]   [log Pr(d | x, y),    log Pr(e | x, y, d)]]

        print(log_probs_completions[1])  # corresponds to examples[1]
        # [[-11.2, -4.7]]  [[log Pr(1 | a, b, c)], log Pr(2 | a, b, c, 1)]]
    """
    bos_token_id = None  # no OpenAI BPE tokenizer adds a BOS token
    token_logprobs_kwargs = dict(
        client=client,
        show_progress_bar=show_progress_bar,
        ask_if_ok=ask_if_ok,
        api_key=api_key,
    )
    return _no_cache.log_probs_conditional_examples(
        examples,
        token_logprobs,
        tiktoken.encoding_for_model(model).encode_batch,
        bos_token_id,
        model,
        **token_logprobs_kwargs,
    )


@classify._predict_proba
def predict_proba(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    client: OpenAI | None = None,
    prior: Sequence[float] | None = None,
    end_of_prompt: Literal[" ", ""] = " ",
    normalize: bool = True,
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
) -> npt.NDArray[np.floating]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    prompts : str | Sequence[str]
        string(s), where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    normalize : bool, optional
        whether or not to normalize completion-after-prompt probabilities into a
        probability distribution over completions. Set this to `False` if you'd like the
        raw completion-after-prompt probability, or you're solving a multi-label
        prediction problem. By default, True
    discount_completions : float, optional
        experimental feature: set it (e.g., 1.0 may work well) if a completion is
        consistently getting too high predicted probabilities. You could instead fudge
        the `prior`, but this hyperparameter may be easier to tune than the `prior`. By
        default 0.0
    log_marg_probs_completions : Sequence[Sequence[float]] | None, optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Pre-compute them by passing `completions`, `model`, and
        `end_of_prompt` to :func:`token_logprobs`. By default, if `not
        discount_completions`, they are (re-)computed
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompt-completion combinations
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    pred_probs : npt.NDArray[np.floating]

        If `prompts` is a string, then an array with shape `len(completions),` is
        returned: `pred_probs[completion_idx]` is the model's estimate of the
        probability that `completions[completion_idx]` comes after `prompt +
        end_of_prompt`.

        If `prompts` is a sequence of strings, then an array with shape `(len(prompts),
        len(completions))` is returned: `pred_probs[prompt_idx, completion_idx]` is the
        model's estimate of the probability that `completions[completion_idx]` comes
        after `prompts[prompt_idx] + end_of_prompt`.

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
        feedback_types = (
            "the product is too expensive",
            "the product uses low quality materials",
            "the product is difficult to use",
            "the product is great",
        )
        prior = (2 / 5, 1 / 5, 1 / 5, 1 / 5)
        # I already expect customers to say it's expensive


        # Write a prompt
        def prompt_func(product_review: str) -> str:
            return f'''
        This product review: {product_review}\n
        is best summarized as:'''

        # Supply the texts you wanna classify
        product_reviews = [
            "I can't figure out how to integrate it into my setup.",
            "Yeah it's pricey, but it's definitely worth it.",
        ]
        prompts = [
            prompt_func(product_review) for product_review in product_reviews
        ]

        pred_probs = predict_proba(
            prompts, completions=feedback_types, model="text-curie-001", prior=prior
        )
        pred_probs_rounded = pred_probs.round(1)  # just for cleaner output

        # predicted probability that 1st product review says it's difficult to use
        print(pred_probs_rounded[0, 2])
        # 0.8

        # predicted probability that 2nd product review says it's great
        print(pred_probs_rounded[1, 3])
        # 0.6

        # predicted probability that 2nd product review says it's too expensive
        print(pred_probs_rounded[1, 0])
        # 0.2
    """
    return log_probs_conditional(**locals())


@classify._predict_proba_examples
def predict_proba_examples(
    examples: Example | Sequence[Example],
    model: openai.api.Model,
    client: OpenAI | None = None,
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
) -> npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]:
    """
    Predict probabilities of each completion coming after each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompt-completion combinations
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    pred_probs : npt.NDArray[np.floating] | list[npt.NDArray[np.floating]]

        If `examples` is an :class:`cappr.Example`, then an array with shape
        `(len(example.completions),)` is returned: `pred_probs[completion_idx]` is the
        model's estimate of the probability that `example.completions[completion_idx]`
        comes after `example.prompt + example.end_of_prompt`.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a list with
        length `len(examples)` is returned: `pred_probs[example_idx][completion_idx]` is
        the model's estimate of the probability that
        `examples[example_idx].completions[completion_idx]` comes after
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`. If the
        number of completions per example is a constant `k`, then an array with shape
        `(len(examples), k)` is returned instead of a list of 1-D arrays.

    Example
    -------
    Let's demo COPA https://people.ict.usc.edu/~gordon/copa.html::

        from cappr import Example
        from cappr.openai.classify import predict_proba_examples

        # Create data from the premises and alternatives
        examples = [
            Example(
                prompt="The man broke his toe because",
                completions=(
                    "he got a hole in his sock.",
                    "he dropped a hammer on his foot."
                ),
            ),
            Example(
                prompt="I tipped the bottle, so",
                completions=(
                    "the liquid in the bottle froze.",
                    "the liquid in the bottle poured out.",
                ),
            ),
        ]

        pred_probs = predict_proba_examples(examples, model="text-curie-001")
        pred_probs_rounded = pred_probs.round(2)  # just for cleaner output

        # predicted probability that 'he dropped a hammer on his foot' is the
        # alternative implied by the 1st premise: 'The man broke his toe'
        print(pred_probs_rounded[0, 1])
        # 0.53

        # predicted probability that 'the liquid in the bottle poured out' is the
        # alternative implied by the 2nd premise: 'I tipped the bottle'
        print(pred_probs_rounded[1, 1])
        # 0.75
    """
    return log_probs_conditional_examples(**locals())


@classify._predict
def predict(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: openai.api.Model,
    client: OpenAI | None = None,
    prior: Sequence[float] | None = None,
    end_of_prompt: Literal[" ", ""] = " ",
    discount_completions: float = 0.0,
    log_marg_probs_completions: Sequence[Sequence[float]] | None = None,
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
) -> str | list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    prompts : str | Sequence[str]
        string(s), where, e.g., each contains the text you want to classify
    completions : Sequence[str]
        strings, where, e.g., each one is the name of a class which could come after a
        prompt
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    prior : Sequence[float] | None, optional
        a probability distribution over `completions`, representing a belief about their
        likelihoods regardless of the prompt. By default, each completion in
        `completions` is assumed to be equally likely
    end_of_prompt : Literal[' ', ''], optional
        whitespace or empty string to join prompt and completion, by default whitespace
    discount_completions : float, optional
        experimental feature: set it to >0.0 (e.g., 1.0 may work well) if a completion
        is consistently getting over-predicted. You could instead fudge the `prior`, but
        this hyperparameter may be easier to tune than the `prior`. By default 0.0
    log_marg_probs_completions : Sequence[Sequence[float]] | None, optional
        experimental feature: pre-computed log probabilities of completion tokens
        conditional on previous completion tokens (not prompt tokens). Only used if `not
        discount_completions`. Pre-compute them by passing `completions`, `model`, and
        `end_of_prompt` to :func:`token_logprobs`. By default, if `not
        discount_completions`, they are (re-)computed
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 prompt-completion combinations
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    preds : str | list[str]

        If `prompts` is a string, then the completion from `completions` which is
        predicted to most likely follow `prompt + end_of_prompt` is returned.

        If `prompts` is a sequence of strings, then a list with length `len(prompts)` is
        returned. `preds[prompt_idx]` is the completion in `completions` which is
        predicted to follow `prompts[prompt_idx] + end_of_prompt`.

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
        feedback_types = (
            "the product is too expensive",
            "the product uses low quality materials",
            "the product is difficult to use",
            "the product is great",
        )
        prior = (2 / 5, 1 / 5, 1 / 5, 1 / 5)
        # I already expect customers to say it's expensive


        # Write a prompt
        def prompt_func(product_review: str) -> str:
            return f'''
        This product review: {product_review}\n
        is best summarized as:'''


        # Supply the texts you wanna classify
        product_reviews = [
            "I can't figure out how to integrate it into my setup.",
            "Yeah it's pricey, but it's definitely worth it.",
        ]
        prompts = [
            prompt_func(product_review) for product_review in product_reviews
        ]

        preds = predict(
            prompts, completions=feedback_types, model="text-curie-001", prior=prior
        )
        print(preds)
        # ['the product is difficult to use',
        #  'the product is great']
    """
    return predict_proba(**locals())


@classify._predict_examples
def predict_examples(
    examples: Example | Sequence[Example],
    model: openai.api.Model,
    client: OpenAI | None = None,
    show_progress_bar: bool | None = None,
    ask_if_ok: bool = False,
    api_key: str | None = None,
) -> str | list[str]:
    """
    Predict which completion is most likely to follow each prompt.

    Parameters
    ----------
    examples : Example | Sequence[Example]
        `Example` object(s), where each contains a prompt and its set of possible
        completions
    model : cappr.openai.api.Model
        string for the name of an OpenAI text-completion model, specifically one from
        the ``/v1/completions`` endpoint:
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    client : OpenAI | None, optional
        an OpenAI client object. By default, the global client instance is used
    show_progress_bar : bool | None, optional
        whether or not to show a progress bar. By default, it will be shown only if
        there are at least 5 texts
    ask_if_ok : bool, optional
        whether or not to prompt you to manually give the go-ahead to run this function,
        after notifying you of the approximate cost of the OpenAI API calls. By default,
        False
    api_key : str | None, optional
        your OpenAI API key. By default, it's set to the OpenAI's module attribute
        ``openai.api_key``, or the environment variable ``OPENAI_API_KEY``

    Returns
    -------
    preds : str | list[str]

        If `examples` is an :class:`cappr.Example`, then the completion from
        `example.completions` which is predicted to most likely follow `example.prompt +
        example.end_of_prompt` is returned.

        If `examples` is a sequence of :class:`cappr.Example` objects, then a list with
        length `len(examples)` is returned: `preds[example_idx]` is the completion in
        `examples[example_idx].completions` which is predicted to most likely follow
        `examples[example_idx].prompt + examples[example_idx].end_of_prompt`.

    Example
    -------
    Let's demo COPA https://people.ict.usc.edu/~gordon/copa.html::

        from cappr import Example
        from cappr.openai.classify import predict_examples

        # Create data from the premises and alternatives
        examples = [
            Example(
                prompt="The man broke his toe because",
                completions=(
                    "he got a hole in his sock.",
                    "he dropped a hammer on his foot."
                ),
            ),
            Example(
                prompt="I tipped the bottle, so",
                completions=(
                    "the liquid in the bottle froze.",
                    "the liquid in the bottle poured out.",
                ),
            ),
        ]

        preds = predict_examples(examples, model="text-curie-001")
        print(preds)
        # ['he dropped a hammer on his foot',
        #  'the liquid in the bottle poured out']
    """
    return predict_proba_examples(**locals())

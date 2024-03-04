"""
Base classes which are parametrized with a set of test cases which every `classify`
module must pass.
"""

from __future__ import annotations
from typing import Sequence

import pandas as pd
import pytest

from cappr import Example

from _protocol import classify_module
import _test_form
import _test_content


class _BaseTest:
    def __setattr__(self, attr, value):
        raise AttributeError(
            f"Tried to set {attr} to {value}. But this class is frozen."
        )

    @property
    def module(self) -> classify_module:
        """
        `classify` module which needs to be tested.
        """
        raise NotImplementedError

    @property
    def module_correct(self) -> classify_module:
        """
        `classify` module which is correct, and serves as a reference implementation.
        Have it return `None` if there's no way to determine whether an implementation
        is correct. Usually that's the case if you're integrating an API endpoint like
        OpenAI.
        """
        raise NotImplementedError

    def _test(self, function: str, *args, **kwargs):
        """
        Format of all tests: test `function`'s outputs for form/structure, and then test
        their content if there exists a reference implementation to test against.
        """
        test_form = getattr(_test_form, function)
        test_form(self.module, *args, **kwargs)
        if self.module_correct is None:
            return
        test_form(self.module_correct, *args, **kwargs)
        test_content = getattr(_test_content, function)
        test_content(self.module_correct, self.module, *args, **kwargs)


@pytest.mark.parametrize(
    "prompts",
    (
        ####################### Single tokens for easy debugging #######################
        ["a b c", "c"],
        ############################### Test single-input ##############################
        "prompts single string",
        ############# Test a different type of sequence w/ corrupt indices #############
        pd.Series(["prompts is", "a", "Series"], index=[9, 0, 9]),
    ),
)
@pytest.mark.parametrize(
    "completions",
    (
        ####################### Single tokens for easy debugging #######################
        ["d e f g", "1 2", "O"],
        ######################## Test Single-token optimization ########################
        ["d", "e", "f"],
        ############# Test a different type of sequence w/ corrupt indices #############
        pd.Series(["completions is", "a", "Series"], index=[8, 10, 8]),
    ),
)
class TestPromptsCompletions(_BaseTest):
    """
    Test non-`_examples` functions with a basic set of combinations of `prompts` and
    `completions`.
    """

    def test_log_probs_conditional(
        self, prompts: str | Sequence[str], completions: Sequence[str], *args, **kwargs
    ):
        self._test("log_probs_conditional", prompts, completions, *args, **kwargs)

    # Fixtures for predict_proba
    # They're not predict_proba parametrizations b/c that raises an error indicating,
    # IIUC, that they're not in scope for this class' subclasses
    @pytest.fixture(scope="class", params=[None, 1])
    def prior(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture(scope="class", params=[True, False])
    def normalize(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(scope="class", params=[0, 1])
    def discount_completions(self, request: pytest.FixtureRequest) -> float:
        return request.param

    def test_predict_proba(
        self,
        prompts: str | Sequence[str],
        completions: Sequence[str],
        *args,
        prior,
        discount_completions: float,
        normalize: bool,
        **kwargs,
    ):
        self._test(
            "predict_proba",
            prompts,
            completions,
            *args,
            prior=prior if prior is None else [1 / len(completions)] * len(completions),
            discount_completions=discount_completions,
            normalize=normalize,
            **kwargs,
        )

    def test_predict(
        self, prompts: str | Sequence[str], completions: Sequence[str], *args, **kwargs
    ):
        self._test("predict", prompts, completions, *args, **kwargs)


@pytest.mark.parametrize(
    "examples",
    (
        ######################## Test non-constant # completions #######################
        [
            Example("a b c", ["d", "e f g"]),
            Example("chi", ["can", "ery"]),
            Example("koyaa", ["nisqatsi"], normalize=False),
        ],
        ############## Test constant # completions, non-constant # tokens ##############
        [
            Example(
                "jelly", ["fin", "is"], prior=pd.Series([0.33, 0.67], index=[1, 1])
            ),
            Example("a great", ["thing.", "shout"]),
            Example("out to", ["open", "source, yo."]),
        ],
        ############################ Test singleton example ############################
        Example("lonesome", ["singleton", "example"]),
        ################### Test singleton example single completion ###################
        Example("lonely", ["loner"], normalize=False),
    ),
)
class TestExamples(_BaseTest):
    """
    Test `_examples` functions with a basic set of :class:`cappr.Example`(s).
    """

    def test_log_probs_conditional_examples(
        self, examples: Example | Sequence[Example], *args, **kwargs
    ):
        self._test("log_probs_conditional_examples", examples, *args, **kwargs)

    def test_predict_proba_examples(
        self, examples: Example | Sequence[Example], *args, **kwargs
    ):
        self._test("predict_proba_examples", examples, *args, **kwargs)

    def test_predict_examples(
        self, examples: Example | Sequence[Example], *args, **kwargs
    ):
        self._test("predict_examples", examples, *args, **kwargs)


@pytest.mark.parametrize("prompt_prefix", ("a b c",))
@pytest.mark.parametrize("prompts", (["d", "d e"],))  # TODO: add single prompt
@pytest.mark.parametrize(
    "completions",
    (
        ["e f", "f g h i j"],  # multiple tokens
        ["e", "f"],  # single tokens
    ),
)
class TestCache(_BaseTest):
    def _test_log_probs_conditional(
        self,
        log_probs_completions_from_cached: list[list[list[float]]],
        prompt_prefix: str,
        prompts: list[str],
        completions: list[str],
        *model_args,
        **kwargs,
    ):
        _test_form._test_log_probs_conditional(
            log_probs_completions_from_cached,
            expected_len=len(prompts),
            num_completions_per_prompt=[len(completions)] * len(prompts),
        )
        prompts_full = [prompt_prefix + " " + prompt for prompt in prompts]
        log_probs_completions_wo_cache = self.module_correct.log_probs_conditional(
            prompts_full, completions, *model_args, **kwargs
        )
        _test_content._test_log_probs_conditional(
            log_probs_completions_from_cached,
            log_probs_completions_wo_cache,
            is_single_input=False,
        )

    def _test_log_probs_conditional_examples(
        self,
        log_probs_completions_from_cached: list[list[list[float]]],
        prompt_prefix: str,
        examples: list[Example],
        *model_args,
        **kwargs,
    ):
        _test_form._test_log_probs_conditional(
            log_probs_completions_from_cached,
            expected_len=len(examples),
            num_completions_per_prompt=[
                len(example.completions) for example in examples
            ],
        )
        examples_full = [
            Example(prompt_prefix + " " + example.prompt, example.completions)
            for example in examples
        ]
        log_probs_completions_wo_cache = (
            self.module_correct.log_probs_conditional_examples(
                examples_full, *model_args, **kwargs
            )
        )
        _test_content._test_log_probs_conditional(
            log_probs_completions_from_cached,
            log_probs_completions_wo_cache,
            is_single_input=False,
        )

    def test_cache(
        self,
        prompt_prefix: str,
        prompts: list[str],
        completions: list[str],
        *model_args,
        **kwargs,
    ):
        examples = [Example(prompt, completions) for prompt in prompts]
        with self.module.cache(*model_args, prompt_prefix) as cached:
            log_probs_completions = self.module.log_probs_conditional(
                prompts, completions, cached, **kwargs
            )
            log_probs_completions_ex = self.module.log_probs_conditional_examples(
                examples, cached, **kwargs
            )
        self._test_log_probs_conditional(
            log_probs_completions,
            prompt_prefix,
            prompts,
            completions,
            *model_args,
            **kwargs,
        )
        self._test_log_probs_conditional_examples(
            log_probs_completions_ex, prompt_prefix, examples, *model_args, **kwargs
        )

    def test_cache_model(
        self,
        prompt_prefix: str,
        prompts: list[str],
        completions: list[str],
        *model_args,
        **kwargs,
    ):
        examples = [Example(prompt, completions) for prompt in prompts]
        cached = self.module.cache_model(*model_args, prompt_prefix)
        log_probs_completions = self.module.log_probs_conditional(
            prompts, completions, cached, **kwargs
        )
        log_probs_completions_ex = self.module.log_probs_conditional_examples(
            examples, cached, **kwargs
        )
        self._test_log_probs_conditional(
            log_probs_completions,
            prompt_prefix,
            prompts,
            completions,
            *model_args,
            **kwargs,
        )
        self._test_log_probs_conditional_examples(
            log_probs_completions_ex, prompt_prefix, examples, *model_args, **kwargs
        )

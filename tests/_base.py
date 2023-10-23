"""
Base classes which are parametrized with a set of test cases which every `classify`
module must pass.
"""
from __future__ import annotations
from typing import Collection, Sequence

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
    def module_correct(self) -> classify_module:
        """
        `classify` module which is correct, and serves as a reference implementation.
        Have it return `None` if there's no way to determine whether an implementation
        is correct. Usually that's the case if you're integrating an API endpoint like
        OpenAI.
        """
        raise NotImplementedError

    @property
    def modules_to_test(self) -> Collection[classify_module]:
        """
        `classify` modules which need to be tested. Usually just::

            (cappr.your_new_backend.classify,)
        """
        raise NotImplementedError

    def _test(self, function: str, *args, **kwargs):
        """
        Format of all tests: test all modules' `function` outputs for form/structure,
        and then test their content if `self.module_correct is not None`, i.e., there
        exists a reference implementation to test against.
        """
        test_form = getattr(_test_form, function)
        for module in self.modules_to_test:
            test_form(module, *args, **kwargs)
        if self.module_correct is None:
            # We can't test the form or content. Done.
            return
        # We can test the form of this module and the content of the others.
        test_form(self.module_correct, *args, **kwargs)
        test_content = getattr(_test_content, function)
        for module in self.modules_to_test:
            test_content(self.module_correct, module, *args, **kwargs)


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
class BaseTestPromptsCompletions(_BaseTest):
    """
    Test non-`_examples` functions with a basic set of combinations of `prompts` and
    `completions`.
    """

    def test_log_probs_conditional(
        self, prompts: str | Sequence, completions: Sequence[str], *args, **kwargs
    ):
        self._test("log_probs_conditional", prompts, completions, *args, **kwargs)

    # Fixtures for predict_proba
    # TODO: this is a bit dirty. Figure out how to make it simpler.
    @pytest.fixture(scope="class", params=[True, False])
    def _use_prior(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(scope="class", params=[True, False])
    def normalize(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(scope="class", params=[0, 1])
    def discount_completions(self, request: pytest.FixtureRequest) -> float:
        return request.param

    def test_predict_proba(
        self,
        prompts: str | Sequence,
        completions: Sequence[str],
        *args,
        _use_prior: bool,
        discount_completions: float,
        normalize: bool,
        **kwargs,
    ):
        if _use_prior:
            kwargs["prior"] = [1 / len(completions)] * len(completions)
        else:
            kwargs["prior"] = None
        self._test(
            "predict_proba",
            prompts,
            completions,
            *args,
            discount_completions=discount_completions,
            normalize=normalize,
            **kwargs,
        )

    def test_predict(
        self, prompts: str | Sequence, completions: Sequence[str], *args, **kwargs
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
            Example("jelly", ["fin", "is"], prior=(1 / 3, 2 / 3)),
            Example("a great", ["thing.", "shout"]),
            Example("out to", ["open", "source, yo."]),
        ],
        ############################ Test singleton example ############################
        Example("lonesome", ["singleton", "example"]),
        ################### Test singleton example single completion ###################
        Example("lonely", ["loner"], normalize=False),
    ),
)
class BaseTestExamples(_BaseTest):
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

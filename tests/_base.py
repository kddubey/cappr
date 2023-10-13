"""
TODO
"""
from __future__ import annotations
from typing import Collection, Sequence

import numpy as np
import pandas as pd
import pytest

from cappr import Example

from _protocol import classify_module
import _test_form
import _test_content


class _BaseTest:
    """
    TODO
    """

    def __setattr__(self, attr, value):
        raise AttributeError(
            f"Tried to set {attr} to {value}. But you cannot set attributes for "
            "instances of this class. It's frozen bb."
        )

    @property
    def module_correct(self) -> classify_module:
        raise NotImplementedError(
            "module_correct has not been set. Have it return None if there's no way to "
            "determine whether an implementation is correct. Usually that's the case "
            "if you're integrating an API endpoint like OpenAI.\n"
        )

    @property
    def modules_to_test(self) -> Collection[classify_module]:
        raise NotImplementedError(
            "modules_to_test has not been set. Set it to the modules which need to be "
            "tested."
        )

    def _test(self, function: str, *args, **kwargs):
        """
        Format of all tests: test all modules for form/structure, and then test their
        contents if `self.module_correct is not None`, i.e., there exists a reference
        implementation to test against.
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
        ["a b c", "c"],
        ############################### Test single-input ##############################
        "prompts single",
        ######################## Test different type of sequence #######################
        pd.Series(["prompts is", "a", "Series"], index=np.random.choice(3, size=3)),
    ),
)
@pytest.mark.parametrize(
    "completions",
    (
        ["d e f g", "1 2", "O"],
        ######################## Test Single-token optimization ########################
        ["d", "e", "f"],
        ########################### Test pandas Series input ###########################
        pd.Series(["completions is", "a", "Series"], index=np.random.choice(3, size=3)),
    ),
)
class BaseTestPromptsCompletions(_BaseTest):
    """
    TODO
    """

    def test_log_probs_conditional(
        self, prompts: str | Sequence, completions: Sequence[str], *args, **kwargs
    ):
        self._test("log_probs_conditional", prompts, completions, *args, **kwargs)

    @pytest.fixture(scope="class", params=[True, False])
    def _use_prior(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(scope="class", params=[True, False])
    def normalize(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture(scope="class", params=[0, 1])
    def discount_completions(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    # @pytest.mark.parametrize("_use_prior", (True, False))
    # @pytest.mark.parametrize("discount_completions", (0.0, 1.0))
    # @pytest.mark.parametrize("normalize", (True, False))
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
        [
            Example("a b c", ["d", "e f g"]),
            Example("C", ["p G C p G", "D E F", "ya later alligator"]),
        ],
        ############################# Next set of examples #############################
        [
            Example("chi", ["can", "ery"]),
            Example("koyaa", ["nisqatsi"], normalize=False),
            Example("hi hi", ["bye bye", "yo yo"]),
        ],
        ############## Test constant # completions, non-constant # tokens ##############
        [
            Example("jelly", ["fin", "is"]),
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
    TODO
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

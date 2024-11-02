"""
This module only exists to ensure that Llama 3's tokenizer supports
`tokenizer.add_bos_token = False`. In the future, it'd be nice to delete this.

Issue: https://github.com/huggingface/transformers/issues/30947
"""

from functools import lru_cache
from typing import Literal

from tokenizers import processors
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from cappr.utils.classify import _setattr


@lru_cache(maxsize=5)
def does_disabling_add_token_disable_adding_token(
    tokenizer: PreTrainedTokenizerBase, token_name: Literal["bos_token", "eos_token"]
) -> bool:
    # NOTE: this function should only return False for Llama 3's BOS token. This fact is
    # tested via:
    #
    # python -m pytest \
    # tests/huggingface/test_huggingface_classify.py \
    # -k test__does_disabling_add_token_disable_adding_token \
    # -x

    if token_name == "bos_token":
        position = 0
    elif token_name == "eos_token":
        position = -1
    else:
        raise ValueError(
            'token_name must be either "bos_token", "eos_token"'
        )  # pragma: no cover

    text = "a"
    tokens_default: list[int] = tokenizer(text)["input_ids"]
    is_token_added = tokens_default[position] == getattr(
        tokenizer, f"{token_name}_id", None
    )
    if not is_token_added:
        # Disabling vacuously works b/c, by default, the token wasn't added
        return True

    with _setattr(tokenizer, f"add_{token_name}", False):
        tokens_after_disabling: list[int] = tokenizer(text)["input_ids"]

    tokens_default_wo_token = tokens_default[:]
    tokens_default_wo_token.pop(position)
    if tokens_after_disabling == tokens_default_wo_token:
        return True
    else:
        # Ensure that disabling really did do nothing / it didn't remove the token and
        # did nothing else.
        condition = tokens_after_disabling == tokens_default
        msg = (
            f"There was an unexpected side effect from disabling add_{token_name}. "
            f"The default setting caused 'a' to be tokenized as {tokens_default}. "
            f"Disabling caused 'a' to be tokenized as {tokens_after_disabling}. "
            "Please raise an issue here: https://github.com/kddubey/cappr/issues"
        )
        assert condition, msg
        return False


def force_support(tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Hack to incorporate:

    https://github.com/huggingface/transformers/pull/31316
    """

    text = "a"
    tokens_default: list[int] = tokenizer(text)["input_ids"]

    # We need to initialize these correctly, not None. The reason is that if we update
    # set add_eos/bos_token later, and then reset it back to None, we'll always have
    # False-y values instead of the original behavior.
    tokenizer._add_eos_token = tokens_default[-1] == getattr(
        tokenizer, "eos_token_id", None
    )
    tokenizer._add_bos_token = tokens_default[0] == getattr(
        tokenizer, "bos_token_id", None
    )

    class _PreTrainedTokenizerFastPatched(type(tokenizer)):
        @property
        def add_eos_token(self):
            return self._add_eos_token

        @property
        def add_bos_token(self):
            return self._add_bos_token

        @add_eos_token.setter
        def add_eos_token(self, value: bool):
            self._add_eos_token = value
            self.update_post_processor()

        @add_bos_token.setter
        def add_bos_token(self, value: bool):
            self._add_bos_token = value
            self.update_post_processor()

        def update_post_processor(self):
            """
            Overwrites the underlying post processor with the current `bos_token` and
            `eos_token`.
            """
            if not isinstance(
                self._tokenizer.post_processor, processors.TemplateProcessing
            ) and not isinstance(self._tokenizer.post_processor, processors.Sequence):
                return

            bos = self.bos_token
            bos_token_id = self.bos_token_id
            if bos is None and self.add_bos_token:
                raise ValueError("add_bos_token = True but bos_token = None")

            eos = self.eos_token
            eos_token_id = self.eos_token_id
            if eos is None and self.add_eos_token:
                raise ValueError("add_eos_token = True but eos_token = None")

            single = (
                f"{(bos + ':0 ') if self.add_bos_token else ''}"
                "$A:0"
                f"{(' ' + eos + ':0') if self.add_eos_token else ''}"
            )
            pair = (
                f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} "
                "$B:1"
                f"{(' ' + eos + ':1') if self.add_eos_token else ''}"
            )

            special_tokens = []
            if self.add_bos_token:
                special_tokens.append((bos, bos_token_id))
            if self.add_eos_token:
                special_tokens.append((eos, eos_token_id))
            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=single, pair=pair, special_tokens=special_tokens
            )

    # https://stackoverflow.com/questions/31590152/monkey-patching-a-property
    tokenizer.__class__ = _PreTrainedTokenizerFastPatched

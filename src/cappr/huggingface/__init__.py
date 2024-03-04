"""
Install the Hugging Face requirements to use this module::

    pip install "cappr[hf]"

See this section of the documentation:

https://cappr.readthedocs.io/en/latest/select_a_language_model.html#hugging-face
"""

from . import _utils, classify, classify_no_cache

__all__ = ["_utils", "classify", "classify_no_cache"]

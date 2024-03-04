"""
Install the Llama CPP requirements to use this module::

    pip install "cappr[llama-cpp]"

See this section of the documentation:

https://cappr.readthedocs.io/en/latest/select_a_language_model.html#llama-cpp
"""

from . import _utils, classify, _classify_no_cache

__all__ = ["_utils", "classify", "_classify_no_cache"]

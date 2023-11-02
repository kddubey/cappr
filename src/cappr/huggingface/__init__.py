"""
Note that you need to install the HuggingFace requirements to use this module::

    pip install "cappr[hf]"

See this section of the documentation: https://cappr.readthedocs.io/en/latest/select_a_language_model.html#huggingface
"""
from . import (
    _utils,
    classify,
    classify_,
    classify_no_batch,
    classify_no_cache,
    classify_no_cache_no_batch,
)

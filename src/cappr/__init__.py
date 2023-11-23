"""
Completion After Prompt Probability. Make your LLM make a choice

https://cappr.readthedocs.io/
"""
__version__ = "0.8.7"

from . import utils
from ._example import Example

try:
    from . import openai  # can be optional if cappr is installed with --no-deps
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    from . import huggingface  # optional
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    from . import llama_cpp  # optional
except ModuleNotFoundError:  # pragma: no cover
    pass

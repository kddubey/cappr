"""
Completion After Prompt Probability. Make your LLM make a choice

https://cappr.readthedocs.io/
"""

__version__ = "0.9.0"

from . import utils
from ._example import Example

try:
    from . import openai
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    from . import huggingface
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    from . import llama_cpp
except ModuleNotFoundError:  # pragma: no cover
    pass

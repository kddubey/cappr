"""
Completion After Prompt Probability. Make your LLM make a choice

https://cappr.readthedocs.io/
"""

__version__ = "0.9.2"

from . import utils  # noqa: F401
from ._example import Example  # noqa: F401

try:
    from . import openai  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    from . import huggingface  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

try:
    from . import llama_cpp  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

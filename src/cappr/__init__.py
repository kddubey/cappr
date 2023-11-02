"""
Completion After Prompt Probability

https://cappr.readthedocs.io/
"""
__version__ = "0.7.0"

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

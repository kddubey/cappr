"""
Completion After Prompt Probability
"""
__version__ = "0.6.3"

from . import utils
from ._example import Example

try:
    from . import openai  # can be optional if cappr is installed with --no-deps
except ModuleNotFoundError:
    pass

try:
    from . import huggingface  # optional
except ModuleNotFoundError:
    pass

try:
    from . import llama_cpp  # optional
except ModuleNotFoundError:
    pass

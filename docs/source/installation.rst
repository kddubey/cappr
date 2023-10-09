Installation
============

To use PyTorch ``transformers`` models:

::

   pip install "cappr[hf]"

To use GGUF models::

   pip install "cappr[llama-cpp]"

Or, to use OpenAI models, `sign up <https://platform.openai.com/signup>`_ for the OpenAI
API, set the environment variable ``OPENAI_API_KEY``, and then:

::

   pip install cappr

(Optional) Install requirements for running the repo's `demos
<https://github.com/kddubey/cappr/tree/main/demos>`_:

::

   pip install "cappr[demos]"

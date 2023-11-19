Installation
============

To use HuggingFace models (including `AutoGPTQ <https://github.com/PanQiWei/AutoGPTQ>`_
and `AutoAWQ <https://github.com/casper-hansen/AutoAWQ>`_)::

   pip install "cappr[hf]"

To use GGUF models::

   pip install "cappr[llama-cpp]"

To use OpenAI `/v1/completions
<https://platform.openai.com/docs/models/model-endpoint-compatibility>`_ models
(`excluding
<https://cappr.readthedocs.io/en/latest/select_a_language_model.html#openai>`_
``gpt-3.5-turbo-instruct``), `sign up <https://platform.openai.com/signup>`_ for the
OpenAI API and then::

   pip install cappr

(Optional) To use any of the above models::

   pip install "cappr[all]"


Without dependencies
--------------------

For backwards compatibility reasons, installing this package installs OpenAI's Python
`API client <https://pypi.org/project/openai/>`_ and its (lightweight) dependencies. You
don't need these packages to run HuggingFace or GGUF models. So in case you'd like a
lighter install, run the pair of commands below instead of the ones above.

To use HuggingFace models:

::

   pip install \
   "numpy>=1.21.0" \
   "tqdm>=4.27.0" \
   "transformers[torch]>=4.31.0"

::

   pip install --no-deps cappr


To use GGUF models:

::

   pip install \
   "numpy>=1.21.0" \
   "tqdm>=4.27.0" \
   "llama-cpp-python>=0.2.11"

::

   pip install --no-deps cappr

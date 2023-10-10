Installation
============

To use PyTorch ``transformers`` models::

   pip install "cappr[hf]"

To use GGUF models::

   pip install "cappr[llama-cpp]"

To use OpenAI `v1/completions
<https://platform.openai.com/docs/models/model-endpoint-compatibility>`_ models
(`excluding
<https://cappr.readthedocs.io/en/latest/select_a_language_model.html#openai>`_
``gpt-3.5-turbo-instruct``), `sign up <https://platform.openai.com/signup>`_ for the
OpenAI API, set the environment variable ``OPENAI_API_KEY``, and then::

   pip install cappr

(Optional) Install requirements for running the repo's `demos
<https://github.com/kddubey/cappr/tree/main/demos>`_::

   pip install "cappr[demos]"


Without dependencies
--------------------

For backwards compatibility reasons, installing this package installs the ``openai``
`API <https://pypi.org/project/openai/>`_ and its (lightweight) dependencies. You don't
need these packages to run ``llama-cpp``/GGUF models with ``cappr``. So in case you'd
like to avoid installing ``openai`` and its dependencies, run these two commands (in any
order) instead of the one above:

::

   pip install \
   "numpy>=1.21.0" \
   "tqdm>=4.27.0" \
   "llama-cpp-python>=0.2.11"

::

   pip install --no-deps cappr

(In the future, it'd be nice if a feature like `this one
<https://github.com/pypa/setuptools/pull/1503>`_ were supported by PyPA.)

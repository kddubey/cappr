Installation
============

If you intend on using OpenAI models, `sign up`_ for the OpenAI API, and then set the
environment variable ``OPENAI_API_KEY`` \. For zero-shot classification, OpenAI models
are currently ahead of others. But using them will cost ya 💰!

.. _sign up: https://platform.openai.com/signup

Install with ``pip``:

::

   pip install cappr

(Optional) Install requirements for HuggingFace models:

::

   pip install "cappr[hf]"

(Optional) Install requirements for running `demos`_:

::

   pip install "cappr[demos]"

.. _demos: https://github.com/kddubey/cappr/tree/main/demos

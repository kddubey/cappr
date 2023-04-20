Installation
============

If you intend on using OpenAI models, `sign up`_ for the OpenAI API, and then set the
environment variable ``OPENAI_API_KEY`` \. For zero-shot classification, OpenAI models
are currently ahead of others. But using them will cost ya 💰!

.. _sign up: https://platform.openai.com/signup

Install with ``pip``:

::

   python -m pip install cappr

(Optional) Install requirements for HuggingFace models:

::

   python -m pip install cappr[hf]

(Optional) Install requirements for running demos:

::

   python -m pip install cappr[demos]

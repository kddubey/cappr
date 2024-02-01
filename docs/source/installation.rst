Installation
============

To use Hugging Face models (including `AutoGPTQ <https://github.com/PanQiWei/AutoGPTQ>`_
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

   pip install "cappr[openai]"

(Optional) To use any of the above models::

   pip install "cappr[all]"

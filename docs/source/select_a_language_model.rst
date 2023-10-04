Select a language model
=======================

CAPPr typically works better with larger, instruction-trained models. But it may be able
to `squeeze more out of smaller models
<https://cappr.readthedocs.io/en/latest/future_research.html>`_ than other methods. So
don't sleep on the little Llamas or Mistrals out there.

Besides that, selecting a language model is almost entirely a process of trial and
error, balancing statistical performance with computational constraints. It should be
easy to plug and play though.


HuggingFace
-----------

Here's a quick example (which will download a small GPT-2 model to your computer):

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   from cappr.huggingface.classify import predict

   # Load a model and its corresponding tokenizer
   model_name = "gpt2"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   prompt = "Which planet is closer to the Sun: Mercury or Earth?"
   completions = ("Mercury", "Earth")

   pred = predict(prompt, completions, model_and_tokenizer=(model, tokenizer))
   print(pred)
   # 'Mercury'

So far, CAPPr has been tested for correctness on the following architectures:

- GPT-2
- GPT-J
- GPT-NeoX (including StableLM, and its tuned/instruct and GPTQd versions)
- Llama
- Llama 2 (chat, raw, and its GPTQd versions)
- Mistral.

You'll need access to beefier hardware to run models from the HuggingFace hub, as
:mod:`cappr.huggingface` currently locally loads models. HuggingFace Inference Endpoints
are not yet supported by this package.

:mod:`cappr.huggingface` is not yet compatible with GGML/GGUF models. I'm waiting on
`this issue`_ in ``ctransformers`` to be resolved.

.. _this issue: https://github.com/marella/ctransformers/issues/150


Which CAPPr HuggingFace module should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two CAPPr HuggingFace modules. In general, stick to
:mod:`cappr.huggingface.classify`.

:mod:`cappr.huggingface.classify` is `faster
<https://cappr.readthedocs.io/en/latest/computational_performance.html>`_ than
:mod:`cappr.huggingface.classify_no_cache`, especially when there are a lot of
completions and you're running a model on batches of prompts.

:mod:`cappr.huggingface.classify_no_cache` may happen to be compatible with a slightly
broader class of causal/autoregressive language models. Here, the model is only assumed
to input token/input IDs + attention mask, and then output logits for each input ID.
Furthermore, in the current implementation, this module may be a bit faster when
``batch_size=1``.

.. warning:: For completions which aren't single tokens, the current HuggingFace
   implementation theoretically takes more RAM than it should, even when
   ``batch_size=1``. It doesn't seem bad, but I'll measure and fix it soon.


Examples
~~~~~~~~

For an example of running Llama 2 with CAPPr, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/llama2/copa.ipynb>`_.

For a minimal example of running an `AutoGPTQ <https://github.com/PanQiWei/AutoGPTQ>`_
model, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/auto_gptq.ipynb>`_.

For simple GPT-2 CPU examples, see the **Example** section for each of these functions:

:func:`cappr.huggingface.classify.predict`

:func:`cappr.huggingface.classify.predict_examples`


OpenAI
------

Here's a quick example:

.. code:: python

   from cappr.openai.classify import predict

   prompt = """
   Tweet about a movie: "Oppenheimer was pretty good. But 3 hrs...cmon Nolan."
   This tweet contains the following criticism:
   """.strip("\n")

   completions = ("bad message", "too long", "unfunny")

   pred = predict(prompt, completions, model="text-ada-001")
   print(pred)
   # 'too long'

CAPPr is currently only compatible with `/v1/completions`_ models (because we can
request log-probabilities of tokens in an *inputted* string). Unfortunately, with the
exception of ``davinci-002`` and ``babbage-002`` (weak, non-instruction-trained models),
**OpenAI will deprecate all instruct models on January 4, 2024**.

.. warning:: While ``gpt-3.5-turbo-instruct`` is compatible with `/v1/completions`_, it
   won't support setting `echo=True` and `logprobs=1` after October 5, 2023. So CAPPr
   can't support this model. I don't know why OpenAI is disabling this setting. CAPPr
   with this model `may be SOTA for zero-shot COPA`_ (see the very last section). I
   contacted support. It's low-key kinda sad, yo.

.. _/v1/completions: https://platform.openai.com/docs/models/model-endpoint-compatibility

.. _may be SOTA for zero-shot COPA: https://github.com/kddubey/cappr/blob/main/demos/superglue/copa.ipynb

.. warning:: Currently, :mod:`cappr.openai.classify` must repeat the ``prompt`` for
             however many completions there are. So if your prompt is long and your
             completions are short, you may end up spending much more with CAPPr.
             (:mod:`cappr.huggingface.classify` does not have to repeat the prompt
             because it caches its representation.)


Examples
~~~~~~~~

Great zero-shot COPA performance is achieved in `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/superglue/copa.ipynb>`_.

Great zero-shot WSC performance with ``text-curie-001`` is achieved in `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/superglue/wsc.ipynb>`_.

For simple examples, see the **Example** section for each of these functions:

:func:`cappr.openai.classify.predict`

:func:`cappr.openai.classify.predict_examples`

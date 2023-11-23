Select a language model
=======================

CAPPr typically works better with larger, instruction-trained models. But it may be able
to `squeeze more out of smaller models
<https://cappr.readthedocs.io/en/latest/statistical_performance.html>`_ than other
methods. So don't sleep on the little Llamas or Mistrals out there, especially if
they've been trained for your application.

Besides that, selecting a language model is almost entirely a process of trial and
error, balancing statistical performance with computational constraints. It should be
easy to plug and play though.

For CAPPr, `GPTQ models <https://huggingface.co/models?sort=trending&search=gptq>`_ are
the most computationally performant. `Mistral trained on OpenOrca
<https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ>`_ is statistically
performant. These models are compatible with :mod:`cappr.huggingface.classify`.


HuggingFace
-----------

To work with models which implement the ``transformers`` CausalLM interface, including
`AutoGPTQ`_ and `AutoAWQ`_ models, CAPPr depends on the ``transformers`` package. Search
the `HuggingFace model hub <https://huggingface.co/models>`_ for these models.

.. note:: For ``transformers>=4.32.0``, GPTQ models `can be loaded
          <https://huggingface.co/docs/transformers/main/en/main_classes/quantization#autogptq-integration>`_
          using ``transformers.AutoModelForCausalLM.from_pretrained``.

Here's a quick example (which will download a small GPT-2 model to your computer):

.. code:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer
   from cappr.huggingface.classify import predict

   # Load a model and its tokenizer
   model_name = "gpt2"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)

   prompt = "Which planet is closer to the Sun: Mercury or Earth?"
   completions = ("Mercury", "Earth")

   pred = predict(prompt, completions, model_and_tokenizer=(model, tokenizer))
   print(pred)
   # Mercury

So far, CAPPr has been tested for code correctness on the following architectures:

- GPT-2
- GPT-J
- GPT-NeoX (including StableLM)
- Llama, Llama 2
- Mistral.

You'll need access to beefier hardware to run models from the HuggingFace hub, as
:mod:`cappr.huggingface` currently assumes you've locally loaded the model. HuggingFace
Inference Endpoints are not yet supported by this package.

``ctransformers`` model objects are not yet supported. (I think I'm just waiting on
`this issue <https://github.com/marella/ctransformers/issues/150>`_.)

``vllm`` model objects are not yet supported.


Which CAPPr HuggingFace module should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two CAPPr HuggingFace modules. In general, stick to
:mod:`cappr.huggingface.classify`.

:mod:`cappr.huggingface.classify` has the greatest `throughput
<https://cappr.readthedocs.io/en/latest/computational_performance.html>`_. By default,
prompts are processed two-at-a-time and completions are processed in parallel. These
settings are controlled by the ``batch_size`` and ``batch_size_completions`` keyword
arguments, respectively. Decreasing their values decreases peak memory usage but costs
runtime. Increasing their values decreases runtime but costs memory.

:mod:`cappr.huggingface.classify` can also cache shared instructions for prompts,
resulting in a modest speedup. See :func:`cappr.huggingface.classify.cache_model`.

:mod:`cappr.huggingface.classify_no_cache` may be compatible with a slightly
broader class of architectures and model interfaces. Here, the model is only assumed to
input token/input IDs and an attention mask, and then output logits for each input ID.

.. note:: For ``transformers>=4.35.0``, AWQ models `can be loaded
          <https://huggingface.co/docs/transformers/main/en/main_classes/quantization#awq-integration>`_
          using ``transformers.AutoModelForCausalLM.from_pretrained``. AWQ models
          loaded this way are compatible with :mod:`cappr.huggingface.classify`.

In particular, :mod:`cappr.huggingface.classify_no_cache` is compatible with models
loaded via:

.. code:: python

   from awq import AutoAWQForCausalLM

   model = AutoAWQForCausalLM.from_quantized(
      model_id,
      ...,
      batch_size=batch_size_completions,
   )
   model.device = "cuda"


Examples
~~~~~~~~

For an example of running Llama 2, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/superglue/copa.ipynb>`_.

For an example of running an `AutoGPTQ`_ Mistral model, where we cache shared prompt
instructions to save time and batch completions to save memory, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/craigslist_bargains.ipynb>`_.

For a minimal example of running an `AutoAWQ`_ Mistral model, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/autoawq.ipynb>`_.

For minimal examples you can quickly run, see the **Example** section for each of these
functions:

:func:`cappr.huggingface.classify.predict`

:func:`cappr.huggingface.classify.predict_examples`

.. _AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ

.. _AutoAWQ: https://github.com/casper-hansen/AutoAWQ


Llama CPP
---------

To work with models stored in the GGUF format, CAPPr depends on the `llama-cpp-python
<https://github.com/abetlen/llama-cpp-python>`_ package. Search the `HuggingFace model
hub <https://huggingface.co/models?sort=trending&search=gguf>`_ for these models.

Here's a quick example (which assumes you've downloaded `this 6 MB model
<https://huggingface.co/aladar/TinyLLama-v0-GGUF>`_):

.. code:: python

   from llama_cpp import Llama
   from cappr.llama_cpp.classify import predict

   # Load model
   model = Llama("./TinyLLama-v0.Q8_0.gguf", verbose=False)

   prompt = """Gary told Spongebob a story:
   There once was a man from Peru; who dreamed he was eating his shoe. He
   woke with a fright, in the middle of the night, to find that his dream
   had come true.

   The moral of the story is to"""

   completions = (
      "look at the bright side",
      "use your imagination",
      "eat shoes",
   )

   pred = predict(prompt, completions, model)
   print(pred)
   # use your imagination


Examples
~~~~~~~~

For an example of running Llama 2 on the COPA challenge, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb>`_.

For an example of running Llama 2 on the AG News challenge, where we cache shared prompt
instructions to save time, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/ag_news.ipynb>`_.

For minimal examples you can quickly run, see the **Example** section for each of these
functions:

:func:`cappr.llama_cpp.classify.predict`

:func:`cappr.llama_cpp.classify.predict_examples`


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
   # too long

CAPPr is currently only compatible with `/v1/completions`_ models where
log-probabilities of *inputted* tokens can be requested, via `echo=True, logprobs=1`. On
January 4, 2024, OpenAI will deprecate all of these models except ``davinci-002`` and
``babbage-002``—weak, non-instruction-trained models. While ``gpt-3.5-turbo-instruct``
is compatible with `/v1/completions`_, this model stopped supporting `echo=True,
logprobs=1` on October 5, 2023. So CAPPr can't support this model.

.. _/v1/completions: https://platform.openai.com/docs/models/model-endpoint-compatibility

.. warning:: Currently, :mod:`cappr.openai.classify` must repeat the ``prompt`` for
             however many completions there are. So if your prompt is long and you have
             many completions, you may end up spending much more with CAPPr.
             (:mod:`cappr.huggingface.classify` and :mod:`cappr.llama_cpp.classify` do
             not repeat the prompt because they cache its representation.)


Examples
~~~~~~~~

`COPA <https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/copa.ipynb>`_

`WSC <https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/wsc.ipynb>`_

Decent performance on RAFT training sets is demonstrated in `these notebooks
<https://github.com/kddubey/cappr/blob/main/demos/openai/raft>`_.

For minimal examples you can quickly run, see the **Example** section for each of
these functions:

:func:`cappr.openai.classify.predict`

:func:`cappr.openai.classify.predict_examples`

Select a language model
=======================

CAPPr typically works better with larger, instruction-trained models. But it may be able
to `squeeze more out of smaller models
<https://cappr.readthedocs.io/en/latest/future_research.html>`_ than other methods. So
don't sleep on the little Llamas or Mistrals out there, especially if they've been
trained for your domain/application.

Besides that, selecting a language model is almost entirely a process of trial and
error, balancing statistical performance with computational constraints. It should be
easy to plug and play though.


HuggingFace
-----------

To work with models which implement the PyTorch ``transformers`` CausalLM interface,
including `AutoGPTQ`_ and `AutoAWQ`_ models, CAPPr depends on the `transformers
<https://github.com/huggingface/transformers>`_ package. You can search the `HuggingFace
model hub <https://huggingface.co/models?library=pytorch>`_ for these models.

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
   # Mercury

So far, CAPPr has been tested for correctness on the following architectures:

- GPT-2
- GPT-J
- GPT-NeoX (including StableLM, and its tuned/instruct and GPTQd versions)
- Llama
- Llama 2 (chat, raw, and its GPTQd versions)
- Mistral.

You'll need access to beefier hardware to run models from the HuggingFace hub, as
:mod:`cappr.huggingface` currently assumes you've locally loaded the model. HuggingFace
Inference Endpoints are not yet supported by this package.

``ctransformers`` model objects are not yet supported. (I think I'm just waiting on
`this issue <https://github.com/marella/ctransformers/issues/150>`_.)

``vllm`` model objects are not yet supported.


Which CAPPr HuggingFace module should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are three CAPPr HuggingFace modules.

:mod:`cappr.huggingface.classify` has the greatest `throughput
<https://cappr.readthedocs.io/en/latest/computational_performance.html>`_, but costs a
bit more memory. Furthermore, this module is currently not compatible with `AutoAWQ`_
models.

:mod:`cappr.huggingface.classify_no_cache` is compatible with `AutoAWQ`_ models. In
general, it may be compatible with a slightly broader class of causal/autoregressive
architectures. Here, the model is only assumed to input token/input IDs and an attention
mask, and then output logits for each input ID. Furthermore, in the current
implementation, this module may be a bit faster when ``batch_size=1``, where the model
processes one prompt at a time.

.. note:: In the above modules, the ``batch_size`` keyword argument refers to the number
   of prompts that are processed at a time; completions are always processed in
   parallel.

:mod:`cappr.huggingface.classify_no_batch` is compatible with all models which
:mod:`cappr.huggingface.classify_no_cache` is compatible with. The difference is that
the no-batch module has the model process one prompt-completion pair at a time,
minimizing memory usage.

.. warning:: When instantiating your `AutoAWQ`_ model, you must set an extra attribute
             indicating the device(s) which the model is on, e.g.,
             ``model.device = "cuda"``.

.. note:: If you're using an `AutoAWQ`_ model, pass ``batch_size=len(completions)`` to
          the model's initialization. If you're processing :class:`cappr.Example`
          objects with a non-constant number of :attr:`cappr.Example.completions`, then
          leave out the ``batch_size`` argument from the model's initialization (or,
          equivalently, set it to 1) and use :mod:`cappr.huggingface.classify_no_batch`
          instead of :mod:`cappr.huggingface.classify_no_cache`.


Examples
~~~~~~~~

For an example of running Llama 2, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/superglue/copa.ipynb>`_.

For a minimal example of running an `AutoGPTQ`_ StableLM model, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/auto_gptq.ipynb>`_.

For a minimal example of running an `AutoAWQ`_ Mistral model, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/autoawq.ipynb>`_.

For simple GPT-2 CPU examples, see the **Example** section for each of these functions:

:func:`cappr.huggingface.classify.predict`

:func:`cappr.huggingface.classify.predict_examples`

.. _AutoGPTQ: https://github.com/PanQiWei/AutoGPTQ

.. _AutoAWQ: https://github.com/casper-hansen/AutoAWQ


Llama CPP
---------

To work with models stored in the GGUF format, CAPPr depends on the `llama-cpp-python
<https://github.com/abetlen/llama-cpp-python>`_ package. You can search the `HuggingFace
model hub <https://huggingface.co/models?sort=trending&search=gguf>`_ for these models.

.. note:: When instantiating your Llama, set ``logits_all=True``.

Here's a quick example (which assumes you've downloaded `this 6 MB model
<https://huggingface.co/aladar/TinyLLama-v0-GGUF/blob/main/TinyLLama-v0.Q8_0.gguf>`_):

.. code:: python

   from llama_cpp import Llama
   from cappr.llama_cpp.classify import predict

   # Load model. Always set logits_all=True for CAPPr
   model = Llama("./TinyLLama-v0.Q8_0.gguf", logits_all=True, verbose=False)

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

So far, CAPPr has been tested for correctness on GGUF models which use SentencePiece
tokenization, e.g., Llama. I'll test on models which use BPE soon. I think you may just
need to add a space before each completion string.


Examples
~~~~~~~~

For an example of running Llama 2 on the COPA challenge, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb>`_.

For an example of running Llama 2 on the AG News challenge, where instructions are
cached, see `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/ag_news.ipynb>`_.

For simple examples, see the **Example** section for each of these functions:

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

CAPPr is currently only compatible with `/v1/completions`_ models (because we can
request log-probabilities of tokens in an *inputted* string). Unfortunately, with the
exception of ``davinci-002`` and ``babbage-002`` (weak, non-instruction-trained models),
**OpenAI will deprecate all instruct models on January 4, 2024**. While
``gpt-3.5-turbo-instruct`` is compatible with `/v1/completions`_, it won't support
setting `echo=True` and `logprobs=1` after October 5, 2023. So CAPPr can't support this
model.

.. _/v1/completions: https://platform.openai.com/docs/models/model-endpoint-compatibility

.. warning:: Currently, :mod:`cappr.openai.classify` must repeat the ``prompt`` for
             however many completions there are. So if your prompt is long and your
             completions are short, you may end up spending much more with CAPPr.
             (:mod:`cappr.huggingface.classify` and :mod:`cappr.llama_cpp.classify` do
             not repeat the prompt because they cache its representation.)


Examples
~~~~~~~~

Great zero-shot COPA performance is achieved in `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/copa.ipynb>`_.

Great zero-shot WSC performance with ``text-curie-001`` is achieved in `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/wsc.ipynb>`_.

Decent performance on RAFT training sets is demonstrated in `these notebooks
<https://github.com/kddubey/cappr/blob/main/demos/openai/raft>`_.

For simple examples, see the **Example** section for each of these functions:

:func:`cappr.openai.classify.predict`

:func:`cappr.openai.classify.predict_examples`

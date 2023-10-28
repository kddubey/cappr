Computational performance
=========================

One concern was that CAPPr requires as many model calls as there are classes. But in the
CAPPr scheme, we can cache each attention block's keys and values for the prompts. This
feature is already supported by ``AutoModelForCausalLM``\ s. See `this module`_ for the
implementation. Note that this caching is not implemented for OpenAI models. So if
you're running :mod:`cappr.openai.classify` functions, you'll be on the *cappr (no
cache)* line :-(

.. _this module: https://github.com/kddubey/cappr/blob/main/src/cappr/huggingface/classify.py

.. figure:: _static/scaling_classes/batch_size_32.png
   :align: center

   `COPA`_ dataset, repeating the choices to simulate multi-class classification tasks.
   `GPT-2 (small)`_ was run on a T4 GPU. 96 classification inputs were processed in
   batches of size 32. For a controlled runtime comparison, GPU RAM was held (roughly)
   constant for each method and each number of classes. Each point in the graph is a
   median of 5 runs. For text generation, exactly 4 tokens were generated for each
   prompt, which is the number of tokens in ``'\n\nAnswer A'``. 1-token times are also
   shown. But for COPA (and other multiple-choice style prompts), that may result in
   lower zero-shot accuracy, as most of the sampled choices come after the first token.

.. _COPA: https://people.ict.usc.edu/~gordon/copa.html

.. _GPT-2 (small): https://huggingface.co/gpt2

See `this notebook
<https://github.com/kddubey/cappr/blob/main/demos/computational_analysis.ipynb>`_ for
the code which produced the figure above.

.. note:: For :mod:`cappr.llama_cpp.classify`, batch inference currently isn't possible.
          As a result, text generation is typically faster when there are many
          completions.


Weaknesses
----------

CAPPr does not computationally perform well when there are 10s of classes and the prompt
is so long that only one prompt fits in memory during processing. In these cases,
CAPPr's memory requirements are higher because :mod:`cappr.huggingface.classify`
currently processes completions in parallel. :mod:`cappr.huggingface.classify_no_batch`
minimizes memory but costs a lot of time because it processes each completion one at a
time.

In the future, 2 things will be explored:

1. Revamp :mod:`cappr.huggingface.classify` by batching completions and supporting
   sub-prompt caching like :func:`cappr.huggingface.classify_no_batch.cache`. Batching
   completions trades off runtime for reduced memory requirements. Sub-prompt caching
   significantly decreases runtime.
2. Are there classification tasks where classes don't need to be provided in context
   (and instead as a completion) for CAPPr to statistically perform well? If so, CAPPr's
   computational issues can be worked around through better prompt engineering. And the
   model's context window can be reduced.

The first thing is tracked by `this ticket
<https://github.com/users/kddubey/projects/1/views/1?pane=issue&itemId=42888520>`_.

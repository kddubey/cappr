Computational performance
=========================

One concern was that CAPPr requires as many model calls as there are classes. But in the
CAPPr scheme, we can simply cache each attention block's keys and values for the
prompts. This feature is already supported by ``AutoModelForCausalLM``\ s. See `this
module`_ for the implementation. Note that this caching is not implemented for OpenAI
models, as I can't control their backend. So if you're running
:mod:`cappr.openai.classify` functions, you'll be on the *cappr (no cache)* line :-(

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


Future work
-----------

Significant memory savings can be achieved by improving the current implementation of
batching. Currently, all of the completions are ran in parallel per prompt. In the
future, I'll enable completions to be batched as well; each prompt-completion pair will
be processed one at a time. This should result in significant memory savings over text
generation because a CAPPr prompt is typically much smaller than a text generation
prompt. A text generation prompt must describe each class in the prompt, and the model
must attend to all of this information to perform well. A CAPPr prompt doesn't need to
include info about the classes to perform well. And a completion is just one of the
classes. As a result, model context and memory requirements can be relaxed.

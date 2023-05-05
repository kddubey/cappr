Computational performance
=========================

One concern was that CAPPr requires as many model calls as there are classes. But in the
CAPPr scheme, we can simply cache each attention block's keys and values for the
prompts. This feature is already supported by ``AutoModelForCausalLM``\ s. See `this
code`_ for the implementation. Note that this caching is not implemented for OpenAI
models, as I can't control their backend. This means that when running
:mod:`cappr.openai.classify` functions, you'll be on the *cappr (no cache)* line :-(

.. _this code: https://github.com/kddubey/cappr/blob/main/src/cappr/huggingface/classify.py

.. figure:: _static/scaling_classes/batch_size_32.png
   :align: center

   `COPA`_ dataset, repeating the choices to simulate multi-class classification tasks.
   `GPT-2 (small)`_ was run on a Tesla K80 GPU (whatever was free in Google Colab in
   March 2023). 96 classification inputs were processed in batches of size 32. Each
   point in the graph is a median of 5 runs. For classification via sampling (CVS),
   exactly 4 tokens were generated for each prompt, which is the number of tokens in
   ``'\n\nAnswer A'``. 1-token times are also shown. But for COPA (and other
   multiple-choice style prompts), that may result in lower zero-shot accuracy, as most
   of the sampled choices come after the first token.

.. _COPA: https://people.ict.usc.edu/~gordon/copa.html

.. _GPT-2 (small): https://huggingface.co/gpt2

See the `this notebook`_ for the experiment code which produced the figure above.

.. _this notebook: https://github.com/kddubey/cappr/blob/main/demos/computational_analysis.ipynb

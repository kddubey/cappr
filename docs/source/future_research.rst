The Misfit Toys Hypothesis
==========================

Modern language models undergo two training stages: pretraining, and instruction
training. When solving a classification task, it's tempting to lean on instruction-style
prompts in combination with text generation. This combination works incredibly well for
multi-GPU, proprietary models. But what about open source ones? Perhaps there are
untapped, smaller, or undertrained models which are not good at generating text from
instructions, but are good at estimating probabilities.

A handful of experiments suggest that CAPPr squeezes more out of smaller LLMs. In the
`OpenAI COPA demo
<https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/copa.ipynb>`_, text
generation using OpenAI's smaller model, ``text-curie-001``, is less than 50% accurate,
while CAPPr using the same model is 80% accurate. Similar results can be seen in:

- the 4-bit 4 GB `Llama 2 COPA demo
  <https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb>`_
- the 4-bit 4 GB `Llama 2 AG News demo
  <https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/ag_news.ipynb>`_
- this (minimal but surprising) 3 GB `StableLM demo
  <https://github.com/kddubey/cappr/blob/main/demos/huggingface/auto_gptq.ipynb>`_.

I'll study how replicable this result is across datasets, model sizes, architectures,
and levels of quantization.

The `calibration`_ of CAPPr estimates has not yet been studied. These estimates are
slightly different than usual next-token probability estimates because:

#. CAPPr hackily takes a mean over next-token probabilities

#. CAPPr incorporates a prior specific to your classification data.

.. _calibration: https://en.wikipedia.org/wiki/Probabilistic_classification#Probability_calibration

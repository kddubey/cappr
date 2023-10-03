Future research
===============

I'm curious to see how much statistically easier estimation is than sampling/generation.
I think estimation could be easier simply because models are trained to maximize
likelihood; they're trained to produce accurate probabilities. And a fundamental idea
from machine learning is that good probability estimation does not necessarily imply
good discrimination. Perhaps there are untapped, smaller, or undertrained models which
are good at estimation, but not good at discrimination.

A handful of experiments suggest that CAPPr squeezes more out of smaller LLMs. In the
`OpenAI COPA demo`_, text generation using OpenAI's ``text-curie-001`` is less than 50%
accurate, while CAPPr using the same model is 80% accurate. Similar results can be seen
in:

- the 4 GB `Llama 2 COPA demo`_
- this (minimal but surprising) 3 GB `StableLM demo`_.

.. _OpenAI COPA demo: https://github.com/kddubey/cappr/blob/main/demos/superglue/copa.ipynb
.. _Llama 2 COPA demo: https://github.com/kddubey/cappr/blob/main/demos/llama2/copa.ipynb
.. _StableLM demo: https://github.com/kddubey/cappr/blob/main/demos/auto_gptq.ipynb

The `calibration`_ of CAPPr estimates has not yet been studied. These estimates are
slightly different than usual next-token probability estimates because:

#. CAPPr hackily takes a mean over next-token probabilities

#. CAPPr incorporates a prior specific to your classification data.

.. _calibration: https://en.wikipedia.org/wiki/Probabilistic_classification#Probability_calibration

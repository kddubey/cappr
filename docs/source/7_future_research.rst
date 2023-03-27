Future research
===============

I have a somewhat unjustified hypothesis that CAPPr squeezes more out of smaller or
less-heavily trained LMs. This hypothesis is based on just 2 experiments, one of which
was unfortunately done on a private dataset. The public experiment is the `COPA demo`_.
It demonstrates that classification-via-sampling using ``text-curie-001`` (a smaller
GPT-3 model) performs worse than random guessing, while CAPPr using ``text-curie-001``
is 80% accurate.

.. _COPA demo: <https://github.com/kddubey/cappr/blob/main/demos/copa.ipynb>`__.

The `calibration`_ of CAPPr estimates has not yet been studied. These estimates are
slightly different than usual next-token probability estimates because:

#. CAPPr hackily takes a mean over next-token probabilities

#. CAPPr incorporates a prior specific to your classification data.

.. _calibration: https://en.wikipedia.org/wiki/Probabilistic_classification#Probability_calibration

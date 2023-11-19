(Optional) Supply a prior
=========================

A prior is a probability distribution over completions indicating how likely you think
each completion is *regardless of the prompt*. It nudges language model probabilities
towards the domain-specific probabilities which are needed to make optimal predictions.

If you have a handful of examples whose correct class/choice is known, then you may
simply compute the fraction of examples belonging to each class, e.g.,

.. code:: python

   # class_labels[i] is the index of the class which example i belongs to
   # There are 3 possible classes, indexed as 0, 1, and 2
   class_labels = [0, 0, 0, 1, 1, 1, 1, 1, 2]

   # prior[k] is the observed fraction of examples which belong to class k
   prior = [3/9, 5/9, 1/9]

There are better but slighly more complicated ways to estimate a prior, e.g., `additive
smoothing <https://en.wikipedia.org/wiki/Additive_smoothing>`_. A prior may be guessed
based on domain knowledge.

If you have absolutely no idea what a reasonable prior could be, then leave out the
``prior`` keyword argument for ``predict`` and ``predict_proba`` functions.


Examples
--------

See the `Banking 77 demo
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/banking_77_classes.ipynb>`_.

For a minimal example of using a prior, see the **Example** section for this function:

:func:`cappr.huggingface.classify.predict`

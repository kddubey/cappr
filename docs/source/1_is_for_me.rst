Is this for me?
===============

CAPPr is for you if you:

#. Want to do *zero-shot* text classification

#. Can pay to use OpenAI models, or you can locally load HuggingFace models (`Inference
   Endpoints`_ are not yet supported by this package).

.. _Inference Endpoints: https://huggingface.co/docs/inference-endpoints/index


Is zero-shot for me?
--------------------

Yes if you only have a few, e.g., 200, labeled examples for a text classification task,
and it's difficult to label more. In such a setting, you should only be using those
labeled examples to evaluate or select among one or two models or prompt formats.\ [#]_

If you're something of an ML engineer and you can label a few hundred examples, you
should strongly consider trying out few-shot methods which use open source language
models. These methods include:

#. `SetFit <https://github.com/huggingface/setfit>`_

#. `Pattern-Exploiting Training <https://github.com/timoschick/pet>`_

#. `Plain old BERT embeddings
   <https://huggingface.co/transformers/v3.3.1/training.html>`_.

You should also consider how "subtle" the association is between your input texts and
their labels. Few-shot methods may be able to pick up more subtle or internally
specified associations between texts and their labels. Zero-shot methods using LLMs may
be better at tasks which require more factual knowledge.\ [#]_


Footnotes
~~~~~~~~~

.. [#] Some quick-and-dirty rationale for this guidance: a Wald 68% confidence interval
   for the expected error rate of a binary classifier—which is estimated to be 80%
   accurate—is (0.72, 0.88) when evaluated on an independent set of 100 labeled
   examples. That's quite wide. Ideally, for zero-shot problems, you still have 200 or
   so labeled examples. I like to separate labeled examples into a set of 50 for
   training and 150 for test. Those 50 are used to estimate a prior, select a prompt
   format, and select an LM. The other 150 are only used for evaluation.

.. [#] You may be interested in the Real-World Few-Shot Text Classification (RAFT)
   benchmark. See the leaderboard `here
   <https://huggingface.co/spaces/ought/raft-leaderboard>`_, and CAPPr's zero-shot
   performance on the RAFT training set `here
   <https://github.com/kddubey/cappr/blob/main/demos/raft>`_.

Related work
============

The idea of aggregating token probabilities is well known. You'll find it as a
subroutine in papers from `GPT-2
<https://paperswithcode.com/paper/language-models-are-unsupervised-multitask>`_\ [#]_ to
`Self-Consistency <https://arxiv.org/abs/2203.11171>`_\ [#]_. The ``cappr``
implementation includes a few computational and statistical optimizations, while
maintaining a simple interface.

Here are some papers which focus on the idea of aggregating token probabilities.

`This paper <https://arxiv.org/abs/1806.02847>`_\ [#]_ presents a transposed version of
CAPPr. Its method was used in CAPPr's `demo for the Winograd Schema Challenge
<https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/wsc.ipynb>`_.

`PET with multiple masks <https://arxiv.org/abs/2009.07118>`_\ [#]_ also aggregates
token probabilities to do prompt-completion classification. But these probabilities are
assumed to come from masked language models like BERT.

References
----------

.. [#] Radford, Alec, et al. "Language models are unsupervised multitask learners."
    OpenAI blog 1.8 (2019): 9.

.. [#] Wang, Xuezhi, et al. "Self-consistency improves chain of thought reasoning in
    language models." arXiv preprint arXiv:2203.11171 (2022).

.. [#] Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning."
    arXiv preprint arXiv:1806.02847 (2018).

.. [#] Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small
    language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).

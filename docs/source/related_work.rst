Related work
============

The idea of aggregating token log-probabilities is well known. You'll find it as a
subroutine in papers from `GPT-2
<https://paperswithcode.com/paper/language-models-are-unsupervised-multitask>`_\ [#]_ to
`Self-Consistency <https://arxiv.org/abs/2203.11171>`_\ [#]_ to `InPars
<https://arxiv.org/abs/2202.05144>`_\ [#]_ to `hallucination detection
<https://arxiv.org/abs/2208.05309>`_\ [#]_ to `SimPO
<https://arxiv.org/abs/2405.14734>`_\ [#]_. The ``cappr`` implementation includes a few
computational and statistical optimizations, while maintaining a simple interface.

Here are some papers which focus on the idea of aggregating token log-probabilities.

`This paper <https://arxiv.org/abs/1806.02847>`_\ [#]_ presents a transposed version of
CAPPr. Its method was used in CAPPr's `demo for the Winograd Schema Challenge
<https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/wsc.ipynb>`_.

`PET with multiple masks <https://arxiv.org/abs/2009.07118>`_\ [#]_ also aggregates
token log-probabilities to do prompt-completion classification. But these
log-probabilities are assumed to come from masked language models like BERT.

References
----------

.. [#] Radford, Alec, et al. "Language models are unsupervised multitask learners."
    OpenAI blog 1.8 (2019): 9.

.. [#] Wang, Xuezhi, et al. "Self-consistency improves chain of thought reasoning in
    language models." arXiv preprint arXiv:2203.11171 (2022).

.. [#] Bonifacio, Luiz, et al. "Inpars: Data augmentation for information retrieval
    using large language models." arXiv preprint arXiv:2202.05144 (2022).

.. [#] Guerreiro, Nuno M., Elena Voita, and André FT Martins. "Looking for a needle in a
    haystack: A comprehensive study of hallucinations in neural machine translation."
    arXiv preprint arXiv:2208.05309 (2022).

.. [#] Meng, Yu, Mengzhou Xia, and Danqi Chen. "SimPO: Simple Preference Optimization
    with a Reference-Free Reward." arXiv preprint arXiv:2405.14734 (2024).

.. [#] Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning."
    arXiv preprint arXiv:1806.02847 (2018).

.. [#] Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small
    language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).

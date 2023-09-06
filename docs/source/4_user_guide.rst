User Guide
==========

There are three factors which influence the performance of CAPPr: the prompt-completion
format, the language model, and the prior.


Select a prompt-completion format
---------------------------------

With CAPPr, your job is to write up your classification task as a string with three
components:

.. code:: python

   {prompt}{end_of_prompt}{completion}

The text you want to classify should appear in the ``prompt``. One of the classes which
the text could belong to should appear in the ``completion``. These are not hard rules.
For example, the `demo for the Winograd Schema Challenge`_ flips the roles of the
``prompt`` and ``completion``. (Just don't use the ``prior`` keyword argument in that
case.)

.. _demo for the Winograd Schema Challenge: https://github.com/kddubey/cappr/blob/main/demos/superglue/wsc.ipynb

One rule is that the ``completion`` text should flow naturally after
``{prompt}{end_of_prompt}``. So pay close attention to the use of white spaces,
newlines, and word casing. :mod:`cappr.openai.classify` does not do any string
processing for you: **it just concatenates the three strings and sends it**!

And yes, you'll likely need to do prompt engineering. It's mostly a matter of trial and
error. Here's an `external guide`_ if you'd like to survey research in this field.\ [#]_
Step-by-step\ [#]_ and chain-of-thought prompting\ [#]_ are highly effective for slighly
more complex classification tasks. While CAPPr is not immediately well-suited to these
sorts of prompts, it may be applied to post-process completions:

1. Get the completion from the step-by-step / chain-of-thought prompt

2. Pass this completion in a second prompt, and have CAPPr classify the answer. You can
   probably get away with using a less expensive model for this task, as it just takes a
   bit of semantic parsing.

.. _external guide: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

Here's an example of these two steps:

.. code:: python

   from cappr.openai.api import gpt_chat_complete
   from cappr.openai.classify import predict

   prompt_raw = """
   Hi Professor. I'm interested in taking ML-101, but I'm struggling to decide which
   course I need to take before that. I've already taken CS-101. Which course should I
   take next?

   Here's a list of courses and their prerequisites which I pulled from the course
   catalog.

   CS-101: no prerequisites
   CS-102: CS-101, MATH-101
   MATH-101: no prerequisites
   MATH-102: MATH-101
   ML-101: CS-101, MATH-102, STAT-101
   STAT-101: MATH-101
   STAT-102: STAT-101, MATH-102
   """

   prompt_step_by_step = prompt_raw + "\n" + "Let's think step by step."

   chat_api_response = gpt_chat_complete(
      texts=[prompt_step_by_step],
      model="gpt-4",
      system_msg=(
         "You are a computer scientist mentoring a student. End your response to "
         "the student's question with the final answer, which is the name of a course."
      ),
      max_tokens=1_000,
      temperature=0,
   )

   step_by_step_answer = chat_api_response[0]["message"]["content"]

   prompt_answer = f'''
   Here is an answer about which course a student needs to take:

   """
   {step_by_step_answer}
   """

   According to this answer, the very next course that the student should take is
   '''

   class_names = (
      "CS-101",
      "CS-102",
      "MATH-101",
      "MATH-102",
      "ML-101",
      "STAT-101",
      "STAT-102",
   )

   answer = predict(
      prompts=[prompt_answer],
      completions=class_names,
      model="text-ada-001",
   )

   print(answer)
   # ['MATH-101']

.. warning:: Currently, :mod:`cappr.openai.classify` must repeat the ``prompt`` for
             however many completions there are. So if your prompt is long and your
             completions are short, you may end up spending much more with CAPPr.
             (:mod:`cappr.huggingface.classify` does not have to repeat the prompt
             because it caches its representation.)

Note that while all of the examples in the documentation are zero-shot prompts, nothing
about CAPPr prevents you from using few-shot prompts. Just make sure you're not paying
too much for a small benefit.


Select a language model
-----------------------

CAPPr typically requires larger language models, as it's a zero-shot method. For OpenAI
models, there's some rough guidance `here
<https://platform.openai.com/docs/models/overview>`_. Other than that, selecting a
language model is almost entirely a process of trial and error. It should be easy to
plug and play though.

Should I use OpenAI or HuggingFace models?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, OpenAI models perform better. But I'll try to document competitive,
instruction-trained LMs which are hosted on HuggingFace as more are released. For now,
you'll need access to beefier hardware to run them, as :mod:`cappr.huggingface`
currently locally loads HuggingFace models.

.. warning:: Some of OpenAI's `GPT-3.5+ models`_ currently don't return token
   probabilities, so they currently can't be used by CAPPr. I hope this changes soon.

.. _GPT-3.5+ models: https://platform.openai.com/docs/models/gpt-3-5

.. note:: `HuggingFace Inference Endpoints`_ are not yet supported by this package.
.. _HuggingFace Inference Endpoints: https://huggingface.co/docs/inference-endpoints/index


(Optional) Supply a prior
-------------------------

A prior is a marginal probability distribution over the classes in your classification
problem. It nudges language model probabilities towards the conditional class
probabilities which are needed to make optimal predictions.

If you have a handful of labeled examples for each possible class, then you may simply
compute the fraction of examples belonging to each class, e.g.,

.. code:: python

   # class_labels[i] is the index of the class which example i belongs to
   class_labels = [0, 0, 0, 1, 1, 1, 1, 1, 2]

   # prior[k] is the observed fraction of examples which belong to class k
   prior = [3/9, 5/9, 1/9]

There are better but slighly more complicated ways to estimate a prior, e.g., `additive
smoothing <https://en.wikipedia.org/wiki/Additive_smoothing>`_.

You may also simply guess a prior if you have some domain knowledge. If you have
absolutely no idea what a reasonable prior could be, then you may leave out the
``prior`` keyword argument for this package's ``predict`` and ``predict_proba``
functions. In this case, a uniform prior is assumed.


References
----------

.. [#] Weng, Lilian. (Mar 2023). Prompt Engineering. Lil'Log.
   https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/.

.. [#] Kojima, Takeshi, et al. "Large language models are zero-shot reasoners." arXiv
    preprint arXiv:2205.11916 (2022).

.. [#] Wei, Jason, et al. "Chain of thought prompting elicits reasoning in large
    language models." arXiv preprint arXiv:2201.11903 (2022).

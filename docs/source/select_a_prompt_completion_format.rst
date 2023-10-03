Select a prompt-completion format
=================================

With CAPPr, your job is to write up your classification task as a string with three
components:

.. code:: python

   {prompt}{end_of_prompt}{completion}

The text you want to classify should appear in the ``prompt``. One of the classes which
the text could belong to should appear in the ``completion``.\ [#]_

The ``completion`` string should flow naturally after ``{prompt}{end_of_prompt}``. So
pay close attention to the use of white spaces, newlines, and word casing. CAPPr doesn't
do any string processing for you; **it just concatenates the three strings and sends
it**! For each completion in your list of possible completions/choices, consider
printing ``{prompt}{end_of_prompt}{completion}`` to ensure it passes the eye test.

And yes, you'll likely need to do a bit of prompt engineering. It's mostly a matter of
trial and error. (Here's an `external guide`_ if you'd like to survey research in this
field.\ [3]_) Empirically, the impact of the prompt-completion format on accuracy
depends on the quality of the language model. For larger, instruction-trained models,
the format is not too consequential (they've seen it all!). For smaller, less
instruction-trained models, it can be critical to get the format right.

.. _external guide: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

Feel free to `open an issue`_ if you're having trouble with writing a prompt. It's not
always an easy task, and models can be surprisingly finicky.

.. _open an issue: https://github.com/kddubey/cappr/issues


Prompt-completion formats
-------------------------

Most successful prompts belong to one of three categories: Concat-Class, Yes-No,
Multiple Choice.


Concat-Class
~~~~~~~~~~~~

In this format, the full name of the class is directly used as the completion. In other
words, the class name is directly concatenated after ``{prompt}{end_of_prompt}`` Here's
an example:

.. code:: python

   prompts = [
      "Stephen Curry is a",
      "Martina Navratilova was a",
      "Dexter, from the TV Series Dexter's Laboratory, is a",
      "LeBron James is a",
   ]
   end_of_prompt = " "

   # Each of the prompts could possibly be completed with one of these:
   class_names = ("basketball player", "tennis player", "scientist")

   # What strings will CAPPr see?
   for i, prompt in enumerate(prompts):
      print(f"For prompt {i + 1}")
      print("------------")
      for completion in class_names:
         print(f"{prompt}{end_of_prompt}{completion}")
      print()
   
   # For prompt 1
   # ------------
   # Stephen Curry is a basketball player
   # Stephen Curry is a tennis player
   # Stephen Curry is a scientist
   # 
   # For prompt 2
   # ------------
   # Martina Navratilova was a basketball player
   # Martina Navratilova was a tennis player
   # Martina Navratilova was a scientist
   # 
   # For prompt 3
   # ------------
   # Dexter, from the TV Series Dexter's Laboratory, is a basketball player
   # Dexter, from the TV Series Dexter's Laboratory, is a tennis player
   # Dexter, from the TV Series Dexter's Laboratory, is a scientist
   # 
   # For prompt 4
   # ------------
   # LeBron James is a basketball player
   # LeBron James is a tennis player
   # LeBron James is a scientist

For each prompt, CAPPr will pick the completion which makes the sentence make the most
sense (according to the model!).

Examples
++++++++

For examples of this prompt-completion format in action, see the **Example** section for
each of these functions:

:func:`cappr.openai.classify.predict`

:func:`cappr.huggingface.classify.predict`

:func:`cappr.huggingface.classify.predict_examples`

:func:`cappr.openai.classify.predict_examples`

.. warning:: I haven't evaluated CAPPr on completion strings which are longer than 15
             tokens long. And I don't think CAPPr works well when there are ≥50 possible
             completions. Consider these domains uncharted and risky for CAPPr.

A reasonable hypothesis is that this prompt-completion format is especially well-suited
to smaller models. See the footnote for a reason for this hypothesis.\ [#]_


Yes-No
~~~~~~

Sometimes, your task can be framed as a yes or no question. Here's an example of a
successful format for instruction-trained models, which was pulled from `this demo
<https://github.com/kddubey/cappr/blob/main/demos/raft/ade.ipynb>`_:

.. code:: python

   def prompt_yes_or_no(text: str) -> str:
      return f"""
   The following sentence was taken from a medical case report:
   {text}
   Does the sentence describe an adverse effect of a pharmaceutical drug or
   substance?
   Answer Yes or No:"""

   end_of_prompt = " "
   completions = ("Yes", "No")

   medical_case_report = """
   We describe the case of a 10-year-old girl with two epileptic seizures and
   subcontinuous spike-waves during sleep, who presented unusual side-effects related
   to clobazam (CLB) monotherapy.
   """

   prompt = prompt_yes_or_no(medical_case_report)
   for completion in completions:
      print(f"{prompt}{end_of_prompt}{completion}")
      print()


Examples
++++++++

For another example of this prompt-completion format in action, see `this demo
<https://github.com/kddubey/cappr/blob/main/demos/raft/over.ipynb>`_.


Multiple Choice
~~~~~~~~~~~~~~~

Big, instruction-trained models are good at answering multiple choice questions, because
they've been trained to do so. One caveat is that the number of choices shouldn't be
more than five, because multiple choice question formats seen during training are
usually limited to the letters from school exams: (A), (B), (C), (D), (E). And ensure
that the system prompt/message is explicit about answering with one of the letters.
Here's an example of the system prompt used for the `COPA demo`_:

.. _COPA demo: https://github.com/kddubey/cappr/blob/main/demos/llama2/copa.ipynb

.. code:: python

   system_prompt_copa = (
      "Identify the cause or effect of a premise given two choices. Each choice "
      "is identified by a letter, A or B.\n"
      "Respond only with the letter corresponding to the correct cause or effect."
   )


Here's a little utility function which automatically writes out the letters and choices:

.. code:: python

   from string import ascii_uppercase

   def multiple_choice(*choices) -> str:
      if len(choices) > len(ascii_uppercase):
         raise ValueError("There are more choices than letters.")
      return "\n".join(
         [f"{letter}. {choice}" for letter, choice in zip(ascii_uppercase, choices)]
      )

   choices = [
      "Don't Wanna Know",
      "Shit",
      "All Time Low",
      "Welcome to the Internet",
      "Bezos II",
   ]
   print(multiple_choice(*choices))

   # A. Don't Wanna Know
   # B. Shit
   # C. All Time Low
   # D. Welcome to the Internet
   # E. Bezos II

Based on a few experiments, multiple choice questions are less appropriate for smaller
or less instruction-trained models.

.. warning:: Currently, :mod:`cappr.openai.classify` must repeat the ``prompt`` for
             however many completions there are. So if your prompt is long and your
             completions are short, you may end up spending much more with CAPPr.
             (:mod:`cappr.huggingface.classify` does not have to repeat the prompt
             because it caches its representation.)


Wrangle step-by-step completions
--------------------------------

Step-by-step\ [4]_ and chain-of-thought prompting\ [5]_ are highly effective for slighly
more complex classification tasks. While CAPPr is not immediately well-suited to these
sorts of prompts, it may be applied to post-process completions:

1. Get the completion from the step-by-step / chain-of-thought prompt

2. Pass this completion in a second prompt, and have CAPPr classify the answer. You can
   probably get away with using a cheap model for this task, as it just takes a bit of
   semantic parsing.

Here's an example:

.. code:: python

   from cappr.openai.api import gpt_chat_complete
   from cappr.openai.classify import predict

   # task: pick the next prereq to take
   class_to_prereqs = {
      "CS-101": "no prerequisites",
      "CS-102": "CS-101, MATH-101",
      "MATH-101": "no prerequisites",
      "MATH-102": "MATH-101",
      "ML-101": "CS-101, MATH-102, STAT-101",
      "STAT-101": "MATH-101",
      "STAT-102": "STAT-101, MATH-102",
   }
   class_to_prereqs_str = "\n".join(
      f"{class_}: {prereqs}" for class_, prereqs in class_to_prereqs.items()
   )

   prompt_raw = f"""
   Hi Professor. I'm interested in taking ML-101, but I'm struggling to decide which
   course I need to take before that. I've already taken CS-101. Which course should
   I take next?

   Here's a list of courses and their prerequisites which I pulled from the course
   catalog.

   {class_to_prereqs_str}
   """

   prompt_step_by_step = prompt_raw + "\n" + "Let's think step by step."

   chat_api_response = gpt_chat_complete(
      prompt_step_by_step,
      model="gpt-4",
      system_msg=(
         "You are a computer scientist mentoring a student. End your response to "
         "the student's question with the final answer, which is the name of a "
         "course."
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

   answer = predict(
      prompt_answer,
      completions=class_to_prereqs.keys(),
      model="text-ada-001",
   )

   print(answer)
   # MATH-101


A note on few-shot prompts
--------------------------

While all of the examples in the documentation are zero-shot prompts, nothing about
CAPPr prevents you from using few-shot prompts. Just make sure you're not paying too
much for a small benefit. And consider that you may not need to label many (or any!)
examples for few-shot prompting to work well. \ [6]_


Footnotes
---------

.. [#] These are not hard rules. For example, the `demo for the Winograd Schema
   Challenge <https://github.com/kddubey/cappr/blob/main/demos/superglue/wsc.ipynb>`_
   flips the roles of the ``prompt`` and ``completion``. (Just don't use the ``prior``
   keyword argument in that case.)

.. [#] CAPPr may be able to lean more on what was learned during pretraining than
   methods which rely on instruction-style prompts. Consider the `COPA task
   <https://github.com/kddubey/cappr/blob/main/demos/llama2/copa.ipynb>`_. A
   smaller language model probably hasn't seen enough of the instruction-style prompt:

   .. code::

      The man broke his toe because
      A. He got a hole in his sock
      B. He dropped a hammer on his foot
      Answer A or B.

   But from pretraining, the model has probably seen many sentences like:

   .. code::

      The man broke his toe because he dropped a hammer on his foot.

   And it would therefore give higher probability to the correct choice: ``he dropped a
   hammer on his foot``.


References
----------

.. [3] Weng, Lilian. (Mar 2023). Prompt Engineering. Lil'Log.
   https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/.

.. [4] Kojima, Takeshi, et al. "Large language models are zero-shot reasoners." arXiv
    preprint arXiv:2205.11916 (2022).

.. [5] Wei, Jason, et al. "Chain of thought prompting elicits reasoning in large
    language models." arXiv preprint arXiv:2201.11903 (2022).

.. [6] Min, Sewon, et al. "Rethinking the role of demonstrations: What makes in-context
    learning work?." arXiv preprint arXiv:2202.12837 (2022).

Select a prompt-completion format
=================================

   *"If you can dodge a wrench, you can dodge a ball!"*

   -- Patches

With CAPPr, your job is to conceptually write up your classification task as a formatted
string, called the prompt-completion format:

.. code:: python

   {prompt}{end_of_prompt}{completion}

The text you want to classify should appear in the ``prompt``. ``end_of_prompt`` is
either a whitespace ``" "`` (default) or the empty string ``""``. Each ``completion``
should contain a choice/class which the text could belong to.\ [#]_

Each ``completion`` should flow naturally after ``{prompt}{end_of_prompt}``. So pay
close attention to the use of whitespaces, newlines, and word casing. CAPPr doesn't do
any string processing for you; it just concatenates the three strings and sends it!
**It's on you to format the prompt according to the model's instruction/chat format**,
assuming that's applicable and beneficial. Before calling any CAPPr functions, consider
printing ``{prompt}{end_of_prompt}{completion}`` for each completion in your list of
possible completions/choices, and ensure that each passes the eye test.

And yes, you'll likely need to do a bit of prompt engineering. But if you can write a
sentence, you can write a prompt. It's mostly a matter of trial and error. Here's an
`external guide`_ if you'd like to survey research in this field.\ [3]_

Empirically, the impact of the prompt-completion format on accuracy depends on the
quality of the language model. For larger, instruction-trained models, the format is not
as consequential. For smaller, less instruction-trained models, it can be critical to
get the format right.

.. _external guide: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/

Feel free to `open an issue <https://github.com/kddubey/cappr/issues>`_ if you're having
trouble with writing a prompt. It's not always an easy task. Models can be surprisingly
finicky.


Prompt-completion formats
-------------------------

Most successful prompt-completion formats adhere to one of these three styles:
Concat-Class, Yes-No, Multiple Choice.


Concat-Class
~~~~~~~~~~~~

In this format, the full name of the class is directly used as the completion. In other
words, the class name is directly concatenated after ``{prompt}{end_of_prompt}``.

Here's an example:

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
sense (according to the model). A reasonable hypothesis is that this prompt-completion
format is especially well-suited to less instruction-trained models. See the footnote
for a reason for this hypothesis.\ [#]_

It's often a good idea to also mention the choices/completions in the prompt, or the
system prompt for an instruction-trained model. For example:

.. code:: python

   class_names = ("basketball player", "tennis player", "scientist")
   class_names_str = '\n'.join(class_names)
   prompt_prefix = f'''Every input belongs to one of these categories:
   {class_names_str}'''
   print(prompt_prefix)
   # Every input belongs to one of these categories:
   # basketball player
   # tennis player
   # scientist

.. warning:: CAPPr hasn't been systematically evaluated on ``completions`` where some
             are longer than 30 tokens. Consider this domain uncharted and risky for
             CAPPr.


Examples
++++++++

For examples of this prompt-completion format in action, see any of the `HuggingFace
demos <https://github.com/kddubey/cappr/blob/main/demos/huggingface>`_ excluding
``sciq.ipynb``, or any of the `Llama CPP demos
<https://github.com/kddubey/cappr/blob/main/demos/llama_cpp>`_.

For minimal examples, see the **Example** section for each of these functions:

:func:`cappr.openai.classify.predict`

:func:`cappr.huggingface.classify.predict`

:func:`cappr.llama_cpp.classify.predict`

:func:`cappr.openai.classify.predict_examples`

:func:`cappr.huggingface.classify.predict_examples`

:func:`cappr.llama_cpp.classify.predict_examples`

For minimal examples with a shared prompt prefix, see the **Example** section for each
of these functions:

:func:`cappr.huggingface.classify.cache_model`

:func:`cappr.llama_cpp.classify.cache_model`


Yes-No
~~~~~~

Sometimes, your task can be framed as a yes or no question.

Here's an example of a successful format for instruction-trained models, which was
pulled from `this demo
<https://github.com/kddubey/cappr/blob/main/demos/openai/raft/ade.ipynb>`_:

.. code:: python

   def prompt_yes_or_no(text: str) -> str:
       return (
           "The following sentence was taken from a medical case report: "
           f'"{text}"\n'
           "Does the sentence describe an adverse effect of a pharmaceutical "
           "drug or substance?\n"
           "Answer Yes or No:"
       )

   end_of_prompt = " "
   completions = ("Yes", "No")

   medical_case_report = (
       "We describe the case of a 10-year-old girl with two "
       "epileptic seizures and subcontinuous spike-waves during sleep, who "
       "presented unusual side-effects related to clobazam (CLB) monotherapy."
   )

   prompt = prompt_yes_or_no(medical_case_report)
   for completion in completions:
       print(f"{prompt}{end_of_prompt}{completion}")
       print()


Examples
++++++++

For another example of this prompt-completion format in action, see `this demo
<https://github.com/kddubey/cappr/blob/main/demos/openai/raft/over.ipynb>`_.


Multiple Choice
~~~~~~~~~~~~~~~

Many models have been extensively trained to answer multiple choice questions. One
caveat is that the number of choices ideally shouldn't be more than five, because
multiple choice question formats seen during training are usually limited to the letters
from school exams: A, B, C, D, E. Based on a few experiments, multiple choice questions
are less appropriate for less instruction-trained models.

Also, ensure that the system prompt is explicit about answering with one of the letters.
Here's an example of the system prompt used for the `Llama 2 COPA demo
<https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb>`_:

.. code:: python

   system_prompt_copa = (
       "Identify the cause or effect of a premise given two choices. Each "
       "choice is identified by a letter, A or B.\n"
       "Respond only with the letter corresponding to the correct cause or "
       "effect."
   )


Here's a little utility function which automatically writes out the letters and choices:

.. code:: python

   from string import ascii_uppercase as alphabet

   def multiple_choice(*choices) -> str:
       if len(choices) > len(alphabet):
           raise ValueError("There are more choices than letters.")
       letters_and_choices = [
           f"{letter}. {choice}" for letter, choice in zip(alphabet, choices)
       ]
       return "\n".join(letters_and_choices)

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


Examples
++++++++

For an example of this prompt-completion format in action, see `this demo
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/sciq.ipynb>`_.


Quirks
------

Most models are sensitive to quirky differences between prompts.

For models using SentencePiece tokenization, e.g., Llama and Mistral, only

.. code:: python

   {prompt} {completion}

formats are possible. In other words, ``end_of_prompt`` is always a whitespace,
regardless of your input. The placement of whitespaces can also affect performance.
Consider adding the whitespace to the end of the ``prompt`` string.

Another quirk is that when using a Concat-Class style prompt with a less
instruction-trained model, it's possible to achieve higher accuracy by abandoning the
chat/instruction format. See, e.g., the `Llama 2 COPA demo`_.

These notes will be updated as more quirks are discovered.


Wrangle step-by-step completions
--------------------------------

Step-by-step\ [4]_ and chain-of-thought\ [5]_ prompts are highly effective for more
involved tasks. While CAPPr is not immediately well-suited to these prompts, it may be
applied to post-process completions:

1. Get the completion from the step-by-step / chain-of-thought prompt

2. Pass this completion in a second prompt, and have CAPPr classify the answer. You can
   probably get away with using a cheap model for this task, as it just takes a bit of
   semantic parsing.

Here's an example:

.. code:: python

   from cappr.openai.api import gpt_chat_complete
   from cappr.openai.classify import predict

   # Task for a student in school: pick the next prereq to take
   class_to_prereqs = {
       "CS-101": "no prerequisites",
       "CS-102": "CS-101",
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
   Hi Professor. I'm interested in taking ML-101, but I'm struggling to decide
   which course I need to take before that. I've already taken CS-101. Which
   course should I take next?

   Here's a list of courses and their prerequisites which I pulled from the
   course catalog.

   {class_to_prereqs_str}
   """

   prompt_step_by_step = prompt_raw + "\n" + "Let's think step by step."

   chat_api_response = gpt_chat_complete(
      prompt_step_by_step,
      model="gpt-4",
      system_msg=(
          "You are a computer scientist mentoring a student. End your response "
          "to the student's question with the final answer, which is the name "
          "of a course."
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

   According to this answer, the very next course that the student should
   take is'''

   answer = predict(
       prompt_answer,
       completions=class_to_prereqs.keys(),
       model="text-babbage-001",
   )

   print(answer)
   # MATH-101


A note on few-shot prompts
--------------------------

While all of the examples in the documentation are "zero-shot" prompts (they don't
include examples of inputs and expected outputs) nothing about CAPPr prevents you from
using few-shot prompts / in-context learning. Just make sure you're not paying too much
money, latency, or data for a small benefit. Use
:func:`cappr.llama_cpp.classify.cache_model` or
:func:`cappr.huggingface.classify.cache_model` if applicable. And consider that you may
not need to label many (or any!) examples for few-shot prompting to work well.\ [6]_


Footnotes
---------

.. [#] These are not hard rules. For example, the `demo for the Winograd Schema
   Challenge
   <https://github.com/kddubey/cappr/blob/main/demos/openai/superglue/wsc.ipynb>`_
   flips the roles of the ``prompt`` and ``completion``. Just don't use the ``prior``
   keyword argument in that case.

.. [#] CAPPr may be able to lean more on what was learned during pretraining than
   methods which rely on instruction-style prompts. Consider the `COPA task
   <https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb>`_.
   A smaller language model probably hasn't seen enough of the instruction-style prompt:

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

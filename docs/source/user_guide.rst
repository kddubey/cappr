User Guide
==========

Given—

- a ``prompt`` string
- a list of possible completion strings
- and a language model

—CAPPr picks the completion which is most likely to follow ``prompt`` according to the
language model.

Here's a quick example:

.. code:: python

   from cappr.openai.classify import predict

   prompt = """
   Tweet about a movie: "Oppenheimer was pretty good. But 3 hrs...cmon Nolan."
   This tweet contains the following criticism:
   """.strip("\n")

   completions = ("bad message", "too long", "unfunny")

   pred = predict(prompt, completions, model="text-ada-001")
   print(pred)
   # too long

There are three factors which influence the performance of CAPPr: the language model,
the prompt-completion format, and the prior.

.. toctree::
   :maxdepth: 2

   select_a_language_model
   select_a_prompt_completion_format
   supply_a_prior
   examples
   why_probability
   a_note_on_workflow

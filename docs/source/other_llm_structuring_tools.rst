Other LLM structuring tools
===========================

There are `other LLM structuring tools
<https://www.reddit.com/r/LocalLLaMA/comments/17a4zlf/reliable_ways_to_get_structured_output_from_llms/>`_
which support "just pick one" functionality. `guidance
<https://github.com/guidance-ai/guidance>`_, for example, provides a ``select`` function
which almost always returns a valid choice. You should strongly consider using these
tools, as they scale well with respect to the number of choices.

One potential weakness of algorithms like this is that they don't always look at the
entire choice: they exit early when the generated choice becomes unambiguous. This
property makes the algorithm highly scalable with respect to the number of tokens in
each choice. But I'm curious to see if there are tasks where looking at all of the
choice's tokens—like CAPPr does—squeezes more out. Taking the tiny task from the
previous page (where CAPPr succeeds):

.. code:: python

   from guidance import models, select

   model = models.OpenAI("text-curie-001")

   class_names = (
       "The product is too expensive",
       "The product uses low quality materials",
       "The product is difficult to use",
       "The product doesn't look good",
       "The product is great",
   )
   class_names_str = "\n".join(class_names)

   product_review = "I can't figure out how to integrate it into my setup."
   prompt = f"""
   A customer left this product review: {product_review}

   Every product review belongs to exactly one of these categories:
   {class_names_str}

   Pick exactly one category which the product review belongs to.
   """

   result = model + prompt + select(class_names, name="pred")
   print(result["pred"])
   # The product is great

(Other prompts, including beefier versions of CAPPr's prompt on the previous page, also
fail.)

.. note:: When you're using other tools, if there are choices with multiple tokens and
          there's some ambiguity in the task, it's necessary to mention the choices in
          the prompt itself.

CAPPr doesn't try to be a heavyweight LLM query/programming language. It's aimed at
solving text classification problems, and is hopefully quite easy to pick up. CAPPr also
let's you easily compute probabilities, which `may be useful
<https://cappr.readthedocs.io/en/latest/why_probability.html>`_ in high-stakes
applications.

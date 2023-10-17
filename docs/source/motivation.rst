Motivation
==========

Why does this package exist? The short answer is to create a more usable text
classification interface.

Now for the long answer, which expands on the meaning of *usable*.


Problem
-------

There are many ways to do text classification. The one that this package is competing
against is text generation.

To make text generation more concrete, let's work through an example. Your first task is
to write up your classification task in a ``prompt`` string. For example, to classify a
product review, text generation code looks like this:

.. code:: python

   from cappr import openai

   class_names = (
       "The product is too expensive",
       "The product uses low quality materials",
       "The product is difficult to use",
       "The product isn't working",
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

   api_resp = openai.api.gpt_complete(
       prompt,
       model="gpt-3.5-turbo-instruct",
       max_tokens=10,
       temperature=0,
   )
   completion = api_resp[0]["text"]
   print(repr(completion))
   # '\nThe product is difficult to use'
   # correct!

This usually works well. But if you've ever run text generation on a slightly larger
scale, then you know that there may be a considerable fraction of cases where the
``completion`` is not actually in ``class_names``. For your LLM application to work
well, you need to handle these cases. So you add:

.. code:: python

   if completion not in class_names:
       completion = post_process(completion)

   assert completion in class_names

Implementing ``post_process`` can be challenging, as the ``completion`` is sampled from
the space of all possible sequences of tokens. This means you'll likely have to deal
with the cases where:

- The ``completion`` includes multiple plausible classes from ``class_names``

- The ``completion`` includes a bit of fluff

- The ``completion``\ 's word casing is different than the one used in ``class_names``,
  or it's spelled or phrased slightly differently

- The LM says ``"I'm not sure"`` in three different ways.

The OpenAI community knows that this can be challenging, so `they suggest
<https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit>`_
that you modify your code in at least 1 of 2 ways:

#. Point to multi-token class names using a single token—as in a multiple choice
   question

#. Transform multi-token class names into a single token, and finetune a model so that
   it learns the mapping to the single tokens.

These can be nontrivial modifications. Single-token references can sacrifice performance
when you have quite a few classes, as it's not a typical instruction format. On the
other hand, single-token transformations sacrifice useful semantics. And finetuning
costs too much time, money, and data.

**The fact is: you can endlessley accomodate text generation, but you'll still have to
write custom code to post-process its arbitrary outputs.** Fundamentally, sampling is
not a clean solution to a classification problem.


Recap
-----

Before moving on to the solution, let's recap the text generation workflow:

#. Design a prompt which asks the model to output exactly one choice from a given set of
   choices
#. Given this prompt, figure out how often the model doesn't output one of the given
   choices—call this an "invalid" output
#. Figure out how to post-process invalid outputs, depending on the way they look. Or
   give up, and figure out how to get your application to gracefully fail on an invalid
   output
#. Figure out if you need to tweak the text generation strategy, loop back to step (2).


Solution
--------

With CAPPr's ``predict`` interface, your job starts and stops at writing up your
classification task as a ``{prompt}{end_of_prompt}{completion}`` string.

Let's now run CAPPr on that product review classification task. Also, let's:

- supply a prior (optional)

- predict a probability distribution over classes (optional)

- use a smaller, "dumber" model—``text-curie-001``

  - Text generation with ``text-curie-001`` typically does not work well for slightly
    complicated tasks, e.g., run that text generation code above with
    ``model="text-curie-001"``\ .

.. code:: python

   from cappr.openai.classify import predict_proba

   class_names = (
       "The product is too expensive",
       "The product uses low quality materials",
       "The product is difficult to use",
       "The product isn't working",
       "The product doesn't look good",
       "The product is great",
   )
   prior = (
       2 / 7,
       1 / 7,
       1 / 7,
       1 / 7,
       1 / 7,
       1 / 7,
   )  # set to None if you don't have a prior
   # 2/7 reflects that perhaps we already expect customers to say it's expensive

   product_review = "I can't figure out how to integrate it into my setup."
   prompt = f"""
   This product review: {product_review}

   is best summarized as:"""

   completions = [class_name.lower() for class_name in class_names]

   pred_probs = predict_proba(
       prompt, completions, model="text-curie-001", prior=prior
   )

   print(repr(pred_probs.round(1)))
   # array([0.1, 0. , 0.7, 0.1, 0. , 0. ])

   pred_class_idx = pred_probs.argmax(axis=-1)
   print(class_names[pred_class_idx])
   # The product is difficult to use

CAPPr is guaranteed to output exactly one choice from a given set of choices. As a
result, your work is reduced to designing a prompt-completion string format.

In the age of large language models, text classification should be boring and easy.
CAPPr aims to be just that.

Motivation
==========

Why does this package exist? The short answer is to create a more usable zero-shot text
classification interface than `classification via sampling`_.

.. _classification via sampling: https://platform.openai.com/docs/guides/completion/classification

Now for a longer answer, which expands on what's meant by *usable*.


Problem
-------

There are many ways to do zero-shot text classification. The one that this package is
competing against is what I'll call **classification via sampling (CVS)**. This method
is pretty common for large language models (LLMs) like GPT-3+, PaLM, etc. CVS is
motiviated by the fact that LLMs are great at generating text. It's currently the method
which the `OpenAI guide on classification`_ covers.

.. _OpenAI guide on classification: https://platform.openai.com/docs/guides/completion/classification

To make CVS more concrete, let's work through an example.

In CVS, your job is to write up your classification task in a ``prompt`` string. For
example, to classify a product review, CVS code looks like this:

.. code:: python

   from cappr import openai

   class_names = ('The product is too expensive',
                  'The product uses low quality materials',
                  'The product is difficult to use',
                  "The product isn't working",
                  "The product doesn't look good",
                  'The product is great')
   class_names_str = '\n'.join(class_names)

   product_review = "I can't figure out how to integrate it into my setup."
   prompt = f'''
   A customer left this product review: {product_review}

   Every product review belongs to exactly one of these categories:
   {class_names_str}

   Pick exactly one category which the product review belongs to.
   '''

   api_resp = openai.api.gpt_complete(texts=[prompt],
                                      model='text-davinci-003',
                                      max_tokens=10,
                                      temperature=0)
   completion = api_resp[0]['text']
   completion
   # '\nThe product is difficult to use'
   # correct!

This usually works well. But if you've ever run CVS on a slightly larger scale, then you
know that there may be a considerable fraction of cases where the ``completion`` is not
actually in ``class_names``. To address these cases, you add:

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
  or it's spelled or phrased slightly differently. (These discrepancies typically only
  occur for domain-specific text.)

- The LM phrases its uncertainty in three different ways.

The OpenAI community knows that this can be challenging, so `they suggest`_ that you
modify your code in at least 1 of 2 ways:

#. Point to multi-token class names using a single token—like a multiple choice question

#. Transform multi-token class names into a single token, and finetune a model so that
   it learns the mapping to the single tokens.

.. _they suggest: https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit

These can be nontrivial modifications. Single-token references can sacrifice performance
when you have quite a few classes, as it's not a typical instruction format.
Single-token transformations sacrifice useful semantics, and finetuning usually requires
spending too much money and too much of your data.

**The fact is: you can endlessley accomodate CVS, but you'll still have to write custom
code to post-process its arbitrary outputs.** Fundamentally, sampling is not a clean
solution to a classification problem.


Solution
--------

With CAPPr's ``predict`` interface, your job starts and stops at writing up your
classification task as a ``{prompt}{end_of_prompt}{completion}`` string.

Let's now run CAPPr on that product review classification task. Also, let's:

- trivially incorporate a prior (optional)

- predict a probability distribution over classes (optional)

- replace the expensive ``text-davinci-003`` model call with a ``text-curie-001`` one

  - CVS with ``text-curie-001`` typically does not work well for slightly complicated
    tasks, e.g., run that CVS code above with ``model='text-curie-001'``\ .

.. code:: python

   from cappr.openai.classify import predict_proba

   class_names = ('The product is too expensive',
                  'The product uses low quality materials',
                  'The product is difficult to use',
                  "The product isn't working",
                  "The product doesn't look good",
                  'The product is great')
   prior = (2/7, 1/7, 1/7, 1/7, 1/7, 1/7)
   # perhaps we already expect customers to say it's expensive

   product_review = "I can't figure out how to integrate it into my setup."
   prompt = f'''
   This product review: {product_review}

   is best summarized as:'''

   completions = [class_name.lower() for class_name in class_names]

   pred_probs = predict_proba(prompts=[prompt],
                              completions=completions,
                              model='text-curie-001',
                              prior=prior)

   pred_probs.round(2)
   # array([[0.08, 0.  , 0.74, 0.11, 0.02, 0.05]])

   pred_class_idxs = pred_probs.argmax(axis=1)
   [class_names[pred_class_idx] for pred_class_idx in pred_class_idxs]
   # ['The product is difficult to use']

In the age of large language models, text classification should be boring and easy.
CAPPr aims to be just that.

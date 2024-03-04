A note on workflow
==================

You've done the hard part of translating a practical problem into a "make a choice"
problem that some LLM may be able to solve. Can you deploy this prompt + LLM system now?
Probably not. Here's what, and how, you need to do.


Gather data
-----------

Without previous experiments, don't assume that an LLM and your first prompt-completion
format are going to work well. Instead, gather data like so:

.. list-table:: A bunch of input-output pairs (200 examples)
   :widths: 3 10 10
   :header-rows: 1

   * - id
     - raw input
     - correct output index
   * - 1
     - "input 1"
     - 2
   * - 2
     - "input 2"
     - 0
   * - 3
     - "input 3"
     - 0
   * - ...
     - ...
     - ...
   * - 200
     - "input 200"
     - 1

The prompt is a transformation of the **raw input**. The **correct output index**
corresponds to the correct output/choice for that input.

In general, you should gather as many of these input-output pairs/examples as is
feasible. If there are only 2 possible choices (and say accuracy is 90%), then gather at
least 200 examples total.\ [#]_ As the number of possible choices increases, or as
accuracy gets closer to random guessing, more examples are needed to evaluate the
system.

If you don't have many input-output examples immediately within reach, then do the hard
but important work of making them up.\ [#]_ Think carefully about the types of inputs
you expect to see in production, and their relative frequencies. Make sure every choice
is included in the dataset. Consider adding a few tricky inputs to understand the limits
of your system. But don't evaluate anything just yet!


Split data into train and test
------------------------------

Now that you have a nice dataset, before you do anything else, `randomly partition
<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_
the dataset into a "training" dataset and a "test" dataset.\ [#]_ The importance of this
step cannot be overstated.

.. list-table:: training dataset (50 examples)
   :widths: 3 10 10
   :header-rows: 1

   * - id
     - raw input
     - correct output index
   * - 105
     - "input 105"
     - 1
   * - ...
     - ...
     - ...
   * - 27
     - "input 27"
     - 0

.. list-table:: test dataset (150 examples)
   :widths: 3 10 10
   :header-rows: 1

   * - id
     - raw input
     - correct output index
   * - 174
     - "input 174"
     - 2
   * - 26
     - "input 26"
     - 2
   * - 91
     - "input 91"
     - 1
   * - ...
     - ...
     - ...
   * - 136
     - "input 136"
     - 1


Iterate on the training dataset
-------------------------------

Evaluate your first prompt-completion format on the training dataset. Examine and
understand failure cases. Is your prompt specific enough? Does it include enough
context? Iterate the format, language model, or prior, and evaluate on the training
dataset again.

Be disciplined about not seeing or evaluating on the test dataset until you've finalized
your selections for a format, langauge model, and prior.


If necessary, bring out the big guns
------------------------------------

Sometimes, you'll find that your task is too difficult for a smaller model and a static
prompt-completion format. In that case, consider the most OP solution: get a
chain-of-thought completion from GPT-4 or Claude 2, and then have a cheap model classify
the answer from this completion using CAPPr. See `this section of the documentation
<https://cappr.readthedocs.io/en/latest/select_a_prompt_completion_format.html#wrangle-step-by-step-completions>`_
for an example. Just keep in mind that the big guns cost quite a bit of latency and
money.


Evaluate on the test dataset once
---------------------------------

After fully specifying everything about how your system is going to work, run that
system on the test dataset **once**. When you're asked for performance metrics, report
the ones from this dataset.


Footnotes
~~~~~~~~~

.. [#] Some quick-and-dirty rationale: a Wald 95% confidence interval for the expected
   accuracy of a binary classifier—which is estimated to be 90% accurate—is (0.84, 0.96)
   when evaluated on an independent/unseen set of 100 examples. For some applications,
   that level of uncertainty may not be acceptable.

.. [#] If you're careful, you may use a powerful LLM to make them up for you. Give it a
    handful of (handcrafted) high quality input-output pairs, and ask it to vary them
    and generate new pairs according to some requirements. Depending on your
    application, the examples it generates may not look like what you'll see in
    production. Iterate carefully and use your best judgement. Prefer quality over
    quantity to some degree.

.. [#] There are some applications where you may not want to *randomly* split the
    dataset. Perhaps your inputs are grouped, or change with time. In these cases,
    consider splitting by groups or by time.

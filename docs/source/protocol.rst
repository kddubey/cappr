Protocol
========

The design is to implement this protocol::

    ├───(module) {LM interface name}
        ├───(module) classify.py

            └───(function) log_probs_conditional:
                    prompts, completions, model, **kwargs
                        ->
                    list[i][j][k] = Pr_model(
                        completions[j][k] | prompts[i] + completions[j][:k]
                    )
            
            └───(function) predict_proba:
                    prompts, completions, model, **kwargs
                        ->
                    array[i, j] = Pr_model(completions[j] | prompts[i])

            └───(function) predict:
                    prompts, completions, model, **kwargs
                        ->
                    list[i] = argmax_j Pr_model(completions[j] | prompts[i])

The ``_examples`` functions are excluded from the diagram because they don't contribute
to understanding the design. They're a convenient way to do batch inference / achieve
higher throughput for arbitrary combinations of prompts and completions.

The `CAPPr scheme <https://stats.stackexchange.com/q/601159>`_ applies to any
autoregressive language model. But different language models have different ways of
outputting token log-probabilities.


How to add a new LM interface
-----------------------------

The only work is in implementing ``log_probs_conditional`` for your new LM interface.
``predict_proba`` is immediately defined by decorating ``log_probs_conditional``. And
``predict`` is immediately defined by decorating ``predict_proba``.

The implementation of ``log_probs_conditional`` must treat ``prompts`` as a sequence of
strings. Process it in batches if possible, else loop over it. The decorator handles the
case where ``prompts`` is a string. The decorators also run all required input checks on
``prompts`` and ``completions``. You may need to run input checks on ``model``, or set
it up so that it's correct. If you want to set up ``model`` so that it's correct, then
tear down your changes before the function is done (or raises an exception).

Let's work through an example. Say we're integrating Anthropic's `Claude
<https://www.anthropic.com/index/introducing-claude>`_. Here's what
``src/cappr/anthropic/classify.py`` would look like:

.. code:: python

   from cappr.utils import classify


   @classify._log_probs_conditional
   def log_probs_conditional(
       prompts: str | Sequence[str],
       completions: Sequence[str],
       model: str,
       **kwargs,
   ) -> list[list[float]] | list[list[list[float]]]:
       # 1. Hit the Anthropic API in batches of requests.
       log_probs_completions = []
       # 2. Process the output into this doubly-nested and ragged list:
       #    log_probs_completions[i][j][k] is the log-probability of the
       #    completion token k in completions[j], conditional on
       #    prompts[i] + previous completion tokens.
       return log_probs_completions


   @classify._predict_proba
   def predict_proba(
       prompts: str | Sequence[str],
       completions: Sequence[str],
       model: str,
       **kwargs,  # omitting the prior and other universal kwargs for brevity
   ) -> npt.NDArray[np.floating]:
       return log_probs_conditional(prompts, completions, model, **kwargs)


   @classify._predict
   def predict(
       prompts: str | Sequence[str],
       completions: Sequence[str],
       model: str,
       **kwargs,
   ) -> str | list[str]:
       return predict_proba(prompts, completions, model, **kwargs)

This decorator-based design always works because ``predict_proba`` is a transformation
of the output of ``log_probs_conditional``—a transformation which is always independent
of the LM interface. ``predict`` is just the argmax of the output of ``predict_proba``.

Add the import to ``src/cappr/__init__.py``, and add its dependencies as extras to
``setup.py``. Add a testing module. The testing module for any API LM interface will
look a lot like `OpenAI's testing module
<https://github.com/kddubey/cappr/blob/main/tests/openai/test_openai_classify.py>`_.
After passing these tests, evaluate the module on a few `demos
<https://github.com/kddubey/cappr/blob/main/demos>`_.

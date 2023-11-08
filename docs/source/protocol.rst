Protocol
========

The design is to implement this protocol:

.. code:: python

   # src/cappr/your_new_lm_interface/classify.py

   # In docstrings, for brevity:
   # p = prompts
   # c = completions

   def log_probs_conditional(prompts, completions, model, **kwargs):
       """
       list[i][j][k] = log Pr_model(c[j][token k] | p[i] + c[j][:token k])
       """

   def predict_proba(prompts, completions, model, **kwargs):
       """
       array[i, j] = (
                         log Pr_model(c[j][token k] | p[i] + c[j][:token k])
                         .mean(axis=-1)
                         .exp()
                         .normalize(axis=1)
                     )
                   = Pr_model(c[j] | p[i])
       """

   def predict(prompts, completions, model, **kwargs):
       """
       list[i] = argmax_j Pr_model(c[j] | p[i])
       """

The ``_examples`` functions are excluded from the diagram because they don't contribute
to understanding the design. They're a convenient way to do batch inference / achieve
higher throughput for arbitrary combinations of prompts and completions.

The `CAPPr method <https://stats.stackexchange.com/q/601159>`_ applies to any
autoregressive language model. But different language models have different ways of
outputting token log-probabilities, hence the need for separate LM interface modules.


How to add a new LM interface
-----------------------------

The only work is in implementing ``log_probs_conditional`` for your new LM interface.
``predict_proba`` is immediately defined by decorating ``log_probs_conditional``. And
``predict`` is immediately defined by decorating ``predict_proba``.

The implementation of ``log_probs_conditional`` should batch over ``prompts`` and
``completions`` if possible. Treat ``prompts`` as a sequence of strings rather than a
string itself. The decorator handles the case where ``prompts`` is a string. The
decorators also run all required input checks on ``prompts`` and ``completions``. You
may need to run input checks on ``model``, or set it up so that it's correct. If you
want to set up ``model`` so that it's correct, then tear down your changes before the
function is done by implementing a context manager.

If the LM interface doesn't support KV caching, then the implementation will look a lot
like the `Llama CPP no-cache module
<https://github.com/kddubey/cappr/blob/main/src/cappr/llama_cpp/_classify_no_cache.py>`_
or the `OpenAI module
<https://github.com/kddubey/cappr/blob/main/src/cappr/openai/classify.py>`_.

After implementing your new module, add the import to ``src/cappr/__init__.py``, and add
its dependencies as extras to ``setup.py``. Add a testing module to ``tests``. The
testing module for any API LM interface will look a lot like `OpenAI's testing module
<https://github.com/kddubey/cappr/blob/main/tests/openai/test_openai_classify.py>`_.
After passing these tests, evaluate the module on a few `demos
<https://github.com/kddubey/cappr/blob/main/demos>`_.

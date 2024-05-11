Design choices
==============

This is the first open source software project I've created. I thought I'd write about
some of its design choices, and what I learned from creating it. These thoughts are more
reflective than educational.


An alternate design
-------------------

As noted on the previous page, ``predict_proba`` is a transformation of the output of
``log_probs_conditional``—a transformation which is always independent of the LM
interface. ``predict`` is just the argmax of the output of ``predict_proba``. These
patterns suggest a design which does some type of function-wrapping.

The current design uses decorators. An alternate design is to create an abstract base
class which all LM interfaces inherit from / implement.

.. code:: python

   from abc import ABC, abstractmethod

   import numpy.typing as npt


   class _AbstractClassifier(ABC):
       @classmethod
       @abstractmethod
       def log_probs_conditional(
           cls,
           prompts: str | Sequence[str],
           completions: Sequence[str],
           model,
           **kwargs,
       ) -> list[list[float]] | list[list[list[float]]]:
           pass


       @classmethod
       def predict_proba(
           cls,
           prompts: str | Sequence[str],
           completions: Sequence[str],
           model,
           **kwargs,
       ) -> npt.NDArray[np.floating]:
           log_probs_completions = cls.log_probs_conditional(
               prompts, completions, model, **kwargs
           )
           return (  # these methods don't exist, but you know what I mean
               log_probs_completions
               .average(axis=-1)
               .exponentiate()
               .normalize(axis=1)
           )


       @classmethod
       def predict(
           cls,
           prompts: str | Sequence[str],
           completions: Sequence[str],
           model,
           **kwargs,
       ) -> str | list[str]:
           pred_probs = cls.predict_proba(prompts, completions, model, **kwargs)
           pred_completions = [
               completions[completion_idx]
               for completion_idx in pred_probs.argmax(axis=1)
           ]
           return pred_completions

As far as I can see, there is no benefit to this design. It's tempting to think that it
reduces writing LM modules to writing sublcasses which implement
``log_probs_conditional`` and nothing else. That's not practically true. Because if
you're writing a new LM module, it most likely has unique ``kwargs`` which you need to
supply in the signature and explain in the docstring. So you need to write out the
method anyway. Moreover, the amount of repeated code that's saved is insignificant.

There is a small cost to this design. Enabling the simple import interface—

::

   from cappr.{LM interface name}.classify import predict_proba


—would require adding a bit of logic to ``__init__.py`` and Sphinx's
``docs/source/conf.py`` for every new module. I could automate this stuff by
implementing some slightly tricky logic. But more logic is worse than less logic.

Overall, this design doesn't align with my style. Inheritance isn't needed to achieve
function wrapping. Decorators are sufficient.


Input checks
------------

There are some silent and some loud errors which would be hard to debug for a user. For
example:

- inputting an unordered iterable will cause an output array of predicted probabilities
  to be meaningless

- inputting an incorrectly structured prior will cause all of the model's compute to be
  a waste

- inputting an empty object causes an obscure index error in some downstream
  model/tokenization functions.

To combat this, every decorator does input checking. These checks aren't too
restrictive. For example, ``isinstance(ordered_object, Sequence)`` is not required.
``ordered_object`` just has to have an obvious, deterministic order when ``x for x in
ordered_object`` is done. As a result, the user can be lazy: they can input a pandas
Series, a dictionary's ``.keys()``, a numpy array, or torch tensor, and the function
will work just fine.


Context managers
----------------

The Hugging Face module requires that the ``model_and_tokenizer`` input is set up in a
particular way. It looks like many other tools solve this problem by creating a loading
function or method which does the required set up. Its returned object is internal to
the package. I don't think this extra abstraction is necessary. And it comes at the
small cost of requiring the user to learn a new way to load their model.

Instead, ``cappr.huggingface`` lets the user initialize the object however they want. It
then internally sets it up as required, rolling back these changes when finished. This
pattern is accomplished by the context managers `here
<https://github.com/kddubey/cappr/blob/main/src/cappr/huggingface/_utils.py>`_.


No string formatting abstractions
---------------------------------

This package will do one thing well: pick a completion from a user-created prompt. If
users need or want to use a string formatter, that's on them.


Repeat docstrings
-----------------

Lots of text in docstrings are repeated. After all, every LM module is implementing a
protocol.

I previously experimented with `an automation
<https://github.com/kddubey/dumpy/tree/main/wrap>`_ that dynamically writes the
docstring via decorators. This pattern is used throughout Hugging Face ``transformers``,
for example. I decided against this pattern because it sacrifices an important
convenience: hovering over a function to see what it does. Code analyzers like Pylance
won't show the ``__doc__`` attribute that was dynamically constructed.

I personally am slightly annoyed when I have to open up a function's documentation in my
browser, and look back and forth at my browser and IDE. I like the convenience of
hovering over the function in my IDE itself. So I opted to do what numpy, scipy, and
scikit-learn do in their docstrings: repeat text. It's definitely tedious to make
modifications. But that tediousness is outweighed by the benefits to the user.


Testing
-------

This package's tests are designed in a sophisticated (complicated) way. It took me a
while to think about what they should look like. The goal was to allow for 2 things:

#. shared test cases universal to all ``classify`` modules—these are the
   parametrizations in ``_base.TestPromptsCompletions`` and ``_base.TestExamples`` (see
   this `module <https://github.com/kddubey/cappr/blob/main/tests/_base.py>`_)
#. LM-interface-specific fixtures and parametrizations to test LM-interface-specific
   setups and arguments, e.g., ``batch_size`` in ``cappr.huggingface``.

The current testing design accomplishes these things through inheritance, because pytest
is incredibly powerful with inheritance. For an example, see the `tests
<https://github.com/kddubey/cappr/blob/main/tests/llama_cpp/test_llama_cpp_classify.py>`_
for llama-cpp models.

There are still a few testing todos. One problem is that there are dependencies in the
tests; if ``test_log_probs_conditional`` fails, the rest of the tests will fail.
Ideally, for example, ``test_predict_proba`` assumes ``log_probs_conditional`` is
correct.


Mistakes were made
------------------

I made plenty of mistakes while developing CAPPr.

Too many breaking changes
~~~~~~~~~~~~~~~~~~~~~~~~~

`Releases <https://github.com/kddubey/cappr/releases>`_ were not as backwards compatible
as they could've been. I was tripped up by the OpenAI v1.0 release. I've been figuring
stuff out on the fly and releasing whenever I think something is good enough for the
short term.

Too many half-measures
~~~~~~~~~~~~~~~~~~~~~~

It's well known that attention keys and values can be cached whenever substrings are
repeated for inference. Getting this feature to align with the CAPPr scheme took nitty
gritty work. My first few implementations of caching were suboptimal from both a
computational and a UI perspective. I got lost in the sauce of making lots and lots of
incremental improvements. Eventually, I `re-did
<https://github.com/kddubey/cappr/commit/d3b52e975918fa83b52c963116b79d5132ba5277>`_ the
whole thing with some success. There are still probably important optimizations I left
on the table, but it'll do for now.

Marketing matters
~~~~~~~~~~~~~~~~~

The first version of the `User Guide
<https://cappr.readthedocs.io/en/latest/user_guide.html>`_ was written for ML types,
when it should've been written for software engineers. What's text classification? What
are "labeled examples"? What's a prior? Why is a probability distribution useful? Docs
for other tools answer, or successfully dodge, these questions much more effectively.


Pleasant surprises
------------------

See `this page of the documentation
<https://cappr.readthedocs.io/en/latest/statistical_performance.html>`_.

Besides the algorithmic stuff, I was pleasantly surprised to find that I enjoyed
engineering this project from the ground up. Mulling over design decisions and managing
myself was fun. I also became much more aware of open source tools and practices. I
appreciate open source software at a higher level.

Design choices
==============

This is the first open source software project I've created. I thought I'd write about
some of its design choices, and what I learned from creating it. These thoughts are more
reflective than educational.


An alternate design
-------------------

An alternate design is to ditch the decorators, and instead create an abstract base
class which all LM interfaces inherit from / implement.

.. code:: python

   from abc import ABC, abstractmethod

   import numpy.typing as npt


   class _BaseClassifier(ABC):
       @classmethod
       @abstractmethod
       def log_probs_conditional(
           cls,
           prompts: str | Sequence[str],
           completions: Sequence[str],
           model: str,
           **kwargs,
       ) -> list[list[float]] | list[list[list[float]]]:
           pass


       @classmethod
       def predict_proba(
           cls,
           prompts: str | Sequence[str],
           completions: Sequence[str],
           model: str,
           **kwargs,
       ) -> npt.NDArray[np.floating]:
           log_probs_completions = cls.log_probs_conditional(
               prompts, completions, model, **kwargs
           )
           return (  # these methods don't exist, but you know what I mean
               log_probs_completions
               .average(axis=2)
               .exponentiate(axis=1)
               .normalize(axis=1)
           )


       @classmethod
       def predict(
           cls,
           prompts: str | Sequence[str],
           completions: Sequence[str],
           model: str,
           **kwargs,
       ) -> str | list[str]:
           pred_probs = cls.predict_proba(prompts, completions, model, **kwargs)
           pred_completions = [
               completions[completion_idx]
               for completion_idx in pred_probs.argmax(axis=1)
           ]
           return pred_completions

As far as I can see, there is no benefit to this design. It's tempting to think that it
reduces writing LM interface modules to writing sublcasses which implement
``log_probs_conditional`` and nothing else. That's not practically true. Because if
you're writing a new LM interface, it most likely has unique ``kwargs`` which you need
to supply in the signature and explain in the docstring. So you need to write out the
method anyway. Moreover, the amount of repeated code that's saved is insignificant.

There are a few small costs to this design. Enabling the simple import interface—

::

   from cappr.{LM interface name}.classify import predict_proba


—would require adding a bit of stuff to ``__init__.py`` and Sphinx's
``docs/source/conf.py`` for every new interface. I could automate this stuff by
implementing even more, slightly tricky code. But more code is worse than less code.

It's also a bit confusing to users (like me) who want to understand why something which
is fundamentally a function is actually a method. Is there some obscure state that I'm
not supposed to know is being maintained? Is there a useful hierarchy? The answer to
both of these questions is categorically no. I don't like when that happens.

Overall, this design doesn't align with my style. Inheritance isn't needed to achieve
function wrapping. Decorators are sufficient.


Input checks
------------

There are some silent and some loud errors would be hard to debug. For example:

- inputting an unordered iterable will cause an output array of predicted probabilities
  to be meaningless

- inputting an incorrectly structured prior will cause all of the model's compute to be
  a waste

- inputting an empty object causes an obscure index error in some downstream
  model/tokenization functions.

To combat this, every decorator does input checking. These checks aren't too
restrictive. For example, ``isinstance(ordered_object, Sequence)`` is not required.
``ordered_object`` just has to have an obvious, deterministic order when ``x for x in
ordered_object`` is done. As a result, the user can be lazy; they can input a pandas
Series, a dictionary's ``.keys()``, a numpy array, or torch tensor, and the function
will work just fine.


Context managers
----------------

The HuggingFace interface requires that the ``model_and_tokenizer`` input is set up in a
particular way. It looks like many other tools solve this problem by creating a loading
function or method which does the required set up. Its returned object is internal to
the package.

I don't like this solution because it sacrifices some user conveniences. The user can't
easily use their package-specific initialization method (which usually comes with a
helpful docstring), or copy-paste initialization code from elsewhere. This design also
forces the user to potentially re-load the model if they're using it elsewhere and the
package doesn't internally cache it, which costs time.

Instead, I chose to let the user initialize the object however they want, then
internally set it up as required, and internally roll it back when CAPPr is done. This
pattern is accomplished by the context managers `here
<https://github.com/kddubey/cappr/blob/main/src/cappr/huggingface/_utils.py>`_. A
developer benefit to context managers is that they self-document requirements without
forcing the user to satisfy them. When you call a function using your model and
tokenizer, you can safely assume it sets up the model as needed.


No string formatting abstractions
---------------------------------

Many tools in this space includes some type of string formatting abstraction. Some
abstract the complex process of structuring a completion or a chain of prompts and
completions. Others format a single prompt to, e.g., abstract the process of writing a
few-shot prompt. Prompt formatters are not as helpful. Not to be too dismissive, but
anyone who uses Python knows how to format a string. Prompt formatters replace the
question of "how do I tell the LM to do what I want?" with "how do I use this string
formatting interface to tell the LM to do what I want?". The latter question takes more
time to answer. And while answering that question, you may end up realizing that the
formatter doesn't let you do what you need to do. Moreover, these formatters sometimes
obfuscate what the prompt actually looks like, which is a risk. For smaller LMs, there
are quirks which prompt writers should be aware of.

I want this package to do one thing well: pick a completion from a user-created prompt.
If users want to use a string formatter to write prompts, that's on them.


Repeating docstrings
--------------------

Lots of text in docstrings are repeated. After all, fundamentally, the three functions
take the same inputs and produce the same outputs regardless of the LM interface.

I previously experimented with `an automation
<https://github.com/kddubey/dumpy/tree/main/wrap>`_ that dynamically writes the
docstring via decorators. This pattern is used throughout HuggingFace ``transformers``,
for example. I decided against this pattern because it sacrifices an important
convenience: hovering over a function to see what it does. Code analyzers like Pylance
are not fully dynamic. They won't show the ``__doc__`` attribute you dynamically
constructed.

I personally am annoyed when I have to open up a function's documentation in my browser,
and look back and forth at my browser and IDE. I like the convenience of hovering over
the function in my IDE itself. So I opted to do what numpy, scipy, and scikit-learn do
in their docstrings: repeat text. It's definitely tedious to make modifications. But
that tediousness is outweighed by the benefits to the user.


Testing
-------

This package's tests are designed in a sophisticated (complicated) way. It took me a
while to think about what they should look like. The goal was to allow for 2 things:

#. share test cases universal to all ``classify`` modules—these are the parametrizations
   in ``BaseTestPromptsCompletions`` and ``BaseTestExamples`` (see this `module
   <https://github.com/kddubey/cappr/blob/main/tests/_base.py>`_)
#. module-specific fixtures and parametrizations to test module-specific setups and
   arguments, e.g., ``batch_size`` in the HuggingFace backend.

The current testing design accomplishes these things through inheritance, because pytest
is incredibly powerful with inheritance. For an example, see the `tests
<https://github.com/kddubey/cappr/blob/main/tests/llama_cpp/test_llama_cpp_classify.py>`_
for llama-cpp models.

There are still a lot of testing TODOs.


Mistakes were made
------------------

Too many breaking changes
~~~~~~~~~~~~~~~~~~~~~~~~~

`Releases <https://github.com/kddubey/cappr/releases>`_ were not as backwards compatible
as they could've been.

Too many half-measures
~~~~~~~~~~~~~~~~~~~~~~

It's well known that attention keys and values can be cached whenever substrings are
repeated for inference. Getting this feature to align with the CAPPr scheme took nitty
gritty work. My first few implementations of caching were suboptimal from both a
computational and a UI perspective. I got lost in the sauce of making lots and lots of
incremental improvements. Eventually I `re-did
<https://github.com/kddubey/cappr/commit/d3b52e975918fa83b52c963116b79d5132ba5277>` the
whole thing with some success.

Marketing matters
~~~~~~~~~~~~~~~~~

The first version of the `User Guide
<https://cappr.readthedocs.io/en/latest/user_guide.html>`_ was written for ML types,
when it should've been written for software engineers. What's text classification? What
are "labeled examples"? What's a prior? Why is a probability distribution useful? Docs
for other tools answer, or successfully dodge, these questions much more effectively. In
the age of LLMs, text classification can be done by any engineer, not just ML engineers.


Pleasant surprises
------------------

Re the algorithm: see the `Misfit Toys Hypothesis
<https://cappr.readthedocs.io/en/latest/future_research.html>`_.

Besides the algorithmic stuff, I was pleasantly surprised to find that I loved
engineering this project from the ground up. Mulling over design decisions and managing
myself was fun. Writing tests was enlightening using pytest. Writing docs was satisfying
(and `almost <https://github.com/kddubey/dumpy/tree/main/sphinx_setup>`_ easy) using
Sphinx and readthedocs. Writing GitHub workflows made releases convenient, and they made
my project feel way more professional lol. I found `ReWrap
<https://stkb.github.io/Rewrap/>`_ and `autoDocstring
<https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_ for the
first time. I'll be using them for every project from now on. Overall, as a result of
working on this project, I appreciate open source software at a much higher level.

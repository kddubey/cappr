Local setup
===========

These instructions work for \*nix systems.


Setup
-----

#. Create a new Python 3.8+ virtual environment. Activate the venv. I use
   `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_. For this
   example, let's create a virtual environment called ``cappr`` using Python's native
   ``venv``::

      cd your/venvs

      python3 -m venv cappr

      source cappr/bin/activate

      python -m pip install wheel --upgrade pip


#. ``cd`` to wherever you store projects, and clone the repo (or fork it and clone that)
   there::

      cd your/projects

      git clone https://github.com/kddubey/cappr.git

#. ``cd`` to the repo and install this package in editable mode, along with development
   requirements (after ensuring that your venv is activated!)

   ::

      cd cappr

      python -m pip install -e ".[dev]"

      pre-commit install

#. Download these tiny GGUF models I uploaded to HF

   ::

      huggingface-cli download \
      aladar/TinyLLama-v0-GGUF \
      TinyLLama-v0.Q8_0.gguf \
      --local-dir ./tests/llama_cpp/fixtures/models \
      --local-dir-use-symlinks False

   ::

      huggingface-cli download \
      aladar/tiny-random-BloomForCausalLM-GGUF \
      tiny-random-BloomForCausalLM.gguf \
      --local-dir ./tests/llama_cpp/fixtures/models \
      --local-dir-use-symlinks False


VS code extensions
------------------

- `autoDocstring
  <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_. Use
  the numpy format, and check "Start On New Line"

- `Rewrap <https://stkb.github.io/Rewrap/>`_. Enable Auto Wrap

- Set Python formatting to ``black``.

And set the vertical line ruler to 88.


Testing
-------

From the repo home directory ``cappr``::

   pytest

Note that a few small transformers and tokenizers will be downloaded to your computer.

Sometimes I get worried about bigger code changes. So consider additionally testing
statistical performance by running an appropriate demo in the repo's `demos
<https://github.com/kddubey/cappr/tree/main/demos>`_.

When you add a new testing module, add it to the list in ``pyproject.toml``. The list is
in order of dependencies: ``Example`` and ``utils`` must pass for the rest of the
modules to pass.

To test a specific module, e.g., ``huggingface``::

   pytest -k huggingface


Docs
----

To test changes to documentation, first locally build them from the repo home directory
``cappr`` via::

   cd docs

   make html

and then preview them by opening ``docs/build/html/index.html`` in your browser.

After merging code to main, the official docs will be automatically built and published.


Release
-------

`Bump the version
<https://github.com/kddubey/cappr/commit/d1f7dd51fa702c123bdfb0bcb97535995641c224>`_,
and then create a new release on GitHub. A new version of the package will then be
automatically published on PyPI.

Try to follow `semantic versioning <https://semver.org/>`_ guidelines, even though I
haven't been great at that so far.
# CAPPr: Completion After Prompt Probability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/cappr/badge/?version=latest)](https://cappr.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/kddubey/cappr/actions/workflows/test.yml/badge.svg)](https://github.com/kddubey/cappr/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/kddubey/cappr/branch/main/graph/badge.svg?token=NYIL076PSM)](https://codecov.io/gh/kddubey/cappr)
[![PyPI - Package Version](https://img.shields.io/pypi/v/cappr?logo=pypi&style=flat&color=orange)](https://pypi.org/project/cappr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

CAPPr performs text classification. No training. No post-processing. **Just have your
LLM pick from a list of choices.** Or compute the probability of a completion given a
prompt. Squeeze more out of open source LLMs.


## Usage

<details>
<summary>Use a model from the OpenAI API</summary>

Specifically, this model must be compatible with the
[/v1/completions](https://platform.openai.com/docs/models/model-endpoint-compatibility)
endpoint
([excluding](https://cappr.readthedocs.io/en/latest/select_a_language_model.html#openai)
``gpt-3.5-turbo-instruct``).

```python
from cappr.openai.classify import predict

prompt = """
Tweet about a movie: "Oppenheimer was pretty good. But 3 hrs...cmon Nolan."
This tweet contains the following criticism:
""".strip("\n")

completions = ("bad message", "too long", "unfunny")

pred = predict(prompt, completions, model="text-ada-001")
print(pred)
# too long
```

Notice that a completion can contain many tokens.

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/select_a_language_model.html#openai)
for more info on using OpenAI models.
</details>


<details>
<summary>Extract the final answer from a step-by-step completion</summary>

Step-by-step and chain-of-thought prompts are highly effective ways to get an LLM to
"reason" about more complex tasks. But if you need a structured output, a step-by-step
completion is unwieldy. Use CAPPr to extract the final answer from these types of
completions, given a list of possible answers.

See this idea in action [here in the
documentation](https://cappr.readthedocs.io/en/latest/select_a_prompt_completion_format.html#wrangle-step-by-step-completions).
CAPPr is **100% guaranteed** to return an output from the list of possible answers.
</details>


<details>
<summary>Use a PyTorch transformers model</summary>

Specifically, this model must be able to be loaded using
[`transformers.AutoModelForCausalLM.from_pretrained`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict

# Load a model and its corresponding tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Which planet is closer to the Sun: Mercury or Earth?"
completions = ("Mercury", "Earth")

pred = predict(prompt, completions, model_and_tokenizer=(model, tokenizer))
print(pred)
# Mercury
```

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/select_a_language_model.html#huggingface)
for more info on using PyTorch ``transformers`` models.
</details>


<details>
<summary>Use a GGUF model</summary>

Specifically, this model must be able to be loaded using
[`llama_cpp.Llama`](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama).

```python
from llama_cpp import Llama
from cappr.llama_cpp.classify import predict

# Load model. Always set logits_all=True for CAPPr
model = Llama("./TinyLLama-v0.Q8_0.gguf", logits_all=True, verbose=False)

prompt = """Gary told Spongebob a story:
There once was a man from Peru; who dreamed he was eating his shoe. He
woke with a fright, in the middle of the night, to find that his dream
had come true.

The moral of the story is to"""

completions = (
  "look at the bright side",
  "use your imagination",
  "eat shoes",
)

pred = predict(prompt, completions, model)
print(pred)
# use your imagination
```

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/select_a_language_model.html#llama-cpp)
for more info on using GGUF models.
</details>


<details>
<summary>Use an AutoGPTQ model</summary>

[`cappr.huggingface`](https://cappr.readthedocs.io/en/latest/cappr.huggingface.html)
seems to play nice with models loaded via
[`auto_gptq.AutoGPTQForCausalLM.from_quantized`](https://github.com/PanQiWei/AutoGPTQ).
But I haven't thoroughly tested that. See [this
notebook](https://github.com/kddubey/cappr/blob/main/demos/huggingface/auto_gptq.ipynb)
for a minimal demo.
</details>


<details>
<summary>Use an AutoAWQ model</summary>

[`cappr.huggingface.classify_no_cache`](https://cappr.readthedocs.io/en/latest/cappr.huggingface.html)
seems to play nice with models loaded via
[`awq.AutoAWQForCausalLM.from_quantized`](https://github.com/casper-hansen/AutoAWQ). But
I haven't thoroughly tested that. See [this
notebook](https://github.com/kddubey/cappr/blob/main/demos/huggingface/autoawq.ipynb)
for a minimal demo.
</details>


<details>
<summary>Run in batches</summary>

Let's use a PyTorch ``transformers`` model. Also, let's predict probabilities instead of
the class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict_proba

# Load a model and its corresponding tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = [
    "Stephen Curry is a",
    "Martina Navratilova was a",
    "Dexter, from the TV Series Dexter's Laboratory, is a",
    "LeBron James is a",
]

# Each of the prompts could be completed with one of these:
class_names = ("basketball player", "tennis player", "scientist")
prior =       (      1/6,                1/6,            2/3    )
# Say I expect most of my data to have scientists

# Run CAPPr
pred_probs = predict_proba(
    prompts=prompts,
    completions=class_names,
    model_and_tokenizer=(model, tokenizer),
    batch_size=32,  # whatever fits on your CPU/GPU
    prior=prior,
)

# pred_probs[i,j] = probability that prompts[i] is classified as class_names[j]
print(pred_probs.round(1))
# [[0.5 0.3 0.2]
#  [0.3 0.6 0.2]
#  [0.1 0.1 0.8]
#  [0.8 0.2 0. ]]

# For each prompt, which completion is most likely?
pred_class_idxs = pred_probs.argmax(axis=-1)
preds = [class_names[pred_class_idx] for pred_class_idx in pred_class_idxs]
print(preds)
# ['basketball player',
#  'tennis player',
#  'scientist',
#  'basketball player']
```
</details>


<details>
<summary>Run in batches, where each prompt has a different set of possible completions
</summary>

Again, let's use a PyTorch ``transformers`` model to predict probabilities.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict_proba_examples
from cappr import Example

# Load a model and its corresponding tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a sequence of Example objects representing your classification tasks
examples = [
    Example(
        prompt="Jodie Foster played",
        completions=("Clarice Starling", "Trinity in The Matrix"),
    ),
    Example(
        prompt="Batman, from Batman: The Animated Series, was played by",
        completions=("Pete Holmes", "Kevin Conroy", "Spongebob!"),
        prior=      (     1/3      ,      2/3     ,      0      ),
    ),
]

# Run CAPPr
pred_probs = predict_proba_examples(examples, model_and_tokenizer=(model, tokenizer))

# pred_probs[i][j] = probability that examples[i].prompt is classified as
# examples[i].completions[j]
print([example_pred_probs.round(2) for example_pred_probs in pred_probs])
# [array([0.7, 0.3]),
#  array([0.03, 0.97, 0.  ])]

# For each example, which completion is most likely?
pred_class_idxs = [example_pred_probs.argmax() for example_pred_probs in pred_probs]
preds = [
    example.completions[pred_class_idx]
    for example, pred_class_idx in zip(examples, pred_class_idxs)
]
print(preds)
# ['Clarice Starling',
#  'Kevin Conroy']
```
</details>

See
[`demos/llama_cpp/superglue/copa.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb)
for a demonstration of a slightly harder classification task.


## Documentation

https://cappr.readthedocs.io


## Installation

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/installation.html).


## Motivation

Minimize engineering complexity.

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/motivation.html) for more info.

<details>
<summary>Cool</summary>

A handful of experiments suggest that CAPPr squeezes more out of smaller LLMs. See [this
page of the
documentation](https://cappr.readthedocs.io/en/latest/future_research.html).
</details>


<details>
<summary>Honest</summary>

am bored. am unemployed.
</details>


## Performance

<details>
<summary>
Statistical performance
</summary>

I'm still evaluating open source models. For now, see

- the 4-bit 4 GB [Llama 2 COPA
  demo](https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/superglue/copa.ipynb)
- the 4-bit 4 GB [Llama 2 AG News
  demo](https://github.com/kddubey/cappr/blob/main/demos/llama_cpp/ag_news.ipynb)
- and this (minimal but surprising) 3 GB [StableLM
  demo](https://github.com/kddubey/cappr/blob/main/demos/auto_gptq.ipynb).

For OpenAI models, see

[2 SuperGLUE
datasets](https://github.com/kddubey/cappr/blob/main/demos/openai/superglue)

[RAFT zero-shot training
sets](https://github.com/kddubey/cappr/blob/main/demos/openai/raft)

TODO: summary tables/spiderwebs
</details>


<details>
<summary>
Computational performance
</summary>

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/computational_performance.html).
</details>


## How it works

You input a `prompt` string, a `end_of_prompt` string (a whitespace or empty) and a set
of candidate `completion` strings such that the string—

```python
{prompt}{end_of_prompt}{completion}
```

—is a naturally flowing thought. CAPPr picks the `completion` which is mostly likely to
follow `prompt` by computing the:

> **C**ompletion<br>
  **A**fter<br>
  **P**rompt<br>
  **Pr**obability<br>

The method is fleshed out in my [question on Cross
Validated](https://stats.stackexchange.com/q/601159/337906).


## Related work

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/related_work.html).


## Local development

### Setup

1. Create a new Python 3.8+ virtual environment. Activate the venv. I use
   [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). For this
   example, let's create a virtual environment called `cappr`
   using Python's native `venv`:

   ```bash
   cd your/venvs

   python3 -m venv cappr

   source cappr/bin/activate

   python -m pip install wheel --upgrade pip
   ```

2. `cd` to wherever you store projects, and clone the repo (or fork it and clone that)
   there

    ```bash
    cd your/projects

    git clone https://github.com/kddubey/cappr.git
    ```

3. `cd` to the repo and install this package in editable mode, along with development
   requirements (after ensuring that your venv is activated!)

   ```bash
   cd cappr

   python -m pip install -e ".[dev]"

   pre-commit install
   ```

4. Download [the tiny GGUF Llama model](https://huggingface.co/aladar/TinyLLama-v0-GGUF)
   I uploaded to HF

   ```bash
   huggingface-cli download \
   aladar/TinyLLama-v0-GGUF \
   TinyLLama-v0.Q8_0.gguf \
   --local-dir ./tests/llama_cpp/fixtures/models \
   --local-dir-use-symlinks False
   ```


### VS code extensions for development

  * [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).
  Use the numpy format, and check "Start On New Line".
  * Set Python formatting to `black`.
  * [Rewrap](https://stkb.github.io/Rewrap/). Enable Auto Wrap.

And set the vertical line ruler to 88.

### Testing

From the repo home directory `cappr`:

```
pytest
```

Note that a few small transformers and tokenizers will be downloaded to your computer.

Sometimes I get worried about bigger code changes. So consider additionally testing
statistical performance by running an appropriate demo in
[`demos`](https://github.com/kddubey/cappr/tree/main/demos).

When you add a new testing module, add it to the list in `pyproject.toml`. The list is
in order of dependencies: `Example` and `utils` must pass for the rest of the modules to
pass.

To test a specific module, e.g., `huggingface`:

```
pytest -k huggingface
```

To see uncovered line changes:

```
pytest --cov=cappr --cov-report term-missing
```

### Docs

To test changes to documentation, first locally build them from the repo home directory
`cappr` via

```
cd docs

make html
```

and then preview them by opening `docs/build/html/index.html` in your browser.

After merging code to main, the official docs will be automatically built and published.

### Release

[Bump the
version](https://github.com/kddubey/cappr/commit/d1f7dd51fa702c123bdfb0bcb97535995641c224),
and then create a new release on GitHub. A new version of the package will then be
automatically published on PyPI.

Try to follow [semantic versioning](https://semver.org/) guidelines, even though I
haven't been great at that so far.

## Todo

I'm dumping TODOs here:

[Code changes](https://github.com/users/kddubey/projects/1/views/1)

[Reseach experiments](https://github.com/users/kddubey/projects/2)

# CAPPr: zero-shot text classification using autoregressive language models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/cappr/badge/?version=latest)](https://cappr.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/kddubey/cappr/actions/workflows/test.yml/badge.svg)](https://github.com/kddubey/cappr/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/kddubey/cappr/branch/main/graph/badge.svg?token=NYIL076PSM)](https://codecov.io/gh/kddubey/cappr)
[![PyPI - Package Version](https://img.shields.io/pypi/v/cappr?logo=pypi&style=flat&color=orange)](https://pypi.org/project/cappr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Perform zero-shot text classification by estimating the probability that an inputted
completion comes after an inputted prompt. Hence the name:

> **C**ompletion<br>
  **A**fter<br>
  **P**rompt<br>
  **Pr**obability<br>

The method is fleshed out in my [question on Cross Validated](https://stats.stackexchange.com/q/601159/337906).


## Usage

<details>
<summary>Use a model from the OpenAI API</summary>

Specifically, this model must be compatible with the
[/v1/completions](https://platform.openai.com/docs/models/model-endpoint-compatibility)
endpoint.

```python
from cappr.openai.classify import predict

prompt = """
This is a tweet about a movie: "Oppenheimer was pretty good. But 3 hrs...cmon Nolan."
This tweet contains the following criticism:
""".strip("\n")

completions = ("bad message", "too long", "unfunny")

pred = predict(prompt, completions, model="text-ada-001")
print(pred)
# 'too long'
```

Notice that the completions can contain many tokens.
</details>

<details>
<summary>Extract the final answer from a step-by-step completion</summary>

Step-by-step and chain-of-thought prompts are highly effective ways to get an LLM to
"reason" about more complex tasks. But if you need a structured output, a step-by-step
completion is unwieldy. Use CAPPr to extract the final answer from these types of
completions, given a list of possible answers.

See this idea in action [here in the
docs](https://cappr.readthedocs.io/en/latest/4_user_guide.html#select-a-prompt-completion-format).
CAPPr is **100% guaranteed** to return an output from the list of answers.
</details>

<details>
<summary>Use a model from the HuggingFace model hub</summary>

Specifically, this model must be able to be loaded using
`transformers.AutoModelForCausalLM.from_pretrained`.

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
# 'Mercury'
```

For an example with Llama 2, see the notebook
[`demos/llama2/copa.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/llama2/copa.ipynb)
or
[`demos/llama2/quick_check_correctness.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/llama2/quick_check_correctness.ipynb).
So far, CAPPr has been tested for correctness on the following architectures:
  - GPT-2
  - GPT-J
  - Llama
  - Llama 2 (chat, raw, and its GPTQd versions).

Raise an issue to lmk that you don't see your architecture on this list.

</details>

<details>
<summary>Run in batches</summary>

Let's use `huggingface` for this example cuz it's free. And let's predict probabilities
instead of the class.

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
pred_class_idxs = pred_probs.argmax(axis=1)
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

Again, let's use `huggingface` to predict probabilities.

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

More examples are linked [here in the
documentation](https://cappr.readthedocs.io/en/latest/5_examples.html).

See
[`demos/superglue/copa.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/superglue/copa.ipynb)
for a demonstration of a slightly harder classification task.


## Documentation

https://cappr.readthedocs.io


## Setup

If you intend on using OpenAI models, [sign up for the OpenAI API
here](https://platform.openai.com/signup), and then set the environment variable
`OPENAI_API_KEY`. For zero-shot classification, OpenAI models are currently far ahead of
others. But using them will cost ya 💰!

Install with `pip`:

```
pip install cappr
```

(Optional) Install requirements for HuggingFace models

```
pip install "cappr[hf]"
```

(Optional) Install requirements for running
[`demos`](https://github.com/kddubey/cappr/tree/main/demos)

```
pip install "cappr[demos]"
```


## Motivation

Create a more usable zero-shot text classification interface than
[classification via sampling (CVS)](https://platform.openai.com/docs/guides/completion/classification).

<details>
<summary>Short</summary>

With CVS, your job is to write up your classification task in a `prompt` string, and
then write custom code to post-process arbitrary `completion`/output strings.

With CAPPr, your job starts and stops at writing up your classification task as a
`{prompt}{end_of_prompt}{completion}` string.
</details>

<details>
<summary>Long</summary>

Please see [this page of the
documentation](https://cappr.readthedocs.io/en/latest/2_motivation.html).

</details>

<details>
<summary>Unstudied</summary>

I'm curious to see how much easier estimation/discrimination is than generation. In
[`demos/superglue/copa.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/superglue/copa.ipynb),
CVS using OpenAI's `text-curie-001` is less than 50% accurate, while CAPPr is 80%
accurate.

</details>

<details>
<summary>Honest</summary>

Keep myself busy

</details>


## Results

<details>
<summary>
Statistical performance
</summary>

Not too shabby. TODO: summary table comparing CVS vs. CAPPr vs. few-shot methods like
SetFit and PET.

[2 SuperGLUE datasets](https://github.com/kddubey/cappr/blob/main/demos/superglue)

[RAFT zero-shot training sets](https://github.com/kddubey/cappr/blob/main/demos/raft)
</details>


<details>
<summary>
Computational performance
</summary>

One concern was that CAPPr requires as many `model()` calls as there are classes. But in
the CAPPr scheme, we can simply cache each attention block's keys and values for the
prompts. This feature is already supported by `AutoModelForCausalLM`s. See [this
code](https://github.com/kddubey/cappr/blob/main/src/cappr/huggingface/classify.py) for
the implementation. Note that this caching is not implemented for OpenAI models, as I
can't control their backend. **This means that when running `cappr.openai` functions,
you'll be on the *cappr (no cache)* line** :-(

![](/docs/source/_static/scaling_classes/batch_size_32.png)

*Figure 1: [COPA](https://people.ict.usc.edu/~gordon/copa.html) dataset, repeating the
choices to simulate multi-class classification tasks. [GPT-2
(small)](https://huggingface.co/gpt2) was run on a Tesla K80 GPU (whatever was free in
Google Colab in March 2023). 96 classification inputs were processed in batches of size
32. Each point in the graph is a median of 5 runs. For classification via sampling
(CVS), exactly 4 tokens were generated for each prompt, which is the number of tokens in
`'\n\nAnswer A'`. 1-token times are also shown. But for COPA (and other multiple-choice
style prompts), that may result in lower zero-shot accuracy, as most of the sampled
choices come after the first token.*

See the [`demos/computational_analysis.ipynb`
notebook](https://github.com/kddubey/cappr/blob/main/demos/computational_analysis.ipynb).

</details>


## Related work

The idea behind CAPPr is very well known. There are many papers where averaging token
log-probabilities is a useful subroutine. Here are some papers which focus on this idea.

While [benchmarking this
method](https://github.com/kddubey/cappr/blob/main/demos/superglue/wsc.ipynb) on the
Winograd Schema Challenge, I found that [this paper](https://arxiv.org/abs/1806.02847)
is very similar:

> Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning." arXiv
> preprint arXiv:1806.02847 (2018).

[PET with multiple masks](https://arxiv.org/abs/2009.07118) also aggregates token
probabilities to do prompt-completion classification, but these probabilities are
assumed to come from masked language models like BERT.

> Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small language
> models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).


## Local development

(If you're on a Windows system, some of the commands below will be different.)

### Setup

1. Create a new Python 3.8+ virtual environment. Activate the venv. I use
   [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/). For
   example, let's create a virtual environment called `cappr`
   using Python's native `venv`:

   ```bash
   cd your/venvs

   python3 -m venv cappr

   source cappr/bin/activate

   python -m pip install wheel --upgrade pip
   ```

2. `cd` to wherever you store projects, and clone the repo (or fork it and clone that) there

    ```bash
    cd your/projects

    git clone https://github.com/kddubey/cappr.git
    ```

3. `cd` to the repo and install this package in editable mode, along with development
   requirements (**ensure your venv is activated**)

   ```
   cd cappr

   python -m pip install -e ".[dev]"
   ```

### VS code extensions for development

  * [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).
  Use the numpy format, and check "Start On New Line".
  * Set Python formatting to `black`.
  * [Rewrap](https://stkb.github.io/Rewrap/). Enable Auto Wrap.

### Testing

From the repo home directory `cappr`:

```
pytest
```

Note that a few small transformers will be downloaded to your computer.

If a code change could affect statistical performance, then additionally test
statistical performance by running an appropriate demo in
[`demos`](https://github.com/kddubey/cappr/tree/main/demos).

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


## Todo

Idk how to use GitHub projects, but I've put TODOs here:

[Code changes](https://github.com/users/kddubey/projects/1/views/1)

[Reseach experiments](https://github.com/users/kddubey/projects/2)

# CAPPr: Completion After Prompt Probability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?logo=python&style=for-the-badge)](https://www.python.org/downloads/release/python-380/)
[![tests](https://img.shields.io/github/actions/workflow/status/kddubey/cappr/test.yml?style=for-the-badge&logo=github&label=tests)](https://github.com/kddubey/cappr/actions/workflows/test.yml)
[![codecov](https://img.shields.io/codecov/c/github/kddubey/cappr?token=NYIL076PSM&style=for-the-badge&logo=codecov&color=%2309BC00)](https://codecov.io/gh/kddubey/cappr)
[![PyPI - Package Version](https://img.shields.io/pypi/v/cappr?logo=pypi&style=for-the-badge&color=orange)](https://pypi.org/project/cappr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-purple.svg?logo=apache&style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)

<!-- [![Documentation Status](https://readthedocs.org/projects/cappr/badge/?version=latest&style=for-the-badge)](https://cappr.readthedocs.io/en/latest/?badge=latest) -->

Make your LLM pick from a list of choices. <br>
Or compute the probability of a completion given a prompt, which may be
[useful](https://cappr.readthedocs.io/en/latest/related_work.html). <br>
Squeeze [more](https://cappr.readthedocs.io/en/latest/statistical_performance.html) out
of open source LLMs.


## Usage

<details>
<summary>Use a GGUF model</summary>

```python
from llama_cpp import Llama
from cappr.llama_cpp.classify import predict

model = Llama("./TinyLLama-v0.Q8_0.gguf", verbose=False)

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
<summary>Use a Hugging Face transformers model</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict

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
documentation](https://cappr.readthedocs.io/en/latest/select_a_language_model.html#hugging-face)
for more info on using ``transformers`` models.
</details>


<details>
<summary>Cache instructions to save time</summary>

Many prompts start with the same set of instructions, e.g., a system prompt plus a
handful of example input-output pairs. Instead of repeatedly running the model on common
instructions, cache them so that future computations are faster.

Here's an
example using
[`cappr.huggingface.classify.cache_model`](https://cappr.readthedocs.io/en/latest/cappr.huggingface.classify.html#cappr.huggingface.classify.cache_model).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import cache_model, predict

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_and_tokenizer = (model, tokenizer)

# Create data
prompt_prefix = '''Instructions: complete the sequence.
Here are examples:
A, B, C => D
1, 2, 3 => 4

Complete this sequence:'''

prompts = ["a, b, c =>", "X, Y =>"]
completions = ["d", "Z", "Hi"]

# Cache prompt_prefix because it's used for all prompts
cached_model_and_tokenizer = cache_model(
    model_and_tokenizer, prompt_prefix
)

# Compute
preds = predict(
    prompts, completions, cached_model_and_tokenizer
)
print(preds)
# ['d', 'Z']
```
</details>


<details>
<summary>Use a model from the OpenAI API</summary>

This model must be compatible with the
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
</details>


<details>
<summary>Run in batches, predict probabilities</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict_proba

# Load a model and its tokenizer
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

Again, let's predict probabilities.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from cappr.huggingface.classify import predict_proba_examples
from cappr import Example

# Load a model and its tokenizer
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
pred_probs = predict_proba_examples(
    examples, model_and_tokenizer=(model, tokenizer)
)

# pred_probs[i][j] = probability that examples[i].prompt is classified as
# examples[i].completions[j]
print([example_pred_probs.round(2) for example_pred_probs in pred_probs])
# [array([0.7, 0.3]),
#  array([0.03, 0.97, 0.  ])]

# For each example, which completion is most likely?
pred_class_idxs = [
    example_pred_probs.argmax() for example_pred_probs in pred_probs
]
preds = [
    example.completions[pred_class_idx]
    for example, pred_class_idx in zip(examples, pred_class_idxs)
]
print(preds)
# ['Clarice Starling',
#  'Kevin Conroy']
```
</details>


See the [`demos`](https://github.com/kddubey/cappr/blob/main/demos/) for demonstrations
of slightly harder classification tasks.

For CAPPr, GPTQ models are the most computationally performant. These models are
compatible with `cappr.huggingface.classify`. See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/select_a_language_model.html#hugging-face)
for more info on using these models.


## Documentation

https://cappr.readthedocs.io


## Installation

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/installation.html).


## Related work

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/related_work.html).


## Motivation

Reduce engineering complexity.

See [this page of the
documentation](https://cappr.readthedocs.io/en/latest/motivation.html) for more info.


## Performance

[Statistical performance](https://cappr.readthedocs.io/en/latest/statistical_performance.html)

[Computational performance](https://cappr.readthedocs.io/en/latest/computational_performance.html)


## How it works

You input a `prompt` string, a `end_of_prompt` string (a whitespace or empty) and a set
of candidate `completion` strings such that the string—

```python
{prompt}{end_of_prompt}{completion}
```

—is a naturally flowing thought. CAPPr picks the `completion` which is mostly likely to
follow `prompt` by computing the—

> **C**ompletion<br>
  **A**fter<br>
  **P**rompt<br>
  **Pr**obability<br>

—as fleshed out in my [question on Cross
Validated](https://stats.stackexchange.com/q/601159/337906).


## Local development

See [this page of the documentation](https://cappr.readthedocs.io/en/latest/local.html).


## Todo

I'm dumping todos here:

[Code changes](https://github.com/users/kddubey/projects/1/views/1)

[Reseach experiments](https://github.com/users/kddubey/projects/2)

Feel free to raise issues ofc

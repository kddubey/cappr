# *CALLM*: zero-shot text *C*lassification using *A*utoregressive *LLM*s

[![test](https://github.com/kddubey/callm/actions/workflows/test.yml/badge.svg)](https://github.com/kddubey/callm/actions/workflows/test.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Perform zero-shot text classification based on the following idea: for a given prompt 
and completion text pair, what's the probability that the completion comes after the 
prompt? The method is fleshed out
[here in CrossValidated](https://stats.stackexchange.com/q/601159/337906).

⚠️ This package is currently under construction. ⚠️

## Usage

<details>
<summary>Use a model from the OpenAI API</summary>

Specifically, this model must be compatible with the
[/v1/completions](https://platform.openai.com/docs/models/model-endpoint-compatibility)
endpoint.

Let's classify
[this sentiment example](https://platform.openai.com/docs/guides/completion/classification)
from the OpenAI text completion docs.

```python
from callm.openai.classify import predict_proba

tweet = 'I loved the new Batman movie!'
prompt = f'Tweet: {tweet}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
prior       = (   1/8    ,    1/8   ,     3/4   )

pred_probs = predict_proba(prompts=[prompt],
                           completions=class_names,
                           prior=prior,
                           model='text-ada-001')

print(pred_probs.round(3))
# [[0.979 0.001 0.02 ]]

pred_class_idxs = pred_probs.argmax(axis=1)
print([class_names[pred_class_idx] for pred_class_idx in pred_class_idxs])
# ['positive']
```

</details>

<details>
<summary>Use a model from the HuggingFace model hub</summary>

Specifically, this model must be able to be loaded using
`transformers.AutoModelForCausalLM.from_pretrained(model)`.

Smaller LMs may not work well. But there will likely be better ones in the hub soon.

```python
from callm.huggingface.classify import predict_proba

tweet = 'I loved the new Batman movie!'
prompt = f'Tweet: {tweet}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
prior = None  # uniform prior

pred_probs = predict_proba(prompts=[prompt],
                           completions=class_names,
                           prior=prior,
                           model='gpt2')

print(pred_probs.round(3))
# [[0.668 0.006 0.326]]

pred_class_idxs = pred_probs.argmax(axis=1)
print([class_names[pred_class_idx] for pred_class_idx in pred_class_idxs])
# ['positive']
```
</details>

<details>
<summary>Run in batches</summary>

Let's use `huggingface` for this example cuz it's free.

```python
from callm.huggingface.classify import predict_proba

prompts = [
    'Stephen Curry is a',
    'Martina Navratilova was a',
    "Dexter, from the TV Series, Dexter's Laboratory, is a",
    'LeBron James is a',    
]

# each of the prompts could be completed with one of these:
class_names = (
    'basketball player',
    'tennis player',
    'scientist'
)

prior = (
    1/6,  # few
    1/6,  # few
    2/3   # there are more
)

pred_probs = predict_proba(prompts=prompts,
                           completions=class_names,
                           prior=prior,
                           batch_size=32,  # whatever fits on your CPU/GPU
                           model='gpt2')

# pred_probs[i,j] = probability that prompts[i] is classified as class_names[j]
print(pred_probs.round(1))
# [[0.5 0.3 0.2]
#  [0.3 0.6 0.2]
#  [0.1 0.1 0.7]
#  [0.8 0.2 0. ]]

# for each prompt, which completion is most likely?
pred_class_idxs = pred_probs.argmax(axis=1)
print([class_names[pred_class_idx] for pred_class_idx in pred_class_idxs])
# ['basketball player',
#  'tennis player',
#  'scientist',
#  'basketball player']
```
</details>

<details>
<summary>Run in batches, where each prompt has a different set of possible completions
</summary>

Again, let's use `huggingface` here.

```python
import numpy as np

from callm.example import Example
from callm.huggingface.classify import predict_proba_examples

examples = [
    Example(prompt='Jodie Foster played',
            completions=('Clarice Starling', 'Trinity in The Matrix')),
    Example(prompt='Batman, from Batman: The Animated Series, was played by',
            completions=('Kevin Conroy', 'Pete Holmes', 'Spongebob!'),
            prior=      (     2/3      ,      1/3     ,      0      ))
]

pred_probs = predict_proba_examples(examples, model='gpt2')

# pred_probs[i][j] = probability that examples[i].prompt is classified as
# examples[i].completions[j]
print([example_pred_probs.round(2)
       for example_pred_probs in pred_probs])
# [array([0.7, 0.3]),
#  array([0.97, 0.03, 0.  ])]

# for each example, which completion is most likely?
pred_class_idxs = [np.argmax(example_pred_probs)
                   for example_pred_probs in pred_probs]
print([example.completions[pred_class_idx]
       for example, pred_class_idx in zip(examples, pred_class_idxs)])
# ['Clarice Starling',
#  'Kevin Conroy']
```
</details>

See [`demos/copa.ipynb`](https://github.com/kddubey/callm/blob/main/demos/copa.ipynb)
for a harder classification task.


## Motivation

Improve my understanding of LMs.

<details>
<summary>Product-y motivation</summary>

Create a more usable zero-shot text classification interface than
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification) (CVS).
[Cookbook here](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit).
With this package's `predict_proba` interface, you no longer have to:
  1. study sampled completion strings which aren't in your label set
  2. figure out how to map them back to the label set
  3. figure out how to transform or point multi-token labels to single tokens, ignoring
  their semantics if they were transformed
  4. ignore your prior over multi-token labels.

This package tries to do one thing well: classification. I'll assess it across
these dimensions: statistical performance, computational performance, and
usability.
</details>

## Setup

If you intend on using OpenAI models,
[sign up for the OpenAI API here](https://openai.com/api/), and then set the environment
variable `OPENAI_API_KEY`. For zero-shot classification, OpenAI models are currently far
ahead of others. But using them will cost ya 💰!

Install from source:

```
python -m pip install git+https://github.com/kddubey/callm.git
```

<details>
<summary>(Optional) Install requirements for HuggingFace models</summary>

```
python -m pip install "callm[hf] @ git+https://github.com/kddubey/callm.git"
```
</details>

<details>
<summary>(Optional) Set up to run demos</summary>

```
python -m pip install "callm[demos] @ git+https://github.com/kddubey/callm.git"
```
</details>


## Related work

While
[benchmarking this method](https://github.com/kddubey/callm/blob/main/demos/wsc.ipynb) 
on the
[Winograd Schema Challenge (WSC)](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html),
I found that [this paper](https://arxiv.org/abs/1806.02847) is pretty similar:

> Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning." arXiv preprint arXiv:1806.02847 (2018).

[This paper](https://arxiv.org/abs/2009.07118) is also similar in spirit:

> Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).


## Testing

### Setup

1. Clone the repo

   ```
   git clone https://github.com/kddubey/callm.git
   ```

2. Create a new Python 3.8+ environment

3. Install this package in editable mode, along with development requirements

   ```
   python -m pip install -e callm[dev]
   ```

### Run tests

```
pytest
```


## Todo

(**) = I'm currently working on this or will work on it really soon

<details>
<summary>Code</summary>

- [ ] Testing
  - [ ] Increase coverage (**)
  - [ ] Standardize (**)
- [ ] Factor out input checks on prompts and completions
- [x] De-automate overzealous auto-docstring stuff
- [ ] HuggingFace `transformers.AutoModelForCausalLM`
  - [x] Optimize backend to allow for parallelization over completions/classes
  - [ ] Fix `end_of_prompt`
  - [ ] Allow user to pass in an instantiated model instead of a string
  - [x] Optional/extra install, so that you can optionally add the hefty
    requirements needed to run `huggingface`
- [x] Put dev requirements in setup extras
- [x] (for me) Auto-enforced code formatting b/c it's getting time-consuming
- [ ] Create a notebook template
- [ ] Docs and user guides (not just docstrings)
</details>

<details>
<summary>Research</summary>

Evaluate on more tasks, and understand its relative advantages and disadvantages vs
other classification methods.

- [ ] Create a user guide, build a table of results comparing competing
  approaches on statistical performance, cost, and computation
- [ ] Make a computational comparison to sampling (**)
  - [ ] Assume I have full freedom to decide how inference works. Demo w/
  GPT-2 (**)
- [ ] More SuperGLUE tasks
- [ ] More real world or harder tasks
  - [ ] Multi-token labels w/ non-uniform prior
- [ ] Calibration
  - [ ] (easy) Is the prior actually effective? Downsample and see
  - [ ] curves
- [ ] Compare against few-shot embeddings
- [ ] Finetune smaller, cheaper model and compare against zero-shot w/ davinci
  - [ ] e.g., GPT-2 from huggingface, `text-ada-001`
  - [ ] Again, compare against sampling
- [ ] Evaluate different aggregation functions. Currently taking mean, but
there was no good motivation for that
- [ ] A bit ambitious: support insertion. For transformers, I think this just
entails manipulating position IDs?
</details>

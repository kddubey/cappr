# CAPPr: zero-shot text classification using autoregressive language models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/cappr/badge/?version=latest)](https://cappr.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/kddubey/cappr/actions/workflows/test.yml/badge.svg)](https://github.com/kddubey/cappr/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/kddubey/cappr/branch/main/graph/badge.svg?token=NYIL076PSM)](https://codecov.io/gh/kddubey/cappr)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - Package Version](https://img.shields.io/pypi/v/cappr?logo=pypi&style=flat&color=orange)](https://pypi.org/project/cappr/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Perform zero-shot text classification by estimating the probability that an inputted
completion comes after an inputted prompt. Hence the name:

> **C**ompletion<br>
  **A**fter<br>
  **P**rompt<br>
  **Pr**obability<br>

The method is fleshed out in my [question on CrossValidated](https://stats.stackexchange.com/q/601159/337906).


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
from cappr.openai.classify import predict

tweet = 'I loved the new Batman movie!'
prompt = f'Tweet: {tweet}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
# optional: let's supply a prior distribution over the classes
prior       = (   1/8    ,    1/8   ,     3/4   )

preds = predict(prompts=[prompt],
                completions=class_names,
                model='text-ada-001',
                prior=prior)
preds
# ['positive']
```
</details>

<details>
<summary>Use a model from the HuggingFace model hub</summary>

Specifically, this model must be able to be loaded using
`transformers.AutoModelForCausalLM.from_pretrained(model)`.

Smaller LMs may not work well. But there will likely be better ones in the hub soon.

```python
from cappr.huggingface.classify import predict

prompt = 'Which planet is closer to the Sun: Mercury or Earth?'

class_names = ('Mercury', 'Earth')
prior = None  # uniform prior

preds = predict(prompts=[prompt],
                completions=class_names,
                model='gpt2',
                prior=prior)
preds
# ['Mercury']
```
</details>

<details>
<summary>Run in batches</summary>

Let's use `huggingface` for this example cuz it's free. And let's predict probabilities
instead of the class.

```python
from cappr.huggingface.classify import predict_proba

prompts = [
    'Stephen Curry is a',
    'Martina Navratilova was a',
    "Dexter, from the TV Series Dexter's Laboratory, is a",
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
                           model='gpt2',
                           batch_size=32,  # whatever fits on your CPU/GPU
                           prior=prior)

# pred_probs[i,j] = probability that prompts[i] is classified as class_names[j]
print(pred_probs.round(1))
# [[0.5 0.3 0.2]
#  [0.3 0.6 0.2]
#  [0.1 0.1 0.8]
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

Again, let's use `huggingface` to predict probabilities. And this time, let's pass in an 
instantiated model and tokenizer instead of its name. That way, the model isn't
re-loaded every time you wanna use it.

```python
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from cappr import Example
from cappr.huggingface.classify import predict_proba_examples

examples = [
    Example(prompt='Jodie Foster played',
            completions=('Clarice Starling', 'Trinity in The Matrix')),
    Example(prompt='Batman, from Batman: The Animated Series, was played by',
            completions=('Pete Holmes', 'Kevin Conroy', 'Spongebob!'),
            prior=      (     1/3      ,      2/3     ,      0      ))
]

model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pred_probs = predict_proba_examples(examples,
                                    model_and_tokenizer=(model, tokenizer))

# pred_probs[i][j] = probability that examples[i].prompt is classified as
# examples[i].completions[j]
print([example_pred_probs.round(2)
       for example_pred_probs in pred_probs])
# [array([0.7, 0.3]),
#  array([0.03, 0.97, 0.  ])]

# for each example, which completion is most likely?
pred_class_idxs = [np.argmax(example_pred_probs)
                   for example_pred_probs in pred_probs]
print([example.completions[pred_class_idx]
       for example, pred_class_idx in zip(examples, pred_class_idxs)])
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

https://cappr.readthedocs.io/en/latest/

Please let me know if you find the writing too dense. The main motivation behind this
project is simplicity :-)


## Setup

If you intend on using OpenAI models, [sign up for the OpenAI API
here](https://platform.openai.com/signup), and then set the environment variable
`OPENAI_API_KEY`. For zero-shot classification, OpenAI models are currently far ahead of
others. But using them will cost ya 💰!

Install with `pip`:

```
python -m pip install cappr
```

<details>
<summary>(Optional) Install requirements for HuggingFace models</summary>

```
python -m pip install cappr[hf]
```
</details>

<details>
<summary>(Optional) Install requirements for running demos</summary>

```
python -m pip install cappr[demos]
```
</details>


## Motivation

Create a more usable zero-shot text classification interface than
[classification via sampling (CVS)](https://platform.openai.com/docs/guides/completion/classification).

<details>
<summary>Short</summary>

In CVS, your job is to write up your classification task in a `prompt` string, and then
write custom code to post-process arbitrary `completion`/output strings.

In CAPPr, your job starts and stops at writing up your classification task as a
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
you'll be on the *cappr (slow)* line** :-(

![](/docs/source/_static/scaling_classes/batch_size_32.png)

*Figure 1: [COPA](https://people.ict.usc.edu/~gordon/copa.html) dataset, repeating the
choices to simulate multi-class classification tasks. [GPT-2
(small)](https://huggingface.co/gpt2) was run on a Tesla K80 GPU (whatever was free in
Google Colab in March 2023, I'm not hardware savvy). 96 classification inputs were
processed in batches of size 32. Each point in the graph is a median of 5 runs. For
classification via sampling (CVS), exactly 4 tokens were generated for each prompt,
which is the number of tokens in `'\n\nAnswer A'`. 1-token times are also shown. But for
COPA (and other multiple-choice style prompts), that may result in lower zero-shot
accuracy, as most of the sampled choices come after the first token.*

See the [`demos/computational_analysis.ipynb`
notebook](https://github.com/kddubey/cappr/blob/main/demos/computational_analysis.ipynb).

</details>


## Related work

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


## Contributing

TODO


## Testing

### Setup

1. Clone the repo

   ```
   git clone https://github.com/kddubey/cappr.git
   ```

2. Create a new Python 3.8+ environment

3. Install this package in editable mode, along with development requirements

   ```
   python -m pip install -e .[dev]
   ```

### Run tests

```
pytest
```

Dumping VS code extensions for development:
  * [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring).
  Use the numpy format.
  * [Set Python formatting to
    `black`](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0).
  * [Rewrap](https://stkb.github.io/Rewrap/). Enable Auto Wrap.
  * [TOML Language
    Support](https://marketplace.visualstudio.com/items?itemName=be5invis.toml)


## Todo

(**) = I'm currently working on this or will work on it really soon

<details>
<summary>Code</summary>

- [ ] Testing
  - [ ] Increase test cases
  - [ ] Some more standardization b/t openai and huggingface tests
  - [x] Add code coverage badge to look cool
  - [ ] Test input checks
- [x] Small CPU speed-ups
  - [x] For constant-completions input, vectorize `agg_log_probs`
  - [x] For `examples` input, if # completions per prompt is constant, vectorize
  `posterior_prob`
- [ ] Make progress bars optional, since inference often isn't batched
- [ ] Factor out input checks (on prompts and completions)
- [x] De-automate overzealous auto-docstring stuff :-(
- [ ] HuggingFace `transformers.AutoModelForCausalLM`
  - [x] Optimize backend to enable greater scaling wrt # completions/classes
  - [x] Get it working on single-GPU, check that it's faster than sampling assuming
  batching
    - [ ] Get to the bottom of why it's slower w/o batching
  - [ ] Allow non-`' '` `end_of_prompt`! I'll have to go back to the drawing board I
  think
  - [ ] Consider batchifying the completions again, since they technically don't go in
  batches of `batch_size`; the actual batch size is the sum of the number of completions
  corresponding to the batch of prompts! Not a huge memory issue I think b/c completions
  are usually half as long. But it should be configurable at the very least.
  - [ ] Factor out repeated code b/t `classify` and `classify_no_cache`
  - [ ] Support [Inference
    Endpoints](https://huggingface.co/docs/inference-endpoints/index)?
  - [ ] Support TensorFlow models if it's easy
  - [ ] Support priming, as in: cache it
- [x] (for me) Auto-enforced code formatting b/c it's getting time-consuming
- [ ] Allow for multi-label classification
  - [ ] Pass `normalize` as an argument to predict_proba functions
  - [ ] For `huggingface`, add note that you'll get faster results by passing all
  labels at once (assuming prompt is identical for each label)
- [ ] Create a notebook template
- [ ] Fill in missing or non-numpy docstrings
</details>

<details>
<summary>Research</summary>

Evaluate on more datasets, and understand its relative advantages and disadvantages vs
other classification methods.

- [ ] RAFT benchmark (**)
  - [x] Zero-shot training scores
  - [ ] Submit zero-shot test predictions
  - [ ] Few-shot (priming) training scores
  - [ ] Submit few-shot test predictions
- [ ] Create a user guide, build a table of results comparing competing approaches on
statistical performance, cost, and computation
- [ ] Make a computational comparison to sampling (**)
  - [x] Assume I have full freedom to decide how inference works. Demo w/
  GPT-2. Process inputs in batches.
  - [ ] Process inputs 1-by-1
- [ ] More SuperGLUE tasks?
  - [ ] Re-run COPA demo w/ left-stripped completions (there are a few which aren't)
- [ ] Calibration
  - [ ] Is the prior actually effective? Downsample and see
  - [ ] curves
- [ ] Compare against few-shot embeddings
- [ ] Finetune smaller, cheaper model and compare against zero-shot w/ davinci
  - [ ] e.g., GPT-2 from huggingface, `text-ada-001`
  - [ ] Again, compare against sampling
- [ ] Evaluate a bigger model like GPT-J
- [ ] Evaluate different aggregation functions. Currently taking mean, but
there was no good theory for that
- [ ] A bit ambitious: support insertion and backwards-completion. Quite ambitious b/c
manipulating position IDs isn't sufficient (I think).
</details>

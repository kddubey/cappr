# CAPPr: zero-shot text classification using autoregressive LMs

[![test](https://github.com/kddubey/cappr/actions/workflows/test.yml/badge.svg)](https://github.com/kddubey/cappr/actions/workflows/test.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Perform zero-shot text classification based on the following idea: for a given prompt 
and completion text pair, what's the probability that the completion comes after the 
prompt? Hence the name:

> **C**ompletion<br>
  **A**fter<br>
  **P**rompt<br>
  **Pr**obability<br>

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
from cappr.openai.classify import predict

tweet = 'I loved the new Batman movie!'
prompt = f'Tweet: {tweet}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
prior       = (   1/8    ,    1/8   ,     3/4   )

preds = predict(prompts=[prompt],
                completions=class_names,
                prior=prior,
                model='text-ada-001')
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
                prior=prior,
                model='gpt2')
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
                           prior=prior,
                           batch_size=32,  # whatever fits on your CPU/GPU
                           model='gpt2')

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
            completions=('Kevin Conroy', 'Pete Holmes', 'Spongebob!'),
            prior=      (     2/3      ,      1/3     ,      0      ))
]

model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
pred_probs = predict_proba_examples(examples,
                                    model_and_tokenizer=(model, tokenizer))

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

TODO: see the user guide for more details 

See [`demos/copa.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/copa.ipynb)
for a demonstration of a slightly harder classification task.


## Setup

If you intend on using OpenAI models,
[sign up for the OpenAI API here](https://openai.com/api/), and then set the environment
variable `OPENAI_API_KEY`. For zero-shot classification, OpenAI models are currently far
ahead of others. But using them will cost ya 💰!

Install from source:

```
python -m pip install git+https://github.com/kddubey/cappr.git
```

<details>
<summary>(Optional) Install requirements for HuggingFace models</summary>

```
python -m pip install "cappr[hf] @ git+https://github.com/kddubey/cappr.git"
```
</details>

<details>
<summary>(Optional) Set up to run demos</summary>

```
python -m pip install "cappr[demos] @ git+https://github.com/kddubey/cappr.git"
```
</details>


## Motivation

Answer my [CrossValidated question](https://stats.stackexchange.com/q/601159/337906),
improve my understanding of LMs.

<details>
<summary>Product-y motivation</summary>

Create a more usable zero-shot text classification interface than
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification) (CVS).
In CVS, completion tokens are sampled from a language model given a prompt.
[Cookbook here](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit).
With this package's `predict` interface, you no longer have to:
  1. study sampled completion strings which aren't in your label set
  2. figure out how to map them back to the label set
  3. figure out how to transform or point multi-token labels to single tokens, ignoring
  their semantics if they were transformed
  4. ignore your prior over multi-token labels.

All you have to do is figure out how to frame your classification task as a
`{prompt}{end_of_prompt}{completion}` string.

</details>


## Related work

While
[benchmarking this method](https://github.com/kddubey/cappr/blob/main/demos/wsc.ipynb) 
on the
[Winograd Schema Challenge (WSC)](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html),
I found that [this paper](https://arxiv.org/abs/1806.02847) is pretty similar:

> Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning." arXiv preprint arXiv:1806.02847 (2018).

[This paper](https://arxiv.org/abs/2009.07118) is also similar in spirit:

> Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).


## Results

<details>
<summary>
Statistical performance
</summary>
Performs ok based on 2 datasets, when compared to classification via sampling (CVS).
I need to run it on more ofc. Will update

  * [`demos/copa.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/copa.ipynb)
  * [`demos/wsc.ipynb`](https://github.com/kddubey/cappr/blob/main/demos/wsc.ipynb)
</details>


<details>
<summary>
Computational performance
</summary>

One concern was that CAPPr requires as many `model()` calls as there are classes. But
in the CAPPr scheme, we can simply cache each attention block's keys and values for the 
prompts. This feature is already supported by `AutoModelForCausalLM`s. See
[this code](https://github.com/kddubey/cappr/blob/main/cappr/huggingface/classify.py)
for the implementation. Note that this caching is not implemented for OpenAI models,
as I can't control their backend.
**This means that when running `cappr.openai` functions, you'll be on the *cappr (slow)* line** :-(

![](/demos/scaling_classes.png)

*Figure 1: [COPA](https://people.ict.usc.edu/~gordon/copa.html) dataset, repeating the choices to simulate multi-class classification tasks. [GPT-2 (small)](https://huggingface.co/gpt2) was run on a Tesla K80 GPU (whatever was free in Google Colab in March 2023, idk a lick of hardware lol). 160 classification inputs were processed in batches of size 32. Each point in the graph is a median of 5 runs. For classification via sampling (CVS), exactly 4 tokens were generated for each prompt, which is the number of tokens in `'\n\nAnswer A'`. 1-token times are also shown. But for COPA (and other multiple-choice style prompts), that may result in lower zero-shot accuracy, as most of the sampled choices come after the first token.*

[See the `demos/computational_analysis.ipynb` notebook](https://github.com/kddubey/cappr/blob/main/demos/computational_analysis.ipynb).

</details>


## Testing

### Setup

1. Clone the repo

   ```
   git clone https://github.com/kddubey/cappr.git
   ```

2. Create a new Python 3.8+ environment

3. Install this package in editable mode, along with development requirements

   ```
   python -m pip install -e cappr[dev]
   ```

### Run tests

```
pytest
```

Dumping VS code extensions for development:
  * [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring). This tool is awesome
    * format: numpy
  * [Set Python formatting to `black`](https://dev.to/adamlombard/how-to-use-the-black-python-code-formatter-in-vscode-3lo0)


## Todo

(**) = I'm currently working on this or will work on it really soon

<details>
<summary>Code</summary>

- [ ] Testing
  - [ ] Increase coverage (**)
  - [ ] Standardize (**)
- [ ] ReadTheDocs w/ user guide (**)
  - [ ] Need to figure out how to cleanly automate some of the manual things needed to
  build from scratch
- [ ] Publish to PyPi
- [ ] Factor out input checks on prompts and completions
- [x] De-automate overzealous auto-docstring stuff
- [ ] Make progress bar optional
- [ ] HuggingFace `transformers.AutoModelForCausalLM`
  - [x] Optimize backend to enable greater scaling wrt # completions/classes
  - [x] Get it working on single-GPU, check that it's faster than sampling
  - [ ] Allow non-`' '` `end_of_prompt`!
  - [ ] Factor out repeated code b/t fast and slow modules? I don't really care
  - [ ] Set device at function level, not globally
  - [ ] Support TensorFlow models
- [x] (for me) Auto-enforced code formatting b/c it's getting time-consuming
- [ ] Allow for multi-label classification
  - [ ] Pass `normalize` as an argument to predict_proba functions
  - [ ] For `huggingface`, add note that you'll get faster results by passing all
  labels at once (assuming prompt is identical for each label)
- [x] Small CPU speed-ups
  - [x] For constant-completions input, vectorize `agg_log_probs`
  - [x] For `examples` input, if # completions per prompt is constant, vectorize
  `posterior_prob`
- [ ] Annotate arrays and tensors using
[this cool strategy](https://stackoverflow.com/a/64032593/18758987),
or [`nptyping`](https://github.com/ramonhagenaars/nptyping) for arrays
- [ ] Create a notebook template
</details>

<details>
<summary>Research</summary>

Evaluate on more tasks, and understand its relative advantages and disadvantages vs
other classification methods.

- [ ] Re-run COPA demo w/ left-stripped completions (there are a few which aren't)
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
- [ ] Evaluate a bigger model like GPT-J
- [ ] Evaluate different aggregation functions. Currently taking mean, but
there was no good motivation for that
- [ ] A bit ambitious: support insertion. For transformers, I think this just
entails manipulating position IDs?
</details>

# Zero-shot text classification

Perform zero-shot text classification based on the following idea: for a given
prompt and completion, what's the probability that the completion follows the
prompt?

The method is fleshed out
[here in CrossValidated](https://stats.stackexchange.com/q/601159/337906)
(I'm chicxulub). The first demo of this method is
[here, in `demos/copa.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demos/copa.ipynb).


## Disclaimers

This package only supports [language models (LMs) in OpenAI's text completion API](https://platform.openai.com/docs/models/gpt-3),
which you gotta pay for. Prices are [here](https://openai.com/api/pricing/).

Moreover, this code may not be ready for production use. And I still need to
evaluate the method on more classification datasets and tasks.

If you're something of an ML engineer, and you have labeled and unlabeled text, 
there are likely far better alternatives to this method.
[PET training](http://timoschick.com/explanatory%20notes/2020/10/23/pattern-exploiting-training.html),
[textual entailment](https://huggingface.co/tasks/zero-shot-classification), or
[plain old BERT embeddings](https://huggingface.co/docs/transformers/tasks/sequence_classification)
are gonna be way less expensive, and are less bad for the environment. This 
method is just trying to beat
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification),
which targets software developers working on zero-shot or few-shot text 
classification tasks.


## Usage

Let's classify [this sentiment example](https://platform.openai.com/docs/guides/completion/classification)
from the OpenAI text completion docs:

```python
from lm_classification.classify import predict_proba

text = 'I loved the new Batman movie!'
prompt = f'Tweet: {text}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
prior = (1/8, 1/8, 3/4) # Twitter amirite

pred_probs = predict_proba(prompts=[prompt],
                           completions=class_names,
                           prior=prior,
                           model='text-ada-001')

print(pred_probs.round(2))
# [[0.98 0.   0.02]]

pred_class_idxs = pred_probs.argmax(axis=1)
print([class_names[pred_class_idx] for pred_class_idx in pred_class_idxs])
# ['positive']
```

See [`demos/copa.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demos/copa.ipynb)
for a slightly harder classification task.


## Motivation

Create a more usable zero-shot text classification interface than
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification) (CVS).
([Cookbook here](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit).)
With this package's `predict_proba` interface, you no longer have to:
  1. study sampled completion strings which aren't in your label set
  2. figure out how to map them back to the label set
  3. figure out how to transform or point multi-token labels to single tokens,
     ignoring their semantics if they were transformed
  4. ignore your prior over multi-token labels.

This package just does one thing well: classification. It should be at least as
good as CVS on single token label sets. It should be significantly better than
CVS on multi-token label sets.


## Setup

You can create a new virtual environment, or pray that things don't break in
an existing one. I'll loosen the requirements later.

(This package isn't published, so might as well install it in editable mode.)

```bash
cd your/venvs

python -m venv lm-research

source lm-research/bin/activate

python -m pip install wheel setuptools --upgrade pip

python -m pip install -r requirements.txt

python -m pip install -e .
```

I may make these steps shorter later.


## User guide

**Use these LMs for what they are; stay close to the way GPT-3 was pretrained as
much as possible**

Models like `text-davinci-003` were trained with human feedback, so you 
*could* simply ask it to classify the text you've given, e.g.,
`What's the sentiment of the tweet?`, and expect good results. But IME, 
thinking for a few minutes about how to frame your task as a prompt-completion 
problem usually takes you far. For example, the
[`demos/copa.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demos/copa.ipynb)
notebook demonstrates that a question approach to the task causes accuracy to
dip from 0.92 to 0.87.

I'll expand on this guide as I run more experiments and learn more.


## Related work

While benchmarking this method on the
[Winograd Schema Challenge (WSC)](https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html),
I found that [this paper](https://arxiv.org/abs/1806.02847) has an
identical motivation:

> Trinh, Trieu H., and Quoc V. Le. "A simple method for commonsense reasoning." arXiv preprint arXiv:1806.02847 (2018).

I saw the same motivation again in
[this paper](https://arxiv.org/abs/2009.07118):

> Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).


## Test code

In your virtual environment, install `pytest-mock`:

```
python -m pip install pytest-mock
```

Then run

```
pytest
```


## Todo

Code:
- [x] Add unit tests
- [ ] Add integration tests
  - [x] `utils` + `classify`
  - [ ] `classify`
- [ ] Docs (not just docstrings)
- [ ] Loosen requirements
- [ ] Add setup instructions and file for conda
- [ ] Install requirements as part of setup.py

Research: evaluate on more tasks, and understand its relative advantages and
disadvantages vs other classification methods

- [ ] Compare against zero-shot sampling
- [ ] Expand user guide
- [ ] Compare against few-shot embeddings
- [ ] More SuperGLUE tasks
- [ ] Understand how sampling works, make a computational comparison
  - [ ] Assume I have full freedom to decide how inference works
- [ ] Calibration
  - [ ] (easy) Is the prior actually effective? Downsample and see
- [ ] More real world or harder tasks
  - [ ] Multi-token labels w/ non-uniform prior
- [ ] Finetune smaller, cheaper model and compare against zero-shot w/ davinci
  - [ ] e.g., GPT-2 from huggingface, `text-ada-001`
  - [ ] Again, compare against sampling
- [ ] Evaluate different aggregation functions. Currently taking mean, but
there was no good motivation for that
- [ ] Give this method and package a more specific name.

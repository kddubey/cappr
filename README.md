# Zero-shot text classification

Perform zero-shot text classification based on the following idea: for a given
prompt and completion, what's the probability that the completion follows the
prompt?

The method is fleshed out
[here in CrossValidated](https://stats.stackexchange.com/q/601159/337906)
(I'm chicxulub). The first demo of this method is
[here, in `demos/copa.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demos/copa.ipynb).


## Usage

Let's classify [this sentiment example](https://platform.openai.com/docs/guides/completion/classification)
from the OpenAI text completion docs.

```python
from lm_classification.classify import predict_proba

tweet = 'I loved the new Batman movie!'
prompt = f'Tweet: {tweet}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
prior       = (   1/8,        1/8,        3/4   )

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


## Disclaimers

This package only supports [language models (LMs) in OpenAI's text completion API](https://platform.openai.com/docs/models/gpt-3),
which you gotta pay for. Prices are [here](https://openai.com/api/pricing/).

If you're something of an ML engineer, and you have labeled and unlabeled text, 
there are likely far better alternatives to this slightly hacky method.
[PET training](http://timoschick.com/explanatory%20notes/2020/10/23/pattern-exploiting-training.html),
[textual entailment](https://huggingface.co/tasks/zero-shot-classification), or
[plain old BERT embeddings](https://huggingface.co/docs/transformers/tasks/sequence_classification)
are gonna be way less expensive, and are less bad for the environment.


## Motivation

Improve my understanding of how LMs work. Big and small. With and without RLHF.

Product-y motivation: create a more usable zero-shot text classification
interface than
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification) (CVS).
([Cookbook here](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit).)
With this package's `predict_proba` interface, you no longer have to:
  1. study sampled completion strings which aren't in your label set
  2. figure out how to map them back to the label set
  3. figure out how to transform or point multi-token labels to single tokens,
     ignoring their semantics if they were transformed
  4. ignore your prior over multi-token labels.

I'm not sure how cumbersome users find these issues. In my last job, I worked
on a challenging multi-class multi-token problem. Our team took at least a few
days to address them.

This package tries to do one thing well: classification. I still need to
evaluate it on more datasets and tasks. But the goals are that:
  1. It performs as well as CVS on single token label sets
  2. It's significantly more usable than CVS on multi-token label sets
  3. It's not much more computation than CVS.


## Setup

Requires Python 3.8+

1. Activate your Python environment

2. Clone this repo somewhere

   ```
   git clone https://github.com/kddubey/lm-classification.git
   ```

3. Install it

   ```
   python -m pip install lm-classification
   ```

4. Set the environment variable `OPENAI_API_KEY`.

(Optional) For testing and demo-ing:

1. Create a new Python 3.8+ environment

2. Install the requirements

   ```
   python -m pip install -r requirements.txt
   ```


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

If you'd like to develop this package, then in your environment, install 
`pytest-mock`:

```
python -m pip install pytest-mock
```

And run

```
pytest
```


## Todo

Code:
- [ ] Add unit tests
  - [x] `classify`
  - [x] `utils.batch`
  - [ ] `utils.api`
- [ ] Add integration tests
  - [x] `utils` + `classify`
  - [ ] `classify`
- [x] Loosen dependencies, separate from requirements
- [x] Install dependencies as part of setup.py
- [ ] Create a notebook template
- [ ] Docs (not just docstrings)
- [ ] Publish to PyPI?

Research: evaluate on more tasks, and understand its relative advantages and
disadvantages vs other classification methods

- [ ] Compare against zero-shot sampling
- [ ] Create a user guide, build a table of results comparing competing
  approaches on statistical performance, cost, and computation
- [ ] Compare against few-shot embeddings
- [ ] More SuperGLUE tasks
- [ ] Understand how sampling works, make a computational comparison
  - [ ] Assume I have full freedom to decide how inference works. Sketch out 
  parallelization, demo w/ GPT-2
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

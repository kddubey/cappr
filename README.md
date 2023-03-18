# Zero-shot text classification

Perform zero-shot text classification based on the following idea: for a given
prompt and completion, what's the probability that the completion follows the
prompt?

The method is fleshed out
[here in CrossValidated](https://stats.stackexchange.com/q/601159/337906)
(I'm chicxulub).


## Usage

Let's classify [this sentiment example](https://platform.openai.com/docs/guides/completion/classification)
from the OpenAI text completion docs.

<details>
<summary>Using an OpenAI model</summary>

These currently seem to be far ahead of other models, but this'll cost ya 💰!

```python
from lm_classification.openai.classify import predict_proba

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
<summary>Using a HuggingFace model </summary>

```python
from lm_classification.huggingface.classify import predict_proba

tweet = 'I loved the new Batman movie!'
prompt = f'Tweet: {tweet}\nSentiment:'

class_names = ('positive', 'neutral', 'negative')
prior = None # uniform prior

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

See [`demos/copa.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demos/copa.ipynb)
for a harder classification task.


## Motivation

Improve my understanding of LMs.

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

This package tries to do one thing well: classification. I'll assess it across
these dimensions: statistical performance, computational performance, and
usability.


## Setup

TODO: separate openai and huggingface

Requires Python 3.8+

1. Activate your Python environment

2. Install from git

   ```
   python -m pip install git+https://github.com/kddubey/lm-classification.git
   ```

3. [Sign up here](https://openai.com/api/) for an OpenAI API account.
   Set the environment variable `OPENAI_API_KEY`.

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

```
pytest
```


## Todo

(**) = I'm currently working on this / will work on it really really soon

Code:
- [ ] Add more unit tests
- [ ] Add more integration tests
- [x] Add support for HuggingFace `transformers.AutoModelForCausalLM`
  - [ ] Optional/targeted install, whatever they call it so that you can pick
  `openai`, `huggingface`, or both (**)
- [ ] Put dev requirements in setup extras (**)
- [ ] Auto-enforced code formatting b/c it's getting time-consuming (**)
- [ ] Create a notebook template
- [ ] Docs and user guides (not just docstrings)
- [ ] De-automate overzealous auto-docstring stuff lol (**)

Research: evaluate on more tasks, and understand its relative advantages and
disadvantages vs other classification methods

- [ ] Create a user guide, build a table of results comparing competing
  approaches on statistical performance, cost, and computation
- [ ] Compare against few-shot embeddings
- [ ] More SuperGLUE tasks
- [ ] Understand how sampling works, make a computational comparison (**)
  - [ ] Assume I have full freedom to decide how inference works. Demo w/
  GPT-2 (**)
- [ ] Calibration
  - [ ] (easy) Is the prior actually effective? Downsample and see
- [ ] More real world or harder tasks
  - [ ] Multi-token labels w/ non-uniform prior
- [ ] Finetune smaller, cheaper model and compare against zero-shot w/ davinci
  - [ ] e.g., GPT-2 from huggingface, `text-ada-001`
  - [ ] Again, compare against sampling
- [ ] Evaluate different aggregation functions. Currently taking mean, but
there was no good motivation for that
- [ ] A bit ambitious: support insertion. For transformers, I think this just
entails manipulating position IDs?
- [ ] Give this method and package a more specific name. I thought of one
  at chiptole: CALLM = Classification using an Autoregressive LLM

# lm-classification

Perform zero-shot text classification: for a given prompt and completion,
what's the probability that the completion follows the prompt? The motivation
for this method was
[discussed here in the OpenAI community forum](https://community.openai.com/t/compute-the-probability-of-input-text-for-classification/29840)
(I'm chicxulub on there). The method is fleshed out
[here in CrossValidated](https://stats.stackexchange.com/q/601159/337906)
(still chicxulub). And finally, a demo is here, in
[`demo.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demo.ipynb).


# Disclaimer

This package only supports LMs in
[OpenAI's text completion API](https://platform.openai.com/docs/models/gpt-3),
which you gotta pay for.

If you're an ML engineer working on a few-shot, many-shot, or semi-supervised
text classification task, there are likely far better alternatives to this
classification method. Alternatives such as
[PET training](http://timoschick.com/explanatory%20notes/2020/10/23/pattern-exploiting-training.html),
[textual entailment](https://huggingface.co/tasks/zero-shot-classification), or
[plain old BERT embeddings](https://huggingface.co/docs/transformers/tasks/sequence_classification)
are gonna be way less expensive, and are less bad for the environment. This 
method is just trying to beat
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification),
which targets software developers working on zero-shot or few-shot text
classification tasks.


# Setup

This package isn't published so might as well install it in editable mode.

You can create a new virtual environment, or pray that things don't break in
an existing one. I may loosen the requirements later.

```bash
cd your/venvs

python -m venv lm-research

source lm-research/bin/activate

python -m pip install wheel setuptools --upgrade pip

python -m pip install -r requirements.txt

python -m pip install -e .
```

I may make this nicer later.


# Usage

Take [this sentiment example](https://platform.openai.com/docs/guides/completion/classification)
from the OpenAI text completion docs.

```python
from lm_classification import classify

text = 'I loved the new Batman movie!'
prompt = f'Tweet: {text}' + '\n' + 'Sentiment: ' 

class_names = ('positive', 'neutral', 'negative')
prior = (1/6, 1/6, 2/3) # Twitter amirite

pred_probs = classify.predict_proba(prompts=[prompt],
                                    completions=class_names,
                                    model='text-ada-001',
                                    prior=prior)

print(pred_probs.round(2))
# [[0.95 0.   0.05]]
```

See [`demo.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demo.ipynb)
for a harder classification task.


# Related work

This classification strategy is a zero-shot and autoregressive variant of (a
small part of) the strategy in [this paper](https://arxiv.org/abs/2009.07118):

> Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).

([I saw](https://stats.stackexchange.com/questions/601159/should-a-language-model-like-gpt-3-be-directly-used-to-perform-classification#comment1122996_601159)
the paper after writing out this algorithm.)


# Motivation

My [ramblings here](https://community.openai.com/t/compute-the-probability-of-input-text-for-classification/29840)
are sufficient to help you understand. I might dump slightly more coherent
thoughts here later. The overall point is: I don't want to be forced to
transform multi-token labels to single tokens, I don't want to study completion
strings which aren't in my label set, and I don't want to have to figure out
how to map them back to the label set. We should be able to just do what's
asked: classify.


# Todo

Code:
- [ ] add unit tests
- [ ] add sampling for benchmarking purposes

Research: understand how robust this method is, and its relative advantages
(and disadvantages) compared to other popular methods

- [ ] compare against sampling
- [ ] add "priming", i.e., few-shot
- [ ] compare against few-shot embeddings
- [ ] more SuperGLUE tasks
- [ ] understand how sampling works, make a computational comparison
  - [ ] assume I have full freedom to decide how inference works
- [ ] more real world or harder tasks
  - [ ] multi-token labels w/ non-uniform prior
- [ ] finetune smaller, cheaper model and compare against zero-shot w/ davinci
  - [ ] e.g., GPT-2 from huggingface, `text-ada-001`
  - [ ] again, compare against sampling
- [ ] calibration
  - [ ] (easy) is the prior actually effective? downsample and see
- [ ] evaluate different aggregation functions. currently taking mean, but
there was no good motivation for that.
- [ ] give this method a name

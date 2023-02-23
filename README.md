# Zero-shot text classification

Perform zero-shot text classification: for a given prompt and completion,
what's the probability that the completion follows the prompt? The motivation
for this method was
[discussed here in the OpenAI community forum](https://community.openai.com/t/compute-the-probability-of-input-text-for-classification/29840)
(I'm chicxulub on there). The method is fleshed out
[here in CrossValidated](https://stats.stackexchange.com/q/601159/337906)
(still chicxulub). And finally, a demo is here, in
[`demo.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demo.ipynb).


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
                                    prior=prior,
                                    model='text-ada-001')

print(pred_probs.round(2))
# [[0.95 0.   0.05]]

pred_class_idxs = pred_probs.argmax(axis=1)
print([class_names[pred_class_idx] for pred_class_idx in pred_class_idxs])
# ['positive']
```

See [`demo.ipynb`](https://github.com/kddubey/lm-classification/blob/main/demo.ipynb)
for a harder classification task.


# Setup

You can create a new virtual environment, or pray that things don't break in
an existing one. I may loosen the requirements later.

This package isn't published, so might as well install it in editable mode.

```bash
cd your/venvs

python -m venv lm-research

source lm-research/bin/activate

python -m pip install wheel setuptools --upgrade pip

python -m pip install -r requirements.txt

python -m pip install -e .
```

I may make this nicer later.


# Related work

This classification strategy is a zero-shot and autoregressive variant of (a
small part of) the strategy in [this paper](https://arxiv.org/abs/2009.07118):

> Schick, Timo, and Hinrich Schütze. "It's not just size that matters: Small language models are also few-shot learners." arXiv preprint arXiv:2009.07118 (2020).

([I saw](https://stats.stackexchange.com/questions/601159/should-a-language-model-like-gpt-3-be-directly-used-to-perform-classification#comment1122996_601159)
the paper after writing out this algorithm.)


# Disclaimer

This package only supports LMs in
[OpenAI's text completion API](https://platform.openai.com/docs/models/gpt-3),
which you gotta pay for.

If you're something of an ML engineer, and you've got some labeled and
unlabeled text, there are likely far better alternatives to this classification 
method. Alternatives such as
[PET training](http://timoschick.com/explanatory%20notes/2020/10/23/pattern-exploiting-training.html),
[textual entailment](https://huggingface.co/tasks/zero-shot-classification), or
[plain old BERT embeddings](https://huggingface.co/docs/transformers/tasks/sequence_classification)
are gonna be way less expensive, and are less bad for the environment. This 
method is just trying to beat
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification),
which targets software developers working on zero-shot or few-shot text 
classification tasks.


# Motivation

Create a more usable zero-shot text classification interface than
[classification via sampling](https://platform.openai.com/docs/guides/completion/classification).
I don't want to be forced to transform multi-token labels to single tokens.
I don't want to study completion strings which aren't in my label set. And I
don't want to have to figure out how to map them back to the label set. We
should just do what's asked: classify.


# Todo

Code:
- [ ] Add unit tests
- [ ] Loosen requirements
- [ ] Add setup instructions and file for conda
- [ ] Install requirements as part of setup.py

Research: evaluate on more tasks, and understand its relative advantages and
disadvantages

- [ ] Compare against sampling
- [ ] Add "priming", i.e., few-shot
- [ ] Compare against few-shot embeddings
- [ ] More SuperGLUE tasks
- [ ] Understand how sampling works, make a computational comparison
  - [ ] Assume I have full freedom to decide how inference works
- [ ] More real world or harder tasks
  - [ ] Multi-token labels w/ non-uniform prior
- [ ] Finetune smaller, cheaper model and compare against zero-shot w/ davinci
  - [ ] e.g., GPT-2 from huggingface, `text-ada-001`
  - [ ] Again, compare against sampling
- [ ] Calibration
  - [ ] (easy) Is the prior actually effective? Downsample and see
- [ ] Evaluate different aggregation functions. Currently taking mean, but
there was no good motivation for that
- [ ] Give this method a name

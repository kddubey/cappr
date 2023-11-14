Why predict probabilities?
==========================

Every module has a ``predict_proba`` and ``predict`` function. ``predict_proba`` returns
the probability of each completion given the prompt, i.e., it returns an array of floats
from 0 to 1 indicating confidence. ``predict`` returns the most likely completion, i.e.,
it returns a string. You might be wondering why you'd ever use ``predict_proba``, when
``predict`` seemingly gives you what you need: a single choice.

In high stakes applications, probability scores can be thresholded to determine whether
or not to bypass intermediate systems. For example, if a model is highly confident that
a social media post contains hate speech, then your system can bypass manual review of
that post. If it isn't confident enough, then manual review is needed.

Another application where predicting probabilities turns out to be useful is in
"multilabel" tasks. In these tasks, a single piece of text can be labeled or tagged with
multiple categories. For example, a tweet can express multiple emotions at the same
time. A simple way to have an LLM tag a tweet's emotions is to predict the probability
of each emotion, and then threshold each probability. All possible emotions can be
processed in parallel, which saves time.


Examples
--------

See `this demo
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/banking_77_classes.ipynb>`_
for calibration curves. `Calibration curves
<https://scikit-learn.org/stable/modules/calibration.html>`_ visualize the accuracy of
predicted probabilities.

See `this demo
<https://github.com/kddubey/cappr/blob/main/demos/huggingface/tweet_emotion_multilabel.ipynb>`_
for an example of solving a multilabel classification task.
# CAPPr + GPT-3.5 RAFT benchmark

So far, these are just zero-shot accuracies on the training sets, which each contain
just 50 observations.

The prompts are often taken from the [original RAFT GPT-3
benchmark](https://github.com/oughtinc/raft-baselines/tree/master/example_prompts).<sup>1</sup>
The only difference is that I do not include training examples (for now), i.e., there's
no "priming" or in-context learning. It's zero-shot.

Contamination notice: I don't know whether `text-davinci-003` trained on this data. That
model was trained on data until [June
2021](https://platform.openai.com/docs/models/gpt-3-5). The RAFT datasets were uploaded
together shortly after that, but they may have been aggregated from public data before
that.

Regardless, you have to take these F1-scores on the training set w/ a grain of salt b/c:
  1. Sometimes I messed w/ including or excluding the prior
  2. There are just 50 observations!

Please wait for me to upload test set predictions to the
[competition](https://huggingface.co/spaces/ought/raft-leaderboard) before interpreting
the scores too seriously.

## References

1. Alex, Neel, et al. "RAFT: A real-world few-shot text classification benchmark." arXiv
   preprint arXiv:2109.14076 (2021).

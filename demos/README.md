# Demos

Measure computational and statistical performance.

```bash
pip install "cappr[demos]"
```


## Methodology for evaluating statistical performance

The workflow described
[here](https://cappr.readthedocs.io/en/latest/a_note_on_workflow.html) is usually
followed.

Demos of text generation intentionally use custom post-processing functions to attempt
to squeeze the most out of text generation.

Because the effect of the classification strategy (text generation or CAPPr) on
statistical performance depends on the prompt format, the prompt format is not always
held constant. It's selected to be the most performant given the strategy, which mimics
what would be done in a real application. The demos follow a slightly lazier methodology
though: tune the text generation format based on a training set. If it seems performant,
use that same format for CAPPr. If CAPPr is at least as performant as text generation on
the training set, stop. Else, tune CAPPr's format as well, and then evaluate both
strategies and their formats on an independent test set.

I still need to investigate how much train-test leakage there is between [this HF
model](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ) and the HF datasets.
It'd be cool if there were an easy way to find this info.


## Acknowledgements

Thank you to the Refuel [autolabel](https://github.com/refuel-ai/autolabel/) team for
finding good classification tasks and writing performant prompts.

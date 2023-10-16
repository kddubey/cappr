This is the first open source software project I've created. I thought I'd write about
some of the design choices, and what I learned from creating it.


## Design

The design is to implement this protocol:

```
├───(module) {LM interface name}
    ├───(module) classify.py

        └───(function) log_probs_conditional:
                prompts, completions, model, **kwargs
                    ->
                list[i][j][k] = Pr_model(
                    completions[j][k] | prompts[i] + completions[j][:k]
                )
        
        └───(function) predict_proba:
                prompts, completions, model, **kwargs
                    ->
                array[i, j] = Pr_model(completions[j] | prompts[i])

        └───(function) predict:
                prompts, completions, model, **kwargs
                    ->
                list[i] = argmax_j Pr_model(completions[j] | prompts[i])
```

(I excluded the `_examples` functions from the diagram because they don't contribute to
understanding the design. They're a convenient way to do batch inference / achieve
higher throughput for arbitrary combinations of prompts and completions.)

The [CAPPr
scheme](https://stats.stackexchange.com/questions/601159/should-a-language-model-like-gpt-3-be-directly-used-to-perform-classification)
applies to any autoregressive language model. But different language models have
different ways of outputting token log-probabilities. OpenAI and other AI companies
expose their LMs via APIs. So we just hit the API and go from there. Open source
`CausalLM`s (read: autoregressive LMs) on Huggingface are PyTorch `nn.Module`s. Calling
a `CausalLM` on tokenized inputs returns next-token logits, which are log-softmax'd and
sliced to get token log-probabilities.


### How to add a new LM interface module

The only work is in implementing `log_probs_conditional` for your new LM interface.
`predict_proba` is immediately defined by decorating `log_probs_conditional`. And
`predict` is immediately defined by decorating `predict_proba`.

For example, say we're integrating Anthropic's
[Claude](https://www.anthropic.com/index/introducing-claude). Here's what
`src/cappr/anthropic/classify.py` would look like: 


```python
from cappr.utils import classify


@classify._log_probs_conditional
def log_probs_conditional(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: str,
    **kwargs,
) -> list[list[float]] | list[list[list[float]]]:
    # 1. Hit the Anthropic API in batches of requests.
    log_probs_completions = []
    # 2. Process the output into this doubly-nested and ragged list:
    #    log_probs_completions[i][j][k] is the log-probability of the
    #    completion token k in completions[j], conditional on
    #    prompts[i] + previous completion tokens.
    return log_probs_completions


@classify._predict_proba
def predict_proba(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: str,
    **kwargs,
) -> npt.NDArray[np.floating]:
    return log_probs_conditional(prompts, completions, model, **kwargs)


@classify._predict
def predict(
    prompts: str | Sequence[str],
    completions: Sequence[str],
    model: str,
    **kwargs,
) -> str | list[str]:
    return predict_proba(prompts, completions, model, **kwargs)
```

This decorator-based design always works because `predict_proba` is a transformation of
the output of `log_probs_conditional`—a transformation which is always independent of
the LM interface. `predict` is just the argmax of the output of `predict_proba`.


### An alternate design I decided against

An alternate design is to ditch the decorators, and instead create an abstract base
class which all LM interfaces inherit from / implement.

```python
from abc import ABC, abstractmethod


class _BaseClassifier(ABC):
    @classmethod
    @abstractmethod
    def log_probs_conditional(
        cls,
        prompts: str | Sequence[str],
        completions: Sequence[str],
        model: str,
        **kwargs,
    ) -> list[list[list[float]]]:
        pass

    @classmethod
    def predict_proba(
        cls,
        prompts: str | Sequence[str],
        completions: Sequence[str],
        model: str,
        **kwargs,
    ) -> npt.NDArray[np.floating]:
        log_probs_completions = cls.log_probs_conditional(
            prompts, completions, model, **kwargs
        )
        return (
            log_probs_completions
            .average()
            .exponentiate()
            .normalize()
        )

    @classmethod
    def predict(
        cls,
        prompts: str | Sequence[str],
        completions: Sequence[str],
        model: str,
        **kwargs,
    ) -> str | list[str]:
        pred_probs = cls.predict_proba(prompts, completions, model, **kwargs)
        pred_completions = [
            completions[completion_idx]
            for completion_idx in pred_probs.argmax(axis=-1)
        ]
        return pred_completions
```

As far as I can see, there is no benefit to this design. It's tempting to think that it
reduces writing LM interface modules to writing sublcasses which implement
`log_probs_conditional` and nothing else. That's not practically true. Because if you're
writing a new LM interface, it probably has unique `kwargs` which you need to supply in
the signature and explain in the docstring. So you need to write out the method anyway.
Moreover, the amount of repeated code that's saved is insignificant.

There are a few small costs to this design. Enabling the simple import interface—

```python
from cappr.{LM interface name}.classify import predict_proba
```

—would require adding a bit of stuff to `__init__.py` and Sphinx's `docs/source/conf.py`
for every new interface. I could automate this stuff by implementing even more, slightly
tricky code. But more code is worse than less code.

It's also a bit confusing to users (like me) who want to understand why something which
is fundamentally a function is actually a method. Is there some obscure state that I'm
not supposed to know is being maintained? Is there a useful hierarchy? The answer to
both of these questions is categorically no. I don't like when that happens.

Overall, this design doesn't align with my style. I got away with writing decorators,
and I'm pretty sure I can keep getting away with it.


### I (probably) won't write string formatting abstractions

Every other tool in this space includes some type of string formatting abstraction,
usually based on LangChain or an internal module which processes a user-created config
file. These string formatters abstract the process of writing a few-shot prompt, for
example. Not to sound too dismissive, but anyone who uses Python knows how to format a
string. Formatters replace the question of "how do I tell the LM to do what I want?"
with "how do I use this string formatting interface to tell the LM to do what I want?".
The latter question takes more time to answer. And while answering that question, you
may end up realizing that the formatter doesn't let you do what you need to do.
Moreover, these formatters sometimes obfuscate what the prompt actually looks like,
which is a risk. For smaller LLMs, there are
[quirks](https://cappr.readthedocs.io/en/latest/select_a_prompt_completion_format.html#quirks)
which prompt writers should be aware of.

I want this package to do one thing well: pick a completion from a user-created prompt.
If users want to use abstract string formatters, that's on them.


### Testing

This package's tests are designed in a sophisticated (complicated) way. It took me a
while to think about what they should look like. The goal was to allow for 2 things:

  1. share test cases universal to all `classify` modules—these are the parametrizations
     in `BaseTestPromptsCompletions` and `BaseTestExamples`
  2. module-specific fixtures and parametrizations to test module-specific setups and
     arguments, e.g., `batch_size` in the HuggingFace backend.

The current testing design accomplishes these things through inheritance, because
`pytest` is incredibly powerful with inheritance. For an example, see the
[tests](https://github.com/kddubey/cappr/blob/main/tests/llama_cpp/test_llama_cpp_classify.py)
for llama-cpp models.


### A note on docstrings

Lots of text in docstrings are repeated. After all, fundamentally, the three functions
take the same inputs and produce the same outputs regardless of the LM interface.

I previously experimented with [an
automation](https://github.com/kddubey/dumpy/tree/main/wrap) that dynamically writes the
docstring via decorators. This pattern is used throughout HuggingFace `transformers`,
for example. I decided against this pattern because it sacrifices an important
convenience: hovering over a function to see what it does. Code analyzers like Pylance
are static. They will only show the immediate `__doc__` attribute of a function, not the
attribute you dynamically constructed. I personally am annoyed when I have to open up a
function's documentation in my browser, and look back and forth at my browser and IDE. I
like the convenience of hovering over the function in my IDE itself. So I opted to do
what numpy, scipy, and scikit-learn do in their docstrings: repeat text. It's definitely
tedious to make modifications. But that tediousness is outweighed by the benefits to the
user.


## Where I struggled

It's well known that attention keys and values can be cached whenever substrings are
repeated for inference. Getting this feature to align with the CAPPr scheme took nitty
gritty handling of pad tokens and position IDs. Many model implementations in
HuggingFace don't always handle them correctly (but this will change
[soon](https://github.com/huggingface/transformers/issues/18104#issuecomment-1465629955)).
Automating testing for caching was also tricky. TBH my current implementation of caching
is far from perfect. It currently doesn't allow for caching of sub-prompt text, and it
batches in a slightly misleading way. But it performed [well
enough](https://cappr.readthedocs.io/en/latest/computational_performance.html).

A major todo was to expand support to more backends, e.g., llama-cpp. I tabled that
because I thought OpenAI and HuggingFace `transformers` models were by far the most
common interfaces. Now that I've seen how popular llama.cpp is, that was clearly not
true. It didn't matter too much though. Model architectures, quantization, file formats,
etc. change so quickly that it's kinda hard to be too late.

On a separate note, it's disappointing that OpenAI doesn't support `echo=True,
logprobs=1` for the highly performant `gpt-3.5-turbo-instruct` model. I'll be focusing
more on open source models from now on.


## Marketing matters

The first version of the [User Guide in my
docs](https://cappr.readthedocs.io/en/latest/user_guide.html) was written for ML types,
when it should've be written for software engineers. What's text classification? What
are "labeled examples"? What's a prior? Why is a probability distribution useful? Docs
for other tools answer, or successfully dodge, these questions much more effectively.
CAPPr could be made way more approachable if the docs are re-worded. In the age of LLMs,
text classification can be done by any engineer, not just ML engineers.


## Pleasant surprises

Re the algorithm: text generation using `text-curie-001` (a small model) performs worse
than random guessing, while CAPPr using `text-curie-001` is 80% accurate. See the
experiments [here](https://cappr.readthedocs.io/en/latest/future_research.html). It'd be
cool to demonstrate that CAPPr generally works better for smaller or under-trained LMs.

Besides the algorithmic stuff, I was pleasantly surprised to find that I loved
engineering this project from the ground up. Mulling over design decisions and managing
myself was fun. Writing tests was enlightening using pytest. Writing docs was satisfying
(and [almost](https://github.com/kddubey/dumpy/tree/main/sphinx_setup) easy) using
Sphinx. Writing GitHub workflows made releases convenient, and it made my project feel
way more professional lol. I found [ReWrap](https://stkb.github.io/Rewrap/) and
[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
for the first time. I'll be using them for every project from now on. Overall, as a
result of working on this project, I appreciated open source at a much higher level.

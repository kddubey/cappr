KEYS_VALUES_PROMPTS = '''
    Returns past key-values, the attention mask, and position offsets after
    efficiently performing this procedure:

    1. Repeat `prompts[i]` `num_completions_per_prompt[i]` times (or, if it's an
    integer, `num_completions_per_prompt` times), e.g., if there are 2 prompts
    and `num_completions_per_prompt=(2,3)`:

    ```
        [prompts[0],
         prompts[0],
         prompts[1],
         prompts[1],
         prompts[1]]
    ```

    2. Apply `tokenizer`

    3. Apply `model`.

    "Efficient" = don't actually repeat each prompt; run the model on each
    prompt and then repeat the output data according to
    `num_completions_per_prompt`.
    '''


TEXTS_FROM_PROMPTS_COMPLETIONS = '''
    If `texts` is

    ```python
    [prompt + end_of_prompt + completions
     for prompt in prompts
     for completion in completions]
    ```

    '''


TEXTS_FROM_EXAMPLES = '''
    If `texts` is

    ```python
    [example.prompt + example.end_of_prompt + completion
     for example in examples
     for completion in example.completions]
    ```

    '''


LOGITS_COMPLETIONS_GIVEN_PROMPTS_OUTPUT = '''
    then this function returns

    1. `logits`: tensor with shape

        (`len(texts)`, max # tokens `{text}s`, `tokenizer.vocab_size`)

    where `logits[i,j]` are the `model`'s logits for token `j+1` of the
    {text} in `texts[i]` given the prompt in `texts[i]`. This tensor
    includes logits for right-padded tokens. Use the `encodings.attention_mask`
    to ignore them before further processing.

    2. `encodings`: `BatchEncoding` containing the input IDs, attention mask,
    and position offsets.
    '''


BATCH_SIZE = '''
    Texts are processed by the model in batches of size `batch_size`.
    '''


LOGITS_TO_LOG_PROBS_COMPLETIONS = '''
    Returns a list `log_probs_completions` where `log_probs_completions[i][j]`
    is the log-probablity of *completion* token

        `encodings['input_ids'][i,j]`

    given its previous tokens

        `encodings['input_ids'][i,:j]`

    Pad tokens, i.e., tokens where `encodings['attention_mask'] == 0` are
    excluded.

    `logits[i,j]` is assumed to be an unnormalized distribution (over tokens in
    the vocab) given tokens `input_ids[i,:j]`.
    '''

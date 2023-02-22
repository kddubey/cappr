import logging
import os
import time
from typing import Callable, Sequence

import numpy as np
import openai

from transformers import GPT2Tokenizer


logger = logging.getLogger(__name__)


openai.api_key = os.getenv('OPENAI_API_KEY')


gpt2_tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
## TODO: I'm not sure whether GPT2Tokenizer or GPT2TokenizerFast is used for the
## text models in the OpenAI API


def openai_method_retry(openai_method: Callable, max_num_tries: int=5,
                        sleep_sec: float=10, **openai_method_kwargs):
    '''
    Returns `openai_method(**openai_method_kwargs)`, retrying up to
    `max_num_tries` times and sleeping `sleep_sec` in between tries if the call
    raises `openai.error.ServiceUnavailableError` or
    `openai.error.RateLimitError`.
    '''
    num_tries = 0
    while num_tries < max_num_tries:
        try:
            return openai_method(**openai_method_kwargs)
        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError) as e:
            num_tries += 1
            logger.info(f'openai error: {e}')
            logger.info(f'Try {num_tries}. Sleeping for {sleep_sec} sec.')
            time.sleep(sleep_sec)
            exception = e ## allow it to be referenced later
    logger.error(f'Max retries exceeded. openai error: {exception}')
    raise exception


def batch(lst: list, size: int):
    '''
    Generates sublists in the order of `lst` which partition `lst`. All sublists
    (except potentially the last) have length `size`.
    '''
    if size <= 0:
        raise ValueError('size must be positive.')
    lst = list(lst) ## 0-index whatever was passed, or fully evaluate generator
    n = len(lst)
    for ndx in range(0, n, size):
        yield lst[ndx:(ndx + size)]


def batch_variable(lst: list, sizes: Sequence[int]):
    '''
    Generates sublists in the order of `lst` which partition `lst`. The `i`'th
    generated sublist has length `sizes[i]`.
    '''
    sizes: np.ndarray = np.array(sizes)
    if np.any(sizes <= 0):
        raise ValueError('sizes must all be positive.')
    if len(sizes.shape) != 1:
        raise ValueError('sizes must be 1-D.')
    cumulative_sizes = np.cumsum(sizes)
    ## We want start and stop idxs. The first start idx must ofc be 0.
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    lst = list(lst) ## 0-index and/or fully evaluate generator
    for start, stop in zip(cumulative_sizes[:-1], cumulative_sizes[1:]):
        yield lst[start:stop]

'''
Batch lists into sublists of constant or variable sizes.
'''
from typing import Sequence

import numpy as np


def constant(lst: list, size: int):
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


def variable(lst: list, sizes: Sequence[int]):
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
    if cumulative_sizes[-1] != len(lst):
        raise ValueError('sizes must sum to len(lst).')
    ## We want start and stop idxs. The first start idx must ofc be 0.
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    lst = list(lst) ## 0-index and/or fully evaluate generator
    for start, stop in zip(cumulative_sizes[:-1], cumulative_sizes[1:]):
        yield lst[start:stop]

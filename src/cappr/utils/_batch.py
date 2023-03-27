"""
Batch lists into sublists of constant or variable sizes, and batchify functions.
"""
from __future__ import annotations
from functools import wraps
import inspect
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm


def constant(lst: list, size: int):
    """
    TODO: numpy docstring

    Generates sublists in the order of `lst` which partition `lst`. All sublists (except
    potentially the last) have length `size`.
    """
    if size <= 0:
        raise ValueError("size must be positive.")
    lst = list(lst)  ## 0-index whatever was passed, or fully evaluate generator
    n = len(lst)
    for ndx in range(0, n, size):
        yield lst[ndx : (ndx + size)]


def variable(lst: list, sizes: Sequence[int]):
    """
    TODO: numpy docstring

    Generates sublists in the order of `lst` which partition `lst`. The `i`'th generated
    sublist has length `sizes[i]`.
    """
    sizes: np.ndarray = np.array(sizes)
    if np.any(sizes <= 0):
        raise ValueError("sizes must all be positive.")
    if len(sizes.shape) != 1:
        raise ValueError("sizes must be 1-D.")
    cumulative_sizes = np.cumsum(sizes)
    if cumulative_sizes[-1] != len(lst):
        raise ValueError("sizes must sum to len(lst).")
    ## We want start and stop idxs. The first start idx must ofc be 0.
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    lst = list(lst)  ## 0-index and/or fully evaluate generator
    for start, stop in zip(cumulative_sizes[:-1], cumulative_sizes[1:]):
        yield lst[start:stop]


def _kwarg_name_to_value(func):
    """
    Returns a dictionary mapping keyword arguments in the signature of `func`
    to their default values.
    """
    ## ty https://stackoverflow.com/a/12627202/18758987
    signature = inspect.signature(func)
    return {
        name: value.default
        for name, value in signature.parameters.items()
        if value.default is not inspect.Parameter.empty
    }


def batchify(batchable_arg: str, batch_size: int = 32, progress_bar_desc: str = ""):
    """
    TODO: numpy docstring

    Returns a decorator which runs the decorated function in batches along its
    `batchable_arg`, returning a list of the function's outputs for each batch.

    If the function includes a `'batch_size'` keyword argument, then its value is used
    as the batch size instead of the decorator's default `batch_size`.
    TODO: allow non-kwarg too.
    """

    def decorator(func):
        _arg_names = inspect.getfullargspec(func).args
        batchable_arg_idx = _arg_names.index(batchable_arg)
        batch_size_default = _kwarg_name_to_value(func).get("batch_size", batch_size)

        @wraps(func)
        def wrapper(*args, **kwargs):
            batchable: Sequence = args[batchable_arg_idx]
            size = kwargs.get("batch_size", batch_size_default)
            outputs = []
            args = list(args)  ## need to modify the batch argument value
            with tqdm(total=len(batchable), desc=progress_bar_desc) as progress_bar:
                for batch_ in constant(batchable, size):
                    args[batchable_arg_idx] = batch_
                    outputs.append(func(*args, **kwargs))
                    progress_bar.update(len(batch_))
            return outputs

        return wrapper

    return decorator


def flatten(batchified_func):
    """
    TODO: numpy docstring

    Decorates a `cappr.utils._batch.batchify`d function. Flattens the output.
    """

    @wraps(batchified_func)
    def wrapper(*args, **kwargs):
        nested_outputs = batchified_func(*args, **kwargs)
        return [output for inner_outputs in nested_outputs for output in inner_outputs]

    return wrapper

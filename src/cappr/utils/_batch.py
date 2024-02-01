"""
Batch lists into sublists of constant or variable sizes, and batchify functions
"""

from __future__ import annotations
from functools import wraps
import inspect
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

import numpy as np
from tqdm.auto import tqdm


_T = TypeVar("_T")


def constant(lst: list[_T], size: int):
    """
    Generates sublists in the order of `lst` which partition `lst`. All sublists (except
    potentially the last) have length `size`.
    """
    if size <= 0:
        raise ValueError("size must be positive.")
    indexable = lst if isinstance(lst, str) else list(lst)
    for idx in range(0, len(indexable), size):
        yield indexable[idx : (idx + size)]


def variable(lst: list[_T], sizes: Sequence[int]):
    """
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
    # We want start and stop idxs. The first start idx must ofc be 0
    cumulative_sizes = np.insert(cumulative_sizes, 0, 0)
    lst = list(lst)  # 0-index and/or fully evaluate generator
    for start, stop in zip(cumulative_sizes[:-1], cumulative_sizes[1:]):
        yield lst[start:stop]


class ProgressBar(tqdm, Generic[_T]):
    """
    `tqdm` progress bar which handles auto-show logic.
    """

    def __init__(
        self,
        iterable: Iterable[_T] = None,
        total: int | None = None,
        *args,
        show_progress_bar: bool | None = None,
        min_total_for_showing_progress_bar: int = 5,
        **kwargs,
    ):
        if iterable is None and total is None:
            disable = None
        else:
            total = total if total is not None else len(iterable)
            if show_progress_bar is None:
                disable = total < min_total_for_showing_progress_bar
            else:
                disable = not show_progress_bar
        kwargs = {"total": total, "disable": disable, **kwargs}
        super().__init__(iterable, *args, **kwargs)

    def __iter__(self) -> Iterator[_T]:  # infer type for elements of the iterable
        return super().__iter__()


def _kwarg_name_to_value(func) -> dict[str, Any]:
    """
    Returns a dictionary mapping keyword arguments to their default values according to
    `func`'s signature.
    """
    # ty https://stackoverflow.com/a/12627202/18758987
    signature = inspect.signature(func)
    return {
        name: value.default
        for name, value in signature.parameters.items()
        if value.default is not inspect.Parameter.empty
    }


def batchify(
    batchable_arg: str,
    batch_size: int = 2,
    progress_bar_desc: str = "",
    show_progress_bar: bool | None = None,
):
    """
    Returns a decorator which runs the decorated function in batches along its
    `batchable_arg`, returning a list of the function's outputs for each batch.

    If the function includes a `'batch_size'` keyword argument, then its value is used
    as the batch size instead of the decorator's default `batch_size`.
    """

    def decorator(func):
        _arg_names = inspect.getfullargspec(func).args
        batchable_arg_idx = _arg_names.index(batchable_arg)
        _kwargs_signature = _kwarg_name_to_value(func)
        batch_size_default: int = _kwargs_signature.get("batch_size", batch_size)
        show_progress_bar_default: bool | None = _kwargs_signature.get(
            "show_progress_bar", show_progress_bar
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            # set up batching
            batchable: Sequence = args[batchable_arg_idx]
            batch_size: int = kwargs.get("batch_size", batch_size_default)
            args = list(args)  # needs to be mutable to modify the batch argument value
            # run func along batches of batchable
            outputs = []
            with ProgressBar(
                total=len(batchable),
                show_progress_bar=kwargs.get(
                    "show_progress_bar", show_progress_bar_default
                ),
                desc=progress_bar_desc,
            ) as progress_bar:
                for batch_ in constant(batchable, batch_size):
                    args[batchable_arg_idx] = batch_
                    outputs.append(func(*args, **kwargs))
                    progress_bar.update(len(batch_))
            return outputs

        return wrapper

    return decorator


def flatten(batchified_func):
    """
    Decorates a `cappr.utils._batch.batchify`d function. Flattens the output.
    """

    @wraps(batchified_func)
    def wrapper(*args, **kwargs):
        nested_outputs = batchified_func(*args, **kwargs)
        return [output for inner_outputs in nested_outputs for output in inner_outputs]

    return wrapper

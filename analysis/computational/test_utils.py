'''
Unit tests functions in the `utils` module.

TODO: cover more.
'''
from functools import wraps

import utils


def func(k: int, lst: list, batch_size=100, dummy_kwarg='dummy', **kwargs):
    return sum(lst) + k


@utils.batchify(batchable_arg='lst')
def func_batchified(k: int, lst: list, **kwargs):
    return func(k, lst, **kwargs)


def test_batchify():
    k = 1
    lst = [1, 1, 2, 2, 3]
    batch_size = 3
    out = func_batchified(k, lst, batch_size=batch_size)
    assert out == [(1+1+2 + k), (2+3 + k)]

    batch_size = 2
    out = func_batchified(k, lst, batch_size=batch_size)
    assert out == [(1+1 + k), (2+2 + k), (3 + k)]

    batch_size = 1
    out = func_batchified(k, lst, batch_size=batch_size)
    assert out == [x + k for x in lst]


def is_even(batchified_func):
    @wraps(batchified_func)
    def wrapper(*args, **kwargs):
        outputs = batchified_func(*args, **kwargs)
        return [output % 2 == 0 for output in outputs]
    return wrapper


@is_even
@utils.batchify(batchable_arg='lst')
def post_process(k: int, lst: list, **kwargs):
    return func(k, lst, **kwargs)


def test_post_process():
    k = 1
    lst = [1, 1, 2, 2, 3]
    batch_size = 3
    out = post_process(k, lst, batch_size=batch_size)
    assert out == [False, True]

'''
Unit tests `lm_classification.utils.batch`.
'''
from __future__ import annotations

import pytest

from lm_classification.utils import batch


def _test_partition_and_order(batches: list[list], source_lst: list):
    '''
    Asserts that `batches` partitions `source_lst` and are in the same order as
    in `source_lst`.
    '''
    lst_from_batches = [x for batch_ in batches for x in batch_]    
    assert lst_from_batches == source_lst


@pytest.fixture(scope='module')
def lst():
    return list(range(10))


@pytest.mark.parametrize('size', (2, 3))
def test_constant(lst, size: int):
    batches = list(batch.constant(lst, size))

    _test_partition_and_order(batches, lst)

    ## Test batch sizes
    for batch_ in batches[:-1]:
        assert len(batch_) == size
    remaining = len(lst) % size
    expected_size_last = remaining if remaining else size
    assert len(batches[-1]) == expected_size_last


@pytest.mark.parametrize('sizes', ([2,4,3,1], [10]))
def test_variable(lst, sizes):
    batches = list(batch.variable(lst, sizes))

    _test_partition_and_order(batches, lst)

    assert len(batches) == len(sizes)

    ## Test batch sizes
    for batch_, expected_size in zip(batches, sizes):
        assert len(batch_) == expected_size


@pytest.mark.parametrize('batch_size', (2,3))
def test_batchify(lst, batch_size):
    ## Setup
    def func(dummy_arg, lst, batch_size=100, dummy_kwarg='dummy', **kwargs):
        return lst

    @batch.batchify(batchable_arg='lst')
    def func_batchified(dummy_arg, lst, **kwargs):
        return func(dummy_arg, lst, **kwargs)

    out = func_batchified('dummy', lst, batch_size=batch_size)
    assert out == list(batch.constant(lst, size=batch_size))

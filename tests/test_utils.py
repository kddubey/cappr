'''
Unit tests `lm_classification.utils`.
'''
from __future__ import annotations
import pytest

from lm_classification import utils


def _test_partition_and_order(batches: list[list], source_lst: list):
    '''
    Asserts that `batches` partitions `source_lst` and are in the same order as
    in `source_lst`.
    '''
    lst_from_batches = [x for batch in batches for x in batch]    
    assert lst_from_batches == source_lst


@pytest.fixture(scope='module')
def lst():
    return list(range(10))


@pytest.mark.parametrize('size', (2, 3))
def test_batch(lst: list, size: int):
    batches = list(utils.batch(lst, size))

    _test_partition_and_order(batches, lst)

    ## Test batch sizes
    for batch in batches[:-1]:
        assert len(batch) == size
    remaining = len(lst) % size
    expected_size_last = remaining if remaining else size
    assert len(batches[-1]) == expected_size_last


@pytest.mark.parametrize('sizes', ([2,4,3,1], [10]))
def test_batch_variable(lst: list, sizes):
    batches = list(utils.batch_variable(lst, sizes))

    _test_partition_and_order(batches, lst)

    assert len(batches) == len(sizes)

    ## Test batch sizes
    for batch, expected_size in zip(batches, sizes):
        assert len(batch) == expected_size

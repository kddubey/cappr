"""
Unit tests `cappr.utils._batch`.
"""
from __future__ import annotations

import pytest

from cappr.utils import _batch


@pytest.fixture(scope="module")
def lst():
    return list(range(10))


def test_ProgressBar(lst: list):
    # test basic loop
    assert [x for x in _batch.ProgressBar(lst)] == lst
    assert [x for x in _batch.ProgressBar(lst, show_progress_bar=False)] == lst
    # test blank init like tqdm
    assert _batch.ProgressBar() is not None
    # test context manager
    lst_after_progress_bar = []
    with _batch.ProgressBar(total=len(lst), desc="context manager") as progress_bar:
        for x in lst:
            lst_after_progress_bar.append(x)
            progress_bar.update()


def _test_partition_and_order(batches: list[list], source_lst: list):
    """
    Asserts that `batches` partitions `source_lst` and are in the same order as in
    `source_lst`.
    """
    lst_from_batches = [x for batch_ in batches for x in batch_]
    assert lst_from_batches == source_lst


@pytest.mark.parametrize("size", (2, 3))
def test_constant(lst: list, size: int):
    batches = list(_batch.constant(lst, size))

    _test_partition_and_order(batches, lst)

    # Test batch sizes
    for batch_ in batches[:-1]:
        assert len(batch_) == size
    remaining = len(lst) % size
    expected_size_last = remaining if remaining else size
    assert len(batches[-1]) == expected_size_last


@pytest.mark.parametrize("sizes", ([2, 4, 3, 1], [10]))
def test_variable(lst, sizes):
    batches = list(_batch.variable(lst, sizes))

    _test_partition_and_order(batches, lst)

    assert len(batches) == len(sizes)

    # Test batch sizes
    for batch_, expected_size in zip(batches, sizes):
        assert len(batch_) == expected_size


@pytest.mark.parametrize("batch_size", (2, 3))
def test_batchify(lst, batch_size):
    @_batch.batchify(batchable_arg="lst")
    def func(dummy_arg, lst, batch_size=100, dummy_kwarg="dummy", **kwargs):
        return lst

    out = func("dummy", lst, batch_size=batch_size)
    assert out == list(_batch.constant(lst, size=batch_size))

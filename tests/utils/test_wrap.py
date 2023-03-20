"""
Unit tests `callm.utils.wrap`.

TODO: cover more.
"""
from __future__ import annotations
from functools import WRAPPER_ASSIGNMENTS

from callm.utils import wrap


def test_wraps_but_keep_wrapper_return_ann():
    def boolify(func):
        @wrap.wraps_but_keep_wrapper_return_ann(func)
        def wrapper(*args, **kwargs) -> bool:
            return bool(func(*args, **kwargs))

        return wrapper

    def sum_numbers(x: int, y: float, z) -> int:
        return x + y + z

    name_to_func = dict()
    name_to_func["source"] = sum_numbers
    ## Need to do this jankyness b/c I want to test that the wrapped function's name is
    ## preserved. Without these references, sum_numbers can only refer to the wrapped
    ## function below.

    @boolify
    def sum_numbers(x: int, y: float, z) -> int:
        return x + y + z

    name_to_func["wrapped"] = sum_numbers

    ## Sanity check that the referenced functions are what we want
    assert name_to_func["source"](1, 2, 3) == 6
    assert name_to_func["wrapped"](1, 2, 3) is True
    assert name_to_func["wrapped"](1, 2, -3) is False

    ## Check that the only modification is that __annotations__['return'] comes from the
    ## wrapper
    for attr in WRAPPER_ASSIGNMENTS:
        if attr == "__annotations__":
            annotations_source: dict = getattr(name_to_func["source"], attr)
            annotations_wrapped: dict = getattr(name_to_func["wrapped"], attr)
            assert annotations_source.keys() == annotations_wrapped.keys()
            for key in annotations_source:
                if key == "return":
                    expected_type = "bool"  ## not int!
                    observed_type = annotations_wrapped[key]
                    if not isinstance(observed_type, str):
                        ## didn't do from __future__ import annotations
                        observed_type = observed_type.__name__
                    assert observed_type == expected_type
                else:
                    assert annotations_source[key] == annotations_wrapped[key]
        else:
            # fmt: off
            assert (getattr(name_to_func["source"], attr)
                    ==
                    getattr(name_to_func["wrapped"], attr))
            # fmt: on

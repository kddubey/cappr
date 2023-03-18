'''
Unit tests `lm_classification.utils.wrap`.

TODO: cover more.
'''
from __future__ import annotations
from functools import WRAPPER_ASSIGNMENTS

from lm_classification.utils import wrap


def test_wraps_but_keep_wrapper_return_ann():
    def boolify(func):
        @wrap.wraps_but_keep_wrapper_return_ann(func)
        def wrapper(*args, **kwargs) -> bool:
            return bool(func(*args, **kwargs))
        return wrapper

    def sum(x: int, y: float, z) -> int:
        return x + y + z

    @boolify
    def sum_wrapped(x: int, y: float, z) -> int:
        return x + y + z

    ## check that the only modification is that __annotations__['return']
    ## comes from the wrapper
    for attr in WRAPPER_ASSIGNMENTS:
        if attr == '__annotations__':
            annotations: dict = getattr(sum, attr)
            annotations_wrapped: dict = getattr(sum_wrapped, attr)
            assert annotations.keys() == annotations_wrapped.keys()
            for key in annotations:
                if key == 'return': ## then value should be bool, not int
                    expected_type = 'bool'
                    observed_type = annotations_wrapped[key]
                    if not isinstance(observed_type, str):
                        ## didn't do from __future__ import annotations
                        observed_type = observed_type.__name__
                    assert observed_type == expected_type
                else:
                    assert annotations[key] == annotations_wrapped[key]
        else:
            assert getattr(sum, attr) == getattr(sum, attr)

    ## check that output is what we expect
    assert sum_wrapped(-1, 2, 3) is True
    assert sum_wrapped(-1, 1, 0) is False

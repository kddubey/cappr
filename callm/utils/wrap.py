"""
Decorators to customize function wrapping.
"""
from copy import deepcopy
import functools


def add_doc_after(docstring: str):
    """
    Returns a decorator which concatenates `docstring` to the decorated function's
    docstring.
    """

    def decorator(func):
        func.__doc__ = (func.__doc__ or "") + docstring
        return func

    return decorator


def add_doc_before(docstring: str):
    """
    Returns a decorator which concatenates the decorated function's docstring to
    `docstring`.
    """

    def decorator(func):
        func.__doc__ = docstring + (func.__doc__ or "")
        return func

    return decorator


def wraps_but_keep_wrapper_return_ann(
    wrapped, assigned=functools.WRAPPER_ASSIGNMENTS, updated=functools.WRAPPER_UPDATES
):
    """
    Returns a decorator which is the same as what
    ```
    functools.wraps(wrapped, assigned=assigned, updated=updated)
    ```
    returns, except that the `'return'` annotation from the wrapper function is used
    instead of that from `wrapped`.
    """

    def update_wrapper(wrapper, wrapped, assigned=assigned, updated=updated):
        wrapper_return_ann = deepcopy(wrapper.__annotations__["return"])
        wrapped.__annotations__["return"] = wrapper_return_ann
        return functools.update_wrapper(
            wrapper, wrapped, assigned=assigned, updated=updated
        )

    return functools.partial(
        update_wrapper, wrapped=wrapped, assigned=assigned, updated=updated
    )

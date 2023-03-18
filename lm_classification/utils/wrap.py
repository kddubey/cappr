'''
Decorators to customize function wrapping. These are only useful for development
purposes of course.
'''
import functools
from typing import Literal


def _add_doc(docstring: str, after_or_before: Literal['after', 'before']):
    '''
    Returns a decorator which concatenates `docstring` `after_or_before` the
    decorated function's docstring.
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        if after_or_before == 'after':
            new_docstring = docstring + (func.__doc__ or '')
        elif after_or_before == 'before':
            new_docstring = (func.__doc__ or '') + docstring
        else:
            raise ValueError("after_or_before must be 'after' or 'before'.")
        wrapper.__doc__ = new_docstring
        return wrapper
    return decorator


def add_doc_after(docstring: str):
    '''
    Returns a decorator which concatenates the `docstring` to the decorated
    function's docstring.
    '''
    return _add_doc(docstring, 'after')


def add_doc_before(docstring: str):
    '''
    Returns a decorator which concatenates the decorated function's docstring to
    `docstring`.
    '''
    return _add_doc(docstring, 'before')


def wraps_but_keep_wrapper_return_ann(wrapped,
                                      assigned=functools.WRAPPER_ASSIGNMENTS,
                                      updated=functools.WRAPPER_UPDATES):
    '''
    Decorator which is the same as `functools.wraps` except that the
    return annotation from the wrapper function are used instead of `wrapped`.
    '''
    def update_wrapper(wrapper, wrapped,
                       assigned=functools.WRAPPER_ASSIGNMENTS,
                       updated=functools.WRAPPER_UPDATES):
        for attr in assigned:
            try:
                value = getattr(wrapped, attr)
            except AttributeError:
                pass
            else: ## here lies the modified bit
                if attr == '__annotations__' and 'return' in value:
                    value['return'] = getattr(wrapper, attr).get('return')
                setattr(wrapper, attr, value)
        for attr in updated:
            getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
        wrapper.__wrapped__ = wrapped
        return wrapper
    return functools.partial(update_wrapper, wrapped=wrapped,
                             assigned=assigned, updated=updated)

# [MODULE DOC]
# Tiny method-forwarding mechanism: methods tagged with the wrapable decorator on a
# source object can be copied onto a target object, optionally routed through a
# transform callable that receives the method name and original method as extra
# keyword arguments (wrap_method_name / wrap_original_method).
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Callable
from functools import partial


def wrapable(name: str | None = None):
    # [FUNCTION DOC]
    # Decorator factory: marks a function with a __wrap_meta__ dict holding the exposed
    # name (given name or the function's own name). Returns the decorator.
    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: Attaches __wrap_meta__ to func and returns func unchanged.
        func.__wrap_meta__ = {
            "name": name or func.__name__,
        }
        return func

    return decorator


def wrap(target: object, source: object, transform: Callable | None = None):
    # [FUNCTION DOC]
    # For every callable member of source carrying __wrap_meta__, sets an attribute on
    # target named by the meta name: either the member itself, or (if transform is
    # given) a partial of transform with wrap_method_name and wrap_original_method
    # bound as keywords. Mutates target; returns None.
    for name, member in inspect.getmembers(source):
        if callable(member) and hasattr(member, "__wrap_meta__"):
            meta = member.__wrap_meta__
            method_name = meta["name"]

            if transform:
                setattr(
                    target,
                    method_name,
                    partial(
                        transform,
                        wrap_method_name=method_name,
                        wrap_original_method=member,
                    ),
                )
            else:
                setattr(target, method_name, member)


def wrap_get_method_name(kwargs) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns kwargs["wrap_method_name"] (KeyError if absent).
    return kwargs["wrap_method_name"]


def wrap_get_method(kwargs) -> Callable:
    # [FUNCTION DOC]
    # HELPER: Returns kwargs["wrap_original_method"] (KeyError if absent).
    return kwargs["wrap_original_method"]


def wrap_remove_meta(kwargs) -> dict:
    # [FUNCTION DOC]
    # HELPER: Deletes both wrap_* keys from kwargs in place and returns the same dict;
    # raises KeyError if missing.
    del kwargs["wrap_method_name"]
    del kwargs["wrap_original_method"]
    return kwargs

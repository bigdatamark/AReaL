# [MODULE DOC]
# Dynamic import helper for resolving dotted "module.path.ObjectName" strings to
# objects.
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any


def import_from_string(module_path: str) -> Any:
    # [FUNCTION DOC]
    # Imports module_path's module part and returns its named attribute. Raises
    # ValueError for empty/non-string input or a path without a dot, ImportError
    # (wrapped) if the module import fails, AttributeError (wrapped) if the attribute is
    # missing. Side effect: module import.
    if not module_path or not isinstance(module_path, str):
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected a non-empty string like 'module.path.ObjectName'."
        )

    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid module path: {module_path!r}. "
            f"Expected format 'module.path.ObjectName', got {len(parts)} part(s)."
        )

    module_name, object_name = parts

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_name}' from path '{module_path}': {e}"
        ) from e

    try:
        obj = getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{object_name}'."
        ) from e

    return obj

# [MODULE DOC]
# Pytest conftest for weight_update tests. On Python versions older than 3.12 it
# pre-registers lightweight namespace module stubs for areal, areal.experimental,
# areal.v2 and areal.v2.weight_update (pointing at the real source directories) so that
# importing areal.v2.weight_update submodules does not execute the top-level
# areal/__init__.py, which uses 3.12-only syntax. Side effect at import time: mutates
# sys.modules.
"""Pre-import hook for Python < 3.12 compatibility.

The top-level ``areal/__init__.py`` imports from modules that use PEP 695 syntax
(``def func[T](...)``) which is only valid on Python 3.12+. When running tests on
Python 3.10/3.11 we register lightweight stubs for the ``areal`` namespace packages
so that importing ``areal.v2.weight_update.*`` never triggers the
problematic top-level init.
"""

from __future__ import annotations

import os
import sys
import types

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

_STUB_PACKAGES = [
    ("areal", os.path.join(_REPO_ROOT, "areal")),
    ("areal.experimental", os.path.join(_REPO_ROOT, "areal", "experimental")),
    ("areal.v2", os.path.join(_REPO_ROOT, "areal", "v2")),
    (
        "areal.v2.weight_update",
        os.path.join(_REPO_ROOT, "areal", "v2", "weight_update"),
    ),
]


def _ensure_namespace_stubs():
    # [FUNCTION DOC]
    # HELPER: For each stub package not already imported, creates an empty ModuleType
    # with
    # __path__ set to the real directory and registers it in sys.modules; then links
    # each
    # child stub as an attribute on its parent module. Returns None.
    """Insert namespace-style modules for ``areal`` ancestors with real paths."""
    for name, path in _STUB_PACKAGES:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            sys.modules[name] = mod

    for name, _path in _STUB_PACKAGES:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_attr = parts
            parent = sys.modules.get(parent_name)
            child = sys.modules.get(name)
            if parent is not None and child is not None:
                setattr(parent, child_attr, child)


if sys.version_info < (3, 12):
    _ensure_namespace_stubs()

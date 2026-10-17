# [MODULE DOC]
# Pytest conftest for the v2 inference_service tests. On Python < 3.12 it pre-registers
# empty namespace ModuleType stubs for "areal", "areal.experimental", "areal.v2" and
# "areal.v2.inference_service" in sys.modules (with __path__ set to the real source
# dirs) so submodules can be imported without executing areal/__init__.py, which uses
# 3.12-only syntax. Runs at import time (global sys.modules mutation).
# KNOWN ISSUE: the inline comment says the repo root is "4 levels up", but the code
# joins three ".." components from this directory (which is correct for
# tests/v2/inference_service).
"""Pre-import hook for Python < 3.12 compatibility.

The top-level ``areal/__init__.py`` imports from modules that use PEP 695 syntax
(``def func[T](...)``) which is only valid on Python 3.12+. When running tests on
Python 3.10/3.11 we register lightweight stubs for the ``areal`` namespace packages
so that importing ``areal.v2.inference_service.data_proxy.*`` never triggers the
problematic top-level init.
"""

from __future__ import annotations

import os
import sys
import types

# Resolve the repo root (4 levels up from this conftest)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Ordered list: parent before child so we can wire up attributes.
_STUB_PACKAGES = [
    ("areal", os.path.join(_REPO_ROOT, "areal")),
    ("areal.experimental", os.path.join(_REPO_ROOT, "areal", "experimental")),
    ("areal.v2", os.path.join(_REPO_ROOT, "areal", "v2")),
    (
        "areal.v2.inference_service",
        os.path.join(_REPO_ROOT, "areal", "v2", "inference_service"),
    ),
]


def _ensure_namespace_stubs():
    # [FUNCTION DOC]
    # HELPER: For each (name, path) in _STUB_PACKAGES not already imported, creates a
    # bare module with __path__ [path] and __package__ name and inserts it into
    # sys.modules. Then, for every dotted name, sets the child module as an attribute on
    # its parent module so attribute traversal (e.g. by unittest.mock.patch) works.
    # Returns None.
    """Insert namespace-style modules for ``areal`` ancestors with real paths."""
    for name, path in _STUB_PACKAGES:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            sys.modules[name] = mod

    # Wire parent → child attributes so unittest.mock.patch can traverse.
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

# [MODULE DOC]
# Tests that the TIR tools package and the AReaL sandbox module import cleanly when the
# optional daytona dependency is absent.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import sys

import tools as tools_module

from areal.infra.sandbox import DaytonaRunner


def test_tools_import_without_daytona():
    # [FUNCTION DOC]
    # Monkeypatches importlib.util.find_spec to report daytona as missing, evicts the
    # cached
    # daytona tool module and reloads the tools package; asserts PythonTool and
    # CalculatorTool
    # are exported but DaytonaPythonTool is not. Reloads tools again afterwards to
    # restore
    # the normal state (global module mutation).
    original_find_spec = importlib.util.find_spec

    with __import__("pytest").MonkeyPatch.context() as monkeypatch:

        def fake_find_spec(name, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: find_spec replacement returning None for "daytona" and delegating
            # to the
            # original find_spec otherwise.
            if name == "daytona":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        sys.modules.pop("tools.daytona_python_tool", None)
        reloaded = importlib.reload(tools_module)

        assert hasattr(reloaded, "PythonTool")
        assert hasattr(reloaded, "CalculatorTool")
        assert not hasattr(reloaded, "DaytonaPythonTool")

    importlib.reload(tools_module)


def test_sandbox_import_is_safe_without_daytona():
    # [FUNCTION DOC]
    # Asserts that DaytonaRunner from areal.infra.sandbox imported successfully (not
    # None).
    assert DaytonaRunner is not None

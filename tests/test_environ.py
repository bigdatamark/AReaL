# [MODULE DOC]
# Tests get_env_var and get_bool_env_var from areal/utils/environ.py, loaded from its
# file path with stub areal/areal.utils/areal.utils.logging modules injected into
# sys.modules (via monkeypatch) so the real package is not imported. Env vars are
# mutated only through monkeypatch.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared environment-variable parsing helpers."""

import importlib.util
import logging as stdlib_logging
import sys
import types
from pathlib import Path

_ENVIRON_PATH = Path(__file__).parents[1] / "areal/utils/environ.py"


def _load_environ(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Installs stub modules "areal", "areal.utils" and "areal.utils.logging"
    # (whose getLogger is stdlib logging.getLogger) into sys.modules via monkeypatch,
    # then executes areal/utils/environ.py as module "areal_test_environ" and returns
    # it.
    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = stdlib_logging.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    spec = importlib.util.spec_from_file_location("areal_test_environ", _ENVIRON_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_env_var_uses_primary_then_fallback(monkeypatch):
    # [FUNCTION DOC]
    # Asserts get_env_var returns the fallback name's value when the primary is unset,
    # and the primary's value once set.
    environ = _load_environ(monkeypatch)
    monkeypatch.delenv("AREAL_TEST_PRIMARY", raising=False)
    monkeypatch.setenv("AREAL_TEST_LEGACY", "legacy")

    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
        )
        == "legacy"
    )

    monkeypatch.setenv("AREAL_TEST_PRIMARY", "primary")
    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
        )
        == "primary"
    )


def test_get_env_var_empty_value_fallback_is_configurable(monkeypatch):
    # [FUNCTION DOC]
    # With primary set to empty string: asserts the fallback is used by default, but ""
    # is returned when allow_empty=True.
    environ = _load_environ(monkeypatch)
    monkeypatch.setenv("AREAL_TEST_PRIMARY", "")
    monkeypatch.setenv("AREAL_TEST_LEGACY", "legacy")

    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
        )
        == "legacy"
    )
    assert (
        environ.get_env_var(
            "AREAL_TEST_PRIMARY",
            fallback_names=("AREAL_TEST_LEGACY",),
            allow_empty=True,
        )
        == ""
    )


def test_get_bool_env_var_preserves_defaults_and_supports_opt_in_values(monkeypatch):
    # [FUNCTION DOC]
    # With value "yes": asserts get_bool_env_var returns False with default truthy set,
    # and True when custom truthy/falsy values including "yes" are passed.
    environ = _load_environ(monkeypatch)
    monkeypatch.setenv("AREAL_TEST_BOOL", "yes")

    assert environ.get_bool_env_var("AREAL_TEST_BOOL") is False
    assert (
        environ.get_bool_env_var(
            "AREAL_TEST_BOOL",
            truthy_values=("true", "1", "yes", "on"),
            falsy_values=("false", "0", "no", "off"),
        )
        is True
    )


def test_get_bool_env_var_can_strip_legacy_dte_values(monkeypatch):
    # [FUNCTION DOC]
    # With value " on " and strip_value=True plus custom truthy values, asserts True
    # (whitespace stripped before matching).
    environ = _load_environ(monkeypatch)
    monkeypatch.setenv("AREAL_TEST_BOOL", " on ")

    assert (
        environ.get_bool_env_var(
            "AREAL_TEST_BOOL",
            truthy_values=("true", "1", "yes", "on"),
            falsy_values=("false", "0", "no", "off"),
            strip_value=True,
        )
        is True
    )

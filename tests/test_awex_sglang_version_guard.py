# [MODULE DOC]
# Tests the SGLang version guard of the awex colocate plugin
# (areal.engine.awex.sglang_plugin): assert_supported_sglang_version must raise
# RuntimeError for versions outside SUPPORTED_SGLANG_VERSIONS. The installed version is
# faked by patching sglang_plugin.pkg_version.get_version.
"""The colocate plugin patches SGLang internals, so it pins the versions it knows."""

from unittest import mock

import pytest

from areal.engine.awex import sglang_plugin


def test_register_rejects_unverified_sglang_version():
    # [FUNCTION DOC]
    # Fakes version "0.5.11" and asserts RuntimeError whose message contains that
    # version.
    with mock.patch.object(
        sglang_plugin.pkg_version, "get_version", return_value="0.5.11"
    ):
        with pytest.raises(RuntimeError, match="0.5.11"):
            sglang_plugin.assert_supported_sglang_version()


def test_register_accepts_verified_sglang_versions():
    # [FUNCTION DOC]
    # For every entry of SUPPORTED_SGLANG_VERSIONS, fakes it and asserts the guard does
    # not raise.
    for version in sglang_plugin.SUPPORTED_SGLANG_VERSIONS:
        with mock.patch.object(
            sglang_plugin.pkg_version, "get_version", return_value=version
        ):
            sglang_plugin.assert_supported_sglang_version()


def test_error_names_the_supported_versions():
    # [FUNCTION DOC]
    # Fakes "0.4.0", captures the RuntimeError and asserts its message lists every
    # supported version.
    with mock.patch.object(
        sglang_plugin.pkg_version, "get_version", return_value="0.4.0"
    ):
        with pytest.raises(RuntimeError) as excinfo:
            sglang_plugin.assert_supported_sglang_version()
    message = str(excinfo.value)
    for version in sglang_plugin.SUPPORTED_SGLANG_VERSIONS:
        assert version in message

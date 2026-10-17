# [MODULE DOC]
# Unit and integration tests for the synchronous DaytonaRunner and DaytonaRunResult
# (areal.infra.sandbox). Integration tests hit the real Daytona service and are skipped
# unless DAYTONA_API_KEY is set.
# SPDX-License-Identifier: Apache-2.0

"""Integration and unit tests for the synchronous Daytona runner."""

from __future__ import annotations

import os

import pytest

import areal.infra.sandbox.runner as runner_module
from areal.infra.sandbox import DaytonaRunner, DaytonaRunResult


def _has_daytona_credentials() -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if DAYTONA_API_KEY is set to a non-empty value.
    return bool(os.environ.get("DAYTONA_API_KEY"))


def test_run_result_fields_round_trip():
    # [FUNCTION DOC]
    # Constructs a DaytonaRunResult and asserts all fields (stdout, stderr, exit_code,
    # charts, error) read back unchanged.
    result = DaytonaRunResult(
        stdout="x",
        stderr="",
        exit_code=0,
        charts=[],
        error=None,
    )

    assert result.stdout == "x"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.charts == []
    assert result.error is None


def test_runner_raises_clear_error_when_daytona_missing(monkeypatch):
    # [FUNCTION DOC]
    # Patches the runner module's SDK loader to raise ImportError and asserts
    # DaytonaRunner() propagates an ImportError mentioning daytona.
    def raise_import_error():
        # [FUNCTION DOC]
        # HELPER: Raises ImportError('daytona is not installed').
        raise ImportError("daytona is not installed")

    monkeypatch.setattr(runner_module, "_load_daytona_sdk", raise_import_error)

    with pytest.raises(ImportError, match="daytona"):
        DaytonaRunner()


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_runs_python_code():
    # [FUNCTION DOC]
    # Integration: runs print('hello') in a sandbox and asserts exit code 0 and 'hello'
    # in stdout. Calls close() twice (in finally and at the end), implicitly exercising
    # close idempotency.
    runner = DaytonaRunner()

    try:
        result = runner.run("print('hello')")
    finally:
        runner.close()

    assert result.exit_code == 0
    assert "hello" in result.stdout

    runner.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_context_manager_cleans_up():
    # [FUNCTION DOC]
    # Integration: uses the runner as a context manager and asserts the code ran
    # successfully and output was captured.
    with DaytonaRunner() as runner:
        result = runner.run("print('context manager')")

    assert result.exit_code == 0
    assert "context manager" in result.stdout


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_captures_runtime_error():
    # [FUNCTION DOC]
    # Integration: running code that raises ValueError yields a nonzero exit code and a
    # truthy error.
    with DaytonaRunner() as runner:
        result = runner.run("raise ValueError('x')")

    assert result.exit_code != 0
    assert result.error


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_extracts_matplotlib_charts():
    # [FUNCTION DOC]
    # Integration: running matplotlib plotting code with plt.show() yields a non-empty
    # charts list.
    code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [3, 2, 1])
plt.title('chart')
plt.show()
"""

    with DaytonaRunner() as runner:
        result = runner.run(code)

    assert result.charts


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
def test_runner_timeout_surfaces_as_error():
    # [FUNCTION DOC]
    # Integration: sleeping 10s with timeout=1 yields a nonzero exit code and an error.
    with DaytonaRunner() as runner:
        result = runner.run("import time; time.sleep(10)", timeout=1)

    assert result.exit_code != 0
    assert result.error

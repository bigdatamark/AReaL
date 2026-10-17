# [MODULE DOC]
# Unit tests for Slurm terminal-state detection: STATUS_MAPPING maps NODE_FAIL to
# FAILED, JobState.active() semantics, and query_terminal_state_sacct parsing of mocked
# sacct output (first record only; None when no record or sacct unavailable).
# SPDX-License-Identifier: Apache-2.0
"""Tests for terminal Slurm job-state detection."""

import subprocess
from unittest import mock

import pytest

from areal.infra.utils.launcher import JobState
from areal.infra.utils.slurm import STATUS_MAPPING, query_terminal_state_sacct


class TestNodeFailIsTerminal:
    # [CLASS DOC]
    # Tests job-state mapping and active() classification.
    def test_node_fail_maps_to_failed(self):
        # [FUNCTION DOC]
        # Asserts NODE_FAIL maps to JobState.FAILED.
        assert STATUS_MAPPING["NODE_FAIL"] is JobState.FAILED

    @pytest.mark.parametrize(
        "state", [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]
    )
    def test_terminal_states_are_not_active(self, state):
        # [FUNCTION DOC]
        # Parametrized; asserts COMPLETED/FAILED/CANCELLED are not active.
        assert not state.active()

    @pytest.mark.parametrize("state", [JobState.PENDING, JobState.RUNNING])
    def test_live_states_are_active(self, state):
        # [FUNCTION DOC]
        # Parametrized; asserts PENDING/RUNNING are active.
        assert state.active()


class TestQueryTerminalStateSacct:
    # [CLASS DOC]
    # Tests query_terminal_state_sacct with subprocess.check_output mocked.
    @pytest.mark.parametrize(
        "output,expected",
        [
            ("COMPLETED", JobState.COMPLETED),
            ("FAILED", JobState.FAILED),
            ("NODE_FAIL", JobState.FAILED),
            ("OUT_OF_MEMORY", JobState.FAILED),
            ("CANCELLED by 0", JobState.CANCELLED),
            ("CANCELLED+", JobState.CANCELLED),
            ("RUNNING", JobState.RUNNING),
        ],
    )
    def test_maps_sacct_output_to_a_job_state(self, output, expected):
        # [FUNCTION DOC]
        # Parametrized sacct outputs (including "CANCELLED by 0", "CANCELLED+",
        # OUT_OF_MEMORY); asserts the returned JobState matches.
        with mock.patch("subprocess.check_output", return_value=output.encode()):
            assert query_terminal_state_sacct(4242) is expected

    def test_returns_none_when_sacct_has_no_record(self):
        # [FUNCTION DOC]
        # Asserts whitespace-only output yields None.
        with mock.patch("subprocess.check_output", return_value=b"   \n"):
            assert query_terminal_state_sacct(4242) is None

    @pytest.mark.parametrize(
        "error",
        [
            subprocess.CalledProcessError(1, "sacct"),
            FileNotFoundError("sacct"),
        ],
    )
    def test_returns_none_when_sacct_is_unavailable(self, error):
        # [FUNCTION DOC]
        # Parametrized CalledProcessError / FileNotFoundError; asserts None is returned.
        with mock.patch("subprocess.check_output", side_effect=error):
            assert query_terminal_state_sacct(4242) is None

    def test_reads_only_the_first_record(self):
        # [FUNCTION DOC]
        # Asserts only the first line ("FAILED") of multi-line output is used.
        with mock.patch("subprocess.check_output", return_value=b"FAILED\nCOMPLETED\n"):
            assert query_terminal_state_sacct(4242) is JobState.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# [MODULE DOC]
# Click CliRunner tests for the "areal agent" command group: expected subcommands are
# listed, session-level verbs are not registered, and "run" has no --session-key.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from click.testing import CliRunner

from areal.v2.cli.cli import cli


def test_agent_help_contains_expected_commands():
    # [FUNCTION DOC]
    # Asserts "agent --help" exits 0 and lists run, stop, status, ps and logs.
    result = CliRunner().invoke(cli, ["agent", "--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "stop" in result.output
    assert "status" in result.output
    assert "ps" in result.output
    assert "logs" in result.output


def test_agent_help_does_not_register_session_commands():
    # [FUNCTION DOC]
    # Asserts new_session, switch_session, chat and reward are unknown agent subcommands
    # (non-zero exit with "No such command").
    runner = CliRunner()

    for verb in ("new_session", "switch_session", "chat", "reward"):
        result = runner.invoke(cli, ["agent", verb])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_agent_run_does_not_accept_initial_session_key():
    # [FUNCTION DOC]
    # Asserts "agent run --help" does not mention --session-key.
    result = CliRunner().invoke(cli, ["agent", "run", "--help"])

    assert result.exit_code == 0
    assert "--session-key" not in result.output

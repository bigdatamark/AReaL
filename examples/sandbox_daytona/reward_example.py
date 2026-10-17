# [MODULE DOC]
# Example showing how to use the Daytona remote sandbox runner as a code-execution
# reward: runs model-written Python and compares stdout to an expected value. Requires
# DAYTONA_API_KEY; running it as a script grades a good and a bad snippet.
"""Example: use DaytonaRunner to grade model-generated Python code.

Usage:
    export DAYTONA_API_KEY=***
    uv run python examples/sandbox_daytona/reward_example.py
"""

from areal.infra.sandbox import DaytonaRunner


def grade(model_code: str, expected_stdout: str) -> float:
    # [FUNCTION DOC]
    # Runs model_code in a fresh DaytonaRunner sandbox (5s timeout; the sandbox is torn
    # down by the context manager) and returns 1.0 if the whitespace-stripped stdout
    # equals the stripped expected_stdout, else 0.0. Sandbox/network errors propagate.
    with DaytonaRunner() as runner:
        result = runner.run(model_code, timeout=5)
        return 1.0 if result.stdout.strip() == expected_stdout.strip() else 0.0


if __name__ == "__main__":
    good = "print(sum(range(10)))"
    bad = "print('oops')"
    print(f"Good reward: {grade(good, '45')}")
    print(f"Bad reward:  {grade(bad, '45')}")

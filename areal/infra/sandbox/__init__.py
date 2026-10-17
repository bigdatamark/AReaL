# [PACKAGE DOC]
# Sandbox package for remote code execution (currently Daytona cloud sandboxes,
# installed via the "sandbox" extra). Re-exports DaytonaClientManager, DaytonaRunner and
# DaytonaRunResult.
# SPDX-License-Identifier: Apache-2.0

"""Sandbox backends for code execution in rollouts and user scripts.

Currently provides Daytona as a cloud sandbox. Install via::

    uv sync --extra sandbox
"""

from ._client import DaytonaClientManager
from .runner import DaytonaRunner, DaytonaRunResult

__all__ = ["DaytonaClientManager", "DaytonaRunResult", "DaytonaRunner"]

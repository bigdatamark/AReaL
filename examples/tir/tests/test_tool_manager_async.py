# [MODULE DOC]
# Integration test for async tool dispatch of ToolManager against the real Daytona
# sandbox backend; skipped unless the daytona package and DAYTONA_API_KEY are present.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest
from tool_manager import ToolManager
from tools import ToolCallStatus


def _has_daytona_credentials() -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the daytona package is importable and the DAYTONA_API_KEY
    # environment variable is non-empty.
    return importlib.util.find_spec("daytona") is not None and bool(
        os.environ.get("DAYTONA_API_KEY")
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
@pytest.mark.asyncio
async def test_tool_manager_async_dispatch_daytona():
    # [FUNCTION DOC]
    # Builds a ToolManager with only daytona_python enabled, runs aexecute_tool_call on
    # a
    # fenced python block printing 1, always cleans up in a worker thread, and asserts
    # SUCCESS
    # status with "1" in the output. Network/sandbox side effects.
    manager = ToolManager(enabled_tools="daytona_python")

    try:
        result, status = await manager.aexecute_tool_call("```python\nprint(1)\n```")
    finally:
        await asyncio.to_thread(manager.cleanup)

    assert status == ToolCallStatus.SUCCESS
    assert "1" in result

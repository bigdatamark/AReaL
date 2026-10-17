# [MODULE DOC]
# Tests that BaseTool's default async aexecute delegates to the synchronous execute.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from tools import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType


class DummyTool(BaseTool):
    # [CLASS DOC]
    # Minimal concrete BaseTool for testing: CALCULATOR type, a "dummy" description,
    # dummy start/end markers, identity parsing and an execute that always succeeds.
    @property
    def tool_type(self) -> ToolType:
        # [FUNCTION DOC]
        # Returns ToolType.CALCULATOR.
        return ToolType.CALCULATOR

    @property
    def description(self) -> ToolDescription:
        # [FUNCTION DOC]
        # Returns a ToolDescription named "dummy" with empty parameters, prompt and
        # example.
        return ToolDescription(
            name="dummy",
            description="dummy",
            parameters={},
            parameter_prompt="",
            example="",
        )

    @property
    def markers(self) -> ToolMarkers:
        # [FUNCTION DOC]
        # Returns ToolMarkers with start marker "<dummy>" and end marker "</dummy>".
        return ToolMarkers(start_markers=["<dummy>"], end_markers=["</dummy>"])

    def parse_parameters(self, text: str) -> dict[str, str]:
        # [FUNCTION DOC]
        # Wraps the raw text as a dict under key "text".
        return {"text": text}

    def execute(self, parameters: dict[str, str]) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # Ignores parameters; returns ("ok", ToolCallStatus.SUCCESS).
        return "ok", ToolCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_base_tool_aexecute_delegates_to_execute_by_default():
    # [FUNCTION DOC]
    # Asserts awaiting DummyTool().aexecute({}) yields ("ok", SUCCESS).
    tool = DummyTool()

    assert await tool.aexecute({}) == ("ok", ToolCallStatus.SUCCESS)

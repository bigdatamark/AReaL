# [MODULE DOC]
# Unit tests for areal.experimental.openai.client._align_tools_with_sglang, which
# normalizes tool definitions (chat-shaped dicts, flat Responses-API dicts, pydantic
# models) into the exact dict shape SGLang's own Tool model produces, so tool rendering
# in chat templates matches SGLang's native path. Skipped entirely if SGLang's protocol
# module cannot be imported; a fixture simulates its absence for fallback-path tests.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from pydantic import BaseModel

from areal.experimental.openai.client import _align_tools_with_sglang

pytest.importorskip(
    "sglang.srt.entrypoints.openai.protocol",
    reason="alignment target is sglang's own Tool model",
)

CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

FLAT_RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _sglang_rendering(tool: dict) -> dict:
    # [FUNCTION DOC]
    # HELPER: Reference rendering: validates the tool dict with SGLang's Tool pydantic
    # model and returns its model_dump (including default fields).
    from sglang.srt.entrypoints.openai.protocol import Tool

    return Tool(**tool).model_dump()


def test_chat_tool_matches_sglang_native_rendering():
    # [FUNCTION DOC]
    # Asserts aligning a chat-shaped tool yields exactly SGLang's native rendering.
    assert _align_tools_with_sglang([CHAT_TOOL]) == [_sglang_rendering(CHAT_TOOL)]


def test_alignment_adds_the_defaults_that_caused_the_drift():
    # [FUNCTION DOC]
    # Asserts alignment fills SGLang defaults (strict False, description None) on the
    # output function while leaving the input CHAT_TOOL constant unmutated.
    function = _align_tools_with_sglang([CHAT_TOOL])[0]["function"]

    assert function["strict"] is False
    assert function["description"] is None
    assert "strict" not in CHAT_TOOL["function"]


def test_flat_responses_tool_is_normalized_to_the_chat_shape():
    # [FUNCTION DOC]
    # Asserts a flat Responses-style tool (name/parameters at top level) aligns to the
    # same result as the nested chat-shaped tool.
    assert _align_tools_with_sglang([FLAT_RESPONSES_TOOL]) == _align_tools_with_sglang(
        [CHAT_TOOL]
    )


def test_pydantic_tool_input_is_accepted():
    # [FUNCTION DOC]
    # Asserts pydantic model tool instances (locally defined Tool/Function models) are
    # accepted and their function name is preserved.
    class Function(BaseModel):
        # [CLASS DOC]
        # HELPER: Minimal pydantic function schema with name and parameters.
        name: str
        parameters: dict

    class Tool(BaseModel):
        # [CLASS DOC]
        # HELPER: Minimal pydantic tool schema with type and nested Function.
        type: str
        function: Function

    aligned = _align_tools_with_sglang(
        [Tool(type="function", function=Function(name="get_weather", parameters={}))]
    )

    assert aligned[0]["function"]["name"] == "get_weather"


def test_alignment_is_idempotent():
    # [FUNCTION DOC]
    # Asserts aligning an already aligned list returns an equal list (idempotence).
    once = _align_tools_with_sglang([CHAT_TOOL])

    assert _align_tools_with_sglang(once) == once


def test_invalid_tool_is_passed_through_without_failing_the_batch():
    # [FUNCTION DOC]
    # Asserts an invalid tool (missing function) is passed through unchanged while valid
    # tools in the same list are still aligned.
    aligned = _align_tools_with_sglang([CHAT_TOOL, {"type": "function"}])

    assert aligned[0] == _sglang_rendering(CHAT_TOOL)
    assert aligned[1] == {"type": "function"}


def test_unsupported_tool_type_is_passed_through():
    # [FUNCTION DOC]
    # Asserts a non-dict, non-model entry (a string) is returned unchanged.
    assert _align_tools_with_sglang(["not-a-tool"]) == ["not-a-tool"]


def test_empty_list_is_unchanged():
    # [FUNCTION DOC]
    # Asserts an empty list aligns to an empty list.
    assert _align_tools_with_sglang([]) == []


@pytest.fixture
def sglang_missing(monkeypatch):
    # [FUNCTION DOC]
    # Fixture simulating SGLang being unavailable: sets the protocol module entry in
    # sys.modules to None (making imports fail) and removes the function's
    # _warned_no_sglang one-time-warning attribute so the fallback warning path runs
    # fresh. Both changes are undone by monkeypatch.
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.openai.protocol", None)
    monkeypatch.delattr(_align_tools_with_sglang, "_warned_no_sglang", raising=False)


def test_flat_responses_tool_is_still_nested_without_sglang(sglang_missing):
    # [FUNCTION DOC]
    # Asserts that without SGLang a flat Responses tool is still nested into the chat
    # shape (without added defaults).
    assert _align_tools_with_sglang([FLAT_RESPONSES_TOOL]) == [CHAT_TOOL]


def test_pydantic_tool_becomes_a_dict_without_sglang(sglang_missing):
    # [FUNCTION DOC]
    # Asserts that without SGLang a pydantic tool is converted to a plain dict equal to
    # CHAT_TOOL.
    class Function(BaseModel):
        # [CLASS DOC]
        # HELPER: Minimal pydantic function schema with name and parameters.
        name: str
        parameters: dict

    class Tool(BaseModel):
        # [CLASS DOC]
        # HELPER: Minimal pydantic tool schema with type and nested Function.
        type: str
        function: Function

    tool = Tool(type="function", function=Function(**CHAT_TOOL["function"]))

    assert _align_tools_with_sglang([tool]) == [CHAT_TOOL]


def test_chat_tool_keeps_its_fields_without_sglang(sglang_missing):
    # [FUNCTION DOC]
    # Asserts that without SGLang a chat tool is returned unchanged with no strict field
    # added.
    aligned = _align_tools_with_sglang([CHAT_TOOL])

    assert aligned == [CHAT_TOOL]
    assert "strict" not in aligned[0]["function"]

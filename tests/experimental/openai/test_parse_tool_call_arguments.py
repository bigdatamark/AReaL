# [MODULE DOC]
# Unit tests for _parse_tool_call_arguments, which converts JSON-string tool-call
# arguments in assistant messages into dicts (needed by chat templates that expect dict
# arguments) while leaving other messages untouched.
# SPDX-License-Identifier: Apache-2.0

from areal.experimental.openai.client import _parse_tool_call_arguments


def test_string_arguments_are_parsed_to_dict():
    # [FUNCTION DOC]
    # Asserts a JSON string arguments field is parsed into a dict.
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "Bash", "arguments": '{"command": "ls"}'}}
            ],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_non_string_arguments_are_left_unchanged():
    # [FUNCTION DOC]
    # Asserts already-dict arguments leave the message object identical (same object
    # returned, not a copy).
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "Bash", "arguments": {"command": "ls"}}}
            ],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0] is messages[0]
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}


def test_invalid_json_arguments_are_left_unchanged():
    # [FUNCTION DOC]
    # Asserts arguments that are not valid JSON are left as the original string (no
    # error).
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "Bash", "arguments": "not json"}}],
        }
    ]

    out = _parse_tool_call_arguments(messages)

    assert out[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_messages_without_tool_calls_pass_through():
    # [FUNCTION DOC]
    # Asserts a message without tool_calls is passed through as the same object.
    messages = [{"role": "user", "content": "hi"}]

    out = _parse_tool_call_arguments(messages)

    assert out[0] is messages[0]

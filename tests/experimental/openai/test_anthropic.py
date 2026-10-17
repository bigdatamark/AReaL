# [MODULE DOC]
# Unit test for translate_anthropic_request, which converts an Anthropic Messages API
# request into an OpenAI chat-completions style request.
from areal.experimental.openai.anthropic import translate_anthropic_request


def test_translate_anthropic_request_preserves_tool_round_trip():
    # [FUNCTION DOC]
    # Translates a request containing an assistant tool_use block and a user tool_result
    # block and asserts the tool call id and function name survive in the assistant
    # message's tool_calls, and the following tool message carries the matching
    # tool_call_id.
    translated = translate_anthropic_request(
        {
            "model": "claude-compatible",
            "max_tokens": 64,
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"path": "/tmp/example"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-1",
                            "content": "example contents",
                        }
                    ],
                },
            ],
        }
    )

    tool_call = translated["messages"][0]["tool_calls"][0]
    assert tool_call["id"] == "tool-1"
    assert tool_call["function"]["name"] == "Read"
    assert translated["messages"][1]["tool_call_id"] == "tool-1"

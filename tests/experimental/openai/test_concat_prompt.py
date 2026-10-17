# [MODULE DOC]
# Unit tests for concat_prompt_token_ids_with_parent, which builds a child turn's prompt
# token ids by appending the newly added messages to the parent interaction's prompt
# plus generated output tokens ("concat" mode). Each test checks the result equals
# applying the Qwen3-0.6B chat template (with tools) to the full conversation with a
# generation prompt, focusing on tool-calling flows. Uses fake ModelResponses built from
# fixed text; no inference server or GPU needed (tokenizer only).
# This file tests the concat_prompt_token_ids_with_parent function.
# It verifies that the concat mode produces tokens consistent with
# applying chat_template to the full conversation messages.
# Focus on tool calling scenarios: user request -> LLM tool_call -> tool result

import json

import pytest
from openai.types.chat import ChatCompletionToolParam

from tests.utils import get_model_path

from areal.api import ModelResponse
from areal.experimental.openai.client import (
    concat_prompt_token_ids_with_parent,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer

MODEL_PATH = "Qwen/Qwen3-0.6B"
LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B"

# Fixed fake response content for testing
FAKE_RESPONSE_CONTENT = "<think>\na\n</think>\n\n111"

# Fake tool call response content
FAKE_TOOL_CALL_CONTENT = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'


@pytest.fixture(scope="module")
def tokenizer():
    # [FUNCTION DOC]
    # Module-scoped fixture loading the Qwen3-0.6B tokenizer (local path or hub id).
    return load_hf_tokenizer(get_model_path(LOCAL_MODEL_PATH, MODEL_PATH))


def create_fake_model_response(
    tokenizer, input_tokens: list[int], response_content: str, stop_reason: str = "stop"
) -> ModelResponse:
    # [FUNCTION DOC]
    # HELPER: Builds a ModelResponse whose output_tokens are the tokenization of
    # response_content without special tokens, plus the eos token id unless stop_reason
    # is "length"; logprobs are all 0.0 and versions all 0. Returns the ModelResponse.
    """Create a fake ModelResponse with the given input tokens and response content."""
    # Tokenize the response content and add the eos token
    output_tokens = tokenizer.encode(response_content, add_special_tokens=False)
    if stop_reason != "length":
        # Add eos_token_id to simulate stop token
        output_tokens.append(tokenizer.eos_token_id)
    return ModelResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[0.0] * len(output_tokens),
        output_versions=[0] * len(output_tokens),
        stop_reason=stop_reason,
        tokenizer=tokenizer,
    )


def create_interaction_with_response(
    tokenizer,
    messages: list[dict],
    parent: InteractionWithTokenLogpReward | None,
    output_message_list: list[dict] | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    response_content: str = FAKE_RESPONSE_CONTENT,
    stop_reason: str = "stop",
) -> InteractionWithTokenLogpReward:
    # [FUNCTION DOC]
    # HELPER: Tokenizes messages via apply_chat_template (with tools, generation prompt)
    # as input tokens, attaches a fake model response with response_content/stop_reason,
    # and returns an InteractionWithTokenLogpReward with the messages, parent,
    # output_message_list and chat_template_type "concat".
    """Create an InteractionWithTokenLogpReward with a fake model response."""
    # Calculate input tokens using chat template
    input_tokens = apply_chat_template(
        tokenizer, messages, tools=tools, add_generation_prompt=True, tokenize=True
    )

    # Create fake model response
    model_response = create_fake_model_response(
        tokenizer, input_tokens, response_content, stop_reason=stop_reason
    )

    return InteractionWithTokenLogpReward(
        messages=messages,
        model_response=model_response,
        parent=parent,
        output_message_list=output_message_list,
        chat_template_type="concat",
    )


# Tool definitions for testing (typed as ChatCompletionToolParam)
WEATHER_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. Paris, London",
                }
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}


class TestConcatPromptTokenIds:
    # [CLASS DOC]
    # Test suite grouping the concat-prompt token equivalence checks; each method takes
    # the tokenizer fixture.
    """Test cases for concat_prompt_token_ids_with_parent function."""

    def test_single_turn_no_parent(self, tokenizer):
        # [FUNCTION DOC]
        # Asserts with no parent the concat result equals the direct chat template
        # tokens.
        """Test single turn conversation without parent - should match direct chat_template."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        # Using concat mode with no parent
        concat_tokens = concat_prompt_token_ids_with_parent(
            message_list=messages,
            parent=None,
            tokenizer=tokenizer,
        )

        # Direct chat_template application
        direct_tokens = apply_chat_template(
            tokenizer, messages, add_generation_prompt=True, tokenize=True
        )

        # With no parent, concat mode should produce the same tokens
        assert concat_tokens == direct_tokens, (
            f"Tokens mismatch for single turn without parent.\n"
            f"Expected: {direct_tokens}\n"
            f"Got: {concat_tokens}"
        )

    def test_tool_call_single_round(self, tokenizer):
        # [FUNCTION DOC]
        # Asserts with no parent and the weather tool provided, concat tokens equal the
        # direct template rendering including the tool schema.
        """Test user request -> LLM returns tool_call.

        Scenario:
        1. User asks about weather
        2. LLM responds with a tool_call
        """
        tools = [WEATHER_TOOL]

        # Round 1: User asks about weather
        messages_round1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        # Using concat mode with no parent
        concat_tokens = concat_prompt_token_ids_with_parent(
            message_list=messages_round1,
            parent=None,
            tokenizer=tokenizer,
            tools=tools,
        )

        # Direct chat_template application
        direct_tokens = apply_chat_template(
            tokenizer,
            messages_round1,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert concat_tokens == direct_tokens, (
            f"Tokens mismatch for tool call single round.\n"
            f"Direct tokens length: {len(direct_tokens)}\n"
            f"Concat tokens length: {len(concat_tokens)}\n"
        )

    def test_tool_call_with_tool_result(self, tokenizer):
        # [FUNCTION DOC]
        # Parent turn: user asks weather, model output is think block + tool_call text
        # ending with eos, recorded assistant message carries one tool call. Child adds
        # only the tool result; asserts concat(parent, [tool result]) equals the
        # template over the full conversation.
        """Test user request -> LLM tool_call -> tool result.

        Scenario:
        1. User asks about weather
        2. LLM responds with a tool_call
        3. Tool returns result
        """
        tools = [WEATHER_TOOL]

        # Round 1: User asks about weather, LLM responds with tool_call
        messages_round1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        content = "<think>\n123\n</think>"
        tool_call_response = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'

        interaction_round1 = create_interaction_with_response(
            tokenizer,
            messages_round1,
            parent=None,
            output_message_list=[
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Paris"}),
                            },
                        }
                    ],
                }
            ],
            tools=tools,
            response_content=content + "\n\n" + tool_call_response,
        )

        # Round 2: Add tool result
        messages_round2 = messages_round1 + [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        remaining_messages_round2 = [
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        # Using concat mode with parent
        concat_tokens = concat_prompt_token_ids_with_parent(
            message_list=remaining_messages_round2,
            parent=interaction_round1,
            tokenizer=tokenizer,
            tools=tools,
        )

        # Direct chat_template application to full conversation
        direct_tokens = apply_chat_template(
            tokenizer,
            messages_round2,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert concat_tokens == direct_tokens, (
            f"Tokens mismatch for tool call with tool result.\n"
            f"Direct tokens length: {len(direct_tokens)}\n"
            f"Concat tokens length: {len(concat_tokens)}\n"
            f"Direct tokens: {tokenizer.decode(direct_tokens)}\n"
            f"Concat tokens: {tokenizer.decode(concat_tokens)}"
        )

    def test_tool_call_stop_by_length(self, tokenizer):
        # [FUNCTION DOC]
        # Same as the tool-result test but the parent's response stopped by "length" (no
        # eos appended), asserting concat tokens still equal the full template rendering
        # (the function must supply the missing end-of-turn tokens).
        # KNOWN ISSUE: docstring is copied from the previous test and does not mention
        # the length-stop scenario.
        """Test user request -> LLM tool_call -> tool result.

        Scenario:
        1. User asks about weather
        2. LLM responds with a tool_call
        3. Tool returns result
        """
        tools = [WEATHER_TOOL]
        stop_reason = "length"

        # Round 1: User asks about weather, LLM responds with tool_call
        messages_round1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        content = "<think>\n123\n</think>"
        tool_call_response = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'

        interaction_round1 = create_interaction_with_response(
            tokenizer,
            messages_round1,
            parent=None,
            output_message_list=[
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Paris"}),
                            },
                        }
                    ],
                }
            ],
            tools=tools,
            response_content=content + "\n\n" + tool_call_response,
            stop_reason=stop_reason,
        )

        # Round 2: Add tool result
        messages_round2 = messages_round1 + [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        remaining_messages_round2 = [
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        # Using concat mode with parent
        concat_tokens = concat_prompt_token_ids_with_parent(
            message_list=remaining_messages_round2,
            parent=interaction_round1,
            tokenizer=tokenizer,
            tools=tools,
        )

        # Direct chat_template application to full conversation
        direct_tokens = apply_chat_template(
            tokenizer,
            messages_round2,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert concat_tokens == direct_tokens, (
            f"Tokens mismatch for tool call with tool result.\n"
            f"Direct tokens length: {len(direct_tokens)}\n"
            f"Concat tokens length: {len(concat_tokens)}\n"
            f"Direct tokens: {tokenizer.decode(direct_tokens)}\n"
            f"Concat tokens: {tokenizer.decode(concat_tokens)}"
        )

    def test_multiple_tool_calls_sequence(self, tokenizer):
        # [FUNCTION DOC]
        # Parent turn issues a get_weather tool call with a different think text;
        # asserts round 2 (tool result appended) concat tokens equal the full template
        # rendering.
        # KNOWN ISSUE: docstring describes a fourth step (final answer generation) that
        # the test never performs.
        """Test multiple sequential tool calls.

        Scenario:
        1. User asks about weather
        2. LLM calls get_weather tool
        3. Tool returns result
        4. LLM provides final answer (simulate as another generation)
        """
        tools = [WEATHER_TOOL]

        # Round 1: User asks about weather
        messages_round1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
        ]

        content_round1 = "<think>\nLet me check the weather.\n</think>"
        tool_call_response = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'

        interaction_round1 = create_interaction_with_response(
            tokenizer,
            messages_round1,
            parent=None,
            output_message_list=[
                {
                    "role": "assistant",
                    "content": content_round1,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Paris"}),
                            },
                        }
                    ],
                }
            ],
            tools=tools,
            response_content=content_round1 + "\n\n" + tool_call_response,
        )

        # Round 2: Tool result added
        messages_round2 = messages_round1 + [
            {
                "role": "assistant",
                "content": content_round1,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        # Verify round 2 tokens match
        remaining_messages_round2 = [
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
        ]

        concat_tokens_round2 = concat_prompt_token_ids_with_parent(
            message_list=remaining_messages_round2,
            parent=interaction_round1,
            tokenizer=tokenizer,
            tools=tools,
        )

        direct_tokens_round2 = apply_chat_template(
            tokenizer,
            messages_round2,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert concat_tokens_round2 == direct_tokens_round2, (
            f"Tokens mismatch for multiple tool calls sequence (round 2).\n"
            f"Direct tokens length: {len(direct_tokens_round2)}\n"
            f"Concat tokens length: {len(concat_tokens_round2)}\n"
            f"Direct tokens: {tokenizer.decode(direct_tokens_round2)}\n"
            f"Concat tokens: {tokenizer.decode(concat_tokens_round2)}"
        )

    def test_tool_call_with_multiple_tools(self, tokenizer):
        # [FUNCTION DOC]
        # With both weather and calculator tools available, parent calls calculate;
        # asserts the concat prompt after the tool result "360" equals the full template
        # rendering.
        """Test with multiple tools available.

        Scenario:
        1. User asks a question that could use multiple tools
        2. LLM calls one of the tools
        3. Tool returns result
        """
        tools = [WEATHER_TOOL, CALCULATOR_TOOL]

        # Round 1: User asks question
        messages_round1 = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to weather and calculator tools.",
            },
            {"role": "user", "content": "What's 15 * 24?"},
        ]

        content = "<think>\nI need to use calculator.\n</think>"
        tool_call_response = '<tool_call>\n{"name": "calculate", "arguments": {"expression": "15 * 24"}}\n</tool_call>'

        interaction_round1 = create_interaction_with_response(
            tokenizer,
            messages_round1,
            parent=None,
            output_message_list=[
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": "call_calc_001",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": json.dumps({"expression": "15 * 24"}),
                            },
                        }
                    ],
                }
            ],
            tools=tools,
            response_content=content + "\n\n" + tool_call_response,
        )

        # Round 2: Tool result
        messages_round2 = messages_round1 + [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "call_calc_001",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps({"expression": "15 * 24"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_calc_001",
                "content": "360",
            },
        ]

        remaining_messages_round2 = [
            {
                "role": "tool",
                "tool_call_id": "call_calc_001",
                "content": "360",
            },
        ]

        concat_tokens = concat_prompt_token_ids_with_parent(
            message_list=remaining_messages_round2,
            parent=interaction_round1,
            tokenizer=tokenizer,
            tools=tools,
        )

        direct_tokens = apply_chat_template(
            tokenizer,
            messages_round2,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert concat_tokens == direct_tokens, (
            f"Tokens mismatch for multiple tools scenario.\n"
            f"Direct tokens length: {len(direct_tokens)}\n"
            f"Concat tokens length: {len(concat_tokens)}\n"
        )

    def test_parallel_tool_calls(self, tokenizer):
        # [FUNCTION DOC]
        # Parent emits two parallel get_weather tool calls; child appends two tool
        # results; asserts concat tokens equal the full template rendering.
        """Test parallel tool calls (multiple tool calls in one response).

        Scenario:
        1. User asks about weather in two cities
        2. LLM calls get_weather twice in parallel
        3. Both tool results are returned
        """
        tools = [WEATHER_TOOL]

        # Round 1: User asks about weather in two cities
        messages_round1 = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris and London?"},
        ]

        content = "<think>\nI need to check weather in both cities.\n</think>"
        # LLM responds with two parallel tool calls
        tool_call_response = (
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "London"}}\n</tool_call>'
        )

        interaction_round1 = create_interaction_with_response(
            tokenizer,
            messages_round1,
            parent=None,
            output_message_list=[
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "Paris"}),
                            },
                        },
                        {
                            "id": "call_002",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "London"}),
                            },
                        },
                    ],
                }
            ],
            tools=tools,
            response_content=content + "\n\n" + tool_call_response,
        )

        # Round 2: Both tool results
        messages_round2 = messages_round1 + [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    },
                    {
                        "id": "call_002",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "London"}),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": "The weather in London is cloudy, 15°C",
            },
        ]

        remaining_messages_round2 = [
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "The weather in Paris is sunny, 22°C",
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": "The weather in London is cloudy, 15°C",
            },
        ]

        concat_tokens = concat_prompt_token_ids_with_parent(
            message_list=remaining_messages_round2,
            parent=interaction_round1,
            tokenizer=tokenizer,
            tools=tools,
        )

        direct_tokens = apply_chat_template(
            tokenizer,
            messages_round2,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        assert concat_tokens == direct_tokens, (
            f"Tokens mismatch for parallel tool calls.\n"
            f"Direct tokens length: {len(direct_tokens)}\n"
            f"Concat tokens length: {len(concat_tokens)}\n"
        )

    def test_parent_without_model_response_raises_error(self, tokenizer):
        # [FUNCTION DOC]
        # Asserts concat_prompt_token_ids_with_parent raises ValueError ("Parent
        # interaction has no model_response") when the parent interaction has
        # model_response None.
        """Test that parent without model_response raises ValueError."""
        # Create parent interaction without model_response
        parent_interaction = InteractionWithTokenLogpReward(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            model_response=None,  # No model response
            parent=None,
            chat_template_type="concat",
        )

        messages = [
            {"role": "assistant", "content": "Hi there!"},
        ]

        with pytest.raises(
            ValueError, match="Parent interaction has no model_response"
        ):
            concat_prompt_token_ids_with_parent(
                message_list=messages,
                parent=parent_interaction,
                tokenizer=tokenizer,
            )

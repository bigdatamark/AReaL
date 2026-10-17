# [MODULE DOC]
# Integration tests for ArealOpenAI tool calling against a real SGLang server
# (Qwen3-0.6B, qwen25 tool parser, qwen3 reasoning parser): streaming tool-call chunks,
# single/multi-round and parallel tool calls, reward discounting across tool turns,
# think-tag stripping while preserving tool messages, and concat-style export with loss
# masks. Defines a module-level list of five function tool schemas (get_weather,
# calculate, get_fact, get_time, translate). Marked sglang; requires a GPU.
# KNOWN ISSUE: the header comment is copied from test_client.py and does not mention
# tools.
# This file tests the functionality of our customized OpenAI client.
# The client should be able to generate completions and correctly assign rewards with back-propagation.
import os
import subprocess
import sys
import time

import pytest
import requests

from tests.utils import get_model_path

from areal.api.cli_args import SGLangConfig
from areal.experimental.openai import ArealOpenAI
from areal.infra.utils.proc import kill_process_tree
from areal.utils import network, seeding
from areal.utils.hf_utils import load_hf_tokenizer

pytestmark = pytest.mark.sglang
EXPR_NAME = "test_openai"
TRIAL_NAME = "trial_0"
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
PORT, DIST_PORT = network.find_free_ports(2)
HOST = network.gethostip()
# set a large timeout since we may need to download the model from hub
RUN_SERVER_TIMEOUT = 180


def check_server_health(base_url):
    # [FUNCTION DOC]
    # HELPER: GETs <base_url>/health (30s timeout); returns True iff status 200, False
    # on any requests exception.
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def sglang_server():
    # [FUNCTION DOC]
    # Module-scoped fixture launching an SGLang server subprocess (seeded, mem fraction
    # 0.3, tp 1, GPU 0) and polling health once per second for up to 180s; raises
    # RuntimeError on timeout, yields, then kills the process tree.
    seeding.set_random_seed(1, EXPR_NAME)
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=MODEL_PATH,
            mem_fraction_static=0.3,
        ),
        host=HOST,
        port=PORT,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
    )
    # Launch process
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://{HOST}:{PORT}"
    tik = time.time()
    while time.time() - tik < RUN_SERVER_TIMEOUT:
        if check_server_health(base_url):
            break
        time.sleep(1)
    if time.time() - tik > RUN_SERVER_TIMEOUT:
        raise RuntimeError("server launch failed")
    yield
    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def tokenizer():
    # [FUNCTION DOC]
    # Module-scoped fixture returning the HF tokenizer for MODEL_PATH.
    return load_hf_tokenizer(MODEL_PATH)


@pytest.fixture
def openai_client(sglang_server, tokenizer):
    # [FUNCTION DOC]
    # Fixture building and initializing a RemoteSGLangEngine (sglang:d1) addressed via
    # the AREAL_LLM_SERVER_ADDRS env var, yielding an ArealOpenAI client with qwen25
    # tool parser, qwen3 reasoning parser and "hf" template type; destroys the engine
    # afterwards.
    from areal.api.cli_args import InferenceEngineConfig
    from areal.engine import RemoteSGLangEngine

    config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        max_concurrent_rollouts=2,
        consumer_batch_size=2,
    )
    os.environ["AREAL_LLM_SERVER_ADDRS"] = f"{HOST}:{PORT}"
    engine = RemoteSGLangEngine(config)
    engine.initialize()
    yield ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        chat_template_type="hf",
    )
    engine.destroy()


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 + 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fact",
            "description": "Get an interesting fact about a number",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number to get a fact about",
                    }
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone, e.g. 'America/New_York'",
                    },
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "Translate text to another language",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to translate"},
                    "target_language": {
                        "type": "string",
                        "description": "Target language code",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
]


@pytest.mark.asyncio
async def test_streaming_with_tool_calling(openai_client):
    # [FUNCTION DOC]
    # Streams a weather question with tools and asserts chunk structure; if any chunks
    # carry tool_calls, they must come in pairs where the first carries id, type
    # function, name and empty arguments and the second has the same index and non-empty
    # arguments, and the single final chunk's finish_reason must be tool_calls.
    """Test streaming mode with tool calling."""
    chunks = []
    stream = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with access to weather information.",
            },
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ],
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=8192,
        stream=True,
    )

    async for chunk in stream:
        chunks.append(chunk)

    # Verify we got multiple chunks
    assert len(chunks) > 0, "Should receive at least one chunk"

    # Verify chunk structure
    for chunk in chunks:
        assert chunk.id is not None
        assert chunk.object == "chat.completion.chunk"
        assert len(chunk.choices) == 1

    # Find chunks with tool calls
    tool_call_chunks = [
        chunk
        for chunk in chunks
        if chunk.choices[0].delta.tool_calls is not None
        and len(chunk.choices[0].delta.tool_calls) > 0
    ]

    # Verify tool calls are present and split into name/id and arguments chunks.
    if tool_call_chunks:
        assert len(tool_call_chunks) % 2 == 0
        for first_chunk, second_chunk in zip(
            tool_call_chunks[::2], tool_call_chunks[1::2], strict=True
        ):
            first_tool_call = first_chunk.choices[0].delta.tool_calls[0]
            second_tool_call = second_chunk.choices[0].delta.tool_calls[0]
            assert first_tool_call.id is not None
            assert first_tool_call.type == "function"
            assert first_tool_call.function is not None
            assert first_tool_call.function.name is not None
            assert first_tool_call.function.arguments == ""
            assert second_tool_call.index == first_tool_call.index
            assert second_tool_call.function is not None
            assert second_tool_call.function.arguments is not None
            assert len(second_tool_call.function.arguments) > 0

    # Find the final chunk with finish_reason
    final_chunks = [
        chunk for chunk in chunks if chunk.choices[0].finish_reason is not None
    ]
    assert len(final_chunks) == 1, "Should have exactly one chunk with finish_reason"

    # If tool calls were made, verify finish_reason is "tool_calls"
    if tool_call_chunks:
        assert final_chunks[0].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_single_round_tool_calling(openai_client):
    # [FUNCTION DOC]
    # Asserts a weather question yields an assistant message with tool_calls and
    # finish_reason tool_calls, then that a reward of 1.5 is exported individually for
    # it.
    """Test single-round conversation with tool calling."""

    c = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant with access to weather information.",
            },
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ],
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=8192,
    )

    # Check if tool call was made (might depend on model capability)
    assert c.id is not None
    assert c.choices[0].message.role == "assistant"
    assert c.choices[0].message.tool_calls is not None
    assert c.choices[0].finish_reason == "tool_calls"

    openai_client.set_reward(c.id, reward=1.5)
    completions = openai_client.export_interactions(style="individual")

    assert len(completions) == 1
    assert completions[c.id].reward == 1.5


@pytest.mark.asyncio
async def test_multi_round_tool_calling(openai_client):
    # [FUNCTION DOC]
    # Runs three rounds with mocked tool results (tool_call_ids not matching real ones),
    # sets rewards 1.0 and 2.0 on rounds 2 and 3, applies discount 0.8 and asserts
    # rewards 2.0, 1.0+0.8*2.0 and 0.8 times round 2's value for round 1.
    """Test multi-round conversation with tool calling across rounds."""

    # Round 1 - Initial tool call request
    c1 = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful calculator assistant."},
            {"role": "user", "content": "Calculate 15 * 7"},
        ],
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=8192,
    )

    # Simulate tool call response
    tool_response = "105"

    # Round 2 - Continue with tool result and new request
    c2 = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful calculator assistant."},
            {"role": "user", "content": "Calculate 15 * 7"},
            c1.choices[0].message,
            {"role": "tool", "content": tool_response, "tool_call_id": "mock_call_id"},
            {
                "role": "user",
                "content": "Now get an interesting fact about this number",
            },
        ],
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=8192,
    )

    # Round 3 - Final response with fact
    c3 = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful calculator assistant."},
            {"role": "user", "content": "Calculate 15 * 7"},
            c1.choices[0].message,
            {"role": "tool", "content": tool_response, "tool_call_id": "mock_call_id"},
            {
                "role": "user",
                "content": "Now get an interesting fact about this number",
            },
            c2.choices[0].message,
            {
                "role": "tool",
                "content": "105 is divisible by 3, 5, 7, 15, 21, and 35!",
                "tool_call_id": "mock_call_id_2",
            },
            {"role": "user", "content": "Thank you!"},
        ],
        max_completion_tokens=8192,
    )

    # Set rewards
    openai_client.set_reward(c2.id, reward=1.0)
    openai_client.set_reward(c3.id, reward=2.0)
    openai_client.apply_reward_discount(turn_discount=0.8)

    completions = openai_client.export_interactions(style="individual")

    assert len(completions) == 3
    # c3 is leaf: gets explicit reward
    assert completions[c3.id].reward == 2.0
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 1.0 + 0.8 * 2.0
    # c1 gets discounted reward from c2
    assert completions[c1.id].reward == 0.8 * (1.0 + 0.8 * 2.0)


@pytest.mark.asyncio
async def test_parallel_tool_calling(openai_client):
    # [FUNCTION DOC]
    # Asks for three tool actions in one request, sets reward 3.0, then follows up with
    # three mocked tool results and sets reward 2.0; after discount 0.9 asserts the
    # follow-up keeps 2.0 and the first gets 3.0+0.9*2.0.
    """Test parallel tool calling within a single round."""

    # Single request that could trigger multiple tool calls
    c1 = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can check weather, time, and translate text.",
            },
            {
                "role": "user",
                "content": "I need you to check the weather in Tokyo, get the current time in Japan, and translate 'hello world' to Japanese. Please do all of these.",
            },
        ],
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=8192,
    )

    # Check the response structure
    assert c1.id is not None
    assert c1.choices[0].message.role == "assistant"

    # Even if parallel tool calling isn't supported by the model,
    # we can test the caching and reward system
    openai_client.set_reward(c1.id, reward=3.0)

    # Test with tool responses in follow-up
    c2 = await openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can check weather, time, and translate text.",
            },
            {
                "role": "user",
                "content": "I need you to check the weather in Tokyo, get the current time in Japan, and translate 'hello world' to Japanese. Please do all of these.",
            },
            c1.choices[0].message,
            {"role": "tool", "content": "Sunny, 25°C", "tool_call_id": "weather_call"},
            {"role": "tool", "content": "14:30 JST", "tool_call_id": "time_call"},
            {
                "role": "tool",
                "content": "こんにちは世界",
                "tool_call_id": "translate_call",
            },
            {"role": "user", "content": "Perfect, thank you!"},
        ],
        max_completion_tokens=8192,
    )

    openai_client.set_reward(c2.id, reward=2.0)
    openai_client.apply_reward_discount(turn_discount=0.9)
    completions = openai_client.export_interactions(style="individual")

    assert len(completions) == 2
    # c2 is leaf
    assert completions[c2.id].reward == 2.0
    # c1 gets explicit reward + discounted reward from c2
    assert completions[c1.id].reward == 3.0 + 0.9 * 2.0


def strip_thinking_tags(content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Removes think-tag blocks (non-greedy, multi-line), collapses whitespace
    # and strips; returns the cleaned string.
    """Remove thinking tags and their content from a message."""
    import re

    # Remove <think>...</think> blocks (including multi-line)
    pattern = r"<think>.*?</think>"
    cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@pytest.mark.asyncio
async def test_multi_round_conversation_with_thinking_and_tool_calling(openai_client):
    # [FUNCTION DOC]
    # Runs three rounds with tools, stripping think blocks from assistant dicts before
    # reuse and appending mocked tool results. Asserts stored assistant messages for
    # rounds 2 and 3 contain no think tags, raises RuntimeError if no tool message is in
    # round 3's stored history, then checks discounted rewards (0.9) of 1.5, 2.0+0.9*1.5
    # and 1.0+0.9*(that).
    # KNOWN ISSUE: an inline comment on the leaf reward assertion says "1.5 + 0.8" which
    # does not match the asserted value.
    """Test multi-round conversation with both thinking and tool calling, ensuring thinking is stripped but tool calls are preserved."""

    # Round 1 - Model thinks before making a tool call
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with calculation abilities. Use <think></think> tags for your internal thoughts.",
        },
        {
            "role": "user",
            "content": "I need to calculate the area of a rectangle that is 25 meters long and 18 meters wide. Please think step-by-step.",
        },
    ]
    c1 = await openai_client.chat.completions.create(
        messages=messages, tools=tools, tool_choice="auto", max_completion_tokens=8192
    )

    # Round 2 - Provide tool result and ask follow-up, stripping thinking from previous response
    assistant_message_1 = c1.choices[0].message.model_dump(exclude_none=True)
    cleaned_content_1 = strip_thinking_tags(assistant_message_1.get("content", ""))
    assistant_message_1["content"] = cleaned_content_1
    messages += [
        assistant_message_1,
        {"role": "tool", "content": "450", "tool_call_id": "calc_call_1"},
        {
            "role": "user",
            "content": "Perfect! Now what if I want to carpet this room and carpet costs $15 per square meter? Please think step-by-step.",
        },
    ]
    c2 = await openai_client.chat.completions.create(
        messages=messages, tools=tools, tool_choice="auto", max_completion_tokens=8192
    )

    # Round 3 - Continue with tool result
    assistant_message_2 = c2.choices[0].message.model_dump(exclude_none=True)
    cleaned_content_2 = strip_thinking_tags(assistant_message_2.get("content", ""))
    assistant_message_2["content"] = cleaned_content_2
    messages += [
        assistant_message_2,
        {"role": "tool", "content": "6750", "tool_call_id": "calc_call_2"},
        {
            "role": "user",
            "content": "That's quite expensive! What would be the cost per square foot instead?  Please think step-by-step.",
        },
    ]
    c3 = await openai_client.chat.completions.create(
        messages=messages, max_completion_tokens=8192
    )

    # Verify conversation history
    stored_messages_c2 = openai_client.get_interaction(c2.id).messages
    stored_messages_c3 = openai_client.get_interaction(c3.id).messages

    # Verify thinking tags are stripped from assistant messages
    for msg_list in [stored_messages_c2, stored_messages_c3]:
        for msg in msg_list:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                assert "<think>" not in content
                assert "</think>" not in content

    # Verify tool calls are preserved (check that tool messages exist in history)
    tool_messages_found = False
    for msg in stored_messages_c3:
        if msg.get("role") == "tool":
            tool_messages_found = True
            break

    if not tool_messages_found:
        raise RuntimeError("Tool messages should be preserved in conversation history")

    # Test reward system with thinking + tool calling
    openai_client.set_reward(c1.id, reward=1.0)
    openai_client.set_reward(c2.id, reward=2.0)
    openai_client.set_reward(c3.id, reward=1.5)

    openai_client.apply_reward_discount(turn_discount=0.9)
    completions = openai_client.export_interactions(style="individual")

    assert len(completions) == 3
    # c3 is leaf
    assert completions[c3.id].reward == 1.5  # 1.5 + 0.8
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 2.0 + 0.9 * 1.5
    # c1 gets explicit reward + discounted reward from c2
    assert completions[c1.id].reward == 1.0 + 0.9 * (2.0 + 0.9 * 1.5)


@pytest.mark.asyncio
async def test_multi_step_tool_calling_concat_style(openai_client):
    # [FUNCTION DOC]
    # Switches to "concat" mode and builds two independent branches (different user
    # questions) each with a tool-calling turn followed by a mocked tool result turn.
    # Asserts both first turns produced tool calls, concat export returns the two
    # leaves, individual returns all four, leaves' parents are their first turns which
    # have no parent, rewards are not propagated, and each leaf loss mask is 1 over both
    # outputs and 0 elsewhere.
    # KNOWN ISSUE: docstring mentions stripping think tags, but message content is
    # reused as-is without stripping.
    """Test multi-round conversation with tool calling and concat-style export.

    This test ensures that when using `export_interactions(style="concat")`,
    the conversation history correctly handles tool calls while stripping
    <think> tags, and the resulting loss masks are correct for the
    concatenated trajectories.
    """
    openai_client: ArealOpenAI
    openai_client.chat_template_type = "concat"
    openai_client.chat.completions.chat_template_type = "concat"

    # Base conversation
    base = [
        {
            "role": "system",
            "content": "You are a helpful assistant with calculation abilities.",
        },
    ]

    # Branch A: root -> a -> a1 (with tool call)
    msgs_a = base + [
        {"role": "user", "content": "Calculate 25 * 4"},
    ]
    c_a = await openai_client.chat.completions.create(
        messages=msgs_a, tools=tools, tool_choice="auto", max_completion_tokens=8192
    )
    cleaned_a_content = c_a.choices[0].message.content
    assert c_a.choices[0].message.tool_calls is not None
    assert c_a.choices[0].finish_reason == "tool_calls"

    # Leaf A1
    msgs_a1 = msgs_a + [
        {
            "role": "assistant",
            "content": cleaned_a_content,
            "tool_calls": c_a.choices[0].message.tool_calls,
        },
        {"role": "tool", "content": "100", "tool_call_id": "mock_id_a1"},
    ]
    c_a1 = await openai_client.chat.completions.create(
        messages=msgs_a1, tools=tools, tool_choice="auto", max_completion_tokens=8192
    )

    # Branch B: root -> b -> b1 (with tool call)
    msgs_b = base + [
        {"role": "user", "content": "What is 120 / 3?"},
    ]
    c_b = await openai_client.chat.completions.create(
        messages=msgs_b, tools=tools, tool_choice="auto", max_completion_tokens=8192
    )
    cleaned_b_content = c_b.choices[0].message.content
    assert c_b.choices[0].message.tool_calls is not None
    assert c_b.choices[0].finish_reason == "tool_calls"

    # Leaf B1
    msgs_b1 = msgs_b + [
        {
            "role": "assistant",
            "content": cleaned_b_content,
            "tool_calls": c_b.choices[0].message.tool_calls,
        },
        {"role": "tool", "content": "40", "tool_call_id": "mock_id_b1"},
    ]
    c_b1 = await openai_client.chat.completions.create(
        messages=msgs_b1, tools=tools, tool_choice="auto", max_completion_tokens=8192
    )

    # Set rewards for leaf nodes
    openai_client.set_reward(c_a1.id, 2.0)
    openai_client.set_reward(c_b1.id, 3.0)

    # Export completions
    leaf_completions = openai_client.export_interactions(style="concat")
    all_completions = openai_client.export_interactions(style="individual")

    # Verify exports
    assert set(leaf_completions.keys()) == {c_a1.id, c_b1.id}
    assert set(all_completions.keys()) == {
        c_a.id,
        c_a1.id,
        c_b.id,
        c_b1.id,
    }

    def wrapped_completion(chat_completion):
        # [FUNCTION DOC]
        # HELPER: Looks up the exported interaction for a chat completion by id.
        return all_completions[chat_completion.id]

    # Verify tree structure
    assert wrapped_completion(c_a1).parent is wrapped_completion(c_a)
    assert wrapped_completion(c_a).parent is None
    assert wrapped_completion(c_b1).parent is wrapped_completion(c_b)
    assert wrapped_completion(c_b).parent is None

    # Verify rewards (no propagation in concat style)
    assert wrapped_completion(c_a1).reward == 2.0
    assert wrapped_completion(c_b1).reward == 3.0
    assert wrapped_completion(c_a).reward is None
    assert wrapped_completion(c_b).reward is None

    # Loss mask for c_a1
    c_a_input_len = wrapped_completion(c_a).model_response.input_len
    c_a_output_len = wrapped_completion(c_a).model_response.output_len
    c_a1_input_len = wrapped_completion(c_a1).model_response.input_len
    c_a1_output_len = wrapped_completion(c_a1).model_response.output_len
    c_a1_loss_mask = wrapped_completion(c_a1).to_tensor_dict()["loss_mask"].squeeze(0)

    assert c_a1_loss_mask.tolist() == (
        [0] * (c_a_input_len)
        + [1] * c_a_output_len
        + [0] * (c_a1_input_len - (c_a_input_len + c_a_output_len))
        + [1] * c_a1_output_len
    )

    # Loss mask for c_b1
    c_b_input_len = wrapped_completion(c_b).model_response.input_len
    c_b_output_len = wrapped_completion(c_b).model_response.output_len
    c_b1_input_len = wrapped_completion(c_b1).model_response.input_len
    c_b1_output_len = wrapped_completion(c_b1).model_response.output_len
    c_b1_loss_mask = wrapped_completion(c_b1).to_tensor_dict()["loss_mask"].squeeze(0)

    assert c_b1_loss_mask.tolist() == (
        [0] * c_b_input_len
        + [1] * c_b_output_len
        + [0] * (c_b1_input_len - (c_b_input_len + c_b_output_len))
        + [1] * c_b1_output_len
    )

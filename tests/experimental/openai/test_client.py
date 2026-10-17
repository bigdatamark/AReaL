# [MODULE DOC]
# Integration tests for ArealOpenAI, the OpenAI-compatible client that routes chat
# completions to a RemoteSGLangEngine and records each interaction for RL training. A
# module-scoped fixture launches a real SGLang server (Qwen3-0.6B) on a free port; tests
# cover single/multi-turn rollouts, streaming, reward setting and turn discounting,
# "concat" tree export with parent links and loss masks, input type validation, and
# prompt length overflow. Marked sglang; requires a GPU.
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
    # HELPER: GETs <base_url>/health with a 30s timeout and returns True iff status is
    # 200; any requests exception returns False.
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def sglang_server():
    # [FUNCTION DOC]
    # Module-scoped fixture: seeds RNGs, builds the SGLang launch command (skip
    # tokenizer init, mem fraction 0.3, tp 1, GPU 0, dist addr on a free port) and
    # spawns it as a subprocess with output to stdout. Polls health every second up to
    # RUN_SERVER_TIMEOUT (180s), raising RuntimeError if not healthy, yields, then kills
    # the process tree gracefully.
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
    # Fixture creating a RemoteSGLangEngine (backend sglang:d1, 2 concurrent rollouts)
    # pointed at the launched server by setting AREAL_LLM_SERVER_ADDRS in os.environ,
    # initializing it, and yielding an ArealOpenAI client with qwen25 tool parser, qwen3
    # reasoning parser, "hf" chat template type and engine_max_tokens 16384. Destroys
    # the engine afterwards.
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
        engine_max_tokens=16384,
    )
    engine.destroy()


@pytest.mark.asyncio
async def test_single_turn_rollout(openai_client):
    # [FUNCTION DOC]
    # Creates one completion, sets reward 0.5, and asserts individual export contains
    # just that interaction with the reward.
    c = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        max_completion_tokens=2048,
    )
    openai_client.set_reward(c.id, reward=0.5)
    completions = openai_client.export_interactions(style="individual")
    assert len(completions) == 1
    assert completions[c.id].reward == 0.5


@pytest.mark.asyncio
async def test_streaming_conversation(openai_client):
    # [FUNCTION DOC]
    # Streams a completion and asserts: at least one chunk, a single shared chunk id,
    # each chunk is a chat.completion.chunk with one choice, some chunks have content,
    # exactly one chunk has finish_reason in stop/length, and that final chunk carries
    # usage with positive prompt/completion tokens summing to total.
    """Test streaming mode for a conversation."""
    chunks = []
    stream = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ],
        max_completion_tokens=256,
        stream=True,
    )

    async for chunk in stream:
        chunks.append(chunk)

    # Verify we got multiple chunks
    assert len(chunks) > 0, "Should receive at least one chunk"

    # Verify all chunks have consistent id
    chunk_ids = {chunk.id for chunk in chunks}
    assert len(chunk_ids) == 1, "All chunks should have the same id"

    # Verify chunk structure
    for chunk in chunks:
        assert chunk.id is not None
        assert chunk.object == "chat.completion.chunk"
        assert len(chunk.choices) == 1

    # Find chunks with content
    content_chunks = [
        chunk
        for chunk in chunks
        if chunk.choices[0].delta.content is not None
        and len(chunk.choices[0].delta.content) > 0
    ]
    assert len(content_chunks) > 0, "Should have at least one chunk with content"

    # Find the final chunk with finish_reason
    final_chunks = [
        chunk for chunk in chunks if chunk.choices[0].finish_reason is not None
    ]
    assert len(final_chunks) == 1, "Should have exactly one chunk with finish_reason"
    assert final_chunks[0].choices[0].finish_reason in ["stop", "length"]

    # Verify usage is included in final chunk
    assert final_chunks[0].usage is not None
    assert final_chunks[0].usage.prompt_tokens > 0
    assert final_chunks[0].usage.completion_tokens > 0
    assert final_chunks[0].usage.total_tokens == (
        final_chunks[0].usage.prompt_tokens + final_chunks[0].usage.completion_tokens
    )


@pytest.mark.asyncio
async def test_multi_round_conversation(openai_client):
    # [FUNCTION DOC]
    # Runs three chained turns (feeding back assistant message objects), sets reward 2.0
    # on the last, applies turn discount 0.9, and asserts individually exported rewards
    # are 2.0, 1.8, 1.62 for turns 3, 2, 1.
    """Test multi-round conversation with reward backpropagation."""
    # Round 1
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    c1 = await openai_client.chat.completions.create(
        messages=messages, max_completion_tokens=2048
    )

    # Round 2 - extends the conversation
    assistant_message_1 = c1.choices[0].message
    messages += [
        assistant_message_1,
        {"role": "user", "content": "What about Germany?"},
    ]
    c2 = await openai_client.chat.completions.create(
        messages=messages, max_completion_tokens=2048
    )

    # Round 3 - further extends the conversation
    assistant_message_2 = c2.choices[0].message
    messages += [
        assistant_message_2,
        {"role": "user", "content": "And Italy?"},
    ]
    c3 = await openai_client.chat.completions.create(
        messages=messages, max_completion_tokens=2048
    )

    # Set rewards - only the final completion gets explicit reward
    openai_client.set_reward(c3.id, reward=2.0)
    openai_client.apply_reward_discount(turn_discount=0.9)

    # Export completions with reward backpropagation
    completions = openai_client.export_interactions(style="individual")

    # Verify structure
    assert len(completions) == 3

    # Verify reward backpropagation: c3 is leaf,
    # c2 gets discounted reward from c3, c1 gets discounted reward from c2
    assert completions[c3.id].reward == 2.0
    assert completions[c2.id].reward == 0.9 * 2.0  # discounted from c3
    assert completions[c1.id].reward == 0.9 * (0.9 * 2.0)  # discounted from c2


def strip_thinking_tags(content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Removes all think-tag blocks (non-greedy, across newlines) from content,
    # collapses runs of whitespace to single spaces and strips. Returns the cleaned
    # string.
    """Remove thinking tags and their content from a message."""
    import re

    # Remove <think>...</think> blocks (including multi-line)
    pattern = r"<think>.*?</think>"
    cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@pytest.mark.asyncio
async def test_multi_round_conversation_with_thinking(openai_client):
    # [FUNCTION DOC]
    # Runs three turns, stripping think blocks from each assistant message (as a dict)
    # before feeding it back. Asserts stored messages for turns 2 and 3 contain no think
    # tags, then sets rewards 1.5 (turn 2) and 2.5 (turn 3), applies discount 0.85 and
    # asserts turn 3 is 2.5, turn 2 is 1.5 + 0.85*2.5 and turn 1 is 0.85 times turn 2's
    # value.
    """Test multi-round conversation where thinking content is excluded from subsequent rounds."""

    # Round 1 - Model generates response with thinking
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use <think></think> tags for your internal thoughts.",
        },
        {"role": "user", "content": "What is 15 * 24? Please think step-by-step."},
    ]
    c1 = await openai_client.chat.completions.create(messages=messages, max_tokens=2048)

    # Round 2 - Strip thinking from previous response
    assistant_message_1 = c1.choices[0].message.model_dump(exclude_none=True)
    cleaned_assistant_content = strip_thinking_tags(
        assistant_message_1.get("content", "")
    )
    assistant_message_1["content"] = cleaned_assistant_content
    messages += [
        assistant_message_1,
        {
            "role": "user",
            "content": "Now what is 360 divided by 12? Please think step-by-step.",
        },
    ]
    c2 = await openai_client.chat.completions.create(messages=messages, max_tokens=2048)

    # Round 3 - Continue conversation, stripping thinking from previous response
    assistant_message_2 = c2.choices[0].message.model_dump(exclude_none=True)
    cleaned_assistant_content_2 = strip_thinking_tags(
        assistant_message_2.get("content", "")
    )
    assistant_message_2["content"] = cleaned_assistant_content_2
    messages += [
        assistant_message_2,
        {
            "role": "user",
            "content": "Great! Can you explain why division by 12 gave us 30?  Please think step-by-step.",
        },
    ]
    c3 = await openai_client.chat.completions.create(messages=messages, max_tokens=2048)

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

    # Test reward system
    openai_client.set_reward(c2.id, reward=1.5)
    openai_client.set_reward(c3.id, reward=2.5)

    openai_client.apply_reward_discount(turn_discount=0.85)
    completions = openai_client.export_interactions(style="individual")

    assert len(completions) == 3
    # c3 is leaf
    assert completions[c3.id].reward == 2.5
    # c2 gets explicit reward + discounted reward from c3
    assert completions[c2.id].reward == 1.5 + 0.85 * 2.5
    # c1 gets discounted reward from c2
    assert completions[c1.id].reward == 0.85 * (1.5 + 0.85 * 2.5)


@pytest.mark.asyncio
async def test_multi_round_conversation_concat_style_export(openai_client):
    # [FUNCTION DOC]
    # Switches the client to "concat" template type, builds a tree root -> a -> {a1, a2}
    # and root -> b -> b1, sets rewards on leaves, and asserts concat export returns
    # only the three leaves, individual export returns all six, parent links match the
    # tree, rewards are not propagated, and each leaf's loss_mask is 1 exactly over
    # every ancestor's (and its own) output tokens and 0 over the interleaved prompt
    # tokens.
    """Create a conversation tree using create() and verify parents and rewards.

    Rewards are explicitly set (no propagation). Export should return only leaves.
    """
    openai_client: ArealOpenAI
    openai_client.chat_template_type = "concat"
    openai_client.chat.completions.chat_template_type = "concat"
    # Base conversation
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Start the session."},
    ]

    # Root
    c_root = await openai_client.chat.completions.create(
        messages=base, max_completion_tokens=2048
    )

    # Branch A1: root -> a -> a1
    msgs_a = base + [
        c_root.choices[0].message,
        {"role": "user", "content": "Question A"},
    ]
    c_a = await openai_client.chat.completions.create(
        messages=msgs_a, max_completion_tokens=2048
    )

    msgs_a1 = msgs_a + [
        c_a.choices[0].message,
        {"role": "user", "content": "Follow-up A1"},
    ]
    c_a1 = await openai_client.chat.completions.create(
        messages=msgs_a1, max_completion_tokens=2048
    )

    # Branch A2: root -> a -> a2
    msgs_a2 = msgs_a + [
        c_a.choices[0].message,
        {"role": "user", "content": "Follow-up A2"},
    ]
    c_a2 = await openai_client.chat.completions.create(
        messages=msgs_a2, max_completion_tokens=2048
    )

    # Branch B: root -> b -> b1
    msgs_b = base + [
        c_root.choices[0].message,
        {"role": "user", "content": "Question B"},
    ]
    c_b = await openai_client.chat.completions.create(
        messages=msgs_b,
    )
    msgs_b1 = msgs_b + [
        c_b.choices[0].message,
        {"role": "user", "content": "Follow-up B1"},
    ]
    c_b1 = await openai_client.chat.completions.create(
        messages=msgs_b1, max_completion_tokens=2048
    )

    # Set rewards to leaf nodes only, which should be c_a1, c_a2, c_b1
    openai_client.set_reward(c_a1.id, 2)
    openai_client.set_reward(c_a2.id, 1.5)
    openai_client.set_reward(c_b1.id, 3)

    # Export completions of leaf nodes, check whether all leaves are present
    leaf_completions = openai_client.export_interactions(style="concat")
    all_completions = openai_client.export_interactions(style="individual")
    assert set(leaf_completions.keys()) == {c_a1.id, c_a2.id, c_b1.id}
    assert set(all_completions.keys()) == {
        c_root.id,
        c_a.id,
        c_a1.id,
        c_a2.id,
        c_b.id,
        c_b1.id,
    }

    def wrapped_completion(chat_completion):
        # [FUNCTION DOC]
        # HELPER: Looks up the exported interaction for a chat completion by id.
        return all_completions[chat_completion.id]

    # Check tree structure
    assert wrapped_completion(c_b1).parent is wrapped_completion(c_b)
    assert wrapped_completion(c_b).parent is wrapped_completion(c_root)
    assert wrapped_completion(c_a2).parent is wrapped_completion(c_a)
    assert wrapped_completion(c_a1).parent is wrapped_completion(c_a)
    assert wrapped_completion(c_a).parent is wrapped_completion(c_root)

    print(f"c_root: {c_root}")

    # Reward is not propagated to tree nodes, check reward values
    assert wrapped_completion(c_b1).reward == 3
    assert wrapped_completion(c_a2).reward == 1.5
    assert wrapped_completion(c_a1).reward == 2

    print(f"c_root: {c_root}")

    # Check loss masks produced by completions
    # Ensure number of 1s in the loss masks is actually the number of tokens output by the model
    c_a1_loss_mask = wrapped_completion(c_a1).to_tensor_dict()["loss_mask"].squeeze(0)
    c_root_input_len = wrapped_completion(c_root).model_response.input_len
    c_root_output_len = wrapped_completion(c_root).model_response.output_len
    c_a_input_len = wrapped_completion(c_a).model_response.input_len
    c_a_output_len = wrapped_completion(c_a).model_response.output_len
    c_a1_input_len = wrapped_completion(c_a1).model_response.input_len
    c_a1_output_len = wrapped_completion(c_a1).model_response.output_len

    # c_a1 loss mask
    assert c_a1_loss_mask.squeeze(0).tolist() == (
        [0] * c_root_input_len
        + [1] * c_root_output_len
        + [0] * (c_a_input_len - (c_root_input_len + c_root_output_len))
        + [1] * c_a_output_len
        + [0] * (c_a1_input_len - (c_a_input_len + c_a_output_len))
        + [1] * c_a1_output_len
    )

    # c_a2 loss mask
    c_a2_loss_mask = wrapped_completion(c_a2).to_tensor_dict()["loss_mask"].squeeze(0)
    c_a2_input_len = wrapped_completion(c_a2).model_response.input_len
    c_a2_output_len = wrapped_completion(c_a2).model_response.output_len
    assert c_a2_loss_mask.squeeze(0).tolist() == (
        [0] * c_root_input_len
        + [1] * c_root_output_len
        + [0] * (c_a_input_len - (c_root_input_len + c_root_output_len))
        + [1] * c_a_output_len
        + [0] * (c_a2_input_len - (c_a_input_len + c_a_output_len))
        + [1] * c_a2_output_len
    )

    # c_b1 loss mask
    c_b1_loss_mask = wrapped_completion(c_b1).to_tensor_dict()["loss_mask"].squeeze(0)
    c_b_input_len = wrapped_completion(c_b).model_response.input_len
    c_b_output_len = wrapped_completion(c_b).model_response.output_len
    c_b1_input_len = wrapped_completion(c_b1).model_response.input_len
    c_b1_output_len = wrapped_completion(c_b1).model_response.output_len
    assert c_b1_loss_mask.squeeze(0).tolist() == (
        [0] * c_root_input_len
        + [1] * c_root_output_len
        + [0] * (c_b_input_len - (c_root_input_len + c_root_output_len))
        + [1] * c_b_output_len
        + [0] * (c_b1_input_len - (c_b_input_len + c_b_output_len))
        + [1] * c_b1_output_len
    )


@pytest.mark.asyncio
async def test_type_checking(openai_client):
    # [FUNCTION DOC]
    # In concat mode, asserts message history may contain ChatCompletionMessage objects
    # or dicts, but raises TypeError when it contains a bare string or when messages is
    # a tuple instead of a list.
    openai_client: ArealOpenAI
    openai_client.chat_template_type = "concat"
    openai_client.chat.completions.chat_template_type = "concat"
    # Base conversation
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Start the session."},
    ]
    # Root
    c_root = await openai_client.chat.completions.create(
        messages=base,
    )
    # Directly use message, should not raise error
    msgs_a = base + [
        c_root.choices[0].message,
        {"role": "user", "content": "Question A"},
    ]
    await openai_client.chat.completions.create(
        messages=msgs_a,
    )
    # Use a dict, should not raise error
    msgs_a = base + [
        c_root.choices[0].message.model_dump(exclude_none=True),
        {"role": "user", "content": "Question A"},
    ]
    await openai_client.chat.completions.create(
        messages=msgs_a,
    )
    with pytest.raises(TypeError):
        # Use other stuff without a model_dump(exclude_none=True), should raise error
        msgs_a = base + [
            c_root.choices[0].message.content,
            {"role": "user", "content": "Question A"},
        ]
        await openai_client.chat.completions.create(
            messages=msgs_a,
        )

    with pytest.raises(TypeError):
        # `messages` is not a list, should raise error
        msgs_a = base + [
            c_root.choices[0].message.content,
            {"role": "user", "content": "Question A"},
        ]
        await openai_client.chat.completions.create(
            messages=tuple(msgs_a),
        )


@pytest.mark.asyncio
async def test_prompt_len_exceed(openai_client):
    # [FUNCTION DOC]
    # In concat mode, asserts a follow-up whose prompt (~16k repeats) exceeds the
    # engine's max tokens raises ValueError and that the failed request is not cached:
    # both export styles contain only the root.
    openai_client: ArealOpenAI
    openai_client.chat_template_type = "concat"
    openai_client.chat.completions.chat_template_type = "concat"
    # Base conversation
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Start the session."},
    ]
    # Root
    c_root = await openai_client.chat.completions.create(
        messages=base,
    )
    # Directly use message, should not raise error
    msgs_a = base + [
        c_root.choices[0].message,
        {"role": "user", "content": "Question A " * 16384},
    ]
    with pytest.raises(ValueError):
        await openai_client.chat.completions.create(
            messages=msgs_a,
        )

    # msgs_a should not be in the cache due to failure
    leaf_completions = openai_client.export_interactions(style="concat")
    all_completions = openai_client.export_interactions(style="individual")
    assert set(leaf_completions.keys()) == {c_root.id}
    assert set(all_completions.keys()) == {c_root.id}


@pytest.mark.asyncio
async def test_multi_round_conversation_end_with_length(openai_client):
    # [FUNCTION DOC]
    # Same as the concat tree test but a single chain root -> a -> a1 with
    # max_completion_tokens 8 (generations end by length). Asserts leaf/all export sets,
    # parent links, unpropagated reward, and that a1's loss mask covers all three
    # outputs.
    # KNOWN ISSUE: docstring is copied from the tree test and mentions rewards/leaves
    # generally rather than the length-truncation scenario actually tested.
    """Create a conversation tree using create() and verify parents and rewards.

    Rewards are explicitly set (no propagation). Export should return only leaves.
    """
    openai_client: ArealOpenAI
    openai_client.chat_template_type = "concat"
    openai_client.chat.completions.chat_template_type = "concat"
    # Base conversation
    base = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Start the session."},
    ]

    # Root
    c_root = await openai_client.chat.completions.create(
        messages=base, max_completion_tokens=8
    )

    # Branch A1: root -> a -> a1
    msgs_a = base + [
        c_root.choices[0].message,
        {"role": "user", "content": "Question A"},
    ]
    c_a = await openai_client.chat.completions.create(
        messages=msgs_a, max_completion_tokens=8
    )
    msgs_a1 = msgs_a + [
        c_a.choices[0].message,
        {"role": "user", "content": "Follow-up A1"},
    ]
    c_a1 = await openai_client.chat.completions.create(
        messages=msgs_a1, max_completion_tokens=8
    )

    # Set rewards to leaf nodes only, which should be c_a1
    openai_client.set_reward(c_a1.id, 2)

    # Export completions of leaf nodes, check whether all leaves are present
    leaf_completions = openai_client.export_interactions(style="concat")
    all_completions = openai_client.export_interactions(style="individual")
    assert set(leaf_completions.keys()) == {c_a1.id}
    assert set(all_completions.keys()) == {
        c_root.id,
        c_a.id,
        c_a1.id,
    }

    def wrapped_completion(chat_completion):
        # [FUNCTION DOC]
        # HELPER: Looks up the exported interaction for a chat completion by id.
        return all_completions[chat_completion.id]

    # Check tree structure
    assert wrapped_completion(c_a1).parent is wrapped_completion(c_a)
    assert wrapped_completion(c_a).parent is wrapped_completion(c_root)

    # Reward is not propagated to tree nodes, check reward values
    assert wrapped_completion(c_a1).reward == 2

    # Check loss masks produced by completions
    # Ensure number of 1s in the loss masks is actually the number of tokens output by the model
    c_a1_loss_mask = wrapped_completion(c_a1).to_tensor_dict()["loss_mask"].squeeze(0)
    c_root_input_len = wrapped_completion(c_root).model_response.input_len
    c_root_output_len = wrapped_completion(c_root).model_response.output_len
    c_a_input_len = wrapped_completion(c_a).model_response.input_len
    c_a_output_len = wrapped_completion(c_a).model_response.output_len
    c_a1_input_len = wrapped_completion(c_a1).model_response.input_len
    c_a1_output_len = wrapped_completion(c_a1).model_response.output_len

    # c_a1 loss mask
    assert c_a1_loss_mask.squeeze(0).tolist() == (
        [0] * c_root_input_len
        + [1] * c_root_output_len
        + [0] * (c_a_input_len - (c_root_input_len + c_root_output_len))
        + [1] * c_a_output_len
        + [0] * (c_a1_input_len - (c_a_input_len + c_a_output_len))
        + [1] * c_a1_output_len
    )

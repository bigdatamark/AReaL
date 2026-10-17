# [MODULE DOC]
# Example single-turn math agent using the Anthropic Messages API (AsyncAnthropic)
# through AReaL's proxy; reward computed with math_verify.
# SPDX-License-Identifier: Apache-2.0

import os

import anthropic
from math_verify import parse, verify

from areal.api import AsyncRewardWrapper


def math_reward_fn(completions: str, answer: str) -> float:
    # [FUNCTION DOC]
    # HELPER: Parses completion text and gold answer with math_verify and returns 1.0 if
    # they verify as equivalent, else 0.0.
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


class MathAgent:
    # [CLASS DOC]
    # Single-turn agent; constructor kwargs (minus max_completion_tokens and max_turns)
    # are forwarded to messages.create.
    """Simple single-turn math agent using Anthropic Messages API."""

    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # Copies kwargs and drops max_completion_tokens and max_turns.
        # KNOWN ISSUE: messages.create requires max_tokens; if callers only pass
        # max_completion_tokens it is dropped and the request may fail.
        # Store kwargs for client.messages.create call
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_completion_tokens", None)
        self.kwargs.pop("max_turns", None)

    async def run(self, data: dict, **extra_kwargs) -> float:
        # [FUNCTION DOC]
        # Builds an AsyncAnthropic client (no retries) from extra_kwargs
        # http_client/base_url/api_key or ANTHROPIC_* env; converts messages by
        # extracting system role messages into the system prompt (last wins) and passing
        # others through; calls messages.create with model 'default'; joins text blocks
        # and returns the async-wrapped math reward vs data['answer'].
        """Run the agent on a single problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url, api_key, and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("ANTHROPIC_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("ANTHROPIC_API_KEY")

        client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )

        # Convert OpenAI-style messages to Anthropic format
        messages = data["messages"]
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Call the Anthropic Messages API (via proxy)
        response = await client.messages.create(
            model="default",
            messages=anthropic_messages,
            system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
            **self.kwargs,
        )

        # Extract response text
        completion_text = "".join(
            block.text
            for block in response.content
            if isinstance(block, anthropic.types.TextBlock)
        )

        # Calculate reward
        reward_fn = AsyncRewardWrapper(math_reward_fn)
        return await reward_fn(completion_text, data["answer"])

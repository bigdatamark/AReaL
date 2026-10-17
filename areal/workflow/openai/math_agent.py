# [MODULE DOC]
# Example OpenAI-SDK math agents for AReaL proxy-based training: single-turn chat
# completion, multi-turn retry agent, and an OpenAI Agents SDK tool agent with
# calculator function tools; all scored with math_verify.
# SPDX-License-Identifier: Apache-2.0

import os

from agents import (
    Agent,
    ModelSettings,
    OpenAIProvider,
    RunConfig,
    SQLiteSession,
    function_tool,
)
from agents import Runner as OpenAIRunner
from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import AsyncRewardWrapper


def math_reward_fn(completions: str, answer: str) -> float:
    # [FUNCTION DOC]
    # HELPER: Returns 1.0 if math_verify judges the completion equivalent to the gold
    # answer, else 0.0.
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


class MathAgent:
    # [CLASS DOC]
    # Single-turn chat-completions agent.
    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # Normalizes kwargs for chat.completions: removes max_new_tokens (and max_turns
        # where applicable) and uses it as max_tokens when max_tokens is not given;
        # otherwise drops max_tokens. Creates an async-wrapped math reward.
        # KNOWN ISSUE: an explicitly passed max_tokens is always discarded (the else
        # branch pops it).
        self.kwargs = kwargs.copy()
        max_new_tokens = self.kwargs.pop("max_new_tokens", None)
        self.kwargs.pop("max_turns", None)
        if max_new_tokens is not None and "max_tokens" not in self.kwargs:
            self.kwargs["max_tokens"] = max_new_tokens
        else:
            self.kwargs.pop("max_tokens", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        # [FUNCTION DOC]
        # Creates AsyncOpenAI (no retries) from extra_kwargs
        # http_client/base_url/api_key or OPENAI_* env; uses data['messages_chat'] or
        # data['messages'] (string wrapped as a user message); calls chat.completions
        # with model 'default' and stored kwargs; returns the reward of the first choice
        # content vs data['answer'].
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        messages = data.get("messages_chat") or data["messages"]
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        comp: ChatCompletion = await client.chat.completions.create(
            messages=messages, model="default", **self.kwargs
        )

        return await self._reward_fn(
            completions=comp.choices[0].message.content, answer=data["answer"]
        )


class MultiTurnMathAgent:
    # [CLASS DOC]
    # Multi-turn agent that retries with a corrective user message up to max_turns until
    # reward == 1.
    def __init__(self, max_turns: int = 8, **kwargs):
        # [FUNCTION DOC]
        # Stores max_turns (default 8) and normalizes kwargs like MathAgent (max_turns
        # is a named arg here).
        # KNOWN ISSUE: an explicitly passed max_tokens is always discarded.
        self.max_turns = max_turns
        self.kwargs = kwargs.copy()
        max_new_tokens = self.kwargs.pop("max_new_tokens", None)
        if max_new_tokens is not None and "max_tokens" not in self.kwargs:
            self.kwargs["max_tokens"] = max_new_tokens
        else:
            self.kwargs.pop("max_tokens", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        # [FUNCTION DOC]
        # Copies the message list, then up to max_turns times: requests a completion,
        # appends the assistant message, computes its reward and records it under the
        # completion id; stops when reward == 1 else appends a retry prompt. Returns
        # dict completion_id -> reward (per-turn rewards for the proxy).
        # KNOWN ISSUE: list() is applied before the string check, so a string prompt
        # would be split into characters and never wrapped.
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        messages = list(data.get("messages_chat") or data["messages"])
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = [m for m in messages]
        rewards = {}
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        for _ in range(self.max_turns):
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                model="default",
                **self.kwargs,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            reward = await self._reward_fn(
                completions=message.content, answer=data["answer"]
            )
            rewards[response.id] = reward
            if reward == 1:
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                        "Please carefully read the original question, check the previous errors, and try to answer it again.",
                    }
                )
        return rewards


@function_tool
def add(a: float, b: float) -> float:
    # [FUNCTION DOC]
    # Agents SDK tool: returns a + b.
    """Add two numbers."""
    return a + b


@function_tool
def subtract(a: float, b: float) -> float:
    # [FUNCTION DOC]
    # Agents SDK tool: returns a - b.
    """Subtract two numbers."""
    return a - b


@function_tool
def multiply(a: float, b: float) -> float:
    # [FUNCTION DOC]
    # Agents SDK tool: returns a * b.
    """Multiply two numbers."""
    return a * b


@function_tool
def divide(a: float, b: float) -> float:
    # [FUNCTION DOC]
    # Agents SDK tool: returns a / b; ValueError if b is 0.
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b


@function_tool
def power(a: float, b: float) -> float:
    # [FUNCTION DOC]
    # Agents SDK tool: returns a ** b.
    """Raise a to the power of b."""
    return a**b


@function_tool
def sqrt(a: float) -> float:
    # [FUNCTION DOC]
    # Agents SDK tool: returns a ** 0.5; ValueError if a < 0.
    """Calculate the square root of a number."""
    if a < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return a**0.5


class MathToolAgent:
    # [CLASS DOC]
    # OpenAI Agents SDK agent using calculator function tools.
    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # Copies kwargs, dropping max_tokens and max_turns (remaining keys become
        # ModelSettings); creates the reward wrapper.
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self._reward_fn = AsyncRewardWrapper(math_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        # [FUNCTION DOC]
        # Builds AsyncOpenAI client and a RunConfig (OpenAIProvider, model 'default',
        # tracing disabled, ModelSettings from kwargs); creates an Agent with calculator
        # tools and tool-forcing instructions; runs it on the last message content with
        # an SQLiteSession('math'); returns reward of final_output vs data['answer'].
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        content = data["messages"][-1]["content"]
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            model="default",
            tracing_disabled=True,
            model_settings=ModelSettings(**self.kwargs),
        )
        agent = Agent(
            name="RLVR Math with Calculator",
            instructions="Answer the user's math questions using the available calculator tools. Don't give the answer directly, you must use tools to do the mathematical calculation.",
            tools=[
                add,
                subtract,
                multiply,
                divide,
                power,
                sqrt,
            ],
        )
        session = SQLiteSession("math")
        result = await OpenAIRunner.run(
            agent, input=content, session=session, run_config=run_config
        )

        return await self._reward_fn(
            completions=result.final_output, answer=data["answer"]
        )

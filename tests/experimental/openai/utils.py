# [MODULE DOC]
# Test helper module providing a minimal OpenAI-SDK based agent used by proxy/rollout
# tests to issue one chat completion through an OpenAI-compatible endpoint.
import os

from openai import AsyncOpenAI


class SimpleAgent:
    # [CLASS DOC]
    # HELPER: Minimal agent with an async run(data, **extra_kwargs) interface compatible
    # with AReaL's agent workflow loading.
    async def run(self, data: dict, **extra_kwargs) -> float:
        # [FUNCTION DOC]
        # HELPER: Builds an AsyncOpenAI client from extra_kwargs http_client, base_url
        # and api_key (falling back to OPENAI_BASE_URL / OPENAI_API_KEY env vars) with
        # retries disabled, sends data["messages"] to chat.completions with model
        # "default", discards the response and returns a constant reward 1.0. Raises
        # KeyError if data has no messages; network errors propagate.
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        _ = await client.chat.completions.create(
            messages=data["messages"], model="default"
        )
        return 1.0

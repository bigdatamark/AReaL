# [MODULE DOC]
# Anthropic Messages API <-> OpenAI chat-completions translation helpers used by the
# proxy servers, built on litellm's AnthropicAdapter (a module-level singleton).
# SPDX-License-Identifier: Apache-2.0

"""Anthropic Messages API translation helpers shared by proxy generations."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterable
from typing import Any

from anthropic.types.message import Message
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.utils import ModelResponse as LitellmModelResponse

MessagePreprocessor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]

_adapter = AnthropicAdapter()


def _flatten_content_lists(messages: list[dict[str, Any]]) -> None:
    # [FUNCTION DOC]
    # HELPER: In place, replaces each message's list-typed content with a single string
    # joining (by newline) the text of "text"-type dict blocks and raw string blocks.
    # Messages whose content is not a list are untouched.
    # KNOWN ISSUE: non-text blocks (images, tool_use/tool_result dicts) are silently
    # dropped.
    """Flatten Anthropic text content blocks to strings in place."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                text_parts.append(block)
        message["content"] = "\n".join(text_parts)


def translate_anthropic_request(
    anthropic_request: dict[str, Any],
    message_preprocessors: Iterable[MessagePreprocessor] = (),
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Translates an Anthropic request dict (shallow-copied) to OpenAI chat-completion
    # kwargs via litellm; raises ValueError if translation returns None. If the result
    # has
    # a messages list, flattens list contents to strings and pipes the messages through
    # each
    # preprocessor in order. Returns the OpenAI request dict.
    """Translate an Anthropic request to OpenAI chat-completion parameters."""
    translated = _adapter.translate_completion_input_params(anthropic_request.copy())
    if translated is None:
        raise ValueError("Failed to translate request")
    openai_request = dict(translated)
    messages = openai_request.get("messages")
    if isinstance(messages, list):
        _flatten_content_lists(messages)
        for preprocessor in message_preprocessors:
            messages = preprocessor(messages)
        openai_request["messages"] = messages
    return openai_request


def translate_anthropic_response(openai_response: Any) -> Message:
    # [FUNCTION DOC]
    # Converts a non-streaming OpenAI ChatCompletion (pydantic, via model_dump) to a
    # litellm ModelResponse, translates to Anthropic format (ValueError if None), dumps
    # content blocks to dicts and returns an anthropic Message.
    """Translate a non-streaming OpenAI chat completion to Anthropic format."""
    model_response = LitellmModelResponse(**openai_response.model_dump())
    translated = _adapter.translate_completion_output_params(model_response)
    if translated is None:
        raise ValueError("Failed to translate response")

    content = translated.get("content")
    if content:
        translated["content"] = [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in content
        ]
    return Message(**translated)


def translate_anthropic_stream(
    openai_stream: AsyncGenerator[Any, None],
    model: str,
) -> AsyncGenerator[Any, None]:
    # [FUNCTION DOC]
    # Wraps an async OpenAI chat-completion chunk stream into litellm's async generator
    # of Anthropic SSE stream events for the given model name. Lazy; no I/O until
    # iterated.
    """Translate an OpenAI chat-completion stream to Anthropic SSE events."""
    return _adapter.translate_completion_output_params_streaming(
        completion_stream=openai_stream,
        model=model,
    )

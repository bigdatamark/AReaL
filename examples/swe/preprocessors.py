# [MODULE DOC]
# Message preprocessors for SWE-bench agent rollouts. Each is a callable that mutates
# and
# returns a list of OpenAI-style message dicts after Anthropic-to-OpenAI translation,
# removing volatile Claude Code metadata so prefix-based parent matching in concat
# export
# stays stable.
"""Message preprocessors for SWE-bench agent rollouts.

Preprocessors transform messages after Anthropic-to-OpenAI translation and
before they reach AReaL's OpenAI client. They remove volatile Claude Code
metadata that otherwise breaks prefix-based parent matching in concat export.
"""

import json
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class MessagePreprocessor(Protocol):
    # [CLASS DOC]
    # Runtime-checkable Protocol: any callable mapping a message list to a message list.
    """Protocol for message preprocessors."""

    # [FUNCTION DOC]
    # Protocol signature only; no implementation.
    # (One-line definition: doc placed above it.)
    def __call__(self, messages: list[dict]) -> list[dict]: ...


class StripAnthropicBillingHeader:
    # [CLASS DOC]
    # Removes lines starting with the x-anthropic-billing-header prefix from the first
    # message if it is a system message with string content.
    """Remove per-request Anthropic billing header lines from system prompts."""

    _PATTERN = re.compile(r"^x-anthropic-billing-header:[^\n]*\n?", re.MULTILINE)

    def __call__(self, messages: list[dict]) -> list[dict]:
        # [FUNCTION DOC]
        # Applies the multiline regex removal in place to messages[0] content when
        # applicable;
        # returns the same list (empty list returned unchanged).
        if not messages:
            return messages
        first = messages[0]
        if first.get("role") == "system" and isinstance(first.get("content"), str):
            first["content"] = self._PATTERN.sub("", first["content"])
        return messages


class NormalizeSystemReminder:
    # [CLASS DOC]
    # Removes "# currentDate" plus "Today's date is YYYY-MM-DD." lines from message
    # contents.
    """Remove volatile ``currentDate`` lines from system reminders."""

    _PATTERN = re.compile(
        r"# currentDate\nToday's date is \d{4}-\d{2}-\d{2}\.\n?",
    )

    def __call__(self, messages: list[dict]) -> list[dict]:
        # [FUNCTION DOC]
        # For each message with string content containing "currentDate", removes the
        # date
        # block in place. Returns the same list.
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and "currentDate" in content:
                msg["content"] = self._PATTERN.sub("", content)
        return messages


class StripAllSystemReminders:
    # [CLASS DOC]
    # Removes every system-reminder tagged block (non-greedy, across newlines).
    """Remove all ``<system-reminder>...</system-reminder>`` blocks."""

    _PATTERN = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")

    def __call__(self, messages: list[dict]) -> list[dict]:
        # [FUNCTION DOC]
        # For each message with string content containing the opening tag, deletes all
        # system-reminder blocks in place. Returns the same list.
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and "<system-reminder>" in content:
                msg["content"] = self._PATTERN.sub("", content)
        return messages


class StripAnthropicCacheFields:
    # [CLASS DOC]
    # Removes Anthropic-only fields not kept in stored output.
    """Strip Anthropic-specific fields that are not preserved in stored output."""

    def __call__(self, messages: list[dict]) -> list[dict]:
        # [FUNCTION DOC]
        # In place, pops cache_control and thinking_blocks from each message, and for
        # dict tool
        # calls pops cache_control and provider_specific_fields, plus cache_control from
        # the
        # nested function dict. Returns the same list.
        for msg in messages:
            msg.pop("cache_control", None)
            msg.pop("thinking_blocks", None)
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        tc.pop("cache_control", None)
                        tc.pop("provider_specific_fields", None)
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            fn.pop("cache_control", None)
        return messages


class NormalizeToolCallArguments:
    # [CLASS DOC]
    # Canonicalizes tool-call argument JSON for deterministic comparison: sorted keys,
    # no
    # ASCII escaping, and removal of arguments equal to known defaults
    # (_TOOL_ARG_DEFAULTS,
    # currently Edit.replace_all == False).
    """Normalize tool_call arguments for deterministic comparison."""

    _TOOL_ARG_DEFAULTS: dict[tuple[str, str], object] = {
        ("Edit", "replace_all"): False,
    }

    @classmethod
    def _normalize_tool_arguments(
        cls,
        tool_name: str | None,
        args_dict: dict,
    ) -> dict:
        # [FUNCTION DOC]
        # HELPER: If tool_name is set, pops each field whose value equals the registered
        # default
        # for that tool from args_dict (in place) and returns it.
        if tool_name is not None:
            for (tn, field), default in cls._TOOL_ARG_DEFAULTS.items():
                if tn == tool_name and args_dict.get(field) == default:
                    args_dict.pop(field, None)
        return args_dict

    def __call__(self, messages: list[dict]) -> list[dict]:
        # [FUNCTION DOC]
        # For each dict tool call with a dict function and string arguments, parses the
        # JSON,
        # drops default-valued args and re-serializes with sorted keys; invalid JSON or
        # TypeError
        # leaves the arguments unchanged. Also sets content to "" on assistant messages
        # that have
        # tool_calls but no content key. Mutates in place and returns the list.
        # KNOWN ISSUE: if the arguments JSON parses to a non-dict (e.g. a list) and the
        # tool has
        # a name, calling get on it raises AttributeError, which is not caught.
        for msg in messages:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            tool_name = fn.get("name")
                            args = fn.get("arguments")
                            if isinstance(args, str):
                                try:
                                    parsed = json.loads(args)
                                    parsed = self._normalize_tool_arguments(
                                        tool_name,
                                        parsed,
                                    )
                                    fn["arguments"] = json.dumps(
                                        parsed,
                                        sort_keys=True,
                                        ensure_ascii=False,
                                    )
                                except (json.JSONDecodeError, TypeError):
                                    pass
                if msg.get("role") == "assistant" and "content" not in msg:
                    msg["content"] = ""
        return messages

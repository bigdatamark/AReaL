# [MODULE DOC]
# Custom prefix matcher for InteractionCache parent/child matching in the SWE example.
# Treats messages as equivalent when role, content, thinking and tool-call IDs match
# even
# if tool arguments or tool outputs were rewritten between turns (as Anthropic-
# compatible
# agent frontends do), so concatenated trajectory export stays stable.
"""Custom prefix matchers for InteractionCache parent-child matching.

The default cache matcher requires exact message equality. SWE-bench agents,
especially Claude Code style agents routed through Anthropic-compatible APIs,
can rewrite tool arguments or tool output formatting between turns while still
representing the same conversation. The matcher here keeps concat export stable
for those known rewrites.
"""

from __future__ import annotations


def _tool_call_ids(tool_calls: list[dict]) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Returns the ordered list of "id" values (missing id becomes "") of the
    # dict
    # entries in a tool_calls list; non-dict entries are skipped.
    """Extract ordered tool_call IDs from a tool_calls list."""
    return [tc.get("id", "") for tc in tool_calls if isinstance(tc, dict)]


def _messages_match(a: dict, b: dict) -> bool:
    # [FUNCTION DOC]
    # HELPER: Semantic equality of two chat message dicts. Roles must match. For role
    # "tool"
    # only tool_call_id is compared (output text may differ). Otherwise content (default
    # "")
    # and thinking must be equal; if either side has tool_calls, both must have them
    # and, if
    # both are lists, the same length and same ordered IDs (arguments ignored); non-list
    # tool_calls fall back to plain equality. Returns bool.
    """Check whether two message dicts are semantically equivalent."""
    if a.get("role") != b.get("role"):
        return False

    role = a.get("role")

    if role == "tool":
        return a.get("tool_call_id") == b.get("tool_call_id")

    if a.get("content", "") != b.get("content", ""):
        return False

    if a.get("thinking") != b.get("thinking"):
        return False

    a_tc = a.get("tool_calls")
    b_tc = b.get("tool_calls")
    if a_tc is not None or b_tc is not None:
        if a_tc is None or b_tc is None:
            return False
        if not isinstance(a_tc, list) or not isinstance(b_tc, list):
            return a_tc == b_tc
        if len(a_tc) != len(b_tc):
            return False
        if _tool_call_ids(a_tc) != _tool_call_ids(b_tc):
            return False

    return True


def swe_prefix_matcher(a: list[dict], b: list[dict]) -> bool:
    # [FUNCTION DOC]
    # Returns True iff message list a is a semantic prefix of b: a is not longer than b
    # and
    # each paired message is either exactly equal or matches per _messages_match.
    """Return True if ``a`` is a semantic prefix of ``b``."""
    if len(a) > len(b):
        return False
    for am, bm in zip(a, b):
        if am == bm:
            continue
        if not _messages_match(am, bm):
            return False
    return True

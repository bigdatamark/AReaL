# [MODULE DOC]
# Unit tests for _resolve_max_total_tokens, which computes the total token budget as
# prompt_len + max_new_tokens capped by engine_max_tokens, or by the module default
# _DEFAULT_MAX_TOTAL_TOKENS when the engine limit is None.
# SPDX-License-Identifier: Apache-2.0

from areal.experimental.openai.client import (
    _DEFAULT_MAX_TOTAL_TOKENS,
    _resolve_max_total_tokens,
)


def test_engine_max_tokens_caps_total_length():
    # [FUNCTION DOC]
    # Asserts a huge request is capped at engine_max_tokens (4096).
    out = _resolve_max_total_tokens(
        prompt_len=1000,
        max_new_tokens=100_000,
        engine_max_tokens=4096,
    )

    assert out == 4096


def test_fallback_ceiling_applies_when_engine_max_tokens_is_none():
    # [FUNCTION DOC]
    # Asserts that with engine_max_tokens None an oversized request is capped at
    # _DEFAULT_MAX_TOTAL_TOKENS.
    out = _resolve_max_total_tokens(
        prompt_len=1000,
        max_new_tokens=100_000,
        engine_max_tokens=None,
    )

    assert out == _DEFAULT_MAX_TOTAL_TOKENS


def test_returns_prompt_plus_generation_when_under_cap():
    # [FUNCTION DOC]
    # Asserts prompt_len + max_new_tokens (300) is returned when under the cap.
    out = _resolve_max_total_tokens(
        prompt_len=100,
        max_new_tokens=200,
        engine_max_tokens=4096,
    )

    assert out == 300

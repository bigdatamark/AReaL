# [MODULE DOC]
# Tests validation in HttpGenerationResult (areal.api.io_struct): construction requires
# exactly one real, finite (non-bool) logprob per output token, raising ValueError
# otherwise; empty results are allowed.
import pytest

from areal.api.io_struct import HttpGenerationResult


@pytest.mark.parametrize(
    ("output_tokens", "output_logprobs"),
    [
        pytest.param([1, 2], [-0.1], id="missing-logprob"),
        pytest.param([1], [-0.1, -0.2], id="extra-logprob"),
    ],
)
def test_http_generation_result_rejects_mismatched_evidence_counts(
    output_tokens: list[int], output_logprobs: list[float]
) -> None:
    # [FUNCTION DOC]
    # Parametrized with one missing and one extra logprob; asserts ValueError whose
    # message reports both counts and "exactly one sampling logprob".
    """A parsed result requires one sampling logprob per output token."""
    with pytest.raises(ValueError) as exc_info:
        HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason="stop",
        )

    message = str(exc_info.value)
    assert f"{len(output_tokens)} output tokens" in message
    assert f"{len(output_logprobs)} output logprobs" in message
    assert "exactly one sampling logprob" in message


@pytest.mark.parametrize(
    "invalid_logprob",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_http_generation_result_rejects_non_finite_logprob(
    invalid_logprob: float,
) -> None:
    # [FUNCTION DOC]
    # Parametrized NaN, +inf, -inf at index 1; asserts ValueError "output_logprobs[1]
    # must be a real, finite number".
    """A parsed result rejects non-finite sampling evidence."""
    with pytest.raises(
        ValueError,
        match=r"output_logprobs\[1\] must be a real, finite number",
    ):
        HttpGenerationResult(
            output_tokens=[1, 2],
            output_logprobs=[-0.1, invalid_logprob],
            stop_reason="stop",
        )


@pytest.mark.parametrize("invalid_logprob", [True, False])
def test_http_generation_result_rejects_boolean_logprob(
    invalid_logprob: bool,
) -> None:
    # [FUNCTION DOC]
    # Parametrized True/False as logprob; asserts ValueError for index 0 (bools
    # rejected).
    """Boolean JSON values are not sampling logprob evidence."""
    with pytest.raises(
        ValueError,
        match=r"output_logprobs\[0\] must be a real, finite number",
    ):
        HttpGenerationResult(
            output_tokens=[1],
            output_logprobs=[invalid_logprob],
            stop_reason="stop",
        )


def test_http_generation_result_accepts_empty_evidence() -> None:
    # [FUNCTION DOC]
    # Asserts empty tokens/logprobs with stop_reason "abort" construct successfully and
    # are preserved.
    """An empty result remains valid for abort-before-prefill paths."""
    result = HttpGenerationResult(
        output_tokens=[],
        output_logprobs=[],
        stop_reason="abort",
    )

    assert result.output_tokens == []
    assert result.output_logprobs == []


def test_http_generation_result_accepts_complete_finite_evidence() -> None:
    # [FUNCTION DOC]
    # Asserts two tokens with two finite logprobs construct and are preserved unchanged.
    """A normal sampled result preserves its complete finite evidence."""
    result = HttpGenerationResult(
        output_tokens=[10, 11],
        output_logprobs=[-0.5, -0.25],
        stop_reason="stop",
    )

    assert result.output_tokens == [10, 11]
    assert result.output_logprobs == [-0.5, -0.25]

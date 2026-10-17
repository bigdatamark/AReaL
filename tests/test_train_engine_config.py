# [MODULE DOC]
# Tests for TrainEngineConfig.logprobs_chunk_size: default value and validation.
# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.api.cli_args import TrainEngineConfig


def test_logprobs_chunk_size_defaults_to_1024() -> None:
    # [FUNCTION DOC]
    # Asserts the default logprobs_chunk_size is 1024.
    """The public train-engine config keeps the historical utility default."""
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
    )

    assert config.logprobs_chunk_size == 1024


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_logprobs_chunk_size_raises_value_error(chunk_size: int) -> None:
    # [FUNCTION DOC]
    # Parametrized 0 and -1; asserts ValueError "logprobs_chunk_size must be positive".
    """Invalid chunk sizes fail during config construction with a clear error."""
    with pytest.raises(ValueError, match="logprobs_chunk_size must be positive"):
        TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test-experiment",
            trial_name="trial0",
            logprobs_chunk_size=chunk_size,
        )

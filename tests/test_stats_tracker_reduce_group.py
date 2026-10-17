# [MODULE DOC]
# Unit tests that per-key reduce_group overrides passed to
# DistributedStatsTracker.scalar / stat are used by export instead of the export-level
# default group (all_reduce patched).
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch

from areal.utils.stats_tracker import DistributedStatsTracker


def test_stats_tracker_scalar_reduce_group_override_used_on_export():
    # [FUNCTION DOC]
    # Records scalar "score" with an override group, exports with a different default;
    # asserts two all_reduce calls, both on the override group.
    # Arrange
    tracker = DistributedStatsTracker()
    default_group = object()
    override_group = object()

    tracker.scalar(score=1.0, reduce_group=override_group)

    # Act
    with patch("areal.utils.stats_tracker.dist.all_reduce") as mock_all_reduce:
        tracker.export(key="score", reduce_group=default_group)

    # Assert
    assert mock_all_reduce.call_count == 2
    assert all(
        call.kwargs["group"] is override_group for call in mock_all_reduce.mock_calls
    )


def test_stats_tracker_stat_reduce_group_override_used_on_export():
    # [FUNCTION DOC]
    # Records a mask denominator and a masked "loss" stat with an override group;
    # exports "loss" and asserts four all_reduce calls, all on the override group.
    # Arrange
    tracker = DistributedStatsTracker()
    default_group = object()
    override_group = object()

    tracker.denominator(mask=torch.tensor([True, False, True]))
    tracker.stat(
        "mask",
        loss=torch.tensor([1.0, 2.0, 3.0]),
        reduce_group=override_group,
    )

    # Act
    with patch("areal.utils.stats_tracker.dist.all_reduce") as mock_all_reduce:
        tracker.export(key="loss", reduce_group=default_group)

    # Assert
    assert mock_all_reduce.call_count == 4
    assert all(
        call.kwargs["group"] is override_group for call in mock_all_reduce.mock_calls
    )

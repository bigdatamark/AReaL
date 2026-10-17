# [MODULE DOC]
# CPU unit tests for variable-size group support in areal.utils.data.Normalization (the
# group_sizes argument used when some rollout samples are dropped) and for batched_call
# passing per-chunk trajectory group sizes via its meta argument.
"""Tests for variable-size group support in reward/advantage normalization.

These cover the ``group_sizes`` argument added to ``Normalization`` so that
group-level normalization slices each trajectory group by its actual sample
count instead of a fixed ``group_size``. Variable-size groups arise when some
rollout samples fail or are filtered, leaving groups smaller than ``n_samples``.
A fixed-size slice would otherwise straddle two groups or leave a tail.
"""

import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization, batched_call


def _group_norm_config() -> NormConfig:
    # [FUNCTION DOC]
    # HELPER: Returns a NormConfig with group-level mean and std, group_size 2, no
    # leave-one-out, biased std and eps 0.
    return NormConfig(
        mean_level="group",
        std_level="group",
        group_size=2,
        mean_leave1out=False,
        std_unbiased=False,
        eps=0.0,
    )


def _ref_group_norm(x: torch.Tensor, boundaries: list[int]) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Reference normalizer: for consecutive groups of the given sizes, subtracts the
    # group mean and divides by the biased std in float64 (0 where std is 0). Returns a
    # float64 tensor.
    """Reference: independently mean/std normalize each variable-size group."""
    out = torch.empty_like(x, dtype=torch.float64)
    offset = 0
    for size in boundaries:
        chunk = x[offset : offset + size].to(torch.float64)
        mean = chunk.mean()
        std = chunk.std(unbiased=False)
        out[offset : offset + size] = (chunk - mean) / std if std > 0 else 0.0
        offset += size
    return out


def test_variable_group_sizes_normalize_each_group_independently():
    # [FUNCTION DOC]
    # With group_sizes [3, 2, 3], output matches the reference and each group has mean zero.
    """Unequal group sizes are sliced exactly, not by a fixed group_size."""
    norm = Normalization(_group_norm_config())
    # Three groups of sizes 3, 2, 3 (total 8); fixed group_size=2 would be wrong.
    x = torch.tensor(
        [0.0, 3.0, 6.0, 10.0, 20.0, 1.0, 2.0, 3.0],
        dtype=torch.float32,
    )
    boundaries = [3, 2, 3]

    out = norm(x, group_sizes=boundaries)
    expected = _ref_group_norm(x, boundaries)

    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-5)
    # Each group is zero-mean after normalization.
    offset = 0
    for size in boundaries:
        torch.testing.assert_close(
            out[offset : offset + size].double().mean(),
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )
        offset += size


def test_group_sizes_none_matches_fixed_group_size():
    # [FUNCTION DOC]
    # Omitting group_sizes equals passing [2, 2] with config group_size 2.
    """Without group_sizes, behavior is unchanged (backward compatible)."""
    norm = Normalization(_group_norm_config())  # group_size=2
    x = torch.tensor([0.0, 2.0, 10.0, 14.0], dtype=torch.float32)

    out_default = norm(x)
    out_explicit = norm(x, group_sizes=[2, 2])

    torch.testing.assert_close(out_default, out_explicit, rtol=1e-6, atol=1e-6)


def test_variable_group_with_zero_variance_group_is_finite():
    # [FUNCTION DOC]
    # With eps > 0, an all-equal group normalizes to exact zeros and all outputs are finite.
    """A zero-variance group stays finite thanks to eps in the denominator.

    When a group's rewards are all equal its std is 0; dividing by ``std + eps``
    (eps > 0) keeps the result finite (and zero, since the numerator is also 0)
    instead of producing NaN/inf. This is the guard that prevents a degenerate
    group from blowing up advantages.
    """
    cfg = NormConfig(
        mean_level="group",
        std_level="group",
        group_size=2,
        mean_leave1out=False,
        std_unbiased=False,
        eps=1e-5,
    )
    norm = Normalization(cfg)
    # Group 0 (size 3) has variance; group 1 (size 2) is all-equal -> std 0.
    x = torch.tensor([0.0, 3.0, 6.0, 5.0, 5.0], dtype=torch.float32)
    boundaries = [3, 2]

    out = norm(x, group_sizes=boundaries)

    assert torch.isfinite(out).all(), "normalized output must be finite"
    # The degenerate group collapses to zero (x_centered=0 / (0+eps)).
    torch.testing.assert_close(
        out[3:].double(),
        torch.zeros(2, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_leave_one_out_singleton_group_outputs_zero():
    # [FUNCTION DOC]
    # With leave-one-out mean and unbiased std, a size-1 group outputs 0 and all outputs are
    # finite.
    """A size-1 leave-one-out group has no peer baseline, so it is zeroed."""
    cfg = NormConfig(
        mean_level="group",
        std_level="group",
        group_size=1,
        mean_leave1out=True,
        std_unbiased=True,
        eps=1e-5,
    )
    norm = Normalization(cfg)
    x = torch.tensor([7.0, 1.0, 5.0], dtype=torch.float32)
    # One singleton group then a size-2 group.
    out = norm(x, group_sizes=[1, 2])

    assert torch.isfinite(out).all()
    assert out[0].item() == 0.0


def test_group_sizes_sum_mismatch_raises():
    # [FUNCTION DOC]
    # group_sizes not summing to the batch size raise ValueError ("must equal").
    """Boundaries whose sizes do not sum to the batch size are rejected."""
    norm = Normalization(_group_norm_config())
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)  # bs=4
    with pytest.raises(ValueError, match="must equal"):
        norm(x, group_sizes=[3, 2])  # sums to 5 != 4


def test_group_sizes_non_positive_raises():
    # [FUNCTION DOC]
    # A zero group size raises ValueError ("positive").
    """Boundaries containing a non-positive size are rejected."""
    norm = Normalization(_group_norm_config())
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)  # bs=4
    with pytest.raises(ValueError, match="positive"):
        norm(x, group_sizes=[4, 0])


def test_adv_norm_style_2d_variable_groups_normalize_per_group():
    # [FUNCTION DOC]
    # A 2-D [batch, seq] tensor with a loss mask and group_sizes [3, 2] normalizes along the
    # batch dimension producing a finite tensor of the same shape.
    # KNOWN ISSUE: only finiteness and shape are checked, not the per-group values the name
    # claims.
    """Token-level (2D) advantages normalize per variable-size group on dim 0.

    Mirrors how PPOActor.adv_norm is applied to the GAE advantage tensor.
    """
    norm = _group_norm_config()  # group mean+std, eps=0
    adv_norm = Normalization(norm)
    # bs=5 (groups [3, 2]), seqlen=2.
    x = torch.tensor(
        [[0.0, 6.0], [3.0, 6.0], [6.0, 6.0], [10.0, 0.0], [20.0, 0.0]],
        dtype=torch.float32,
    )
    loss_mask = torch.ones_like(x)
    out = adv_norm(x, loss_mask, group_sizes=[3, 2])

    assert torch.isfinite(out).all()
    assert out.shape == x.shape


def test_batched_call_pass_meta_provides_traj_group_sizes():
    # [FUNCTION DOC]
    # batched_call with pass_meta=True concatenates the two trajectory dicts, calls fn once
    # with meta.traj_group_sizes == [2, 1] (per-input batch sizes), and splits the result
    # back into per-input dicts in order.
    data = [
        {
            "attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "values": torch.tensor([[1.0], [2.0]]),
        },
        {
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "values": torch.tensor([[3.0]]),
        },
    ]

    def fn(batch, meta):
        # [FUNCTION DOC]
        # HELPER: Asserts meta.traj_group_sizes == [2, 1] and returns values + 1.
        assert meta.traj_group_sizes == [2, 1]
        return {"values": batch["values"] + 1}

    out = batched_call(fn, data, pass_meta=True)

    assert len(out) == 2
    torch.testing.assert_close(out[0]["values"], torch.tensor([[2.0], [3.0]]))
    torch.testing.assert_close(out[1]["values"], torch.tensor([[4.0]]))

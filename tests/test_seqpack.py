# [MODULE DOC]
# Tests for sequence-packing allocation (balanced_greedy_partition, ffd_allocate), the
# TrainController trajectory dispatch path (_dispatch_tensors, _prepare_dispatch,
# _is_tensor_like) and batch packing utilities in areal.utils.data
# (concat_padded_tensors, split_and_unpad_tensor, concat_batch/split_batch). Includes
# seeded synthetic sequence-length generators mimicking RL/chat/code/math workloads.
# Pure CPU; RTensors use meta tensors.
"""Tests for seqpack allocation functions."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from areal.api import TrainEngine
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.infra import TrainController
from areal.infra.controller.train_controller import (
    _dispatch_tensors,
    _is_tensor_like,
)
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.utils.data import (
    concat_padded_tensors,
    split_and_unpad_tensor,
)
from areal.utils.seqpack import (
    balanced_greedy_partition,
    ffd_allocate,
)

# =============================================================================
# Test Data Generators
# =============================================================================


def generate_bimodal_seqlens(
    n_long: int, n_short: int, long_range: tuple, short_range: tuple, seed: int = 42
):
    # [FUNCTION DOC]
    # HELPER: Returns a shuffled list of n_long ints drawn uniformly from long_range and
    # n_short from short_range (half-open ranges) using a numpy Generator seeded with
    # seed.
    """Generate bimodal distribution of sequence lengths (common in RL with varied prompts)."""
    rng = np.random.default_rng(seed)
    long_seqs = rng.integers(long_range[0], long_range[1], size=n_long).tolist()
    short_seqs = rng.integers(short_range[0], short_range[1], size=n_short).tolist()
    all_seqs = long_seqs + short_seqs
    rng.shuffle(all_seqs)
    return list(all_seqs)


def generate_uniform_seqlens(n: int, low: int, high: int, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Returns n ints uniform in [low, high) from a seeded numpy Generator.
    """Generate uniformly distributed sequence lengths."""
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=n).tolist()


def generate_skewed_seqlens(n: int, max_len: int, skew: float = 2.0, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Returns n ints = floor(Beta(1, skew) * max_len) + 1, giving many short and
    # few long lengths; seeded.
    """Generate skewed distribution (many short, few long - typical for chat/code)."""
    rng = np.random.default_rng(seed)
    # Use beta distribution to create skew
    samples = rng.beta(1, skew, size=n)
    return [int(s * max_len) + 1 for s in samples]


def generate_exponential_seqlens(n: int, scale: float = 500.0, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Returns n ints sampled from an exponential with the given scale, truncated
    # to int and clipped to [1, 8192]; seeded.
    """Generate exponentially distributed sequence lengths (common in NLP tasks)."""
    rng = np.random.default_rng(seed)
    samples = rng.exponential(scale=scale, size=n)
    # Clip to reasonable range and convert to int
    return [max(1, min(int(s), 8192)) for s in samples]


def generate_multimodal_seqlens(
    n: int, modes: list[tuple[int, int, float]], seed: int = 42
):
    # [FUNCTION DOC]
    # HELPER: modes is a list of (mean, std, weight). For each mode draws
    # floor(n*weight/total) normal samples clipped to >=1, tops up to n using the first
    # mode, shuffles, returns the first n ints.
    """Generate multimodal distribution with multiple peaks.

    Args:
        n: Total number of sequences
        modes: List of (mean, std, weight) tuples for each mode
        seed: Random seed
    """
    rng = np.random.default_rng(seed)
    total_weight = sum(w for _, _, w in modes)
    seqlens = []

    for mean, std, weight in modes:
        count = int(n * weight / total_weight)
        samples = rng.normal(mean, std, size=count)
        seqlens.extend([max(1, int(s)) for s in samples])

    # Fill remaining with first mode
    while len(seqlens) < n:
        mean, std, _ = modes[0]
        seqlens.append(max(1, int(rng.normal(mean, std))))

    rng.shuffle(seqlens)
    return seqlens[:n]


def generate_power_law_seqlens(n: int, alpha: float = 2.0, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Returns n ints = (Pareto(alpha)+1)*100 truncated and clipped to [1, 8192];
    # seeded.
    """Generate power-law distributed sequence lengths (Zipf-like)."""
    rng = np.random.default_rng(seed)
    # Use Pareto distribution (power law)
    samples = (rng.pareto(alpha, size=n) + 1) * 100
    return [max(1, min(int(s), 8192)) for s in samples]


def generate_batch_realistic_seqlens(
    batch_size: int, prompt_range: tuple, response_range: tuple, seed: int = 42
):
    # [FUNCTION DOC]
    # HELPER: Returns batch_size totals of a uniform prompt length from prompt_range
    # plus a uniform response length from response_range (half-open); seeded.
    """Generate realistic batch with prompt+response patterns (typical in RL training).

    In RL training, each sample has a prompt (input) and response (generated).
    The total sequence length varies based on both components.
    """
    rng = np.random.default_rng(seed)
    prompts = rng.integers(prompt_range[0], prompt_range[1], size=batch_size)
    responses = rng.integers(response_range[0], response_range[1], size=batch_size)
    return [int(p + r) for p, r in zip(prompts, responses)]


def generate_code_seqlens(n: int, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Code-like lengths: multimodal mixture of (50,30,.4), (250,100,.35),
    # (800,300,.2), (2000,500,.05).
    """Generate sequence lengths typical for code generation tasks.

    Code has characteristic length distribution:
    - Many short snippets (1-100 tokens)
    - Medium functions (100-500 tokens)
    - Fewer long functions (500-2000 tokens)
    - Rare very long files (2000+ tokens)
    """
    return generate_multimodal_seqlens(
        n=n,
        modes=[
            (50, 30, 0.4),  # Short snippets
            (250, 100, 0.35),  # Medium functions
            (800, 300, 0.2),  # Long functions
            (2000, 500, 0.05),  # Very long files
        ],
        seed=seed,
    )


def generate_chat_seqlens(n: int, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Chat-like lengths: multimodal mixture of (50,30,.5), (200,80,.35),
    # (800,300,.15).
    """Generate sequence lengths typical for chat/conversation data.

    Chat has characteristic length distribution:
    - Many short messages (10-100 tokens)
    - Medium responses (100-500 tokens)
    - Fewer long explanations (500-1500 tokens)
    """
    return generate_multimodal_seqlens(
        n=n,
        modes=[
            (50, 30, 0.5),  # Short messages
            (200, 80, 0.35),  # Medium responses
            (800, 300, 0.15),  # Long explanations
        ],
        seed=seed,
    )


def generate_math_seqlens(n: int, seed: int = 42):
    # [FUNCTION DOC]
    # HELPER: Math-like lengths: prompt in [50,200) plus response in [100,800).
    """Generate sequence lengths typical for math problem solving.

    Math problems have:
    - Short problem statements (50-200 tokens)
    - Variable solution lengths (100-1000 tokens)
    """
    return generate_batch_realistic_seqlens(
        batch_size=n,
        prompt_range=(50, 200),
        response_range=(100, 800),
        seed=seed,
    )


class TestBalancedGreedyPartition:
    # [CLASS DOC]
    # Tests for balanced_greedy_partition(nums, K), which splits n items (n divisible by
    # K, n >= K) into K equal-size groups of indices with balanced sums.
    """Tests for balanced_greedy_partition function."""

    @pytest.mark.parametrize("K", [2, 4, 8])
    def test_basic_partition(self, K):
        # [FUNCTION DOC]
        # Parametrized on K; asserts K groups each with exactly 10 indices.
        """Test basic partition returns correct structure."""
        n = K * 10  # 10 items per group
        nums = list(range(100, 100 + n))

        groups = balanced_greedy_partition(nums, K)

        assert len(groups) == K
        # Each group should have n/K items
        for g in groups:
            assert len(g) == n // K

    def test_returns_indices_not_values(self):
        # [FUNCTION DOC]
        # Asserts returned groups hold indices 0..3 (not values) covering all items.
        """Test that function returns indices, not values."""
        nums = [100, 200, 300, 400]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        # Groups should contain indices (0-3), not values (100-400)
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == [0, 1, 2, 3]

        # Verify we can use indices to get original values
        for g in groups:
            values = [nums[i] for i in g]
            assert all(100 <= v <= 400 for v in values)

    def test_preserves_all_indices(self):
        # [FUNCTION DOC]
        # Asserts every index appears exactly once across groups.
        """Test that all indices are assigned exactly once."""
        nums = [50, 100, 150, 200, 250, 300]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(nums)))

    def test_equal_group_sizes(self):
        # [FUNCTION DOC]
        # Asserts all groups have len(nums)/K items.
        """Test that all groups have equal size."""
        nums = list(range(24))
        K = 4

        groups = balanced_greedy_partition(nums, K)

        expected_size = len(nums) // K
        for g in groups:
            assert len(g) == expected_size

    def test_raises_on_non_divisible(self):
        # [FUNCTION DOC]
        # Asserts ValueError "must be divisible by K" when n is not a multiple of K.
        """Test error when n is not divisible by K."""
        nums = [1, 2, 3, 4, 5]
        K = 2

        with pytest.raises(ValueError, match="must be divisible by K"):
            balanced_greedy_partition(nums, K)

    def test_raises_on_too_few_items(self):
        # [FUNCTION DOC]
        # Asserts ValueError "must be >= K" when n < K.
        """Test error when n < K."""
        nums = [1, 2, 3]
        K = 5

        with pytest.raises(ValueError, match="must be >= K"):
            balanced_greedy_partition(nums, K)

    def test_raises_on_empty_input(self):
        # [FUNCTION DOC]
        # Asserts ValueError "must be >= K" for empty input.
        """Test error when input is empty."""
        nums = []
        K = 4

        with pytest.raises(ValueError, match="must be >= K"):
            balanced_greedy_partition(nums, K)

    def test_balances_sums(self):
        # [FUNCTION DOC]
        # Asserts each group's sum is within 30% of the average for high-variance
        # values.
        # KNOWN ISSUE: inline comment says "within 20%" but the assertion uses a 0.3
        # threshold.
        """Test that group sums are well balanced."""
        # Create values with high variance
        nums = [1000, 900, 800, 700, 100, 200, 300, 400]
        K = 2

        groups = balanced_greedy_partition(nums, K)

        sums = [sum(nums[i] for i in g) for g in groups]

        # Sums should be reasonably balanced
        total = sum(nums)
        expected_avg = total / K
        # Each group should be within 20% of average
        for s in sums:
            assert abs(s - expected_avg) / expected_avg < 0.3

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_bimodal_distribution(self, seed):
        # [FUNCTION DOC]
        # Parametrized seeds; bimodal lengths (8 long, 24 short), K=4; asserts full
        # index coverage and max-min sum spread under 50% of average.
        """Test with bimodal sequence lengths (typical for RL with varied prompts)."""
        n_long, n_short = 8, 24
        values = generate_bimodal_seqlens(
            n_long=n_long,
            n_short=n_short,
            long_range=(1000, 2000),
            short_range=(100, 400),
            seed=seed,
        )
        K = 4

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))

        # Verify balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        # Difference should be reasonable
        assert max_diff / avg_sum < 0.5

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_uniform_distribution(self, seed):
        # [FUNCTION DOC]
        # Parametrized seeds; 200 uniform lengths, K=8; asserts 25 items per group and
        # spread < 20%.
        """Test with uniformly distributed sequence lengths."""
        values = generate_uniform_seqlens(n=200, low=512, high=2048, seed=seed)
        K = 8

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        for g in groups:
            assert len(g) == 25  # 200 / 8

        # Verify balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        assert max_diff / avg_sum < 0.2

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_skewed_distribution(self, seed):
        # [FUNCTION DOC]
        # Parametrized seeds; skewed lengths, K=4; asserts structure and coverage only.
        """Test with skewed distribution (many short, few long - like chat data)."""
        values = generate_skewed_seqlens(n=160, max_len=4096, skew=3.0, seed=seed)
        K = 4

        groups = balanced_greedy_partition(values, K)

        # Verify structure
        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))

    def test_edge_case_identical_values(self):
        # [FUNCTION DOC]
        # Asserts identical values produce identical group sums.
        """Test when all values are identical."""
        nums = [100] * 20
        K = 4

        groups = balanced_greedy_partition(nums, K)

        # All groups should have equal size and equal sum
        assert len(groups) == K
        sums = [sum(nums[i] for i in g) for g in groups]
        assert all(s == sums[0] for s in sums)

    def test_edge_case_two_values(self):
        # [FUNCTION DOC]
        # Four 1000s and four 1s, K=2; asserts the two sums differ by at most 2.
        """Test with extreme two-value distribution."""
        nums = [1000] * 4 + [1] * 4
        K = 2

        groups = balanced_greedy_partition(nums, K)

        # Each group should ideally have 2 large and 2 small
        sums = [sum(nums[i] for i in g) for g in groups]
        # Both sums should be close
        assert abs(sums[0] - sums[1]) <= 2  # Small difference due to 1s

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_realistic_dp_sizes(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized DP size; 16 bimodal sequences per rank; asserts 16 per group and
        # spread < 30%.
        """Test with realistic data parallel sizes."""
        n_seqs = dp_size * 16  # 16 sequences per DP rank
        values = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(1000, 2000),
            short_range=(200, 600),
            seed=42,
        )

        groups = balanced_greedy_partition(values, dp_size)

        assert len(groups) == dp_size
        for g in groups:
            assert len(g) == 16

        # Check balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / dp_size
        assert max_diff / avg_sum < 0.3

    def test_large_scale(self):
        # [FUNCTION DOC]
        # 1000 uniform values into 10 groups; asserts coverage and spread < 10%.
        """Test with larger number of items."""
        n = 1000
        K = 10
        values = generate_uniform_seqlens(n=n, low=100, high=1000, seed=42)

        groups = balanced_greedy_partition(values, K)

        assert len(groups) == K
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(n))

        # Check balance
        sums = [sum(values[i] for i in g) for g in groups]
        max_diff = max(sums) - min(sums)
        avg_sum = sum(sums) / K
        assert max_diff / avg_sum < 0.1  # Should be very balanced

    def test_single_item_per_group(self):
        # [FUNCTION DOC]
        # Asserts n == K yields one index per group.
        """Test edge case where each group gets exactly one item."""
        nums = [100, 200, 300, 400]
        K = 4

        groups = balanced_greedy_partition(nums, K)

        assert len(groups) == K
        for g in groups:
            assert len(g) == 1

    def test_deterministic(self):
        # [FUNCTION DOC]
        # Asserts two calls with the same input return identical groups.
        """Test that function is deterministic."""
        nums = [300, 100, 400, 200, 500, 600]
        K = 2

        groups1 = balanced_greedy_partition(nums, K)
        groups2 = balanced_greedy_partition(nums, K)

        # Should produce same result
        assert groups1 == groups2


class TestFFDAllocate:
    # [CLASS DOC]
    # Regression tests for ffd_allocate(values, capacity, min_groups) (first-fit-
    # decreasing).
    """Tests for existing ffd_allocate function to ensure no regression."""

    def test_basic_allocation(self):
        # [FUNCTION DOC]
        # Asserts at least min_groups groups, full index coverage, and each group sum <=
        # capacity.
        """Test basic FFD allocation."""
        values = [100, 200, 300, 150, 250]
        capacity = 500
        min_groups = 2

        groups = ffd_allocate(values, capacity, min_groups)

        assert len(groups) >= min_groups
        # All indices should be present
        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(len(values)))
        # Each group should respect capacity
        for g in groups:
            total = sum(values[i] for i in g)
            assert total <= capacity

    def test_respects_min_groups(self):
        # [FUNCTION DOC]
        # Asserts min_groups=4 is honoured even when everything fits in one bin.
        """Test that min_groups constraint is respected."""
        values = [50] * 10
        capacity = 1000
        min_groups = 4

        groups = ffd_allocate(values, capacity, min_groups)

        assert len(groups) >= min_groups

    def test_raises_on_value_exceeds_capacity(self):
        # [FUNCTION DOC]
        # Asserts RuntimeError "larger than capacity" when a value exceeds capacity.
        """Test error when a value exceeds capacity."""
        values = [100, 600, 200]
        capacity = 500

        with pytest.raises(RuntimeError, match="larger than capacity"):
            ffd_allocate(values, capacity, min_groups=1)

    def test_raises_on_insufficient_values(self):
        # [FUNCTION DOC]
        # Asserts RuntimeError "smaller than min_groups" when there are fewer values
        # than min_groups.
        """Test error when not enough values for min_groups."""
        values = [100, 200]
        capacity = 500
        min_groups = 5

        with pytest.raises(RuntimeError, match="smaller than min_groups"):
            ffd_allocate(values, capacity, min_groups)


# =============================================================================
# Integration Tests: _dispatch_tensors
# =============================================================================


class TestDispatchTrajList:
    # [CLASS DOC]
    # Tests _dispatch_tensors(traj_list, dp_size), which returns (splits, group_indices)
    # assigning trajectory dicts to DP ranks in equal-size groups using sequence length
    # from RTensor shapes.
    """Integration tests for _dispatch_tensors with balanced_greedy_partition.

    These tests verify that trajectory lists (list[dict[str, RTensor]]) are
    partitioned into equal-size groups for different DP ranks.
    """

    def _create_traj_list(self, seqlens: list[int]) -> list[dict[str, RTensor]]:
        # [FUNCTION DOC]
        # HELPER: Builds one dict per seqlen with "input_ids" an RTensor over a [1,
        # seqlen] meta tensor (dummy shard). Returns the list.
        """Helper to create a list of trajectory dicts (one per sequence)."""
        traj_list = []
        for slen in seqlens:
            shard = TensorShardInfo(shard_id="test", node_addr="")
            data = torch.empty(1, slen, device="meta")
            traj_list.append({"input_ids": RTensor(shard=shard, data=data)})
        return traj_list

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_uniform_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; uniform lengths; asserts dp_size splits each of
        # n/dp_size items and matching group_indices sizes.
        """Test that uniform distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=1000, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        assert len(splits) == dp_size
        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size, (
                f"DP rank {i} got {len(group)} trajectories, expected {expected_size}"
            )
            assert len(group_indices[i]) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_bimodal_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; bimodal lengths; asserts equal-size splits.
        """Test that bimodal distribution splits into equal-size groups."""
        n_seqs = dp_size * 20
        seqlens = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(1000, 2000),
            short_range=(100, 400),
            seed=42,
        )
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_skewed_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; skewed lengths; asserts equal-size splits.
        """Test that skewed distribution splits into equal-size groups."""
        n_seqs = dp_size * 24
        seqlens = generate_skewed_seqlens(n=n_seqs, max_len=2000, skew=3.0, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_exponential_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; exponential lengths; asserts equal-size splits.
        """Test that exponential distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_exponential_seqlens(n=n_seqs, scale=500.0, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_code_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; code-like lengths; asserts equal-size splits.
        """Test that code-like distribution splits into equal-size groups."""
        n_seqs = dp_size * 32
        seqlens = generate_code_seqlens(n=n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_chat_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; chat-like lengths; asserts equal-size splits.
        """Test that chat-like distribution splits into equal-size groups."""
        n_seqs = dp_size * 24
        seqlens = generate_chat_seqlens(n=n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_math_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; math-like lengths; asserts equal-size splits.
        """Test that math problem distribution splits into equal-size groups."""
        n_seqs = dp_size * 16
        seqlens = generate_math_seqlens(n=n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_equal_split_power_law_distribution(self, dp_size):
        # [FUNCTION DOC]
        # Parametrized dp_size; power-law lengths; asserts equal-size splits.
        """Test that power-law distribution splits into equal-size groups."""
        n_seqs = dp_size * 20
        seqlens = generate_power_law_seqlens(n=n_seqs, alpha=2.0, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000, 2024, 3141, 9999])
    def test_equal_split_various_seeds(self, seed):
        # [FUNCTION DOC]
        # Parametrized seeds; 64 bimodal lengths over 4 ranks; asserts 16 per split.
        """Test equal split consistency across many random seeds."""
        dp_size = 4
        n_seqs = 64
        seqlens = generate_bimodal_seqlens(
            n_long=16,
            n_short=48,
            long_range=(500, 1500),
            short_range=(50, 300),
            seed=seed,
        )
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    def test_all_indices_preserved_after_dispatch(self):
        # [FUNCTION DOC]
        # Asserts group_indices cover 0..99 exactly once for 100 trajectories over 4
        # ranks.
        """Test that all original indices are preserved after dispatch."""
        dp_size = 4
        n_seqs = 100
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        traj_list = self._create_traj_list(seqlens)

        _, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        all_indices = sorted(i for g in group_indices for i in g)
        assert all_indices == list(range(n_seqs))

    def test_token_balance_across_dp_ranks(self):
        # [FUNCTION DOC]
        # Asserts per-rank token totals for bimodal data have max-min spread under 30%
        # of average.
        """Test that total tokens are reasonably balanced across DP ranks."""
        dp_size = 4
        seqlens = generate_bimodal_seqlens(
            n_long=20,
            n_short=60,
            long_range=(1000, 2000),
            short_range=(100, 300),
            seed=42,
        )
        traj_list = self._create_traj_list(seqlens)

        _, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        tokens_per_rank = [sum(seqlens[i] for i in g) for g in group_indices]
        avg_tokens = sum(tokens_per_rank) / dp_size
        max_diff = max(tokens_per_rank) - min(tokens_per_rank)

        assert max_diff / avg_tokens < 0.3, (
            f"Token imbalance too high: {max_diff / avg_tokens:.2%}"
        )

    @pytest.mark.parametrize(
        "batch_size,dp_size",
        [
            (32, 2),
            (64, 4),
            (128, 8),
            (256, 8),
            (512, 8),
            (1024, 8),
        ],
    )
    def test_large_batch_equal_split(self, batch_size, dp_size):
        # [FUNCTION DOC]
        # Parametrized (batch_size, dp_size) up to 1024; asserts equal-size splits.
        """Test equal split for various large batch sizes."""
        seqlens = generate_uniform_seqlens(n=batch_size, low=100, high=2000, seed=42)
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = batch_size // dp_size
        for i, group in enumerate(splits):
            assert len(group) == expected_size

    def test_dispatch_preserves_trajectory_content(self):
        # [FUNCTION DOC]
        # Asserts dispatched items are the original dicts with an RTensor "input_ids".
        """Test that dispatched trajectories contain the original dict content."""
        dp_size = 2
        seqlens = [100, 200, 300, 400]
        traj_list = self._create_traj_list(seqlens)

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        # Every dispatched trajectory should be a dict with 'input_ids' RTensor
        for group in splits:
            for traj in group:
                assert isinstance(traj, dict)
                assert "input_ids" in traj
                assert isinstance(traj["input_ids"], RTensor)

    def test_dispatch_with_multi_key_trajectories(self):
        # [FUNCTION DOC]
        # Trajectories with two RTensor keys; asserts equal splits and both keys
        # preserved.
        """Test dispatch with trajectory dicts containing multiple RTensor keys."""
        dp_size = 2
        n_seqs = 8
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        traj_list = []
        for slen in seqlens:
            shard = TensorShardInfo(shard_id="test", node_addr="")
            data = torch.empty(1, slen, device="meta")
            traj_list.append(
                {
                    "input_ids": RTensor(shard=shard, data=data),
                    "labels": RTensor(shard=shard, data=data),
                }
            )

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        expected_size = n_seqs // dp_size
        for group in splits:
            assert len(group) == expected_size
            for traj in group:
                assert "input_ids" in traj
                assert "labels" in traj

    def test_dispatch_fallback_for_non_rtensor_dicts(self):
        # [FUNCTION DOC]
        # Dicts with only string values (seqlen falls back to 1); asserts dp_size splits
        # covering 4 items.
        """Test that dicts without RTensors get seqlen=1 fallback."""
        dp_size = 2
        traj_list = [
            {"text": "hello"},
            {"text": "world"},
            {"text": "foo"},
            {"text": "bar"},
        ]

        splits, group_indices = _dispatch_tensors(traj_list, dp_size=dp_size)

        assert len(splits) == dp_size
        total = sum(len(g) for g in splits)
        assert total == 4


class TestTrainControllerDispatchIntegration:
    # [CLASS DOC]
    # Tests TrainController._prepare_dispatch with a mocked scheduler and a forced
    # train_alloc, verifying equal-size, index-preserving, deterministic DP grouping.
    """Integration tests for TrainController._prepare_dispatch with equal-size splits.

    These tests simulate the full dispatch flow from TrainController to verify
    that batches are split into equal sizes for different DP ranks.
    """

    @pytest.fixture
    def mock_scheduler(self):
        # [FUNCTION DOC]
        # Fixture: Mock scheduler whose async_call_engine is an AsyncMock returning
        # None.
        """Create a mock scheduler for testing."""
        scheduler = Mock()
        scheduler.async_call_engine = AsyncMock(return_value=None)
        return scheduler

    @pytest.fixture
    def train_config(self):
        # [FUNCTION DOC]
        # Fixture: TrainEngineConfig with backend "fsdp:d1" and one dummy
        # SchedulingSpec.
        """Create a TrainEngineConfig for testing."""
        return TrainEngineConfig(
            backend="fsdp:d1",
            scheduling_spec=(
                SchedulingSpec(cpu=4, gpu=1, mem=16000, port_count=2, cmd="dummy"),
            ),
        )

    def _create_traj_list(self, seqlens: list[int]) -> list[dict[str, RTensor]]:
        # [FUNCTION DOC]
        # HELPER: Same as TestDispatchTrajList._create_traj_list: one meta-RTensor
        # "input_ids" dict per seqlen.
        """Helper to create a list of trajectory dicts (one per sequence)."""
        traj_list = []
        for slen in seqlens:
            shard = TensorShardInfo(shard_id="test", node_addr="")
            data = torch.empty(1, slen, device="meta")
            traj_list.append({"input_ids": RTensor(shard=shard, data=data)})
        return traj_list

    @pytest.mark.parametrize("dp_size", [2, 4, 8])
    def test_train_controller_dispatch_equal_split(
        self, mock_scheduler, train_config, dp_size
    ):
        # [FUNCTION DOC]
        # Parametrized dp_size; builds a TrainController, sets train_alloc to
        # fsdp:d{dp}p1t1, and asserts _prepare_dispatch yields dp_size groups of equal
        # size on bimodal data.
        """Test TrainController dispatches batches to equal-size DP groups."""

        class MockTrainEngine(TrainEngine):
            # [CLASS DOC]
            # HELPER: Empty TrainEngine subclass used only as the engine class argument
            # to TrainController; never instantiated.
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        controller.train_alloc = ModelAllocation.from_str(f"fsdp:d{dp_size}p1t1")

        # Create batch with sequences that should split equally
        n_seqs = dp_size * 16
        seqlens = generate_bimodal_seqlens(
            n_long=n_seqs // 4,
            n_short=n_seqs - n_seqs // 4,
            long_range=(500, 1500),
            short_range=(100, 300),
            seed=42,
        )
        traj_list = self._create_traj_list(seqlens)

        split_args, split_kwargs, group_indices = controller._prepare_dispatch(
            traj_list
        )

        # Verify equal split
        expected_size = n_seqs // dp_size
        assert len(group_indices) == dp_size
        for i, g in enumerate(group_indices):
            assert len(g) == expected_size, (
                f"DP rank {i} got {len(g)} sequences, expected {expected_size}"
            )

    @pytest.mark.parametrize(
        "generator_name,generator_func",
        [
            ("uniform", lambda n, seed: generate_uniform_seqlens(n, 100, 1000, seed)),
            (
                "bimodal",
                lambda n, seed: generate_bimodal_seqlens(
                    n // 4, n - n // 4, (800, 1500), (100, 300), seed
                ),
            ),
            ("skewed", lambda n, seed: generate_skewed_seqlens(n, 2000, 3.0, seed)),
            (
                "exponential",
                lambda n, seed: generate_exponential_seqlens(n, 500.0, seed),
            ),
            ("code", lambda n, seed: generate_code_seqlens(n, seed)),
            ("chat", lambda n, seed: generate_chat_seqlens(n, seed)),
            ("math", lambda n, seed: generate_math_seqlens(n, seed)),
            ("power_law", lambda n, seed: generate_power_law_seqlens(n, 2.0, seed)),
        ],
    )
    def test_train_controller_dispatch_all_distributions(
        self, mock_scheduler, train_config, generator_name, generator_func
    ):
        # [FUNCTION DOC]
        # Parametrized over eight length generators with dp_size 4 and 80 sequences;
        # asserts each group has 20 indices.
        """Test TrainController dispatch with all distribution types."""

        class MockTrainEngine(TrainEngine):
            # [CLASS DOC]
            # HELPER: Empty TrainEngine subclass used only as the engine class argument
            # to TrainController; never instantiated.
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        controller.train_alloc = ModelAllocation.from_str(f"fsdp:d{dp_size}p1t1")

        n_seqs = dp_size * 20
        seqlens = generator_func(n_seqs, seed=42)
        traj_list = self._create_traj_list(seqlens)

        _, _, group_indices = controller._prepare_dispatch(traj_list)

        expected_size = n_seqs // dp_size
        for i, g in enumerate(group_indices):
            assert len(g) == expected_size, (
                f"Distribution '{generator_name}': DP rank {i} got {len(g)} "
                f"sequences, expected {expected_size}"
            )

    def test_train_controller_dispatch_preserves_indices(
        self, mock_scheduler, train_config
    ):
        # [FUNCTION DOC]
        # Asserts _prepare_dispatch group indices cover all 100 sequences exactly once.
        """Test that all sequence indices are preserved after dispatch."""

        class MockTrainEngine(TrainEngine):
            # [CLASS DOC]
            # HELPER: Empty TrainEngine subclass used only as the engine class argument
            # to TrainController; never instantiated.
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        controller.train_alloc = ModelAllocation.from_str(f"fsdp:d{dp_size}p1t1")

        n_seqs = 100
        seqlens = generate_uniform_seqlens(n=n_seqs, low=100, high=500, seed=42)
        traj_list = self._create_traj_list(seqlens)

        _, _, group_indices = controller._prepare_dispatch(traj_list)

        all_indices = sorted(i for g in group_indices for i in g)
        assert all_indices == list(range(n_seqs))

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_train_controller_dispatch_deterministic(
        self, mock_scheduler, train_config, seed
    ):
        # [FUNCTION DOC]
        # Parametrized seeds; asserts two _prepare_dispatch calls on equivalent inputs
        # give the same group indices.
        """Test that dispatch is deterministic for the same input."""

        class MockTrainEngine(TrainEngine):
            # [CLASS DOC]
            # HELPER: Empty TrainEngine subclass used only as the engine class argument
            # to TrainController; never instantiated.
            pass

        controller = TrainController(
            train_engine=MockTrainEngine,
            config=train_config,
            scheduler=mock_scheduler,
        )

        dp_size = 4
        controller.train_alloc = ModelAllocation.from_str(f"fsdp:d{dp_size}p1t1")

        seqlens = generate_bimodal_seqlens(
            n_long=16,
            n_short=48,
            long_range=(500, 1500),
            short_range=(100, 300),
            seed=seed,
        )
        traj_list = self._create_traj_list(seqlens)

        _, _, group_indices1 = controller._prepare_dispatch(traj_list)

        # Dispatch again with same data
        traj_list2 = self._create_traj_list(seqlens)
        _, _, group_indices2 = controller._prepare_dispatch(traj_list2)

        assert group_indices1 == group_indices2


class TestPadAndConcatTensors:
    # [CLASS DOC]
    # Tests concat_padded_tensors, which right-pads tensors along dim 1 and concatenates
    # along dim 0 (lists are concatenated).
    """Test concat_padded_tensors utility."""

    def test_concat_single_dict_returns_same(self):
        # [FUNCTION DOC]
        # Asserts a single dict is returned with equal tensors.
        d = {"x": torch.randn(2, 3)}
        result = concat_padded_tensors([d])
        assert set(result.keys()) == set(d.keys())
        for k in d:
            torch.testing.assert_close(result[k], d[k])

    def test_concat_two_dicts_pads_and_cats(self):
        # [FUNCTION DOC]
        # Asserts dicts with differing seq dims are padded to the max and concatenated,
        # preserving original values in their slices.
        d1 = {"x": torch.ones(2, 3), "y": torch.ones(2, 5)}
        d2 = {"x": torch.zeros(1, 4), "y": torch.zeros(1, 5)}
        result = concat_padded_tensors([d1, d2])
        assert result["x"].shape == (3, 4)  # padded to max dim-1
        assert result["y"].shape == (3, 5)
        assert torch.allclose(result["x"][:2, :3], torch.ones(2, 3))
        assert torch.allclose(result["x"][2:, :4], torch.zeros(1, 4))

    def test_concat_empty_returns_empty(self):
        # [FUNCTION DOC]
        # Asserts empty input returns {}.
        assert concat_padded_tensors([]) == {}

    def test_concat_list_values(self):
        # [FUNCTION DOC]
        # Asserts list values are concatenated as lists.
        d1 = {"ids": [1, 2]}
        d2 = {"ids": [3, 4, 5]}
        result = concat_padded_tensors([d1, d2])
        assert result["ids"] == [1, 2, 3, 4, 5]


class TestSplitAndUnpadTensor:
    # [CLASS DOC]
    # Tests split_and_unpad_tensor(obj, n_trajs) splitting batched results per
    # trajectory.
    """Tests for split_and_unpad_tensor."""

    def test_split_tensor(self):
        # [FUNCTION DOC]
        # Asserts a [3,2] tensor splits into three [1,2] tensors.
        """Split a batched tensor into per-traj tensors."""
        batched = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = split_and_unpad_tensor(batched, n_trajs=3)
        assert isinstance(result, list)
        assert len(result) == 3
        torch.testing.assert_close(result[0], torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(result[2], torch.tensor([[5.0, 6.0]]))

    def test_split_dict(self):
        # [FUNCTION DOC]
        # Asserts a dict of 1-D tensors splits per item and non-tensor values are copied
        # to each split.
        """Split a batched dict into per-traj dicts."""
        batched = {
            "logp": torch.tensor([1.0, 2.0]),
            "values": torch.tensor([3.0, 4.0]),
            "scalar": "shared",
        }
        result = split_and_unpad_tensor(batched, n_trajs=2)
        assert isinstance(result, list)
        assert len(result) == 2
        torch.testing.assert_close(result[0]["logp"], torch.tensor([1.0]))
        torch.testing.assert_close(result[1]["values"], torch.tensor([4.0]))
        assert result[0]["scalar"] == "shared"

    def test_split_none(self):
        # [FUNCTION DOC]
        # Asserts None input returns None.
        """None input returns None."""
        assert split_and_unpad_tensor(None, n_trajs=3) is None

    def test_split_passthrough(self):
        # [FUNCTION DOC]
        # Asserts non-tensor, non-dict input (an int) is returned unchanged.
        """Non-tensor/dict/None input is returned as-is."""
        assert split_and_unpad_tensor(42, n_trajs=2) == 42


# =============================================================================
# Wave 1 & 2 Fix Verification Tests
# =============================================================================


class TestPadAndConcatTensorsKeyValidation:
    # [CLASS DOC]
    # Tests that concat_padded_tensors validates identical key sets across dicts.
    """Tests for M4: concat_padded_tensors key validation."""

    def test_consistent_keys_succeeds(self):
        # [FUNCTION DOC]
        # Asserts consistent keys concatenate successfully.
        """concat_padded_tensors works when all dicts have same keys."""
        dicts = [{"a": torch.tensor([1, 2])}, {"a": torch.tensor([3, 4])}]
        result = concat_padded_tensors(dicts)
        assert "a" in result

    def test_inconsistent_keys_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError on disjoint key sets.
        """concat_padded_tensors raises ValueError on mismatched keys."""
        dicts = [{"a": torch.tensor([1])}, {"b": torch.tensor([2])}]
        with pytest.raises(ValueError):
            concat_padded_tensors(dicts)

    def test_extra_key_in_later_dict_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError when a later dict has an extra key.
        """concat_padded_tensors raises ValueError when later dict has extra key."""
        dicts = [
            {"a": torch.tensor([1])},
            {"a": torch.tensor([2]), "b": torch.tensor([3])},
        ]
        with pytest.raises(ValueError):
            concat_padded_tensors(dicts)

    def test_missing_key_in_later_dict_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError when a later dict lacks a key.
        """concat_padded_tensors raises ValueError when later dict misses key."""
        dicts = [
            {"a": torch.tensor([1]), "b": torch.tensor([2])},
            {"a": torch.tensor([3])},
        ]
        with pytest.raises(ValueError):
            concat_padded_tensors(dicts)


class TestSplitAndUnpadTensorRefSafety:
    # [CLASS DOC]
    # Tests that split_and_unpad_tensor gives each split independent copies of non-
    # tensor values.
    """Tests for H3: split_and_unpad_tensor reference safety."""

    def test_non_tensor_values_are_copied(self):
        # [FUNCTION DOC]
        # Mutates one split's list value and asserts the other split is unaffected.
        """Non-tensor values in split results are copies, not shared references."""
        shared_list = [1, 2, 3]
        result = {"data": torch.tensor([[1.0], [2.0]]), "meta": shared_list}
        splits = split_and_unpad_tensor(result, n_trajs=2, traj_group_sizes=1)
        # Modifying one split's meta should not affect the other
        splits[0]["meta"].append(99)
        assert 99 not in splits[1]["meta"], (
            "Non-tensor values should be independent copies"
        )

    def test_non_tensor_dict_values_are_copied(self):
        # [FUNCTION DOC]
        # Asserts dict values are distinct objects across splits.
        """Non-tensor dict values should be shallow-copied per split."""
        shared_dict = {"key": "value"}
        result = {"data": torch.tensor([[1.0], [2.0]]), "config": shared_dict}
        splits = split_and_unpad_tensor(result, n_trajs=2, traj_group_sizes=1)
        # The config should be a copy, not the same object
        assert splits[0]["config"] is not splits[1]["config"]

    def test_scalar_values_preserved_across_splits(self):
        # [FUNCTION DOC]
        # Asserts scalar values appear unchanged in every split.
        """Scalar values should be preserved in all splits."""
        result = {"data": torch.tensor([[1.0], [2.0]]), "count": 42}
        splits = split_and_unpad_tensor(result, n_trajs=2, traj_group_sizes=1)
        assert splits[0]["count"] == 42
        assert splits[1]["count"] == 42


class TestPackUnpackBatch:
    # [CLASS DOC]
    # Tests concat_batch (list of traj dicts -> batched dict + TrajBatchMeta) and
    # split_batch (inverse, trimming to per-trajectory seqlens).
    """Tests for M2: concat_batch and split_batch."""

    def test_pack_batch_basic(self):
        # [FUNCTION DOC]
        # Asserts two [1,3] trajectories batch to [2,3] with meta n_trajs 2, group sizes
        # [1,1], seqlens [3,3].
        """concat_batch concatenates trajectory dicts and returns TrajBatchMeta."""
        from areal.utils.data import TrajBatchMeta, concat_batch

        traj_list = [
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
            {
                "input_ids": torch.tensor([[4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 0]]),
            },
        ]
        batched, meta = concat_batch(traj_list)
        assert batched["input_ids"].shape == (2, 3)
        assert isinstance(meta, TrajBatchMeta)
        assert meta.n_trajs == 2
        assert meta.traj_group_sizes == [1, 1]
        assert meta.traj_seqlens == [3, 3]

    def test_pack_batch_rejects_non_list(self):
        # [FUNCTION DOC]
        # Asserts concat_batch raises AssertionError on a dict input.
        """concat_batch raises AssertionError for non-list input."""
        from areal.utils.data import concat_batch

        with pytest.raises(AssertionError):
            concat_batch({"a": 1})

    def test_pack_batch_rejects_non_dict_elements(self):
        # [FUNCTION DOC]
        # Asserts concat_batch raises AssertionError on a list of non-dicts.
        """concat_batch raises AssertionError for list of non-dicts."""
        from areal.utils.data import concat_batch

        with pytest.raises(AssertionError):
            concat_batch(["not", "dicts"])

    def test_unpack_batch_basic(self):
        # [FUNCTION DOC]
        # Asserts split_batch returns two [1,2] splits for a [2,2] batch.
        """split_batch splits batched dict back into trajectory list."""
        from areal.utils.data import TrajBatchMeta, split_batch

        batched = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        meta = TrajBatchMeta(n_trajs=2, traj_group_sizes=[1, 1], traj_seqlens=[2, 2])
        splits = split_batch(batched, meta)
        assert len(splits) == 2
        assert splits[0]["input_ids"].shape == (1, 2)

    def test_unpack_batch_with_traj_group_sizes(self):
        # [FUNCTION DOC]
        # Asserts split_batch with group sizes [2,2] yields two [2,1] splits.
        """split_batch handles traj_group_sizes > 1."""
        from areal.utils.data import TrajBatchMeta, split_batch

        batched = {"x": torch.tensor([[1], [2], [3], [4]])}
        meta = TrajBatchMeta(n_trajs=2, traj_group_sizes=[2, 2], traj_seqlens=[1, 1])
        splits = split_batch(batched, meta)
        assert len(splits) == 2
        assert splits[0]["x"].shape == (2, 1)

    def test_roundtrip(self):
        # [FUNCTION DOC]
        # Asserts concat_batch then split_batch reproduces the original "x" tensors.
        """concat_batch and split_batch are inverses."""
        from areal.utils.data import concat_batch, split_batch

        traj_list = [
            {"x": torch.tensor([[1.0, 2.0]]), "attention_mask": torch.tensor([[1, 1]])},
            {"x": torch.tensor([[3.0, 4.0]]), "attention_mask": torch.tensor([[1, 1]])},
        ]
        batched, meta = concat_batch(traj_list)
        recovered = split_batch(batched, meta)
        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0]["x"], traj_list[0]["x"])
        torch.testing.assert_close(recovered[1]["x"], traj_list[1]["x"])

    def test_roundtrip_multiple_keys(self):
        # [FUNCTION DOC]
        # Asserts roundtrip preserves multiple keys per trajectory.
        """Roundtrip preserves multiple keys per trajectory dict."""
        from areal.utils.data import concat_batch, split_batch

        traj_list = [
            {
                "a": torch.tensor([[1.0]]),
                "b": torch.tensor([[2.0]]),
                "attention_mask": torch.tensor([[1]]),
            },
            {
                "a": torch.tensor([[3.0]]),
                "b": torch.tensor([[4.0]]),
                "attention_mask": torch.tensor([[1]]),
            },
        ]
        batched, meta = concat_batch(traj_list)
        recovered = split_batch(batched, meta)
        assert len(recovered) == 2
        torch.testing.assert_close(recovered[0]["a"], traj_list[0]["a"])
        torch.testing.assert_close(recovered[0]["b"], traj_list[0]["b"])
        torch.testing.assert_close(recovered[1]["a"], traj_list[1]["a"])
        torch.testing.assert_close(recovered[1]["b"], traj_list[1]["b"])

    def test_roundtrip_unpadding(self):
        # [FUNCTION DOC]
        # Asserts split_batch trims a [2,5] result to seqlens 3 and 4 per trajectory.
        """Roundtrip unpadding: split_batch trims over-padded tensors."""
        from areal.utils.data import TrajBatchMeta, split_batch

        # Simulate engine output padded to global max (5) while traj seqlens are 3 and 4
        result = {"logp": torch.randn(2, 5)}
        meta = TrajBatchMeta(n_trajs=2, traj_group_sizes=[1, 1], traj_seqlens=[3, 4])
        splits = split_batch(result, meta)
        assert splits[0]["logp"].shape == (1, 3)
        assert splits[1]["logp"].shape == (1, 4)


class TestIsTrajList:
    # [CLASS DOC]
    # Tests _is_tensor_like: True only for a non-empty list of dicts where some value is
    # a torch.Tensor or RTensor.
    """Tests for H1: _is_tensor_like tensor check."""

    def test_rejects_non_tensor_dict_list(self):
        # [FUNCTION DOC]
        # Asserts False for dicts with only string values.
        """_is_tensor_like returns False for list of dicts without tensor values."""
        assert not _is_tensor_like([{"key": "string_value"}])

    def test_rejects_dict_with_int_values(self):
        # [FUNCTION DOC]
        # Asserts False for dicts with only int values.
        """_is_tensor_like returns False for list of dicts with int values."""
        assert not _is_tensor_like([{"count": 42}])

    def test_rejects_dict_with_nested_dict(self):
        # [FUNCTION DOC]
        # Asserts False for dicts with nested dict values.
        """_is_tensor_like returns False for list of dicts with nested dict."""
        assert not _is_tensor_like([{"config": {"key": "value"}}])

    def test_accepts_torch_tensor_dict_list(self):
        # [FUNCTION DOC]
        # Asserts True for dicts with a torch.Tensor value.
        """_is_tensor_like returns True for list of dicts with torch.Tensor values."""
        assert _is_tensor_like([{"logits": torch.tensor([1.0])}])

    def test_accepts_rtensor_dict_list(self):
        # [FUNCTION DOC]
        # Asserts True for dicts with an RTensor value.
        """_is_tensor_like returns True for list of dicts with RTensor values."""
        shard = TensorShardInfo(shard_id="test", node_addr="")
        rtensor = RTensor(shard=shard, data=torch.empty(1, 5, device="meta"))
        assert _is_tensor_like([{"input_ids": rtensor}])

    def test_rejects_empty_list(self):
        # [FUNCTION DOC]
        # Asserts False for an empty list.
        """_is_tensor_like returns False for empty list."""
        assert not _is_tensor_like([])

    def test_rejects_non_list(self):
        # [FUNCTION DOC]
        # Asserts False for a dict (non-list) input.
        """_is_tensor_like returns False for non-list input."""
        assert not _is_tensor_like({"key": "value"})

    def test_rejects_list_of_non_dicts(self):
        # [FUNCTION DOC]
        # Asserts False for a list of ints.
        """_is_tensor_like returns False for list of non-dicts."""
        assert not _is_tensor_like([1, 2, 3])

    def test_accepts_dict_with_mixed_tensor_and_non_tensor(self):
        # [FUNCTION DOC]
        # Asserts True when at least one value is a tensor alongside non-tensor values.
        """_is_tensor_like returns True when at least one value is tensor."""
        assert _is_tensor_like([{"data": torch.tensor([1.0]), "meta": "string"}])

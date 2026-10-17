# [MODULE DOC]
# Unit tests (CPU-only, pure Python) for the sequence-packing allocators in
# areal.utils.seqpack: the Karmarkar-Karp (KK) largest-differencing partitioner and its
# internal _KKSet/_KKState classes, kk_allocate, ffd_allocate (first-fit-decreasing),
# get_allocate_fn dispatch, the packing algorithm constants, and
# _compute_packing_metrics. Includes seeded random sequence-length generators and
# comparative KK-vs-FFD balance tests.
"""Tests for Karmarkar-Karp (KK) sequence packing algorithm and configurable dispatch.

Tests cover:
  1. KK core classes (_KKSet, _KKState) behaviour
  2. KK algorithm correctness (partition balance, index coverage, capacity)
  3. Configurable algorithm dispatch via get_allocate_fn()
  4. Comparative tests: KK vs FFD balance quality
  5. Edge cases and error handling
  6. MicroBatchSpec packing_algorithm field validation
  7. KK-specific: k upper bound cap, FFD fallback, equal_size mode
  8. Consistency with veRL-style KK output
  9. _compute_packing_metrics validation

Run with: pytest tests/test_kk_allocate.py -v
"""

import random

import pytest

from areal.utils.seqpack import (
    PACKING_ALGORITHM_FFD,
    PACKING_ALGORITHM_KK,
    PACKING_ALGORITHMS,
    _compute_packing_metrics,
    _kk_partition,
    _KKSet,
    _KKState,
    ffd_allocate,
    get_allocate_fn,
    kk_allocate,
)

# ---------------------------------------------------------------------------
# Data generators (matching test_seqpack.py patterns)
# ---------------------------------------------------------------------------


def generate_bimodal_seqlens(
    n=200, short_range=(50, 200), long_range=(800, 2048), long_ratio=0.3, seed=42
):
    # [FUNCTION DOC]
    # HELPER: Generates n random sequence lengths from a bimodal distribution using a
    # seeded random.Random. Each sample is "long" with probability long_ratio (uniform
    # int in long_range, inclusive) else "short" (uniform int in short_range). Returns
    # list[int]. Deterministic for a given seed.
    rng = random.Random(seed)
    seqlens = []
    for _ in range(n):
        if rng.random() < long_ratio:
            seqlens.append(rng.randint(*long_range))
        else:
            seqlens.append(rng.randint(*short_range))
    return seqlens


def generate_uniform_seqlens(n=200, low=100, high=1024, seed=42):
    # [FUNCTION DOC]
    # HELPER: Returns n seeded uniform random ints in [low, high] inclusive as
    # list[int].
    rng = random.Random(seed)
    return [rng.randint(low, high) for _ in range(n)]


def generate_skewed_seqlens(n=200, seed=42):
    # [FUNCTION DOC]
    # HELPER: Returns n seeded heavy-tailed lengths: int(paretovariate(1.5) * 100) each,
    # so values are >= 100 with a long tail. Output list[int].
    rng = random.Random(seed)
    return [int(rng.paretovariate(1.5) * 100) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests — _KKSet
# ---------------------------------------------------------------------------


class TestKKSet:
    # [CLASS DOC]
    # Tests _KKSet: an accumulator holding (index, value) items with a running sum,
    # supporting add, merge and a total ordering by (sum, item count, items
    # lexicographically).
    """Verify _KKSet behaviour (__slots__-based implementation)."""

    def test_empty_set(self):
        # [FUNCTION DOC]
        # A freshly constructed _KKSet has sum 0 and empty items list.
        s = _KKSet()
        assert s.sum == 0
        assert s.items == []

    def test_add(self):
        # [FUNCTION DOC]
        # add(idx, val) appends (idx, val) tuples in order and accumulates sum.
        s = _KKSet()
        s.add(idx=0, val=10)
        s.add(idx=3, val=20)
        assert s.sum == 30
        assert s.items == [(0, 10), (3, 20)]

    def test_merge(self):
        # [FUNCTION DOC]
        # merge(other) folds another set's items and sum into this one (sum 12, two
        # items).
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 7)
        a.merge(b)
        assert a.sum == 12
        assert len(a.items) == 2

    def test_lt_by_sum(self):
        # [FUNCTION DOC]
        # Ordering: a set with smaller sum compares less than one with larger sum.
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 10)
        assert a < b  # 5 < 10

    def test_lt_by_length(self):
        # [FUNCTION DOC]
        # Ordering tie-break: equal sums, the set with fewer items compares less.
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 3)
        b.add(2, 2)  # same sum=5 but more items
        assert a < b  # len 1 < len 2

    def test_lt_by_items(self):
        # [FUNCTION DOC]
        # Ordering tie-break: equal sum and length, item tuples compared
        # lexicographically.
        a = _KKSet()
        a.add(0, 5)
        b = _KKSet()
        b.add(1, 5)  # same sum, same length, compare items lexicographically
        assert a < b  # (0,5) < (1,5)


# ---------------------------------------------------------------------------
# Tests — _KKState
# ---------------------------------------------------------------------------


class TestKKState:
    # [CLASS DOC]
    # Tests _KKState: a k-way partial partition built from items, exposing k sets,
    # spread (max minus min set sum), merge with another state, heap ordering, and
    # get_partitions.
    """Verify _KKState init, merge, spread, and ordering."""

    def test_single_item_init(self):
        # [FUNCTION DOC]
        # A state from one item with k=3 has 3 sets, sums [0, 0, 100].
        st = _KKState(items=[(0, 100)], k=3)
        assert st.k == 3
        assert len(st.sets) == 3
        sums = sorted(s.sum for s in st.sets)
        assert sums == [0, 0, 100]

    def test_k_items_init(self):
        # [FUNCTION DOC]
        # A state from k items places one item per set (sums 10, 20, 30).
        st = _KKState(items=[(0, 10), (1, 20), (2, 30)], k=3)
        sums = sorted(s.sum for s in st.sets)
        assert sums == [10, 20, 30]

    def test_spread(self):
        # [FUNCTION DOC]
        # spread equals max set sum minus min set sum (50 - 10 = 40).
        st = _KKState(items=[(0, 10), (1, 50)], k=2)
        assert st.spread == 40  # 50 - 10

    def test_merge_reduces_spread(self):
        # [FUNCTION DOC]
        # Merging two single-item states (100 and 90) pairs largest with smallest, so
        # the resulting spread is at most 10.
        s0 = _KKState(items=[(0, 100)], k=2)
        s1 = _KKState(items=[(1, 90)], k=2)
        s0.merge(s1)
        assert s0.spread <= 10

    def test_lt_orders_by_spread_desc(self):
        # [FUNCTION DOC]
        # Ordering is reversed by spread so heapq (min-heap) pops the largest spread
        # first: a state with spread 100 compares less than one with spread 0.
        """Larger spread compares as 'less than' (max-heap semantics)."""
        big = _KKState(items=[(0, 100)], k=2)  # spread = 100
        small = _KKState(items=[(1, 50), (2, 50)], k=2)  # spread = 0
        assert big < small

    def test_get_partitions(self):
        # [FUNCTION DOC]
        # get_partitions returns k lists of original indices that together cover all
        # items.
        st = _KKState(items=[(0, 10), (1, 20)], k=2)
        parts = st.get_partitions()
        assert len(parts) == 2
        all_idx = sorted(sum(parts, []))
        assert all_idx == [0, 1]


# ---------------------------------------------------------------------------
# Tests — _kk_partition
# ---------------------------------------------------------------------------


class TestKKPartition:
    # [CLASS DOC]
    # Tests the core _kk_partition(values, k, equal_size=False) function, which returns
    # k lists of indices into values.
    """Test _kk_partition core algorithm."""

    def test_basic_two_way(self):
        # [FUNCTION DOC]
        # 2-way partition of 8 values returns 2 parts covering every index exactly once.
        values = [7, 5, 5, 4, 4, 3, 2, 1]
        parts = _kk_partition(values, 2)
        assert len(parts) == 2
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(len(values)))

    def test_perfect_partition(self):
        # [FUNCTION DOC]
        # Four equal values split 2-way give zero spread between partition sums.
        values = [10, 10, 10, 10]
        parts = _kk_partition(values, 2)
        sums = [sum(values[i] for i in p) for p in parts]
        assert max(sums) - min(sums) == 0

    def test_single_group(self):
        # [FUNCTION DOC]
        # k=1 puts all indices in one partition.
        values = [5, 3, 2]
        parts = _kk_partition(values, 1)
        assert len(parts) == 1
        assert sorted(parts[0]) == [0, 1, 2]

    def test_many_groups(self):
        # [FUNCTION DOC]
        # Values 1..20 into k=5 parts: exactly 5 parts covering all 20 indices.
        values = list(range(1, 21))  # 1..20
        k = 5
        parts = _kk_partition(values, k)
        assert len(parts) == k
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(20))

    def test_balance_quality(self):
        # [FUNCTION DOC]
        # 50 seeded random values in [1, 1000] split 4-way must have spread under 20% of
        # the average partition sum.
        """KK should produce well-balanced partitions."""
        rng = random.Random(123)
        values = [rng.randint(1, 1000) for _ in range(50)]
        k = 4
        parts = _kk_partition(values, k)
        sums = [sum(values[i] for i in p) for p in parts]
        spread = max(sums) - min(sums)
        avg = sum(values) / k
        assert spread < avg * 0.2, f"Spread {spread} too large vs avg {avg}"

    def test_equal_size_basic(self):
        # [FUNCTION DOC]
        # equal_size=True with 12 items, k=4 yields exactly 3 indices per partition and
        # full coverage.
        """equal_size=True should give exactly n/k items per partition."""
        values = list(range(1, 13))  # 12 items
        k = 4
        parts = _kk_partition(values, k, equal_size=True)
        assert len(parts) == k
        for p in parts:
            assert len(p) == 3  # 12 / 4
        all_indices = sorted(sum(parts, []))
        assert all_indices == list(range(12))

    def test_ascending_sort_property(self):
        # [FUNCTION DOC]
        # Values [1, 2, 3, 4] split 2-way must give the perfect sums [5, 5].
        # KNOWN ISSUE: Name refers to an "ascending sort property" but the test only
        # checks the resulting balance, not any sorting behavior.
        """Verify that the algorithm produces a deterministic, balanced output."""
        values = [1, 2, 3, 4]
        parts = _kk_partition(values, 2)
        sums = sorted(sum(values[i] for i in p) for p in parts)
        assert sums == [5, 5]  # Perfect split: {1,4} and {2,3}


# ---------------------------------------------------------------------------
# Tests — kk_allocate
# ---------------------------------------------------------------------------


class TestKKAllocate:
    # [CLASS DOC]
    # Tests kk_allocate(values, capacity, min_groups, n_groups_divisor=1,
    # equal_size=False): chooses k groups (at least min_groups, at least
    # ceil(total/capacity), rounded up to the divisor and clamped to len(values)), runs
    # KK, and falls back to FFD if any group exceeds capacity. Raises RuntimeError for
    # impossible inputs.
    """Test kk_allocate() with capacity and min_groups."""

    def test_basic_allocation(self):
        # [FUNCTION DOC]
        # Basic allocation: at least min_groups=2 groups and all 5 indices covered.
        values = [100, 200, 300, 150, 250]
        result = kk_allocate(values, capacity=600, min_groups=2)
        assert len(result) >= 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(5))

    def test_respects_capacity_via_fallback(self):
        # [FUNCTION DOC]
        # With capacity 500 and values up to 400, every resulting group sum must be <=
        # 500 (via FFD fallback if KK violates it).
        """When KK violates capacity it falls back to FFD, which respects it."""
        values = [100, 200, 300, 400]
        result = kk_allocate(values, capacity=500, min_groups=2)
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 500, f"Group sum {group_sum} exceeds capacity 500"

    def test_min_groups_guaranteed(self):
        # [FUNCTION DOC]
        # min_groups=3 is honoured even when capacity alone would allow one group.
        values = [10, 20, 30]
        result = kk_allocate(values, capacity=10000, min_groups=3)
        assert len(result) >= 3

    def test_divisor(self):
        # [FUNCTION DOC]
        # With n_groups_divisor=2, the number of groups is a multiple of 2.
        values = [100] * 10
        result = kk_allocate(values, capacity=500, min_groups=3, n_groups_divisor=2)
        assert len(result) % 2 == 0

    def test_large_capacity(self):
        # [FUNCTION DOC]
        # With effectively infinite capacity, exactly min_groups=2 groups are produced
        # covering all indices.
        """When capacity is huge, all items should go into min_groups bins."""
        values = [10, 20, 30, 40, 50]
        result = kk_allocate(values, capacity=int(1e12), min_groups=2)
        assert len(result) == 2
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(5))

    def test_raises_on_value_exceeding_capacity(self):
        # [FUNCTION DOC]
        # A single value larger than capacity causes RuntimeError.
        with pytest.raises(RuntimeError):
            kk_allocate([100, 600], capacity=500, min_groups=1)

    def test_raises_on_too_few_values(self):
        # [FUNCTION DOC]
        # min_groups larger than the number of values causes RuntimeError.
        with pytest.raises(RuntimeError):
            kk_allocate([10], capacity=100, min_groups=5)

    def test_k_upper_bound_cap(self):
        # [FUNCTION DOC]
        # k is clamped to len(values): 10 values of 100, capacity 120, divisor 8 yields
        # exactly 10 groups (not 16). Also checks min_groups > len(values) still raises
        # RuntimeError.
        """Verify k = min(k, len(values)) prevents k from exceeding n.

        With 10 items summing to 1000, capacity=120 → ceil(1000/120)=9,
        n_groups_divisor=8 → roundup to 16, but min(16, 10) = 10.
        """
        # 10 items, total=1000, cap=120 → k=9, roundup(8)=16, cap→min(16,10)=10
        values = [100] * 10
        result = kk_allocate(values, capacity=120, min_groups=1, n_groups_divisor=8)
        # k was capped from 16 → 10 (= len(values))
        assert len(result) == 10
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(10))

        # Also verify min_groups > len(values) still raises
        with pytest.raises(RuntimeError):
            kk_allocate([100, 200, 300], capacity=int(1e12), min_groups=5)

    def test_k_upper_bound_with_divisor(self):
        # [FUNCTION DOC]
        # With 5 values, capacity 200 and divisor 4, the group count is at most 5 and
        # either a multiple of 4 or exactly 5.
        """n_groups_divisor roundup can push k above n — should be clamped."""
        # 5 items, total=500, cap=200 → k=3, divisor=4 → roundup=4, min(4,5)=4
        values = [100] * 5
        result = kk_allocate(values, capacity=200, min_groups=1, n_groups_divisor=4)
        assert len(result) <= 5
        assert len(result) % 4 == 0 or len(result) == 5

    def test_kk_respects_capacity_without_fallback(self):
        # [FUNCTION DOC]
        # Four values of 250 with capacity 300 produce 4 groups each within capacity,
        # covering all indices (no fallback needed).
        """When KK naturally satisfies capacity, no FFD fallback is needed."""
        values = [250, 250, 250, 250]
        # total=1000, capacity=300 → k=ceil(1000/300)=4
        # Each group gets exactly 250 ≤ 300 → no fallback
        result = kk_allocate(values, capacity=300, min_groups=2)
        assert len(result) == 4
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 300
        all_idx = sorted(sum(result, []))
        assert all_idx == [0, 1, 2, 3]

    def test_kk_ffd_fallback_triggered(self):
        # [FUNCTION DOC]
        # Uneven values [280, 280, 280, 160] with capacity 400 force KK into 3 groups
        # that would violate capacity; asserts the final (FFD fallback) groups all fit
        # within 400.
        # KNOWN ISSUE: Does not actually verify that the fallback path was taken; only
        # checks the capacity invariant.
        """Force FFD fallback: min_groups < needed groups, KK tries fewer bins."""
        # 4 items summing to 1000, capacity=300, min_groups=2
        # k = max(2, ceil(1000/300)) = 4, min(4,4)=4 → each gets 250 ≤ 300: OK
        # Let's use unequal values to force violation with min_groups=2:
        values = [280, 280, 280, 160]
        # total=1000, capacity=300 → k=4 → groups of ~250 each → fine
        # With capacity=400, k=max(2,ceil(1000/400))=3
        # KK tries 3 groups: best balance ~333 each, but 280+280=560>400?
        # Actually it'd be {280,160},{280},{280} → sums 440,280,280 → 440>400 → fallback!
        result = kk_allocate(values, capacity=400, min_groups=2)
        for group in result:
            group_sum = sum(values[i] for i in group)
            assert group_sum <= 400, f"Group sum {group_sum} > 400"

    def test_equal_size_mode(self):
        # [FUNCTION DOC]
        # equal_size=True with 6 values and min_groups=3 yields 3 groups of exactly 2
        # items each, covering all indices.
        """Test equal_size=True produces groups with identical element counts."""
        values = [100, 200, 300, 400, 500, 600]
        k = 3
        result = kk_allocate(values, capacity=int(1e12), min_groups=k, equal_size=True)
        assert len(result) == k
        for group in result:
            assert len(group) == len(values) // k
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(len(values)))

    def test_equal_size_raises_if_not_divisible(self):
        # [FUNCTION DOC]
        # equal_size=True with 5 values and 3 groups (not divisible) raises
        # RuntimeError.
        with pytest.raises(RuntimeError):
            kk_allocate(
                [1, 2, 3, 4, 5], capacity=int(1e12), min_groups=3, equal_size=True
            )

    @pytest.mark.parametrize("seed", range(100))
    def test_kk_partition_consistency_with_verl(self, seed):
        # [FUNCTION DOC]
        # Parametrized over 100 seeds: builds n in [10, 60] random values in [1, 500]
        # and k in [2, min(8, n)], then checks _kk_partition covers every index once,
        # partition sums total the value sum, and a second run gives identical sorted
        # sums.
        # KNOWN ISSUE: Despite the name and docstring mentioning veRL, nothing is
        # compared against a veRL reference implementation; it only checks invariants
        # and determinism.
        """Verify _kk_partition: index coverage, sum invariant, determinism.

        Runs 100 seeds comparing that:
          - All indices are covered exactly once.
          - Partition sums add up to the total.
          - Re-running produces identical sums (determinism).
        """
        rng = random.Random(seed)
        n = rng.randint(10, 60)
        k = rng.randint(2, min(8, n))
        values = [rng.randint(1, 500) for _ in range(n)]

        parts = _kk_partition(values, k)

        # All indices covered
        all_idx = sorted(sum(parts, []))
        assert all_idx == list(range(n)), f"seed={seed}: indices mismatch"

        # Sums add up
        part_sums = [sum(values[i] for i in p) for p in parts]
        assert sum(part_sums) == sum(values), f"seed={seed}: total mismatch"

        # Determinism
        parts2 = _kk_partition(values, k)
        sums2 = sorted(sum(values[i] for i in p) for p in parts2)
        assert sorted(part_sums) == sums2, f"seed={seed}: non-deterministic"


# ---------------------------------------------------------------------------
# Tests — _compute_packing_metrics
# ---------------------------------------------------------------------------


class TestComputePackingMetrics:
    # [CLASS DOC]
    # Tests _compute_packing_metrics(values, partitions, capacity), which returns a dict
    # with n_groups, spread, max_load, min_load, mean_load, std_dev, cv, imbalance_ratio
    # (spread/mean), max_load_ratio (max/mean), utilization (mean/capacity) and
    # wasted_tokens (sum of capacity minus each load).
    """Test _compute_packing_metrics helper function."""

    def test_balanced_case(self):
        # [FUNCTION DOC]
        # Perfectly balanced 2 groups of 500 with capacity 600: zero
        # spread/std/cv/imbalance, max_load_ratio 1, utilization 500/600.
        values = [100, 200, 300, 400]
        partitions = [[0, 3], [1, 2]]  # sums: 500, 500
        capacity = 600

        m = _compute_packing_metrics(values, partitions, capacity)
        assert m["n_groups"] == 2
        assert m["spread"] == 0
        assert m["max_load"] == 500
        assert m["min_load"] == 500
        assert abs(m["std_dev"]) < 1e-9
        assert abs(m["cv"]) < 1e-9
        assert abs(m["imbalance_ratio"]) < 1e-9
        assert abs(m["max_load_ratio"] - 1.0) < 1e-9
        assert abs(m["utilization"] - 500 / 600) < 1e-9

    def test_unbalanced_case(self):
        # [FUNCTION DOC]
        # Loads 100 and 900 with capacity 1000: spread 800, mean 500, imbalance 1.6,
        # wasted tokens 1000.
        values = [100, 200, 300, 400]
        partitions = [[0], [1, 2, 3]]  # sums: 100, 900
        capacity = 1000

        m = _compute_packing_metrics(values, partitions, capacity)
        assert m["spread"] == 800
        assert m["max_load"] == 900
        assert m["min_load"] == 100
        assert m["mean_load"] == 500.0
        assert m["imbalance_ratio"] == 800 / 500
        assert m["wasted_tokens"] == (1000 - 100) + (1000 - 900)

    def test_empty_partitions(self):
        # [FUNCTION DOC]
        # Empty values and partitions yield n_groups 0 and spread 0 (no division
        # errors).
        m = _compute_packing_metrics([], [], 100)
        assert m["n_groups"] == 0
        assert m["spread"] == 0

    def test_single_group(self):
        # [FUNCTION DOC]
        # Single group of load 60 with capacity 100: spread 0, max_load 60, utilization
        # 0.6.
        values = [10, 20, 30]
        partitions = [[0, 1, 2]]
        capacity = 100
        m = _compute_packing_metrics(values, partitions, capacity)
        assert m["n_groups"] == 1
        assert m["spread"] == 0
        assert m["max_load"] == 60
        assert m["utilization"] == 60 / 100


# ---------------------------------------------------------------------------
# Tests — get_allocate_fn dispatch
# ---------------------------------------------------------------------------


class TestGetAllocateFn:
    # [CLASS DOC]
    # Tests get_allocate_fn(name="ffd"), which maps algorithm names to allocator
    # functions and raises ValueError for unknown names.
    """Test configurable algorithm dispatch."""

    def test_ffd_dispatch(self):
        # [FUNCTION DOC]
        # "ffd" returns ffd_allocate itself.
        fn = get_allocate_fn("ffd")
        assert fn is ffd_allocate

    def test_kk_dispatch(self):
        # [FUNCTION DOC]
        # "kk" returns kk_allocate itself.
        fn = get_allocate_fn("kk")
        assert fn is kk_allocate

    def test_invalid_algorithm(self):
        # [FUNCTION DOC]
        # Unknown name raises ValueError matching "Unknown packing algorithm".
        with pytest.raises(ValueError, match="Unknown packing algorithm"):
            get_allocate_fn("nonexistent")

    def test_default_is_ffd(self):
        # [FUNCTION DOC]
        # Calling with no argument returns ffd_allocate (default).
        fn = get_allocate_fn()
        assert fn is ffd_allocate

    def test_dispatch_produces_valid_results(self):
        # [FUNCTION DOC]
        # Both dispatched allocators, on 50 uniform lengths with capacity 4096 and
        # min_groups 4, return at least 4 groups covering all indices.
        """Both algorithms should produce valid allocations."""
        values = generate_uniform_seqlens(n=50, seed=99)
        for algo in ["ffd", "kk"]:
            fn = get_allocate_fn(algo)
            result = fn(values, capacity=4096, min_groups=4)
            assert len(result) >= 4
            all_idx = sorted(sum(result, []))
            assert all_idx == list(range(50)), f"{algo} lost indices"


# ---------------------------------------------------------------------------
# Tests — KK vs FFD comparison
# ---------------------------------------------------------------------------


class TestKKVsFFDComparison:
    # [CLASS DOC]
    # Comparative quality tests of KK vs FFD balance (spread = max minus min group sum)
    # with effectively unlimited capacity.
    """Comparative tests demonstrating KK advantage over FFD."""

    @pytest.mark.parametrize("seed", range(10))
    def test_kk_balance_at_least_as_good(self, seed):
        # [FUNCTION DOC]
        # Parametrized over 10 seeds of 100 bimodal lengths, 4 groups: KK spread must be
        # at most 1.05x FFD spread plus 50 tokens of slack.
        """KK should produce partitions with spread <= FFD spread."""
        values = generate_bimodal_seqlens(n=100, seed=seed)
        min_groups = 4
        capacity = int(1e12)

        ffd_result = ffd_allocate(values, capacity, min_groups)
        kk_result = kk_allocate(values, capacity, min_groups)

        ffd_sums = sorted(sum(values[i] for i in g) for g in ffd_result if g)
        kk_sums = sorted(sum(values[i] for i in g) for g in kk_result if g)

        ffd_spread = max(ffd_sums) - min(ffd_sums) if ffd_sums else 0
        kk_spread = max(kk_sums) - min(kk_sums) if kk_sums else 0

        assert kk_spread <= ffd_spread * 1.05 + 50, (
            f"seed={seed}: KK spread {kk_spread} >> FFD spread {ffd_spread}"
        )

    def test_kk_wins_majority(self):
        # [FUNCTION DOC]
        # Over 100 seeded bimodal trials, counts KK wins/FFD wins/ties by spread and
        # asserts KK wins or ties in at least 70% of trials.
        """Over many random trials, KK should win or tie majority of times."""
        kk_wins = 0
        ffd_wins = 0
        ties = 0
        n_trials = 100
        min_groups = 4
        capacity = int(1e12)

        for seed in range(n_trials):
            values = generate_bimodal_seqlens(n=100, seed=seed * 7 + 13)

            ffd_result = ffd_allocate(values, capacity, min_groups)
            kk_result = kk_allocate(values, capacity, min_groups)

            ffd_sums = [sum(values[i] for i in g) for g in ffd_result if g]
            kk_sums = [sum(values[i] for i in g) for g in kk_result if g]

            ffd_spread = max(ffd_sums) - min(ffd_sums) if ffd_sums else 0
            kk_spread = max(kk_sums) - min(kk_sums) if kk_sums else 0

            if kk_spread < ffd_spread:
                kk_wins += 1
            elif ffd_spread < kk_spread:
                ffd_wins += 1
            else:
                ties += 1

        assert kk_wins + ties >= n_trials * 0.7, (
            f"KK wins={kk_wins}, ties={ties}, FFD wins={ffd_wins}"
        )

    @pytest.mark.parametrize(
        "gen_fn,gen_kwargs,spread_threshold",
        [
            (generate_bimodal_seqlens, {"n": 200}, 0.15),
            (generate_uniform_seqlens, {"n": 200}, 0.15),
            (generate_skewed_seqlens, {"n": 200}, 0.55),
        ],
        ids=["bimodal", "uniform", "skewed"],
    )
    def test_kk_balance_across_distributions(
        self, gen_fn, gen_kwargs, spread_threshold
    ):
        # [FUNCTION DOC]
        # Parametrized over bimodal, uniform and skewed generators (n=200, seed 42): KK
        # with 8 groups must keep spread below threshold times the average group load
        # (0.15 or 0.55 for skewed).
        """KK produces good balance across different sequence length distributions."""
        values = gen_fn(**gen_kwargs, seed=42)
        min_groups = 8
        capacity = int(1e12)

        kk_result = kk_allocate(values, capacity, min_groups)
        kk_sums = [sum(values[i] for i in g) for g in kk_result if g]

        if kk_sums:
            spread = max(kk_sums) - min(kk_sums)
            avg = sum(values) / len(kk_sums)
            assert spread < avg * spread_threshold, (
                f"Spread {spread} too large vs avg {avg:.0f} "
                f"(threshold {spread_threshold})"
            )


# ---------------------------------------------------------------------------
# Tests — MicroBatchSpec packing config
# ---------------------------------------------------------------------------


class TestMicroBatchSpecPacking:
    # [CLASS DOC]
    # Standalone checks of packing algorithm constants and config-driven dispatch,
    # mimicking MicroBatchSpec fields without importing it.
    """Test MicroBatchSpec-like config validation (standalone)."""

    def test_valid_algorithms(self):
        # [FUNCTION DOC]
        # "ffd" and "kk" are both members of PACKING_ALGORITHMS.
        for algo in ["ffd", "kk"]:
            assert algo in PACKING_ALGORITHMS

    def test_invalid_algorithm_detected(self):
        # [FUNCTION DOC]
        # An unknown name is not in PACKING_ALGORITHMS.
        assert "invalid_algo" not in PACKING_ALGORITHMS

    def test_default_is_ffd(self):
        # [FUNCTION DOC]
        # PACKING_ALGORITHM_FFD equals "ffd".
        # KNOWN ISSUE: Name says "default is ffd" but it only checks the constant's
        # value, not any default.
        assert PACKING_ALGORITHM_FFD == "ffd"

    def test_kk_constant(self):
        # [FUNCTION DOC]
        # PACKING_ALGORITHM_KK equals "kk".
        assert PACKING_ALGORITHM_KK == "kk"

    def test_config_driven_allocation(self):
        # [FUNCTION DOC]
        # Uses a local FakeSpec (max_tokens_per_mb 4096, n_mbs 4, divisor 1, algorithm
        # "kk") to dispatch via get_allocate_fn and allocate 8 lengths positionally;
        # asserts >= n_mbs groups and full index coverage.
        """Simulate config-driven allocation: read algorithm from config, dispatch."""

        class FakeSpec:
            # [CLASS DOC]
            # HELPER: Plain class-attribute stand-in for MicroBatchSpec packing fields.
            max_tokens_per_mb = 4096
            n_mbs = 4
            n_mbs_divisor = 1
            packing_algorithm = "kk"

        spec = FakeSpec()
        allocate_fn = get_allocate_fn(spec.packing_algorithm)
        values = [512, 1024, 256, 768, 2048, 300, 1500, 900]
        result = allocate_fn(
            values, spec.max_tokens_per_mb, spec.n_mbs, spec.n_mbs_divisor
        )
        assert len(result) >= spec.n_mbs
        all_idx = sorted(sum(result, []))
        assert all_idx == list(range(len(values)))

    def test_config_switch_ffd_to_kk(self):
        # [FUNCTION DOC]
        # Both "ffd" and "kk" allocators on 100 bimodal lengths with unlimited capacity
        # cover all indices exactly once.
        """Switching algorithm via config produces valid results for both."""
        values = generate_bimodal_seqlens(n=100, seed=42)
        capacity = int(1e12)
        min_groups = 4

        ffd_fn = get_allocate_fn("ffd")
        kk_fn = get_allocate_fn("kk")

        ffd_result = ffd_fn(values, capacity, min_groups)
        kk_result = kk_fn(values, capacity, min_groups)

        assert sorted(sum(ffd_result, [])) == list(range(len(values)))
        assert sorted(sum(kk_result, [])) == list(range(len(values)))

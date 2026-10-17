# [MODULE DOC]
# Sequence packing / load-balancing algorithms used to split variable-length
# sequences into micro-batches or across workers: contiguous balanced partitioning
# (numba DP), balanced batch reordering, First-Fit-Decreasing (FFD) and
# Karmarkar-Karp (KK, largest differencing) bin packing with a name registry
# ("ffd", "kk"), packing quality metrics and an equal-count greedy partition. Pure
# CPU computations; running the module directly benchmarks ffd_allocate on random
# data and prints results.
# SPDX-License-Identifier: Apache-2.0

import bisect
import heapq
import itertools
import math
import time
from typing import Any

import numba
import numpy as np

from areal.utils import logging

logger = logging.getLogger("SeqPack")


def flat2d(arr: list[list[Any]]) -> list[Any]:
    # [FUNCTION DOC]
    # HELPER: Flattens a list of lists into one list (one level).
    return list(itertools.chain(*arr))


@numba.njit
def partition_balanced(nums: np.ndarray, k: int, min_size: int = 1):
    # [FUNCTION DOC]
    # Numba-JIT dynamic program splitting nums into k contiguous subarrays (each at
    # least min_size long) minimizing the spread between the largest and smallest
    # subarray sums. dp[i, j] holds the best spread for the first i items in j parts,
    # with running max/min part sums and the chosen split point; the answer is
    # reconstructed by backtracking from n. Returns k + 1 boundary indices starting at
    # 0 and ending at len(nums). Integer sums (int64); O(k * n^2).
    """Partition an array into k subarrays with a minimum absolute difference
    of sums and minimum subarray size.

    Dynamic programming solution.

    Args:
        nums (np.ndarray): The array to be partitioned.
        k (int): Number of partitions.
        min_size (int): Minimum size of each subarray.

    Returns:
        List[int]: Partition slicing point indices in a list including start and end points.
                   Length equals to k + 1.
    """
    n = len(nums)

    dp = np.full((n + 1, k + 1), dtype=np.int64, fill_value=int(1e10))
    maxval = np.full((n + 1, k + 1), dtype=np.int64, fill_value=-int(1e10))
    minval = np.full((n + 1, k + 1), dtype=np.int64, fill_value=int(1e10))
    prefix_sums = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(nums)), axis=0)
    split = np.zeros((n + 1, k + 1), dtype=np.int64)

    for i in range(n + 1):
        dp[i, 1] = 0
        maxval[i, 1] = prefix_sums[i] - prefix_sums[0]
        minval[i, 1] = prefix_sums[i] - prefix_sums[0]

    for j in range(2, k + 1):
        for i in range(j * min_size, n + 1):
            for x in range(min_size, i - min_size + 1):
                xx = prefix_sums[i] - prefix_sums[x]
                min_diff = max(
                    dp[x, j - 1], maxval[x, j - 1] - xx, xx - minval[x, j - 1]
                )
                dp[i, j] = min(dp[i, j], min_diff)

                if dp[i, j] == min_diff:
                    split[i][j] = x
                    if dp[i, j] == maxval[x, j - 1] - xx:
                        maxval[i, j] = maxval[x, j - 1]
                        minval[i, j] = xx
                    elif dp[i, j] == xx - minval[x, j - 1]:
                        maxval[i, j] = xx
                        minval[i, j] = minval[x, j - 1]
                    else:
                        maxval[i, j] = maxval[x, j - 1]
                        minval[i, j] = minval[x, j - 1]
    res = [n]
    idx = n
    for i in range(k, 0, -1):
        idx = split[idx][i]
        res.append(idx)
    return res[::-1]


def partition_balanced_tuples(
    nums: np.ndarray, k: int, min_size: int = 1
) -> list[tuple[int, int]]:
    # [FUNCTION DOC]
    # Runs partition_balanced and returns the k (start, end) index pairs.
    lst = partition_balanced(nums, k, min_size)
    return [(lst[i], lst[i + 1]) for i in range(k)]


def min_abs_diff_partition(
    arr: np.ndarray | list, k: int, min_size: int = 1
) -> list[tuple[int, int]]:
    # [FUNCTION DOC]
    # Validated wrapper around partition_balanced_tuples: converts lists to arrays,
    # requires a 1-D array with length >= k and >= k * min_size (ValueError otherwise),
    # then checks the result is contiguous and has no empty parts (ValueError with a
    # hint about dataset/batch-size bugs). Returns the (start, end) pairs.
    err_hint = (
        " Errors should not be reported in this function. It is probably a bug in the dataset code"
        " or too small batch size with pipeline parallelism."
    )

    if isinstance(arr, list):
        arr = np.array(arr)
    if len(arr.shape) > 1:
        raise ValueError(f"The array to be partitioned must be 1D. ({arr})" + err_hint)
    if len(arr) < k:
        raise ValueError(
            f"The array to be partitioned must have length >= k. (array {arr}, k={k})"
            + err_hint
        )
    if len(arr) < k * min_size:
        raise ValueError(
            f"Length of the array to be partitioned must be at least k * min_size ({k} * {min_size}), current length {len(arr)}."
        )
    partitions = partition_balanced_tuples(arr, k, min_size)
    last_end = 0

    err_type = None
    err_msg = f"Lengths to be partitioned: {arr}, k={k}, current partition result {partitions}."
    for start, end in partitions:
        if start != last_end:
            err_type = "not contiguous"
        if end <= start:
            err_type = "empty"
        if err_type:
            raise ValueError(
                f"Partition {start}-{end} is {err_type}. " + err_msg + err_hint
            )
        last_end = end
    return partitions


# @numba.njit
def reorder_to_balanced_batches(
    seqlens: np.ndarray,
    n_seqs_per_batch: int,
) -> tuple[np.ndarray, int]:
    # [FUNCTION DOC]
    # Greedily assigns sequences (longest first) to ceil(n / n_seqs_per_batch) bins,
    # each time choosing the non-full bin with the smallest total length. Returns
    # (indices concatenated bin by bin in descending bin-total order, maximum pairwise
    # difference of bin totals). Consecutive slices of n_seqs_per_batch indices thus
    # form balanced batches.
    max_bins = (len(seqlens) + n_seqs_per_batch - 1) // n_seqs_per_batch

    bins = [[] for _ in range(max_bins)]
    bin_sizes = np.zeros(max_bins, dtype=np.int32)
    bin_seqlens = np.zeros(max_bins, dtype=np.int32)
    for i in seqlens.argsort()[::-1]:
        idx = np.where(
            bin_sizes + 1 <= n_seqs_per_batch,
            bin_seqlens,
            np.iinfo(np.int32).max,
        ).argmin()
        bins[idx].append(i)
        bin_sizes[idx] += 1
        bin_seqlens[idx] += seqlens[i]

    assert np.all(bin_sizes <= n_seqs_per_batch), (bin_sizes, n_seqs_per_batch)
    max_diff = 0
    for i in range(max_bins):
        for j in range(i + 1, max_bins):
            max_diff = max(max_diff, abs(bin_seqlens[i] - bin_seqlens[j]))

    reordered_indices = []
    for i in bin_seqlens.argsort()[::-1]:
        reordered_indices.extend(bins[i])
    return np.array(reordered_indices), max_diff


# =============================================================================
# Packing Algorithm Registry
# =============================================================================

# Supported packing algorithm names (used in MicroBatchSpec.packing_algorithm)
PACKING_ALGORITHM_FFD = "ffd"
PACKING_ALGORITHM_KK = "kk"
PACKING_ALGORITHMS = {PACKING_ALGORITHM_FFD, PACKING_ALGORITHM_KK}


def get_allocate_fn(algorithm: str = PACKING_ALGORITHM_FFD):
    # [FUNCTION DOC]
    # Maps an algorithm name ("ffd" default or "kk") to ffd_allocate / kk_allocate;
    # raises ValueError for unknown names.
    """Return the allocation function for the given algorithm name.

    Args:
        algorithm: One of ``"ffd"`` or ``"kk"``.

    Returns:
        The corresponding allocation function (``ffd_allocate`` or ``kk_allocate``).

    Raises:
        ValueError: If the algorithm name is not recognized.
    """
    if algorithm == PACKING_ALGORITHM_FFD:
        return ffd_allocate
    elif algorithm == PACKING_ALGORITHM_KK:
        return kk_allocate
    else:
        raise ValueError(
            f"Unknown packing algorithm '{algorithm}'. "
            f"Supported algorithms: {sorted(PACKING_ALGORITHMS)}"
        )


# =============================================================================
# FFD (First Fit Decreasing) Algorithm
# =============================================================================


# @numba.njit
def _ffd_allocate(
    values: np.ndarray, capacity: int, min_groups: int, n_groups_divisor: int = 1
) -> list[list[int]]:
    # [FUNCTION DOC]
    # HELPER: One FFD pass. Processes values in descending order keeping groups sorted
    # by current sum: opens a new group while fewer than min_groups exist or when even
    # the smallest group cannot fit the value; otherwise among groups whose sum plus the
    # value fits capacity it picks the one with the fewest items and adds the value.
    # Returns groups as lists of indices.
    # KNOWN ISSUE: n_groups_divisor is accepted but unused (divisibility is enforced by
    # the caller).
    """A greedy allocation algorithm that partitions a list of numbers
    into k groups, where the summation of each group is less than capacity
    and (k >= min_groups and k % n_groups_divisor == 0). We want to minimize
    k and make partitions as balanced as possible.

    1. Sort the numbers in reverse order.
    2. If the number of groups is less than `min_groups`, create a new group.
    3. For a new number, find all groups with the capacity to hold the new number.
       Put the new number into the group with the smallest size.
    4. Otherwise, create a new group.
    """
    value_indices = np.argsort(-values)
    group_indices: list[list[int]] = []
    group_values: list[tuple[float, int]] = []
    group_cnt = 0
    for idx in value_indices:
        if (
            len(group_values) < min_groups
            or group_values[0][0] + values[idx] > capacity
        ):
            bisect.insort(group_values, (float(values[idx]), group_cnt))
            group_indices.append([idx])
            group_cnt += 1
        else:
            i = bisect.bisect_right(group_values, (capacity - values[idx], len(values)))
            candidates = [group_values[j][1] for j in range(i)]
            lens = [len(group_indices[g]) for g in candidates]
            j = np.argmin(lens)
            v, group_idx = group_values.pop(j)
            assert group_idx == candidates[j]
            bisect.insort(group_values, (float(values[idx] + v), group_idx))
            group_indices[group_idx].append(idx)
    return group_indices


def ffd_allocate(
    values: list[int], capacity: int, min_groups: int, n_groups_divisor: int = 1
) -> list[list[int]]:
    # [FUNCTION DOC]
    # FFD packing satisfying capacity, at least min_groups groups (raised to
    # n_groups_divisor if smaller or None) and a group count divisible by
    # n_groups_divisor. Raises RuntimeError if any value exceeds capacity or there are
    # fewer values than min_groups. Repeats _ffd_allocate, bumping min_groups to the
    # next multiple of the divisor, until the group count is divisible; raises
    # RuntimeError if min_groups exceeds the number of values first. Returns lists of
    # indices.
    if min_groups is None or min_groups < n_groups_divisor:
        min_groups = n_groups_divisor
    if any(v > capacity for v in values):
        raise RuntimeError(f"Values {values} is larger than capacity {capacity}")
    if len(values) < min_groups:
        raise RuntimeError(
            f"Number of values {len(values)} is smaller than min_groups {min_groups}"
        )
    while True:
        res = _ffd_allocate(np.array(values), capacity, min_groups)
        min_groups += n_groups_divisor - min_groups % n_groups_divisor
        if len(res) % n_groups_divisor == 0:
            break
        if len(values) < min_groups:
            raise RuntimeError(
                f"Cannot allocate values {values} that satisfies capacity {capacity}, "
                f"min_groups {min_groups} and n_groups_divisor {n_groups_divisor}."
            )
    return res


# =============================================================================
# Karmarkar-Karp (KK) Algorithm — Largest Differencing Method
# =============================================================================
#
# The KK algorithm produces more balanced partitions than FFD by iteratively
# combining the two most "unbalanced" partial partitions. It is especially
# effective when the goal is to minimise the max-min spread of group sums —
# a common objective for balancing GPU workloads in RL training.
#
# References:
#   - R.E. Korf, "Multi-Way Number Partitioning", IJCAI 2009
# =============================================================================


class _KKSet:
    # [CLASS DOC]
    # HELPER: Mutable set of (index, value) items with a running sum; ordered by sum,
    # then item count, then items lexicographically.
    """A set of items with a running sum, used inside KK partitioning."""

    __slots__ = ("sum", "items")

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Starts empty with sum 0.
        self.sum: int = 0
        self.items: list[tuple[int, int]] = []  # (original_index, value)

    def add(self, idx: int, val: int) -> None:
        # [FUNCTION DOC]
        # Appends one (idx, val) item and adds val to the sum.
        self.items.append((idx, val))
        self.sum += val

    def merge(self, other: "_KKSet") -> None:
        # [FUNCTION DOC]
        # Appends all items of other and adds their values to the sum (other unchanged).
        for idx, val in other.items:
            self.items.append((idx, val))
            self.sum += val

    def __lt__(self, other: "_KKSet") -> bool:
        # [FUNCTION DOC]
        # Ordering by (sum, number of items, items).
        if self.sum != other.sum:
            return self.sum < other.sum
        if len(self.items) != len(other.items):
            return len(self.items) < len(other.items)
        return self.items < other.items


class _KKState:
    # [CLASS DOC]
    # HELPER: One partial k-way partition in the KK priority queue: k _KKSets kept
    # sorted by descending sum.
    """Represents one candidate partitioning state in the KK priority queue.

    Each state contains *k* sets.  Two states are merged by pairing the
    largest set of one with the smallest set of the other, which minimises
    the spread (max_sum − min_sum) across partitions.
    """

    __slots__ = ("k", "sets")

    def __init__(self, items: list[tuple[int, int]], k: int) -> None:
        # [FUNCTION DOC]
        # Creates k empty sets, places the given items (exactly 1 or k items, asserted)
        # one per set, and sorts sets descending.
        self.k = k
        self.sets = [_KKSet() for _ in range(k)]
        assert len(items) in (1, k), f"{len(items)} not in [1, {k}]"
        for i, (idx, val) in enumerate(items):
            self.sets[i].add(idx=idx, val=val)
        self.sets.sort(reverse=True)

    def get_partitions(self) -> list[list[int]]:
        # [FUNCTION DOC]
        # Returns the indices of each set, in the current (descending-sum) set order.
        return [[idx for idx, _ in s.items] for s in self.sets]

    def merge(self, other: "_KKState") -> None:
        # [FUNCTION DOC]
        # Merges other into self by pairing self's i-th largest set with other's i-th
        # smallest set, then re-sorts. Mutates self (and shares item lists' contents).
        # Pair the largest set of *self* with the smallest set of *other*
        # and vice-versa, so that the combined spread is minimised.
        for i in range(self.k):
            self.sets[i].merge(other.sets[self.k - 1 - i])
        self.sets.sort(reverse=True)

    @property
    def spread(self) -> int:
        # [FUNCTION DOC]
        # Property: largest set sum minus smallest set sum.
        return self.sets[0].sum - self.sets[-1].sum

    def __lt__(self, other: "_KKState") -> bool:
        # [FUNCTION DOC]
        # Inverted ordering so heapq pops the state with the largest spread first (ties:
        # larger top set first).
        # Max-heap by spread: largest spread is popped first.
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]


def _kk_partition(
    values: list[int], k: int, equal_size: bool = False
) -> list[list[int]]:
    # [FUNCTION DOC]
    # HELPER: Largest differencing method. Seeds the heap with one state per value
    # (or, if equal_size, one state per consecutive run of k ascending-sorted values,
    # asserting len(values) divisible by k), then repeatedly pops the two largest-spread
    # states, merges them and pushes the result until one remains. Returns its k index
    # partitions.
    """Core KK partitioning using the Largest Differencing Method.

    Args:
        values: List of integer values to partition.
        k: Number of partitions to create.
        equal_size: If True, each partition will have exactly
            ``len(values) // k`` items.

    Returns:
        List of k partitions, each containing indices into *values*.
    """
    sorted_items = sorted(
        [(val, idx) for idx, val in enumerate(values)]
    )  # ascending by value

    states_pq: list[_KKState] = []

    if equal_size:
        assert len(values) % k == 0, f"{len(values)} % {k} != 0"
        for offset in range(0, len(sorted_items), k):
            items = []
            for i in range(k):
                val, idx = sorted_items[offset + i]
                items.append((idx, val))
            heapq.heappush(states_pq, _KKState(items=items, k=k))
    else:
        for val, idx in sorted_items:
            heapq.heappush(states_pq, _KKState(items=[(idx, val)], k=k))

    while len(states_pq) > 1:
        s0 = heapq.heappop(states_pq)
        s1 = heapq.heappop(states_pq)
        s0.merge(s1)
        heapq.heappush(states_pq, s0)

    final = states_pq[0]
    return final.get_partitions()


# =============================================================================
# Packing Metrics — KK vs FFD comparison logging
# =============================================================================


def _compute_packing_metrics(
    values: list[int], partitions: list[list[int]], capacity: int
) -> dict:
    # [FUNCTION DOC]
    # HELPER: Computes load statistics for a partition: number of groups, per-group
    # sums, max/min/mean load, spread, imbalance ratio (spread/mean), standard
    # deviation, coefficient of variation, max/mean ratio, utilization (mean/capacity)
    # and wasted tokens (sum of capacity minus load). Returns zeros for no groups.
    # Ratios are 0 when the denominator is 0.
    """Compute comprehensive packing quality metrics for a given partition.

    Args:
        values: Original sequence lengths / token counts.
        partitions: List of groups, each group a list of indices into *values*.
        capacity: Maximum allowed sum per group.

    Returns:
        Dictionary with keys: n_groups, group_sums, max_load, min_load,
        mean_load, spread, imbalance_ratio, std_dev, cv, max_load_ratio,
        utilization, wasted_tokens.
    """
    group_sums = [sum(values[i] for i in group) for group in partitions]
    n_groups = len(group_sums)

    if n_groups == 0:
        return {
            "n_groups": 0,
            "group_sums": [],
            "max_load": 0,
            "min_load": 0,
            "mean_load": 0.0,
            "spread": 0,
            "imbalance_ratio": 0.0,
            "std_dev": 0.0,
            "cv": 0.0,
            "max_load_ratio": 0.0,
            "utilization": 0.0,
            "wasted_tokens": 0,
        }

    max_load = max(group_sums)
    min_load = min(group_sums)
    mean_load = sum(group_sums) / n_groups
    spread = max_load - min_load
    imbalance_ratio = spread / mean_load if mean_load > 0 else 0.0
    variance = sum((s - mean_load) ** 2 for s in group_sums) / n_groups
    std_dev = variance**0.5
    cv = std_dev / mean_load if mean_load > 0 else 0.0
    max_load_ratio = max_load / mean_load if mean_load > 0 else 0.0
    utilization = mean_load / capacity if capacity > 0 else 0.0
    wasted_tokens = sum(max(0, capacity - s) for s in group_sums)

    return {
        "n_groups": n_groups,
        "group_sums": group_sums,
        "max_load": max_load,
        "min_load": min_load,
        "mean_load": mean_load,
        "spread": spread,
        "imbalance_ratio": imbalance_ratio,
        "std_dev": std_dev,
        "cv": cv,
        "max_load_ratio": max_load_ratio,
        "utilization": utilization,
        "wasted_tokens": wasted_tokens,
    }


def kk_allocate(
    values: list[int],
    capacity: int,
    min_groups: int,
    n_groups_divisor: int = 1,
    equal_size: bool = False,
) -> list[list[int]]:
    # [FUNCTION DOC]
    # KK packing as a drop-in replacement for ffd_allocate. Normalizes min_groups like
    # ffd_allocate and raises RuntimeError for oversized values or too few values.
    # Chooses k = max(min_groups, ceil(total / capacity)), rounded up to a multiple of
    # n_groups_divisor and capped at len(values); with equal_size, k must divide
    # len(values) (RuntimeError). Runs _kk_partition; if any group exceeds capacity it
    # logs a warning and falls back to ffd_allocate. Returns lists of indices.
    # KNOWN ISSUE: capping k at len(values) can break divisibility by
    # n_groups_divisor, and the FFD fallback ignores equal_size.
    """Partition *values* into groups using the Karmarkar-Karp differencing method.

    This is a **drop-in replacement** for :func:`ffd_allocate` that typically
    produces more balanced partitions (lower max-min spread across groups),
    at the cost of slightly higher computation.

    The algorithm works by iteratively combining the two most "unbalanced"
    partial partitions.  It is especially effective when sequence lengths are
    highly variable (e.g. in RL rollouts with diverse prompt/response lengths).

    The number of groups is determined by ``max(min_groups, ceil(total / capacity))``,
    matching the approach used by veRL's ``rearrange_micro_batches`` which computes
    ``num_micro_batches = ceildiv(total_seqlen, max_token_len)`` before calling KK.

    Args:
        values: Sequence lengths (or token counts) to pack.
        capacity: Maximum sum of values per group.  When set to a very large
            number the algorithm effectively ignores the capacity constraint
            and just balances the groups.
        min_groups: Minimum number of output groups.
        n_groups_divisor: The number of groups must be divisible by this value.
            Useful for pipeline parallelism.
        equal_size: If ``True``, every group will contain exactly
            ``len(values) // k`` items.  Requires ``len(values) % k == 0``.

    Returns:
        A list of groups, where each group is a list of original indices into
        *values*.

    Raises:
        RuntimeError: If any single value exceeds *capacity*, or if there are
            fewer values than *min_groups*.

    Ref:
        https://en.wikipedia.org/wiki/Largest_differencing_method

    Example::

        >>> kk_allocate([100, 200, 300, 150, 250], capacity=500, min_groups=2)
        [[2, 0], [4, 1, 3]]
    """
    if min_groups is None or min_groups < n_groups_divisor:
        min_groups = n_groups_divisor

    if any(v > capacity for v in values):
        raise RuntimeError(
            f"Some values exceed capacity {capacity}. "
            "Cannot pack a single item that is larger than the bin."
        )
    if len(values) < min_groups:
        raise RuntimeError(
            f"Number of values {len(values)} is smaller than min_groups {min_groups}"
        )

    total = sum(values)
    k = max(min_groups, math.ceil(total / capacity))

    # Respect n_groups_divisor
    if n_groups_divisor > 1:
        k = math.ceil(k / n_groups_divisor) * n_groups_divisor

    # k cannot exceed the number of values
    k = min(k, len(values))

    if equal_size and len(values) % k != 0:
        raise RuntimeError(
            f"equal_size=True requires len(values) ({len(values)}) "
            f"to be divisible by k ({k})"
        )

    partitions = _kk_partition(values, k, equal_size=equal_size)

    # Safety net: if any group still exceeds capacity, fall back to FFD.
    for group in partitions:
        group_sum = sum(values[i] for i in group)
        if group_sum > capacity:
            logger.warning(
                "KK partition violates capacity constraint (%d > %d), "
                "falling back to FFD.",
                group_sum,
                capacity,
            )
            return ffd_allocate(values, capacity, min_groups, n_groups_divisor)

    return partitions


def balanced_greedy_partition(nums: list[int], K: int) -> list[list[int]]:
    # [FUNCTION DOC]
    # Splits nums into K groups of exactly len(nums) / K items each: processes values in
    # descending order and assigns each to the non-full group with the smallest current
    # sum. Returns index lists. Raises ValueError if len(nums) < K or not divisible by K.
    """
    Splits `nums` into K groups such that the maximum difference between group sums is minimized.

    Returns indices (not values) for each group.

    Greedy with capacity-aware assignment.

    Args:
        nums: List of values to partition
        K: Number of groups to partition into

    Returns:
        List of K lists, where each inner list contains the indices assigned to that group

    Raises:
        ValueError: If len(nums) is not divisible by K or if len(nums) < K
    """
    n = len(nums)
    if n < K:
        raise ValueError(f"Number of items ({n}) must be >= K ({K}).")
    if n % K != 0:
        raise ValueError("The length of nums must be divisible by K.")
    m = n // K

    # Sort indices by value in descending order
    sorted_indices = sorted(range(n), key=lambda i: -nums[i])

    groups: list[list[int]] = [[] for _ in range(K)]
    sums = [0 for _ in range(K)]
    counts = [0 for _ in range(K)]

    for idx in sorted_indices:
        num = nums[idx]
        # Find the non-full group with the smallest current sum
        chosen_group = -1
        min_sum = float("inf")
        for i in range(K):
            if counts[i] < m and sums[i] < min_sum:
                min_sum = sums[i]
                chosen_group = i

        groups[chosen_group].append(idx)
        sums[chosen_group] += num
        counts[chosen_group] += 1

    return groups


if __name__ == "__main__":
    import time

    for i in range(100):
        st = time.monotonic()
        nums = np.random.randint(1024, 8192, size=(100,)).tolist()
        max_tokens_per_mb = 163840
        min_n_groups = np.random.randint(1, 8)
        groups = ffd_allocate(nums, max_tokens_per_mb, min_n_groups)
        assert len(groups) >= min_n_groups
        import itertools

        indices = list(itertools.chain(*groups))
        assert len(set(indices)) == len(indices)
        group_percent = [
            sum(nums[i] for i in group) / max_tokens_per_mb for group in groups
        ]

        print(
            len(groups),
            min_n_groups,
            [sum(nums[i] for i in group) for group in groups],
            max(group_percent),
            min(group_percent),
            np.mean(group_percent),
            time.monotonic() - st,
        )

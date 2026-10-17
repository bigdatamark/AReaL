# [MODULE DOC]
# Shared helpers for profiling scripts; currently only random sequence-length generation
# for synthetic packed batches. Uses the global `random` module state.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for profiling tools."""

import random


def generate_random_seq_lens(
    batch_size: int,
    total_len: int,
    min_seq_len: int = 64,
) -> list[int]:
    # [FUNCTION DOC]
    # Splits total_len into batch_size random integer lengths summing exactly to
    # total_len. For each of the first batch_size-1 items: avg = remaining //
    # items_left; lower bound = max(min_seq_len, avg//2); upper bound = min(remaining -
    # min_seq_len*(items_still_after), 2*avg); draw uniformly (inclusive) with
    # random.randint and subtract from remaining. The last item gets whatever remains.
    # Returns list[int]. Side effect: consumes global RNG.
    # KNOWN ISSUE: if total_len < batch_size*min_seq_len (or avg//2 exceeds the upper
    # bound) the lower bound can exceed the upper bound and random.randint raises
    # ValueError; no input validation, and batch_size<=0 is not handled.
    """Generate random sequence lengths that sum to total_len.

    Generates variable-length sequences for profiling with packed/padded batches.
    Each sequence length is randomized around the average remaining length.

    Args:
        batch_size: Number of sequences to generate.
        total_len: Target total length (sum of all sequence lengths).
        min_seq_len: Minimum length for each sequence.

    Returns:
        List of sequence lengths that sum to total_len.
    """
    seq_lens = []
    remaining = total_len

    for i in range(batch_size - 1):
        avg_remaining = remaining // (batch_size - i)
        min_len = max(min_seq_len, avg_remaining // 2)
        max_len = min(remaining - min_seq_len * (batch_size - i - 1), avg_remaining * 2)
        length = random.randint(min_len, max_len)
        seq_lens.append(length)
        remaining -= length

    seq_lens.append(remaining)
    return seq_lens

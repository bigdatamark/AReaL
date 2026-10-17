# [MODULE DOC]
# Integer math helpers for ceiling division and alignment.
# SPDX-License-Identifier: Apache-2.0

# Common math utilities


def ceil_div(x: int, y: int) -> int:
    # [FUNCTION DOC]
    # Returns ceil(x / y) for positive ints using integer arithmetic.
    """Ceiling division."""
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    # [FUNCTION DOC]
    # Rounds x up to the next multiple of y.
    """Align x to the next multiple of y."""
    return ceil_div(x, y) * y

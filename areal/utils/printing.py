# [MODULE DOC]
# Console formatting helper for printing scalar statistics as a grid table.
# SPDX-License-Identifier: Apache-2.0

from tabulate import tabulate


def tabulate_stats(data: dict[str, float], col=4, floatfmt=".4e") -> str:
    # [FUNCTION DOC]
    # Lays out the (name, value) pairs of data into a table with `col` name/value
    # column pairs, filled column-major (items go down the first column pair, then the
    # next), and renders it with tabulate in "fancy_grid" style using floatfmt for
    # numbers (default ".4e"). Returns the table string; no side effects.
    items = list(data.items())
    # Calculate how many rows we'll need
    row_count = (len(items) + col - 1) // col

    # Reorganize items in column-major order
    column_major = []
    for i in range(row_count):
        row = []
        for j in range(col):
            index = i + j * row_count
            if index < len(items):
                row.extend(items[index])
        column_major.append(row)

    return tabulate(column_major, floatfmt=floatfmt, tablefmt="fancy_grid")

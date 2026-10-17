#!/usr/bin/env python3
# [MODULE DOC]
# Pre-commit hook script that ensures every given Python file contains the SPDX
# Apache-2.0 license header, inserting it (after a shebang line if present, followed by
# a blank line) when missing. Exits 1 if any file was changed, else 0.
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook that ensures every Python file under areal/ contains the
SPDX license header.  Files missing the header are fixed automatically.

Exit codes:
  0 - all files already had the header (nothing changed).
  1 - one or more files were fixed (pre-commit will re-stage them).
"""

from __future__ import annotations

import sys
from pathlib import Path

HEADER = "# SPDX-License-Identifier: Apache-2.0"
SHEBANG_PREFIX = "#!"


def needs_header(path: Path) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if the file's UTF-8 text does not contain the SPDX header
    # string anywhere. Returns False if the file cannot be read.
    """Return True if *path* does not contain the SPDX header."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return HEADER not in text


def fix_file(path: Path) -> None:
    # [FUNCTION DOC]
    # HELPER: Rewrites the file in place inserting the header as line 1, or as line 2
    # when line 1 is a shebang, and ensures an empty line follows the header. Reads and
    # writes UTF-8; may raise on I/O errors.
    """Insert the SPDX header into *path*, respecting shebangs."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    if lines and lines[0].startswith(SHEBANG_PREFIX):
        lines.insert(1, HEADER)
        if len(lines) < 3 or lines[2] != "":
            lines.insert(2, "")
    else:
        lines.insert(0, HEADER)
        if len(lines) < 2 or lines[1] != "":
            lines.insert(1, "")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    # [FUNCTION DOC]
    # Iterates file paths (argv, or sys.argv[1:] when argv is None/empty), skipping
    # non-.py files, fixing those that need the header. Prints a summary of fixed files
    # to stdout and returns 1 if any were fixed, else 0.
    # KNOWN ISSUE: an explicitly passed empty list falls back to sys.argv[1:] because
    # the check is on truthiness rather than None.
    files = argv if argv else sys.argv[1:]
    fixed: list[str] = []

    for name in files:
        path = Path(name)
        if not path.suffix == ".py":
            continue
        if needs_header(path):
            fix_file(path)
            fixed.append(name)

    if fixed:
        print(f"Added SPDX license header to {len(fixed)} file(s):")
        for f in fixed:
            print(f"  {f}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

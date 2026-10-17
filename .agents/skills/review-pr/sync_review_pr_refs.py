#!/usr/bin/env python3
# [MODULE DOC]
# CLI script that keeps derived copies of the /review-pr reference markdown files in
# sync with the canonical copies under .agents/skills/review-pr/references. Derived
# targets live in .opencode/data and .claude/data; each copy has the SKILL.md path
# rewritten for its platform. Run with exactly one of --write (rewrite stale targets) or
# --check (print unified diffs and exit 1 if stale). Exit code comes from main().
"""Sync review-pr reference data from canonical .agents files.

Canonical source:
  - .agents/skills/review-pr/references/review-pr-domains-and-signals.md
  - .agents/skills/review-pr/references/review-pr-templates.md

Derived targets:
  - .opencode/data/review-pr-domains-and-signals.md
  - .opencode/data/review-pr-templates.md
  - .claude/data/review-pr-domains-and-signals.md
  - .claude/data/review-pr-templates.md

Usage:
  python .agents/skills/review-pr/sync_review_pr_refs.py --write
  python .agents/skills/review-pr/sync_review_pr_refs.py --check
"""

from __future__ import annotations

import difflib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

Transform = Callable[[str], str]


@dataclass(frozen=True)
class SyncSpec:
    # [CLASS DOC]
    # Frozen dataclass describing one sync job: src (canonical Path), dst (derived Path)
    # and transform (str -> str function applied to the source text before writing).
    src: Path
    dst: Path
    transform: Transform


def transform_for_opencode(text: str) -> str:
    # [FUNCTION DOC]
    # Returns text with the literal backticked path to .agents/skills/review-pr/SKILL.md
    # replaced by the .opencode/command/review-pr.md path. Pure function.
    return text.replace(
        "`.agents/skills/review-pr/SKILL.md`", "`.opencode/command/review-pr.md`"
    )


def transform_for_claude(text: str) -> str:
    # [FUNCTION DOC]
    # Returns text with the backticked .agents/skills/review-pr/SKILL.md path replaced
    # by .claude/commands/review-pr.md. Pure function.
    out = text
    out = out.replace(
        "`.agents/skills/review-pr/SKILL.md`", "`.claude/commands/review-pr.md`"
    )
    return out


def find_repo_root(start: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: Walks from the resolved start path up through its parents and returns the
    # first directory containing a .git entry (file or dir). Raises RuntimeError if none
    # is found.
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Unable to locate repository root (missing .git)")


def normalized(text: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Converts CRLF and lone CR line endings to LF and guarantees a trailing
    # newline. Returns the new string.
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    if not norm.endswith("\n"):
        norm += "\n"
    return norm


def sync_one(spec: SyncSpec, check_only: bool) -> tuple[bool, str]:
    # [FUNCTION DOC]
    # Syncs one SyncSpec. Reads and normalizes the source, applies the transform and
    # normalizes again to get the expected text; reads the existing destination (empty
    # string if missing). If equal returns (False, ""). In check mode returns (True,
    # unified diff of existing vs expected) without writing. Otherwise creates parent
    # dirs, writes the expected text as UTF-8 and returns (True, ""). Any OSError on
    # read/write is re-raised as RuntimeError naming the file.
    try:
        source_text = normalized(spec.src.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"Cannot read canonical source: {spec.src}") from exc
    expected = normalized(spec.transform(source_text))

    existing = ""
    if spec.dst.exists():
        try:
            existing = normalized(spec.dst.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Cannot read sync target: {spec.dst}") from exc

    if existing == expected:
        return False, ""

    if check_only:
        diff = "".join(
            difflib.unified_diff(
                existing.splitlines(keepends=True),
                expected.splitlines(keepends=True),
                fromfile=str(spec.dst),
                tofile=f"{spec.dst} (expected)",
            )
        )
        return True, diff

    spec.dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        _ = spec.dst.write_text(expected, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot write sync target: {spec.dst}") from exc
    return True, ""


def build_specs(repo_root: Path) -> list[SyncSpec]:
    # [FUNCTION DOC]
    # Builds the fixed list of four SyncSpecs: the domains-and-signals and templates
    # markdown files, each mapped to .opencode/data (opencode transform) and
    # .claude/data (claude transform), all relative to repo_root.
    canonical_dir = repo_root / ".agents/skills/review-pr/references"
    domains_and_signals = canonical_dir / "review-pr-domains-and-signals.md"
    templates = canonical_dir / "review-pr-templates.md"

    return [
        SyncSpec(
            domains_and_signals,
            repo_root / ".opencode/data/review-pr-domains-and-signals.md",
            transform_for_opencode,
        ),
        SyncSpec(
            templates,
            repo_root / ".opencode/data/review-pr-templates.md",
            transform_for_opencode,
        ),
        SyncSpec(
            domains_and_signals,
            repo_root / ".claude/data/review-pr-domains-and-signals.md",
            transform_for_claude,
        ),
        SyncSpec(
            templates,
            repo_root / ".claude/data/review-pr-templates.md",
            transform_for_claude,
        ),
    ]


def parse_mode(argv: list[str]) -> str:
    # [FUNCTION DOC]
    # Parses argv (without program name). -h/--help prints usage to stdout and raises
    # SystemExit(0). Otherwise requires exactly one of --write/--check (other args are
    # ignored); if zero or both are given prints an error to stderr and raises
    # SystemExit(2). Returns the chosen flag string.
    if "-h" in argv or "--help" in argv:
        print("usage: sync_review_pr_refs.py [--write | --check]")
        print()
        print("Sync /review-pr reference files across platforms")
        print()
        print("options:")
        print("  --write     Write derived files")
        print("  --check     Check derived files are up to date")
        raise SystemExit(0)

    modes = [arg for arg in argv if arg in {"--write", "--check"}]
    if len(modes) != 1:
        print(
            "error: exactly one mode is required: --write or --check", file=sys.stderr
        )
        raise SystemExit(2)
    return modes[0]


def main() -> int:
    # [FUNCTION DOC]
    # Entry point: parses the mode, locates the repo root from this file's location,
    # runs sync_one on every spec. In write mode prints "updated:" or "up-to-date:" per
    # target. In check mode, if anything is stale prints a message plus all diffs to
    # stderr and returns 1, else prints an in-sync message. Returns 0 otherwise.
    mode = parse_mode(sys.argv[1:])
    check_only = mode == "--check"
    write_mode = mode == "--write"
    repo_root = find_repo_root(Path(__file__))
    specs = build_specs(repo_root)

    changed_any = False
    diffs: list[str] = []

    for spec in specs:
        changed, diff = sync_one(spec, check_only=check_only)
        changed_any = changed_any or changed
        if changed and check_only and diff:
            diffs.append(diff)
        if changed and write_mode:
            print(f"updated: {spec.dst}")
        if not changed and write_mode:
            print(f"up-to-date: {spec.dst}")

    if check_only:
        if changed_any:
            print(
                "/review-pr reference files are out of sync. Run with --write.",
                file=sys.stderr,
            )
            for d in diffs:
                print(d, file=sys.stderr)
            return 1
        print("/review-pr reference files are in sync.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

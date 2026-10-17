# [MODULE DOC]
# HELPER module: ANSI color constants and small stdout print helpers shared by the
# OpenClaw online-RL CLI scripts (imported as a top-level module "_fmt", so scripts must
# run from this directory).
"""Shared formatting helpers for online RL CLI scripts."""

from __future__ import annotations

import json
import sys

# =============================================================================
# ANSI color codes
# =============================================================================
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
MAGENTA = "\033[0;35m"
BLUE = "\033[0;34m"


# =============================================================================
# Output helpers
# =============================================================================


def info(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an indented cyan info line.
    print(f"  {CYAN}ℹ{RESET}  {msg}")


def success(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an indented green success line.
    print(f"  {GREEN}✔{RESET}  {msg}")


def error(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an indented red error line.
    print(f"  {RED}✘{RESET}  {msg}")


def dim(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an indented dimmed line.
    print(f"  {DIM}{msg}{RESET}")


def arrow(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an indented yellow arrow line.
    print(f"  {YELLOW}→{RESET} {msg}")


def die(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints msg as an error and exits the process with status 1.
    error(msg)
    sys.exit(1)


def show_request(method: str, path: str, auth_label: str, gateway_url: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints the method, "<gateway_url>/<path>" and an auth label, dimmed.
    print(f"  {DIM}{method} {gateway_url}/{path}{RESET}")
    print(f"  {DIM}Auth: {auth_label}{RESET}")


def show_response(status_code: int, body: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints the HTTP status (green for 2xx, red otherwise) and the body,
    # pretty-printed as indented JSON if parseable, else raw.
    if 200 <= status_code < 300:
        print(f"  {GREEN}HTTP {status_code}{RESET}")
    else:
        print(f"  {RED}HTTP {status_code}{RESET}")
    if body:
        try:
            formatted = json.dumps(json.loads(body), indent=2)
            for line in formatted.split("\n"):
                print(f"  {DIM}{line}{RESET}")
        except (json.JSONDecodeError, ValueError):
            for line in body.split("\n"):
                print(f"  {DIM}{line}{RESET}")


def header(title: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints a blank line and a bold blue boxed title.
    """Print a boxed header."""
    print()
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )

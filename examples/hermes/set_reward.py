#!/usr/bin/env python3
# [MODULE DOC]
# CLI script: assigns a scalar reward to the last (or a specified) interaction of an
# active RL session by POSTing to <gateway>/rl/set_reward with the per-session sk-sess-*
# key as Bearer auth. Includes small ANSI-colored print helpers.
"""Set reward for the last interaction in an active session.

Requires the per-session ``sk-sess-*`` key minted by ``start_session.py``.

Usage:
    python set_reward.py http://host:port --api-key <KEY> --reward 1.0
    python set_reward.py http://host:port --api-key <KEY> --reward 0.0 --interaction-id cmpl_abc
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import requests

# =============================================================================
# CLI formatting helpers (ANSI colors + output helpers)
# =============================================================================
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"


def info(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints an indented cyan info line. No return.
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


def die(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints msg as an error, then exits the process with status 1 (SystemExit).
    error(msg)
    sys.exit(1)


def show_request(method: str, path: str, auth_label: str, gateway_url: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints the HTTP method plus "<gateway_url>/<path>" and an auth label,
    # dimmed.
    print(f"  {DIM}{method} {gateway_url}/{path}{RESET}")
    print(f"  {DIM}Auth: {auth_label}{RESET}")


def show_response(status_code: int, body: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints the HTTP status (green for 2xx, red otherwise) and the body;
    # pretty-prints the body as indented JSON when parseable, else prints it raw line by
    # line. Empty body prints nothing more.
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
    # HELPER: Prints a blank line then a bold blue boxed title (rule lines above and
    # below).
    """Print a boxed header."""
    print()
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(
        f"{BOLD}{BLUE}══════════════════════════════════════════════════════════════{RESET}"
    )


def main() -> None:
    # [FUNCTION DOC]
    # Entry point: parses gateway_url, required --api-key, --reward (float, default from
    # env REWARD or 1.0), optional --interaction-id. Builds a body with reward and
    # optional interaction_id, POSTs it (10s timeout) with JSON content type and Bearer
    # auth, prints request/response. Exits 1 via die() on network error or non-200
    # status; otherwise prints success.
    parser = argparse.ArgumentParser(
        description="Set reward for an interaction in an AReaL RL session"
    )
    parser.add_argument("gateway_url", help="Proxy gateway URL")
    parser.add_argument(
        "--api-key",
        required=True,
        help="Per-session sk-sess-* key from start_session.py",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=float(os.getenv("REWARD", "1.0")),
        help="Reward value (default: 1.0, env: REWARD)",
    )
    parser.add_argument(
        "--interaction-id",
        default=None,
        help="Specific interaction ID (default: last interaction)",
    )
    args = parser.parse_args()

    header("Set Reward")
    info(f"Assigning reward={args.reward} to the last interaction.")
    show_request("POST", "rl/set_reward", "Bearer ***", args.gateway_url)

    reward_body: dict = {"reward": args.reward}
    if args.interaction_id is not None:
        reward_body["interaction_id"] = args.interaction_id

    dim(f"Request body: {json.dumps(reward_body)}")

    try:
        resp = requests.post(
            f"{args.gateway_url}/rl/set_reward",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {args.api_key}",
            },
            json=reward_body,
            timeout=10,
        )
    except requests.RequestException as e:
        die(f"Failed to reach gateway: {e}")

    show_response(resp.status_code, resp.text)

    if resp.status_code != 200:
        die(f"set_reward failed (HTTP {resp.status_code}).")

    success(f"Reward {args.reward} applied")


if __name__ == "__main__":
    main()

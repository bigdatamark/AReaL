# [MODULE DOC]
# Authentication helpers for the training gateway (bearer token extraction and admin key
# check).
# KNOWN ISSUE: the module docstring says "inference gateway" but this is the training
# gateway's copy.
# SPDX-License-Identifier: Apache-2.0

"""Authentication helpers for the inference gateway."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import HTTPException, Request


@dataclass
class AuthResult:
    # [CLASS DOC]
    # Dataclass describing an authenticated caller: key_type ("admin" or "session") and
    # api_key. Not used by the training gateway.
    """Result of API key authentication."""

    key_type: str  # "admin" | "session"
    api_key: str


class AuthError(Exception):
    # [CLASS DOC]
    # Exception carrying status_code and detail for auth failures.
    """Raised when auth fails."""

    def __init__(self, status_code: int, detail: str):
        # [FUNCTION DOC]
        # Stores status_code and detail.
        # KNOWN ISSUE: does not call the base Exception initializer, so str(exc) is
        # empty.
        self.status_code = status_code
        self.detail = detail


def extract_bearer_token(request: Request) -> str:
    # [FUNCTION DOC]
    # Returns the token from an "Authorization: Bearer <tok>" header (prefix case-
    # insensitive, stripped). Raises HTTPException 401 otherwise. No x-api-key fallback.
    """Extract API token from Authorization header.

    Raises HTTPException(401) if missing or malformed.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.",
    )


def require_admin_key(request: Request, admin_api_key: str) -> str:
    # [FUNCTION DOC]
    # Extracts the token and constant-time compares it to admin_api_key; returns it or
    # raises HTTPException 403.
    """Validate that the request carries the admin API key.

    Returns the bearer token on success. Raises HTTPException(403) on failure.
    """
    token = extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_api_key):
        raise HTTPException(status_code=403, detail="Admin API key required.")
    return token

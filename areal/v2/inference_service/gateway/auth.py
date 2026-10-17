# [MODULE DOC]
# Authentication helpers for the inference gateway: bearer / x-api-key token extraction
# and constant-time admin key check.
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
    # the raw api_key.
    """Result of API key authentication."""

    key_type: str  # "admin" | "session"
    api_key: str


class AuthError(Exception):
    # [CLASS DOC]
    # Exception carrying an HTTP status_code and detail message for auth failures.
    """Raised when auth fails."""

    def __init__(self, status_code: int, detail: str):
        # [FUNCTION DOC]
        # Stores status_code and detail as attributes.
        # KNOWN ISSUE: does not call the base Exception initializer, so str(exc) is
        # empty.
        self.status_code = status_code
        self.detail = detail


def extract_bearer_token(request: Request) -> str:
    # [FUNCTION DOC]
    # Returns the API token from the request. Prefers an "Authorization: Bearer <tok>"
    # header (prefix case-insensitive, token stripped), else the "x-api-key" header
    # value. Raises FastAPI HTTPException 401 if neither is present.
    """Extract API token from an OpenAI or Anthropic authentication header.

    Raises HTTPException(401) if missing or malformed.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key:
        return x_api_key
    raise HTTPException(
        status_code=401,
        detail=(
            "Missing or malformed authentication header. Expected "
            "'Bearer <token>' or 'x-api-key: <token>'."
        ),
    )


def require_admin_key(request: Request, admin_api_key: str) -> str:
    # [FUNCTION DOC]
    # Extracts the token (may raise 401) and compares it with admin_api_key using
    # hmac.compare_digest. Returns the token on match; raises HTTPException 403
    # otherwise.
    """Validate that the request carries the admin API key.

    Returns the bearer token on success. Raises HTTPException(403) on failure.
    """
    token = extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_api_key):
        raise HTTPException(status_code=403, detail="Admin API key required.")
    return token

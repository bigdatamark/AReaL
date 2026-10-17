# [MODULE DOC]
# Bearer-token helpers for gateway endpoints.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request  # pyright: ignore[reportMissingImports]


def extract_bearer_token(request: Request) -> str:
    # [FUNCTION DOC]
    # Returns the token from an 'Authorization: Bearer <token>' header (case-insensitive
    # scheme, stripped). Raises HTTPException 401 if missing or malformed.
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>'.",
    )


def require_admin_key(request: Request, admin_api_key: str) -> str:
    # [FUNCTION DOC]
    # Extracts the bearer token and compares it to admin_api_key in constant time;
    # raises HTTPException 403 on mismatch (401 if missing). Returns the token.
    token = extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_api_key):
        raise HTTPException(status_code=403, detail="Admin API key required.")
    return token

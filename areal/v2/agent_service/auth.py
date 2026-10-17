# [MODULE DOC]
# Shared bearer-token admin authentication helpers used by the Router and Gateway
# FastAPI apps and by HTTP clients that call their admin endpoints. Defines
# DEFAULT_ADMIN_API_KEY, the fallback admin key when none is configured.
# SPDX-License-Identifier: Apache-2.0

"""Shared authentication utilities for the Agent Service."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

DEFAULT_ADMIN_API_KEY = "areal-agent-admin"


async def verify_admin_key(
    authorization: str = Header(alias="Authorization"),
    *,
    expected_key: str,
) -> None:
    # [FUNCTION DOC]
    # Checks that the Authorization header equals "Bearer <expected_key>" using a
    # constant-time comparison. Inputs: authorization header string (FastAPI Header
    # dependency), keyword-only expected_key. Returns None on success; raises
    # HTTPException 401 "Invalid admin key" on mismatch.
    expected = f"Bearer {expected_key}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid admin key")


def make_admin_dependency(admin_api_key: str):
    # [FUNCTION DOC]
    # Builds a FastAPI dependency closure bound to admin_api_key. Returns an async
    # callable that reads the Authorization header and delegates to verify_admin_key.
    async def _dep(authorization: str = Header(alias="Authorization")) -> None:
        # [FUNCTION DOC]
        # HELPER: FastAPI dependency body; validates the request's Authorization header
        # against the captured admin_api_key (raises HTTPException 401 on mismatch).
        await verify_admin_key(authorization, expected_key=admin_api_key)

    return _dep


def admin_headers(admin_api_key: str) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: Returns a headers dict with "Authorization: Bearer <admin_api_key>" for
    # outgoing admin requests.
    return {"Authorization": f"Bearer {admin_api_key}"}

# [MODULE DOC]
# Shared JSON-over-HTTP client utilities (stdlib urllib only) used by CLI subcommands to
# talk to service daemons.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client base for subcommand CLIs.

Provides:

- ``ServiceHTTPError`` / ``ServiceUnreachable`` exceptions to distinguish
  "server replied with 4xx/5xx" from "couldn't reach the server".
- A low-level ``request_json`` helper that builds the request, handles
  optional Bearer auth, decodes JSON, and translates ``urllib`` errors
  into the two exception types above.
- ``BaseHTTPClient`` — minimal base for per-component clients
  (gateway / router / data-proxy / etc.). Subclasses add their own RPC
  methods that call ``self._get`` / ``self._post`` / ``self._request``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ServiceHTTPError(Exception):
    # [CLASS DOC]
    # Exception for a non-2xx HTTP response; stores status (int) and body (str)
    # attributes and formats the message as "HTTP <status>: <body>".
    """Non-2xx response from the service. Body is captured so callers
    can inspect application-level error payloads."""

    def __init__(self, status: int, body: str) -> None:
        # [FUNCTION DOC]
        # Stores status and body and initializes the exception message.
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class ServiceUnreachable(Exception):
    # [CLASS DOC]
    # Exception for network-level failures reaching a service (refused, DNS, timeout).
    """Network-level error reaching the service: connection refused,
    DNS failure, timeout, etc."""


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
    bearer: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Sends one HTTP request and returns the decoded JSON body. Inputs: url, method
    # (GET), payload (dict JSON-encoded with Content-Type if given), bearer token for
    # Authorization, timeout seconds (5.0). Always sends Accept: application/json.
    # Returns the parsed body or an empty dict for an empty body. Raises
    # ServiceHTTPError (with the decoded error body) on HTTP errors and
    # ServiceUnreachable on URLError/connection/timeout/OS errors. A non-JSON 2xx body
    # raises json.JSONDecodeError.
    """Issue an HTTP request returning parsed JSON.

    A ``payload`` dict triggers JSON encoding + ``Content-Type``.
    ``bearer`` adds ``Authorization: Bearer <token>``. 2xx → returns the
    parsed body (empty dict if body is empty). 4xx/5xx → ``ServiceHTTPError``.
    Network errors → ``ServiceUnreachable``.
    """

    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        raise ServiceHTTPError(exc.code, exc.read().decode(errors="replace")) from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        raise ServiceUnreachable(str(exc)) from exc


class BaseHTTPClient:
    # [CLASS DOC]
    # Base class holding base (URL without trailing slash) and optional api_key, with
    # GET/POST helpers that attach the key as Bearer auth.
    """Minimal HTTP client base. Subclasses add business RPC methods.

    Holds ``base_url`` and an optional ``api_key`` used as Bearer auth on
    every helper call. ``health()`` is provided as it's universal across
    all daemon types.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        # [FUNCTION DOC]
        # Stores base_url stripped of trailing slashes and the api_key.
        self.base = base_url.rstrip("/")
        self.api_key = api_key

    def health(self, *, timeout: float = 2.0) -> dict[str, Any]:
        # [FUNCTION DOC]
        # GET /health without auth (timeout 2s default); returns the JSON dict.
        return self._get("/health", timeout=timeout, auth=False)

    # ------------------------------------------------------------------
    # Helpers for subclass RPC methods

    def _get(self, path: str, *, timeout: float = 5.0, auth: bool = True) -> dict:
        # [FUNCTION DOC]
        # HELPER: GET base+path with optional Bearer auth (auth True by default),
        # timeout 5s.
        return request_json(
            f"{self.base}{path}",
            timeout=timeout,
            bearer=self.api_key if auth else None,
        )

    def _post(
        self,
        path: str,
        payload: dict | None = None,
        *,
        timeout: float = 10.0,
        auth: bool = True,
    ) -> dict:
        # [FUNCTION DOC]
        # HELPER: POST base+path with optional JSON payload and Bearer auth, timeout
        # 10s.
        return request_json(
            f"{self.base}{path}",
            method="POST",
            payload=payload,
            timeout=timeout,
            bearer=self.api_key if auth else None,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        timeout: float = 10.0,
        auth: bool = True,
    ) -> dict:
        # [FUNCTION DOC]
        # HELPER: Arbitrary-method request to base+path with optional JSON payload and
        # Bearer auth, timeout 10s.
        return request_json(
            f"{self.base}{path}",
            method=method,
            payload=payload,
            timeout=timeout,
            bearer=self.api_key if auth else None,
        )

# [MODULE DOC]
# Authentication helpers for the data service gateway: an in-memory registry mapping
# per-dataset API keys to dataset IDs, plus bearer-token extraction and admin-key checks
# that raise FastAPI HTTP errors.
# SPDX-License-Identifier: Apache-2.0

"""Authentication and API key registry for the data service gateway."""

from __future__ import annotations

import hmac
import uuid

from fastapi import HTTPException, Request

from areal.utils import logging

logger = logging.getLogger("DataGatewayAuth")


class DatasetKeyRegistry:
    # [CLASS DOC]
    # In-memory bidirectional map between dataset API keys and dataset IDs, plus the
    # admin key. Not thread-safe and not persisted; one key per dataset (regenerating
    # overwrites the dataset-to-key entry but leaves the old key resolvable).
    """Maps API keys to dataset IDs. Manages admin + dataset keys."""

    def __init__(self, admin_api_key: str):
        # [FUNCTION DOC]
        # Stores the admin API key and initializes two empty dicts: key-to-dataset and
        # dataset-to-key.
        self._admin_key = admin_api_key
        self._key_to_dataset: dict[str, str] = {}  # api_key → dataset_id
        self._dataset_to_key: dict[str, str] = {}  # dataset_id → api_key

    def generate_key(self, dataset_id: str) -> str:
        # [FUNCTION DOC]
        # Creates a key "ds-" followed by 16 random hex characters (from a UUID4),
        # records it in both maps for dataset_id, logs at INFO and returns the key.
        # KNOWN ISSUE: if the dataset already had a key, the old key is not removed from
        # the key-to-dataset map, so it remains valid.
        """Generate a new API key for a dataset."""
        api_key = f"ds-{uuid.uuid4().hex[:16]}"
        self._key_to_dataset[api_key] = dataset_id
        self._dataset_to_key[dataset_id] = api_key
        logger.info("Generated API key for dataset %s", dataset_id)
        return api_key

    def resolve(self, api_key: str) -> str | None:
        # [FUNCTION DOC]
        # Returns the dataset ID bound to api_key, or None if unknown.
        """Resolve API key to dataset_id. Returns None if not found."""
        return self._key_to_dataset.get(api_key)

    def revoke(self, dataset_id: str) -> str | None:
        # [FUNCTION DOC]
        # Removes the key for dataset_id from both maps. Returns the removed key, or
        # None if the dataset had none.
        """Revoke API key for a dataset. Returns the revoked key."""
        api_key = self._dataset_to_key.pop(dataset_id, None)
        if api_key:
            self._key_to_dataset.pop(api_key, None)
        return api_key

    def is_admin(self, api_key: str) -> bool:
        # [FUNCTION DOC]
        # Returns True if api_key equals the admin key, using a constant-time
        # comparison.
        return hmac.compare_digest(api_key, self._admin_key)

    def is_valid_dataset_key(self, api_key: str) -> bool:
        # [FUNCTION DOC]
        # Returns True if api_key is a currently registered dataset key.
        return api_key in self._key_to_dataset


def extract_bearer_token(request: Request) -> str:
    # [FUNCTION DOC]
    # HELPER: Reads the Authorization header of a FastAPI request; if it starts with
    # "bearer " (case-insensitive) returns the remaining token stripped. Otherwise
    # raises HTTPException 401.
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401, detail="Missing or malformed Authorization header."
    )


def require_admin_key(request: Request, admin_api_key: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Extracts the bearer token (401 if missing) and compares it in constant
    # time to admin_api_key; raises HTTPException 403 on mismatch. Returns the token.
    token = extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_api_key):
        raise HTTPException(status_code=403, detail="Admin API key required.")
    return token

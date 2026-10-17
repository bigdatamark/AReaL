# [MODULE DOC]
# Unit tests for data service gateway auth helpers
# (areal.infra.data_service.gateway.auth): the DatasetKeyRegistry that maps per-dataset
# API keys to dataset ids and checks the admin key, extract_bearer_token, and
# require_admin_key. Requests are faked with SimpleNamespace objects carrying a headers
# dict.
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from areal.infra.data_service.gateway.auth import (
    DatasetKeyRegistry,
    extract_bearer_token,
    require_admin_key,
)


class TestDatasetKeyRegistry:
    # [CLASS DOC]
    # Tests key generation, resolution, revocation and admin checks of
    # DatasetKeyRegistry (admin key 'admin-key').
    def test_generate_key_returns_string(self):
        # [FUNCTION DOC]
        # generate_key returns a string starting with 'ds-'.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert isinstance(key, str)
        assert key.startswith("ds-")

    def test_generate_key_unique(self):
        # [FUNCTION DOC]
        # Keys generated for two datasets differ.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key1 = registry.generate_key("dataset-a")
        key2 = registry.generate_key("dataset-b")
        assert key1 != key2

    def test_resolve_returns_dataset_id(self):
        # [FUNCTION DOC]
        # resolve maps a generated key back to its dataset id.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert registry.resolve(key) == "dataset-a"

    def test_resolve_unknown_returns_none(self):
        # [FUNCTION DOC]
        # resolve returns None for an unknown key.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.resolve("ds-does-not-exist") is None

    def test_revoke_removes_key(self):
        # [FUNCTION DOC]
        # revoke(dataset_id) returns the revoked key and the key no longer resolves.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert registry.revoke("dataset-a") == key
        assert registry.resolve(key) is None

    def test_revoke_unknown_returns_none(self):
        # [FUNCTION DOC]
        # revoke of an unknown dataset returns None.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.revoke("dataset-missing") is None

    def test_is_admin_correct_key(self):
        # [FUNCTION DOC]
        # is_admin returns True for the configured admin key.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_admin("admin-key") is True

    def test_is_admin_wrong_key(self):
        # [FUNCTION DOC]
        # is_admin returns False for a wrong key.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_admin("not-admin") is False

    def test_is_admin_timing_safe(self):
        # [FUNCTION DOC]
        # is_admin rejects near-miss keys (extra char, case change). Despite the name,
        # timing is not measured; only correctness of the constant-time comparison path.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_admin("admin-key") is True
        assert registry.is_admin("admin-keyx") is False
        assert registry.is_admin("admin-keY") is False

    def test_is_valid_dataset_key_after_generate(self):
        # [FUNCTION DOC]
        # is_valid_dataset_key is True for a freshly generated key.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert registry.is_valid_dataset_key(key) is True

    def test_is_valid_dataset_key_unknown(self):
        # [FUNCTION DOC]
        # is_valid_dataset_key is False for an unknown key.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_valid_dataset_key("ds-unknown") is False

    def test_is_valid_dataset_key_after_revoke(self):
        # [FUNCTION DOC]
        # is_valid_dataset_key is False after the dataset's key is revoked.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        registry.revoke("dataset-a")
        assert registry.is_valid_dataset_key(key) is False

    def test_generate_revoke_generate_new_key(self):
        # [FUNCTION DOC]
        # Regenerating a key after revoke yields a new key that resolves to the same
        # dataset.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        first_key = registry.generate_key("dataset-a")
        registry.revoke("dataset-a")
        second_key = registry.generate_key("dataset-a")
        assert second_key != first_key
        assert registry.resolve(second_key) == "dataset-a"

    def test_multiple_datasets_independent(self):
        # [FUNCTION DOC]
        # Keys for different datasets are independent: revoking one leaves the other
        # resolvable.
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key_a = registry.generate_key("dataset-a")
        key_b = registry.generate_key("dataset-b")
        assert registry.resolve(key_a) == "dataset-a"
        assert registry.resolve(key_b) == "dataset-b"
        registry.revoke("dataset-a")
        assert registry.resolve(key_a) is None
        assert registry.resolve(key_b) == "dataset-b"


class TestExtractBearerToken:
    # [CLASS DOC]
    # Tests parsing of the Authorization header.
    def test_extract_bearer_token_with_valid_header(self):
        # [FUNCTION DOC]
        # A 'Bearer token-123' header yields 'token-123'.
        request = SimpleNamespace(headers={"authorization": "Bearer token-123"})
        assert extract_bearer_token(request) == "token-123"

    def test_extract_bearer_token_missing_header_raises_401(self):
        # [FUNCTION DOC]
        # A missing Authorization header raises HTTPException 401.
        request = SimpleNamespace(headers={})
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(request)
        assert exc_info.value.status_code == 401

    def test_extract_bearer_token_basic_auth_raises_401(self):
        # [FUNCTION DOC]
        # A Basic auth header raises HTTPException 401.
        request = SimpleNamespace(headers={"authorization": "Basic token-123"})
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(request)
        assert exc_info.value.status_code == 401

    def test_extract_bearer_token_empty_header_raises_401(self):
        # [FUNCTION DOC]
        # An empty Authorization header raises HTTPException 401.
        request = SimpleNamespace(headers={"authorization": ""})
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(request)
        assert exc_info.value.status_code == 401


class TestRequireAdminKey:
    # [CLASS DOC]
    # Tests admin-key enforcement.
    def test_require_admin_key_accepts_valid_admin_token(self):
        # [FUNCTION DOC]
        # A bearer token equal to the admin key is accepted and returned.
        request = SimpleNamespace(headers={"authorization": "Bearer admin-key"})
        assert require_admin_key(request, "admin-key") == "admin-key"

    def test_require_admin_key_rejects_non_admin_token(self):
        # [FUNCTION DOC]
        # A non-admin bearer token raises HTTPException 403.
        request = SimpleNamespace(headers={"authorization": "Bearer user-key"})
        with pytest.raises(HTTPException) as exc_info:
            require_admin_key(request, "admin-key")
        assert exc_info.value.status_code == 403

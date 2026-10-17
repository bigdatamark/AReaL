# [MODULE DOC]
# Unit tests for areal.v2.weight_update.gateway.pair_registry.PairRegistry: an in-memory
# name -> PairInfo map with register (rejects duplicates via ValueError), get_by_name,
# unregister (returns removed entry or None) and list_pairs. Pure CPU, no I/O.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from areal.v2.weight_update.gateway.config import PairInfo
from areal.v2.weight_update.gateway.pair_registry import PairRegistry


@pytest.fixture
def registry() -> PairRegistry:
    # [FUNCTION DOC]
    # Fixture: returns a fresh empty PairRegistry per test.
    return PairRegistry()


@pytest.fixture
def sample_pair_info() -> PairInfo:
    # [FUNCTION DOC]
    # Fixture: returns a PairInfo named 'test_pair' with one train URL, one inference
    # URL, train_world_size=4 and inference_world_size=2.
    return PairInfo(
        pair_name="test_pair",
        train_worker_urls=["http://train:8000"],
        inference_worker_urls=["http://infer:9000"],
        train_world_size=4,
        inference_world_size=2,
    )


def test_get_by_name_returns_pair_info(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    # [FUNCTION DOC]
    # Registers the sample pair and asserts get_by_name returns the identical object.
    registry.register(sample_pair_info)
    retrieved = registry.get_by_name("test_pair")
    assert retrieved is sample_pair_info


def test_get_by_name_returns_none_for_unregistered_name(registry: PairRegistry) -> None:
    # [FUNCTION DOC]
    # Asserts get_by_name returns None for an unknown pair name.
    result = registry.get_by_name("nonexistent_pair")
    assert result is None


def test_register_raises_for_duplicate_pair_name(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    # [FUNCTION DOC]
    # Registers the sample pair, then registering another PairInfo with the same name
    # must raise ValueError with message "Pair 'test_pair' already registered".
    registry.register(sample_pair_info)

    duplicate_pair = PairInfo(
        pair_name="test_pair",
        train_worker_urls=["http://train:8000"],
        inference_worker_urls=["http://infer:9000"],
    )

    with pytest.raises(ValueError, match="Pair 'test_pair' already registered"):
        registry.register(duplicate_pair)


def test_unregister_removes_pair(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    # [FUNCTION DOC]
    # Asserts unregister returns the identical registered object and that the name is no
    # longer resolvable afterwards.
    registry.register(sample_pair_info)
    removed = registry.unregister("test_pair")

    assert removed is sample_pair_info
    assert registry.get_by_name("test_pair") is None


def test_unregister_returns_none_for_nonexistent_pair(registry: PairRegistry) -> None:
    # [FUNCTION DOC]
    # Asserts unregister of an unknown name returns None without raising.
    result = registry.unregister("nonexistent_pair")
    assert result is None


def test_list_pairs_returns_all_names(registry: PairRegistry) -> None:
    # [FUNCTION DOC]
    # Registers two pairs and asserts list_pairs contains exactly both names (order-
    # insensitive).
    pair1 = PairInfo(
        pair_name="pair_1",
        train_worker_urls=["http://train1:8000"],
        inference_worker_urls=["http://infer1:9000"],
    )
    pair2 = PairInfo(
        pair_name="pair_2",
        train_worker_urls=["http://train2:8000"],
        inference_worker_urls=["http://infer2:9000"],
    )

    registry.register(pair1)
    registry.register(pair2)

    pairs = registry.list_pairs()
    assert set(pairs) == {"pair_1", "pair_2"}


def test_list_pairs_returns_empty_for_no_pairs(registry: PairRegistry) -> None:
    # [FUNCTION DOC]
    # Asserts list_pairs returns an empty list on an empty registry.
    pairs = registry.list_pairs()
    assert pairs == []


def test_list_pairs_after_unregister(
    registry: PairRegistry, sample_pair_info: PairInfo
) -> None:
    # [FUNCTION DOC]
    # Asserts list_pairs reflects a register (one name) followed by an unregister (empty
    # list).
    registry.register(sample_pair_info)
    assert registry.list_pairs() == ["test_pair"]

    registry.unregister("test_pair")
    assert registry.list_pairs() == []

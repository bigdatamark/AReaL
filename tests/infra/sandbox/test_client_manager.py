# [MODULE DOC]
# Unit tests for DaytonaClientManager (areal.infra.sandbox._client), the process-wide
# lazy singleton for the async Daytona client. The real Daytona SDK classes are replaced
# with fakes; tests cover singleton behavior, configuration overrides, atexit
# registration and idempotent close.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import areal.infra.sandbox._client as client_module
from areal.infra.sandbox._client import DaytonaClientManager


class FakeDaytonaConfig:
    # [CLASS DOC]
    # Fake stand-in for the Daytona SDK config class that simply records constructor
    # kwargs.
    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # Stores all keyword arguments in self.kwargs.
        self.kwargs = kwargs


class FakeAsyncDaytona:
    # [CLASS DOC]
    # Fake async Daytona client that records its config and counts close() calls.
    def __init__(self, config):
        # [FUNCTION DOC]
        # Stores config and initializes close_calls to 0.
        self.config = config
        self.close_calls = 0

    async def close(self):
        # [FUNCTION DOC]
        # Async; increments close_calls.
        self.close_calls += 1


@pytest.fixture(autouse=True)
def reset_daytona_client_manager(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture: resets DaytonaClientManager class state (_client,
    # _config_overrides, _atexit_registered, _loop) before and after each test and
    # monkeypatches AsyncDaytona/DaytonaConfig in the client module with the fakes.
    DaytonaClientManager._client = None
    DaytonaClientManager._config_overrides = {}
    DaytonaClientManager._atexit_registered = False
    DaytonaClientManager._loop = None
    monkeypatch.setattr(client_module, "AsyncDaytona", FakeAsyncDaytona, raising=False)
    monkeypatch.setattr(
        client_module, "DaytonaConfig", FakeDaytonaConfig, raising=False
    )
    yield
    DaytonaClientManager._client = None
    DaytonaClientManager._config_overrides = {}
    DaytonaClientManager._atexit_registered = False
    DaytonaClientManager._loop = None


@pytest.mark.asyncio
async def test_get_client_returns_singleton():
    # [FUNCTION DOC]
    # Asserts two get_client calls return the same FakeAsyncDaytona instance whose
    # config kwargs are only connection_pool_maxsize=None.
    first_client = await DaytonaClientManager.get_client()
    second_client = await DaytonaClientManager.get_client()

    assert first_client is second_client
    assert isinstance(first_client, FakeAsyncDaytona)
    assert first_client.config.kwargs == {"connection_pool_maxsize": None}


@pytest.mark.asyncio
async def test_configure_before_first_use_applies_overrides():
    # [FUNCTION DOC]
    # Calls configure(api_url=...) before first use and asserts the created client
    # config contains both the default connection_pool_maxsize=None and the override.
    DaytonaClientManager.configure(api_url="https://example.daytona.test")

    client = await DaytonaClientManager.get_client()

    assert client.config.kwargs == {
        "connection_pool_maxsize": None,
        "api_url": "https://example.daytona.test",
    }


@pytest.mark.asyncio
async def test_configure_after_init_raises():
    # [FUNCTION DOC]
    # Asserts configure raises RuntimeError mentioning 'already initialized' once a
    # client exists.
    await DaytonaClientManager.get_client()

    with pytest.raises(RuntimeError, match="already initialized"):
        DaytonaClientManager.configure(api_url="https://example.daytona.test")


@pytest.mark.asyncio
async def test_atexit_registered_on_first_get_client(monkeypatch):
    # [FUNCTION DOC]
    # Captures atexit.register calls and asserts only one registration happens across
    # two get_client calls.
    registrations: list[object] = []
    monkeypatch.setattr(client_module.atexit, "register", registrations.append)

    await DaytonaClientManager.get_client()
    await DaytonaClientManager.get_client()

    assert len(registrations) == 1


@pytest.mark.asyncio
async def test_close_is_idempotent():
    # [FUNCTION DOC]
    # Asserts calling close twice closes the underlying client only once and resets
    # _client to None.
    client = await DaytonaClientManager.get_client()

    await DaytonaClientManager.close()
    await DaytonaClientManager.close()

    assert client.close_calls == 1
    assert DaytonaClientManager._client is None

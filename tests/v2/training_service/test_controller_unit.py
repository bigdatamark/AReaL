# [MODULE DOC]
# Unit tests for GatewayTrainController initialization in the v2 training service. The
# scheduler, guard HTTP calls, process forking, engine creation and router registration
# are all mocked so the async bootstrap sequence can be verified without processes.
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

MODULE = "areal.v2.training_service.controller.controller"


def _make_response(method: str, url: str, *, json=None) -> httpx.Response:
    # [FUNCTION DOC]
    # HELPER: Builds a 200 httpx.Response with the given JSON body and an attached
    # request
    # (method, url) so raise_for_status works.
    return httpx.Response(
        200,
        json=json,
        request=httpx.Request(method, url),
    )


def _make_controller(scheduler: MagicMock | None = None) -> GatewayTrainController:
    # [FUNCTION DOC]
    # HELPER: Builds a GatewayTrainController for FSDPEngine with backend fsdp:d2, a
    # single
    # rpc_server SchedulingSpec, admin key and 5s request/setup timeouts, using the
    # given
    # scheduler mock (or a new MagicMock).
    return GatewayTrainController(
        train_engine="areal.engine.FSDPEngine",
        scheduler=scheduler or MagicMock(),
        config=TrainEngineConfig(
            experiment_name="test-exp",
            trial_name="trial-0",
            backend="fsdp:d2",
            scheduling_spec=(
                SchedulingSpec(
                    cpu=1,
                    gpu=1,
                    mem=1024,
                    port_count=1,
                    cmd="python -m areal.infra.rpc.rpc_server",
                ),
            ),
            admin_api_key="test-admin-key",
            request_timeout=5.0,
            setup_timeout=5.0,
        ),
    )


class _FakeAsyncClient:
    # [CLASS DOC]
    # HELPER: Async-context-manager stand-in for httpx.AsyncClient that returns queued
    # responses (or raises queued exceptions) in FIFO order for get/post, which are
    # AsyncMocks so calls can be inspected.
    def __init__(self, responses_or_errors):
        # [FUNCTION DOC]
        # Stores a copy of the queued responses/errors and wraps _get/_post in
        # AsyncMocks.
        self._responses_or_errors = list(responses_or_errors)
        self.get = AsyncMock(side_effect=self._get)
        self.post = AsyncMock(side_effect=self._post)

    async def __aenter__(self):
        # [FUNCTION DOC]
        # Returns self on async context entry.
        return self

    async def __aexit__(self, *_args):
        # [FUNCTION DOC]
        # No-op async context exit; does not suppress exceptions.
        return None

    async def _get(self, _url: str):
        # [FUNCTION DOC]
        # HELPER: Pops the next queued item; raises it if it is an exception, else
        # returns it.
        # Raises IndexError when the queue is empty.
        next_item = self._responses_or_errors.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    async def _post(self, _url: str, json=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Same as _get for POST; json and kwargs are ignored.
        _ = json
        next_item = self._responses_or_errors.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class TestGatewayTrainControllerInitialization:
    # [CLASS DOC]
    # Tests for GatewayTrainController._async_initialize.
    @pytest.mark.asyncio
    async def test_async_initialize_offloads_scheduler_and_uses_async_helpers(self):
        # [FUNCTION DOC]
        # With two guard workers, a fake HTTP client answering /alloc_ports,
        # asyncio.to_thread
        # patched to run inline and all fork/engine/register helpers mocked: asserts
        # scheduler
        # create_workers and get_workers(role "train-role-guard", timeout 5) run via
        # to_thread,
        # guard env set once, five async forks (2 workers, router, model gateway/data
        # proxy,
        # gateway) and no sync forks, 2 engine creations, 4 engine endpoint calls,
        # router
        # registration with router/model addresses and api key;
        # worker/router/model/gateway
        # addresses are recorded and the api key starts with "ak-train-role-".
        worker0 = MagicMock(ip="127.0.0.1", worker_ports=[18000], id="guard-0")
        worker1 = MagicMock(ip="127.0.0.1", worker_ports=[18001], id="guard-1")

        scheduler = MagicMock()
        scheduler.create_workers.return_value = ["guard-0", "guard-1"]
        scheduler.get_workers.return_value = [worker0, worker1]

        controller = _make_controller(scheduler)
        controller._role = "train-role"

        port_client = _FakeAsyncClient(
            [
                _make_response(
                    "POST",
                    "http://127.0.0.1:18000/alloc_ports",
                    json={"ports": [29500]},
                )
            ]
        )

        async def _run_in_thread(func, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Replacement for asyncio.to_thread that calls func synchronously.
            return func(*args, **kwargs)

        with (
            patch("httpx.AsyncClient", return_value=port_client),
            patch(
                f"{MODULE}.asyncio.to_thread", side_effect=_run_in_thread
            ) as mock_to_thread,
            patch.object(
                controller, "_async_set_guards_env", new_callable=AsyncMock
            ) as mock_set_env,
            patch.object(
                controller,
                "_async_fork_on_guard",
                new_callable=AsyncMock,
                side_effect=[
                    ("127.0.0.1", 19001),
                    ("127.0.0.1", 19002),
                    ("127.0.0.1", 18081),
                    ("127.0.0.1", 18082),
                    ("127.0.0.1", 18080),
                ],
            ) as mock_async_fork,
            patch.object(controller, "_fork_on_guard", autospec=True) as mock_sync_fork,
            patch.object(
                controller, "_create_engine_on_worker", new_callable=AsyncMock
            ) as mock_create_engine,
            patch.object(
                controller,
                "_call_worker_engine_endpoint",
                new_callable=AsyncMock,
            ) as mock_call_engine,
            patch.object(
                controller, "_register_in_router", new_callable=AsyncMock
            ) as mock_register,
        ):
            await controller._async_initialize(role="train-role")

        assert mock_to_thread.await_count == 2
        create_call = mock_to_thread.await_args_list[0]
        get_call = mock_to_thread.await_args_list[1]
        assert create_call.args[0] is scheduler.create_workers
        assert get_call.args[0] is scheduler.get_workers
        assert get_call.kwargs == {
            "role": "train-role-guard",
            "timeout": 5,
        }

        mock_set_env.assert_awaited_once()
        assert mock_async_fork.await_count == 5
        mock_sync_fork.assert_not_called()
        assert mock_create_engine.await_count == 2
        assert mock_call_engine.await_count == 4
        mock_register.assert_awaited_once_with(
            "http://127.0.0.1:18081",
            "http://127.0.0.1:18082",
            controller.api_key,
        )

        assert controller._worker_addrs == [
            "http://127.0.0.1:19001",
            "http://127.0.0.1:19002",
        ]
        assert controller._router_addr == "http://127.0.0.1:18081"
        assert controller._model_addr == "http://127.0.0.1:18082"
        assert controller._gateway_addr == "http://127.0.0.1:18080"
        assert controller.api_key is not None
        assert controller.api_key.startswith("ak-train-role-")

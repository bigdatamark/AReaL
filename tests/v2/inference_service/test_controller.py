# [MODULE DOC]
# Unit tests for RolloutControllerV2 (areal.v2.inference_service.controller): the
# InferenceEngineConfig fields it relies on, workflow/agent resolution into
# InferenceServiceWorkflow, public API surface, construction-time validation and local
# state, stats export, and further behaviours below. Controllers are built with a
# MagicMock scheduler and are generally not initialize()d; private fields are injected
# and HTTP/scheduler interactions are mocked.
"""Tests for RolloutControllerV2."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.utils import stats_tracker
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.v2.inference_service.controller.workflow import (
    InferenceServiceWorkflow,
)


def _make_scheduler(n_gpus_per_node: int = 8) -> MagicMock:
    # [FUNCTION DOC]
    # HELPER: Returns a MagicMock scheduler with n_gpus_per_node (default 8).
    scheduler = MagicMock()
    scheduler.n_gpus_per_node = n_gpus_per_node
    return scheduler


# =============================================================================
# InferenceEngineConfig
# =============================================================================


class TestInferenceEngineConfigForInferenceService:
    # [CLASS DOC]
    # Tests for InferenceEngineConfig defaults/overrides used by the inference service.
    def test_defaults(self):
        # [FUNCTION DOC]
        # Asserts defaults for backend "sglang:d1": admin key "areal-admin-key", model
        # "default", consumer_batch_size 1, max_concurrent_rollouts None,
        # max_head_offpolicyness 0, tracing off, and a default agent config pointing at
        # the online _OnlineAgent class path.
        cfg = InferenceEngineConfig(backend="sglang:d1")
        assert cfg.admin_api_key == "areal-admin-key"
        assert cfg.model == "default"
        assert cfg.consumer_batch_size == 1
        assert cfg.max_concurrent_rollouts is None
        assert cfg.max_head_offpolicyness == 0
        assert cfg.enable_rollout_tracing is False
        assert cfg.agent is not None
        assert (
            cfg.agent.agent_cls_path
            == "areal.experimental.openai.proxy.online_agent._OnlineAgent"
        )

    def test_custom_values(self):
        # [FUNCTION DOC]
        # Asserts custom admin key, batch size, concurrency, offpolicyness and an
        # AgentConfig (set_reward_finish_timeout 3.0) are stored as given.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="custom-key",
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            max_head_offpolicyness=5,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=3.0,
            ),
        )
        assert cfg.admin_api_key == "custom-key"
        assert cfg.consumer_batch_size == 32
        assert cfg.max_concurrent_rollouts == 64
        assert cfg.max_head_offpolicyness == 5
        assert cfg.agent is not None
        assert cfg.agent.set_reward_finish_timeout == 3.0

    def test_scheduling_fields(self):
        # [FUNCTION DOC]
        # Asserts request_timeout 60.0 and setup_timeout 600.0 are stored.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            request_timeout=60.0,
            setup_timeout=600.0,
        )
        assert cfg.request_timeout == 60.0
        assert cfg.setup_timeout == 600.0

    def test_dump_to_file_defaults_to_false(self):
        # [FUNCTION DOC]
        # Asserts dump_to_file defaults to False.
        cfg = InferenceEngineConfig(backend="sglang:d1")
        assert cfg.dump_to_file is False

    def test_deterministic_sampling_with_offpolicy_head_warns(self):
        # [FUNCTION DOC]
        # Patches the cli_args logger; asserts deterministic_sampling with
        # max_head_offpolicyness 1 logs exactly one warning mentioning
        # "task-to-weight-version".
        with patch("areal.api.cli_args.logger") as mock_logger:
            InferenceEngineConfig(
                backend="sglang:d1",
                deterministic_sampling=True,
                max_head_offpolicyness=1,
            )

        mock_logger.warning.assert_called_once()
        assert "task-to-weight-version" in mock_logger.warning.call_args.args[0]

    def test_deterministic_sampling_onpolicy_does_not_warn(self):
        # [FUNCTION DOC]
        # Asserts deterministic sampling with offpolicyness 0 logs no warning.
        with patch("areal.api.cli_args.logger") as mock_logger:
            InferenceEngineConfig(
                backend="sglang:d1",
                deterministic_sampling=True,
                max_head_offpolicyness=0,
            )

        mock_logger.warning.assert_not_called()

    def test_nondeterministic_sampling_with_offpolicy_head_does_not_warn(self):
        # [FUNCTION DOC]
        # Asserts non-deterministic sampling with offpolicyness 1 logs no warning.
        with patch("areal.api.cli_args.logger") as mock_logger:
            InferenceEngineConfig(
                backend="sglang:d1",
                deterministic_sampling=False,
                max_head_offpolicyness=1,
            )

        mock_logger.warning.assert_not_called()


# =============================================================================
# RolloutControllerV2 — workflow resolution helpers
# =============================================================================


class TestControllerWorkflowResolution:
    # [CLASS DOC]
    # Tests for RolloutControllerV2._resolve_workflow and _resolve_should_accept_fn.
    def test_resolve_workflow_with_instance(self):
        # [FUNCTION DOC]
        # Asserts _resolve_workflow(12345) raises TypeError mentioning a "callable run()
        # method".
        # KNOWN ISSUE: the name says "with_instance" but it actually tests rejection of
        # a non-callable, non-agent object (an int).
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        with pytest.raises(TypeError, match=r"callable run\(\) method"):
            controller._resolve_workflow(12345)

    def test_resolve_workflow_none_creates_online_inference_service_workflow(self):
        # [FUNCTION DOC]
        # With an AgentConfig (drop_retry_orphans True) and a gateway address, asserts
        # _resolve_workflow(None, workflow_kwargs={"timeout": 3.0}) returns an online
        # InferenceServiceWorkflow bound to this controller with agent None, timeout 3.0
        # and drop_retry_orphans True.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                drop_retry_orphans=True,
            ),
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        resolved = controller._resolve_workflow(
            None,
            workflow_kwargs={"timeout": 3.0},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.controller is controller
        assert resolved.agent is None
        assert resolved.timeout == 3.0
        assert resolved.drop_retry_orphans is True

    def test_resolve_workflow_agent_class_creates_offline_workflow(self):
        # [FUNCTION DOC]
        # Asserts passing an agent class produces an InferenceServiceWorkflow holding an
        # instance of that class and inheriting drop_retry_orphans True from the config.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                drop_retry_orphans=True,
            ),
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            # [CLASS DOC]
            # HELPER: Minimal agent class whose async run() returns reward 1.0.
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns 1.0 regardless of inputs.
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert isinstance(resolved.agent, MockAgent)
        assert resolved.drop_retry_orphans is True

    def test_resolve_workflow_forwards_reward_normalization(self):
        # [FUNCTION DOC]
        # Asserts group_size=2 and reward_normalization=True passed to _resolve_workflow
        # are forwarded onto the created workflow.
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1",
                admin_api_key="test-admin-key",
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            # [CLASS DOC]
            # HELPER: Minimal agent class whose async run() returns reward 1.0.
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns 1.0 regardless of inputs.
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            group_size=2,
            reward_normalization=True,
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.group_size == 2
        assert resolved.reward_normalization is True

    def test_resolve_should_accept_fn_none(self):
        # [FUNCTION DOC]
        # Asserts the static _resolve_should_accept_fn(None) returns None.
        assert RolloutControllerV2._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        # [FUNCTION DOC]
        # Asserts _resolve_should_accept_fn returns a given callable unchanged.
        fn = lambda x: True  # noqa: E731
        assert RolloutControllerV2._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        # [FUNCTION DOC]
        # With serialize_group_samples True, asserts an agent class is wrapped in an
        # InferenceServiceWorkflow that has an agent, an arun_episode method and
        # serialize_group_samples True.
        """Test _resolve_workflow wraps agent-like classes in InferenceServiceWorkflow."""
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-key",
            serialize_group_samples=True,
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            # [CLASS DOC]
            # HELPER: Minimal agent class whose async run() returns reward 1.0.
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns 1.0 regardless of inputs.
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )
        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert hasattr(resolved, "arun_episode")
        assert resolved.serialize_group_samples is True

    def test_resolve_workflow_agent_class_without_gateway_raises(self):
        # [FUNCTION DOC]
        # Asserts resolving an agent class without a gateway address raises ValueError
        # "Gateway address is unavailable".
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )

        class MockAgent:
            # [CLASS DOC]
            # HELPER: Minimal agent class whose async run() returns reward 1.0.
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns 1.0 regardless of inputs.
                return 1.0

        with pytest.raises(ValueError, match="Gateway address is unavailable"):
            controller._resolve_workflow(MockAgent, workflow_kwargs={})

    def test_resolve_workflow_rollout_workflow_instance_raises(self):
        # [FUNCTION DOC]
        # Asserts passing an InferenceServiceWorkflow instance raises TypeError (direct
        # RolloutWorkflow instances not supported).
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
        )

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow instances are not supported",
        ):
            controller._resolve_workflow(workflow)

    def test_resolve_workflow_rollout_workflow_class_raises(self):
        # [FUNCTION DOC]
        # Asserts passing the import path string of a RolloutWorkflow class raises
        # TypeError (direct RolloutWorkflow classes not supported).
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow classes are not supported",
        ):
            controller._resolve_workflow(
                "areal.v2.inference_service.controller.workflow.InferenceServiceWorkflow"
            )


# =============================================================================
# RolloutControllerV2 — API surface
# =============================================================================


class TestRolloutControllerV2APISurface:
    # [CLASS DOC]
    # Static API-surface checks on the RolloutControllerV2 class.
    def test_has_all_public_methods(self):
        # [FUNCTION DOC]
        # Asserts the class defines each expected public method (initialize, destroy,
        # submit, wait, rollout_batch, prepare_batch, chat_completion, version,
        # capacity, pause/resume, stats, generation pause/continue and perf tracer
        # methods).
        methods = [
            "initialize",
            "destroy",
            "submit",
            "wait",
            "rollout_batch",
            "prepare_batch",
            "chat_completion",
            "set_version",
            "get_version",
            "get_capacity",
            "pause",
            "resume",
            "export_stats",
            "pause_generation",
            "continue_generation",
            "config_perf_tracer",
            "save_perf_tracer",
        ]
        for m in methods:
            assert hasattr(RolloutControllerV2, m), f"Missing method: {m}"

    def test_has_properties(self):
        # [FUNCTION DOC]
        # Asserts the class exposes staleness_manager, workflow_executor,
        # proxy_gateway_addr and worker_ids attributes.
        properties = [
            "staleness_manager",
            "workflow_executor",
            "proxy_gateway_addr",
            "worker_ids",
        ]
        for p in properties:
            assert hasattr(RolloutControllerV2, p), f"Missing property: {p}"

    def test_not_subclass_of_rollout_controller(self):
        # [FUNCTION DOC]
        # Asserts RolloutControllerV2's only base class is object (composition, not
        # inheritance).
        """RolloutControllerV2 must NOT be a subclass of RolloutController."""
        # Verify it doesn't inherit from any class except object
        bases = RolloutControllerV2.__bases__
        assert bases == (object,), f"Unexpected bases: {bases}"


# =============================================================================
# RolloutControllerV2 — construction + state
# =============================================================================


class TestRolloutControllerV2Construction:
    # [CLASS DOC]
    # Tests for constructor validation and initial local state.
    def test_admin_api_key_none_raises(self):
        # [FUNCTION DOC]
        # Sets admin_api_key to "" and asserts construction raises ValueError
        # "admin_api_key must be set".
        # KNOWN ISSUE: the name says "none" but the test uses an empty string, not None.
        cfg = InferenceEngineConfig(backend="sglang:d1")
        cfg.admin_api_key = ""
        with pytest.raises(ValueError, match="admin_api_key must be set"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=8))

    def test_model_empty_raises(self):
        # [FUNCTION DOC]
        # Asserts model "" makes construction raise ValueError "model must not be
        # empty".
        cfg = InferenceEngineConfig(
            backend="sglang:d1", admin_api_key="test-key", model=""
        )
        with pytest.raises(ValueError, match="model must not be empty"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=8))

    def test_constructor(self):
        # [FUNCTION DOC]
        # Asserts a fresh controller stores config/scheduler, has no workers or server
        # infos, version 0, staleness_manager None and empty worker id maps.
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        assert controller.config is cfg
        assert controller.scheduler is scheduler
        assert controller.workers == []
        assert controller.server_infos == []
        assert controller.get_version() == 0
        assert controller.staleness_manager is None
        assert controller._worker_ids == {}
        assert controller.worker_ids == {}

    def test_admin_api_key_defaults(self):
        # [FUNCTION DOC]
        # Asserts the configured admin key is retained on controller.config.
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        assert controller.config.admin_api_key == "test-key"

    def test_version_management_without_services(self):
        # [FUNCTION DOC]
        # Sets the private _version to 42 and asserts get_version() returns 42.
        # KNOWN ISSUE: the docstring mentions set_version, but only the private field is
        # set directly; set_version is not exercised.
        """set_version / get_version work even without gateway services."""
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        # No gateway services started, but version management is local
        controller._version = 42
        assert controller.get_version() == 42

    def test_export_stats_returns_dict(self):
        # [FUNCTION DOC]
        # Asserts export_stats() returns a dict on a fresh controller.
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        stats = controller.export_stats()
        assert isinstance(stats, dict)

    def test_export_stats_drains_local_workflow_metrics(self):
        # [FUNCTION DOC]
        # Resets the global stats_tracker, records rollout reward 0.75, and asserts the
        # first export_stats() returns reward and count while the second returns {}
        # (drained). Resets the global tracker again in a finally block.
        stats_tracker.export_all(reset=True)
        try:
            stats_tracker.get("rollout").scalar(reward=0.75)
            controller = RolloutControllerV2(
                config=InferenceEngineConfig(
                    backend="sglang:d1", admin_api_key="test-key"
                ),
                scheduler=MagicMock(n_gpus_per_node=8),
            )

            assert controller.export_stats() == {
                "rollout/reward": 0.75,
                "rollout/reward__count": 1,
            }
            assert controller.export_stats() == {}
        finally:
            stats_tracker.export_all(reset=True)

    def test_proxy_gateway_addr(self):
        # [FUNCTION DOC]
        # Asserts proxy_gateway_addr is "" before initialize().
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        # Before initialize, proxy_gateway_addr returns the empty _gateway_addr
        assert controller.proxy_gateway_addr == ""

    def test_callback_addr_formats_ipv6_hostport(self):
        # [FUNCTION DOC]
        # Sets private _callback_host to an IPv6 literal and _callback_port 19000;
        # asserts callback_addr brackets the host as "[2001:db8::10]:19000".
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "2001:db8::10"
        controller._callback_port = 19000

        assert controller.callback_addr == "[2001:db8::10]:19000"

    def test_workflow_executor_raises_before_init(self):
        # [FUNCTION DOC]
        # Asserts accessing workflow_executor before initialize raises RuntimeError
        # mentioning "initialize".
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="initialize"):
            _ = controller.workflow_executor

    def test_config_perf_tracer_is_noop(self):
        # [FUNCTION DOC]
        # Asserts config_perf_tracer() and save_perf_tracer() can be called without
        # raising (no-ops).
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.config_perf_tracer()
        controller.save_perf_tracer()

    @pytest.mark.parametrize("deterministic_sampling", [False, True])
    @pytest.mark.asyncio
    async def test_async_initialize_passes_config_to_data_proxy(
        self, deterministic_sampling
    ):
        # [FUNCTION DOC]
        # Parametrized over deterministic_sampling. Builds a config with tokenizer path,
        # timeout, an AgentConfig (finish timeout 7.5, two message preprocessors, a
        # prefix matcher) and a guard SchedulingSpec, a scheduler returning one fake
        # worker, and callback 127.0.0.1:19000. Patches _async_fork_on_guard to return
        # three (host, port) pairs and runs _async_initialize with one
        # LocalInfServerInfo. Asserts exactly one fork with role "data-proxy" whose
        # raw_cmd contains --set-reward-finish-timeout 7.5, --callback-server-addr
        # http://127.0.0.1:19000, --deterministic-sampling only when enabled, two
        # --message-preprocessor flags in config order, and the --prefix-matcher path.
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker = MagicMock()
        worker.ip = "127.0.0.1"
        worker.worker_ports = [18000]

        scheduler = MagicMock(n_gpus_per_node=8)
        scheduler.get_workers.return_value = [worker]

        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            tokenizer_path="mock-tokenizer",
            request_timeout=15.0,
            deterministic_sampling=deterministic_sampling,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=7.5,
                message_preprocessors=[
                    "examples.swe.preprocessors.StripAnthropicBillingHeader",
                    "examples.swe.preprocessors.StripAllSystemReminders",
                ],
                prefix_matcher="examples.swe.prefix_matchers.swe_prefix_matcher",
            ),
            scheduling_spec=(
                SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=1,
                    cmd="python -m areal.v2.inference_service.guard",
                ),
            ),
            admin_api_key="test-admin-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        with patch.object(controller, "_async_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),
                ("127.0.0.1", 18082),
                ("127.0.0.1", 18080),
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=[
                    LocalInfServerInfo(
                        host="127.0.0.1", port=30000, process=MagicMock()
                    )
                ],
            )

        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1
        data_proxy_cmd = data_proxy_calls[0].kwargs["raw_cmd"]
        assert "--set-reward-finish-timeout" in data_proxy_cmd
        assert "7.5" in data_proxy_cmd
        assert "--callback-server-addr" in data_proxy_cmd
        assert "http://127.0.0.1:19000" in data_proxy_cmd
        assert ("--deterministic-sampling" in data_proxy_cmd) is deterministic_sampling
        assert data_proxy_cmd.count("--message-preprocessor") == 2
        first = data_proxy_cmd.index("--message-preprocessor")
        second = data_proxy_cmd.index("--message-preprocessor", first + 1)
        assert data_proxy_cmd[first + 1].endswith("StripAnthropicBillingHeader")
        assert data_proxy_cmd[second + 1].endswith("StripAllSystemReminders")
        matcher = data_proxy_cmd.index("--prefix-matcher")
        assert data_proxy_cmd[matcher + 1] == (
            "examples.swe.prefix_matchers.swe_prefix_matcher"
        )


class TestOnlineCallbackFlow:
    # [CLASS DOC]
    # Tests for the controller's online-mode HTTP callback server
    # (/callback/online_ready). Each starts a real local callback server via
    # _start_online_callback_server, posts with the admin bearer key using a real httpx
    # client, and stops it in finally.
    @pytest.mark.asyncio
    async def test_online_callback_without_waiter_buffers_export_request(self):
        # [FUNCTION DOC]
        # Posts a ready callback with no waiter present; asserts 200 and that a later
        # wait_for_online_trajectory(timeout=1.0) returns the buffered payload.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200
            buffered = await controller.wait_for_online_trajectory(timeout=1.0)
            assert buffered == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_online_callback_settles_waiter_once(self):
        # [FUNCTION DOC]
        # Starts a waiter task first, posts a ready callback, asserts 200 and that the
        # waiter resolves to the posted payload.
        # KNOWN ISSUE: despite the name, it does not verify the waiter is settled only
        # once.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200
            result = await waiter_task
            assert result == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_online_callback_invalid_payload_keeps_waiter_pending(self):
        # [FUNCTION DOC]
        # Starts a waiter, posts a payload missing trajectory_id; asserts HTTP 425 and
        # that the waiter is still pending, then cancels it (cancellation not awaited).
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a"},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 425
            assert not waiter_task.done()
            waiter_task.cancel()
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_cancelled_waiter_buffers_completed_online_result(self):
        # [FUNCTION DOC]
        # Starts and cancels a waiter (asserting CancelledError), then posts a ready
        # callback (200) and asserts a new wait_for_online_trajectory returns the
        # payload, i.e. results arriving after a cancelled waiter are buffered rather
        # than lost.
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200

            buffered = await controller.wait_for_online_trajectory(timeout=1.0)
            assert buffered == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()


class TestInferenceServiceWorkflow:
    # [CLASS DOC]
    # Tests for InferenceServiceWorkflow (offline agent mode, online mode, group
    # sampling, export request options and stats).
    async def _run_offline_group(
        self,
        *,
        serialize_group_samples: bool,
        failing_member: int | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Runs one offline arun_episode for a group of 4 with a MockAgent that
        # records start order and peak concurrency; when serialize_group_samples is
        # False all 4 members must be concurrently active (they rendezvous on an Event
        # with a 1s timeout), and the member whose index equals failing_member raises
        # RuntimeError. Uses a MagicMock controller (version 3), mocks _start_session
        # (group "grp-test-42", 4 session/key pairs), _set_last_reward and
        # _export_interactions, and patches workflow_context and stats_tracker in the
        # workflow module. Asserts _export_interactions was awaited once with all
        # session ids, group_id "grp-test-42" and discard_trajectory True iff a member
        # failed. Returns (result, max_active, start_order, tracker mock, workflow).
        active = 0
        max_active = 0
        start_order: list[int] = []
        all_started = asyncio.Event()

        class MockAgent:
            # [CLASS DOC]
            # HELPER: Agent whose run() derives its member index from the api_key suffix
            # and tracks concurrency via closure variables.
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Records start order, increments/decrements the active counter
                # (updating max_active), waits per the serialization mode, raises for
                # failing_member, else returns the member index as a float reward.
                del data
                nonlocal active, max_active
                member_index = int(kwargs["api_key"].rsplit("-", 1)[1])
                start_order.append(member_index)
                active += 1
                max_active = max(max_active, active)
                try:
                    if serialize_group_samples:
                        await asyncio.sleep(0)
                    else:
                        if active == 4:
                            all_started.set()
                        await asyncio.wait_for(all_started.wait(), timeout=1.0)
                    if member_index == failing_member:
                        raise RuntimeError(f"member {member_index} failed")
                    return float(member_index)
                finally:
                    active -= 1

        controller = MagicMock()
        controller.get_version.return_value = 3
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            group_size=4,
            serialize_group_samples=serialize_group_samples,
        )
        sessions = [(f"task-42-{i}", f"session-key-{i}") for i in range(4)]
        workflow._start_session = AsyncMock(return_value=("grp-test-42", sessions))
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": MagicMock(reward=1.0)}
        )

        tracker = MagicMock()
        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = tracker

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session,
            [session_id for session_id, _ in sessions],
            group_id="grp-test-42",
            discard_trajectory=failing_member is not None,
        )
        return result, max_active, start_order, tracker, workflow

    @pytest.mark.asyncio
    async def test_export_interactions_forwards_drop_retry_orphans(self):
        # [FUNCTION DOC]
        # Builds a workflow with drop_retry_orphans True and a mocked aiohttp-style
        # session whose post() returns an async context manager yielding {"traj": {}};
        # asserts _export_interactions returns {} and the POST JSON has
        # drop_retry_orphans True.
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            drop_retry_orphans=True,
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={"traj": {}})
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=context)

        result = await workflow._export_interactions(session, ["session-1"])

        assert result == {}
        assert session.post.call_args.kwargs["json"]["drop_retry_orphans"] is True

    @pytest.mark.asyncio
    async def test_export_interactions_forwards_reward_normalization(self):
        # [FUNCTION DOC]
        # Same as above with reward_normalization True; asserts {} is returned and the
        # POST JSON has reward_normalization True.
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            reward_normalization=True,
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={"traj": {}})
        response_context = MagicMock()
        response_context.__aenter__ = AsyncMock(return_value=response)
        response_context.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post.return_value = response_context

        result = await workflow._export_interactions(session, ["session-1"])

        assert result == {}
        assert session.post.call_args.kwargs["json"]["reward_normalization"] is True

    @pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
    @pytest.mark.asyncio
    async def test_online_mode_waits_on_controller(self):
        # [FUNCTION DOC]
        # Skipped (pending traj schema migration). Online mode: controller's
        # wait_for_online_trajectory returns a session/trajectory id; mocks the HTTP
        # session, workflow_context, stats_tracker and deserialize_interactions; asserts
        # arun_episode returns interactions containing "chatcmpl-1", waited once with
        # timeout 3.0, posted once and deserialized the "interactions" payload.
        mock_interaction = MagicMock(reward=1.0)
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "sess-1", "trajectory_id": 7}
        )

        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=None,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            timeout=3.0,
        )

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch(
                "areal.v2.inference_service.controller.workflow.deserialize_interactions"
            ) as mock_deserialize,
        ):
            mock_deserialize.return_value = {"chatcmpl-1": mock_interaction}

            # _run_online uses ``async with http_session.post(...)`` directly,
            # so the mock must support the async context-manager protocol.
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(
                return_value={"interactions": {"chatcmpl-1": {}}}
            )

            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_http_session = MagicMock()
            mock_http_session.post = MagicMock(return_value=mock_cm)

            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        controller.wait_for_online_trajectory.assert_awaited_once_with(timeout=3.0)
        mock_http_session.post.assert_called_once()
        mock_deserialize.assert_called_once_with({"chatcmpl-1": {}})

    @pytest.mark.asyncio
    async def test_offline_mode_runs_agent(self):
        # [FUNCTION DOC]
        # Offline mode with a single-member group: mocks _start_session,
        # _set_last_reward and _export_interactions plus workflow_context/stats_tracker;
        # asserts arun_episode returns the exported interactions, starts the session and
        # sets the reward once, and exports ["sess-1"] for group "grp-test-1" with
        # discard_trajectory False.
        controller = MagicMock()

        class MockAgent:
            # [CLASS DOC]
            # HELPER: Minimal agent whose async run() returns reward 1.0.
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns 1.0.
                return 1.0

        mock_interaction = MagicMock(reward=1.0)
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        workflow._start_session = AsyncMock(
            return_value=("grp-test-1", [("sess-1", "sess-api-key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": mock_interaction}
        )

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        workflow._start_session.assert_awaited_once()
        workflow._set_last_reward.assert_awaited_once()
        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session,
            ["sess-1"],
            group_id="grp-test-1",
            discard_trajectory=False,
        )

    @pytest.mark.asyncio
    async def test_offline_mode_discards_export_when_agent_fails(self):
        # [FUNCTION DOC]
        # Group of 2 with a FailingAgent and reward normalization; asserts arun_episode
        # returns None and the export is still called for both sessions with
        # discard_trajectory True.
        class FailingAgent:
            # [CLASS DOC]
            # HELPER: Agent whose run() always raises RuntimeError("agent failed").
            async def run(self, data, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Always raises RuntimeError.
                raise RuntimeError("agent failed")

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=FailingAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            group_size=2,
            reward_normalization=True,
        )
        workflow._start_session = AsyncMock(
            return_value=(
                "grp-test-1",
                [("sess-1", "key-1"), ("sess-2", "key-2")],
            )
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(return_value={})

        with patch(
            "areal.v2.inference_service.controller.workflow.workflow_context"
        ) as context:
            context.get_aiohttp_session = AsyncMock(return_value=AsyncMock())
            context.get.return_value = MagicMock(task_id=42)
            context.get_httpx_client = AsyncMock(return_value=MagicMock())
            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is None
        workflow._export_interactions.assert_awaited_once_with(
            context.get_aiohttp_session.return_value,
            ["sess-1", "sess-2"],
            group_id="grp-test-1",
            discard_trajectory=True,
        )

    @pytest.mark.asyncio
    async def test_offline_group_is_concurrent_by_default(self):
        # [FUNCTION DOC]
        # Asserts that by default the 4 group members run concurrently (peak active 4),
        # all indices start, and per-member rewards 0..3 are recorded via tracker.scalar
        # in member order.
        result, max_active, start_order, tracker, _ = await self._run_offline_group(
            serialize_group_samples=False,
        )

        assert result is not None
        assert max_active == 4
        assert sorted(start_order) == [0, 1, 2, 3]
        assert tracker.scalar.call_args_list == [
            call(reward=0.0),
            call(reward=1.0),
            call(reward=2.0),
            call(reward=3.0),
        ]

    @pytest.mark.asyncio
    async def test_offline_group_serial_flag_preserves_within_group_order(self):
        # [FUNCTION DOC]
        # Asserts serialize_group_samples runs members one at a time (peak 1) in index
        # order and records rewards 0..3 in order.
        result, max_active, start_order, tracker, _ = await self._run_offline_group(
            serialize_group_samples=True,
        )

        assert result is not None
        assert max_active == 1
        assert start_order == [0, 1, 2, 3]
        assert tracker.scalar.call_args_list == [
            call(reward=0.0),
            call(reward=1.0),
            call(reward=2.0),
            call(reward=3.0),
        ]

    @pytest.mark.asyncio
    async def test_offline_group_serial_flag_exports_after_failure(self):
        # [FUNCTION DOC]
        # Serialized group with member 1 failing: asserts result None, still serial in
        # order, _set_last_reward awaited for all 4 members, and no reward stats
        # recorded.
        (
            result,
            max_active,
            start_order,
            tracker,
            workflow,
        ) = await self._run_offline_group(
            serialize_group_samples=True,
            failing_member=1,
        )

        assert result is None
        assert max_active == 1
        assert start_order == [0, 1, 2, 3]
        assert workflow._set_last_reward.await_count == 4
        assert tracker.scalar.call_count == 0


# =============================================================================
# Multi-node inference configuration
# =============================================================================


class TestMultiNodeConfig:
    # [CLASS DOC]
    # Tests for multi-node instance sizing (tensor parallel size > GPUs per node).
    def test_scheduler_zero_gpus_raises(self):
        # [FUNCTION DOC]
        # Asserts a scheduler with 0 GPUs per node makes construction raise ValueError
        # "n_gpus_per_node must be >= 1".
        # KNOWN ISSUE: the local `scheduler` from _make_scheduler is configured but
        # unused; a separate MagicMock is passed to the constructor.
        cfg = InferenceEngineConfig(backend="sglang:d1t8", admin_api_key="test-key")
        scheduler = _make_scheduler()
        scheduler.n_gpus_per_node = 0
        with pytest.raises(ValueError, match="n_gpus_per_node must be >= 1"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=0))

    def test_gpus_not_divisible_raises(self):
        # [FUNCTION DOC]
        # Asserts tp 8 with 3 GPUs per node raises ValueError "must be divisible by
        # n_gpus_per_node".
        # KNOWN ISSUE: the local `scheduler` is configured but unused (same as above).
        cfg = InferenceEngineConfig(backend="sglang:d1t8", admin_api_key="test-key")
        scheduler = _make_scheduler()
        scheduler.n_gpus_per_node = 3
        with pytest.raises(ValueError, match="must be divisible by n_gpus_per_node"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=3))

    def test_single_node_backward_compat(self):
        # [FUNCTION DOC]
        # Asserts sglang:d2t4 on 8-GPU nodes yields _nnodes_per_instance 1.
        cfg = InferenceEngineConfig(backend="sglang:d2t4", admin_api_key="test-key")
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        assert controller._nnodes_per_instance == 1

    def test_multi_node_valid_config(self):
        # [FUNCTION DOC]
        # Asserts sglang:d1t16 on 8-GPU nodes yields _nnodes_per_instance 2.
        # tp=16, n_gpus_per_node=8 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(backend="sglang:d1t16", admin_api_key="test-key")
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        assert controller._nnodes_per_instance == 2

    @pytest.mark.asyncio
    async def test_async_initialize_multinode_worker_count(self):
        # [FUNCTION DOC]
        # Multi-node (tp 8, 4 GPUs/node) with pre-existing server_infos: patches
        # _async_fork_on_guard (router, data proxy, gateway) and runs _async_initialize;
        # asserts the scheduler's create_workers job has replicas 1 (dp_size only),
        # exactly three forks occurred and only one was a data proxy.
        # KNOWN ISSUE: worker1 is built but never used.
        """With multi-node and pre-existing server_infos, should create dp_size workers."""
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker0 = MagicMock()
        worker0.ip = "10.0.0.1"
        worker0.worker_ports = [18000]
        worker0.id = "w0"

        worker1 = MagicMock()
        worker1.ip = "10.0.0.2"
        worker1.worker_ports = [18000]
        worker1.id = "w1"

        scheduler = MagicMock(n_gpus_per_node=4)
        scheduler.get_workers.return_value = [worker0]

        # tp=8, n_gpus_per_node=4 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1t8",
            scheduling_spec=(SchedulingSpec(gpu=1, cpu=1, mem=1, cmd="mock"),),
            admin_api_key="test-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        with patch.object(controller, "_async_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),  # router
                ("127.0.0.1", 18082),  # data proxy (only 1, on head)
                ("127.0.0.1", 18080),  # gateway
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=[
                    LocalInfServerInfo(
                        host="10.0.0.1", port=30000, process=MagicMock()
                    ),
                ],
            )

        # With server_infos, total_workers = dp_size = 1 (not dp_size * nnodes_per_instance)
        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.replicas == 1

        # 3 forks: router + data-proxy + gateway (all on head worker)
        assert mock_fork.call_count == 3
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1

    @pytest.mark.asyncio
    async def test_async_initialize_multinode_fork_path(self, monkeypatch):
        # [FUNCTION DOC]
        # Full multi-node fork path (server_infos None) with TRITON_CACHE_DIR/PATH env
        # set. Patches _get_async_client with a fake whose post() answers /alloc_ports
        # with incrementing ports and records /fork payloads, and patches
        # _async_fork_on_guard, _async_wait_for_service and SGLang version checks.
        # Asserts create_workers replicas 2, three port allocations (1 rendezvous + 2
        # per node), two inf-server forks with worker_index 0 and 1, unique per-fork
        # Triton cache dirs/paths prefixed by the env values plus "inf-server-", a
        # dist-init-addr flag in every fork command, and one data proxy fork.
        # KNOWN ISSUE: the inline comment mentions "2 forks = 5" async posts, but only
        # the 3 allocs and 2 forks are asserted separately.
        """Exercise the full multi-node fork path (server_infos=None)."""
        from areal.api.cli_args import SchedulingSpec

        monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/controller-triton-dir")
        monkeypatch.setenv("TRITON_CACHE_PATH", "/tmp/controller-triton-path")

        worker0 = MagicMock()
        worker0.ip = "10.0.0.1"
        worker0.worker_ports = [18000]
        worker0.id = "w0"

        worker1 = MagicMock()
        worker1.ip = "10.0.0.2"
        worker1.worker_ports = [18000]
        worker1.id = "w1"

        scheduler = MagicMock(n_gpus_per_node=4)
        scheduler.get_workers.return_value = [worker0, worker1]

        # tp=8, n_gpus_per_node=4 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1t8",
            scheduling_spec=(SchedulingSpec(gpu=1, cpu=1, mem=1, cmd="mock"),),
            admin_api_key="test-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        # Track async client .post calls to /alloc_ports and /fork
        alloc_port_counter = 0
        fork_calls = []

        async def mock_async_post(url, json=None, timeout=None):
            # [FUNCTION DOC]
            # HELPER: Fake async POST returning a MagicMock 200 response; /alloc_ports
            # increments the counter and returns the URL's host with port 30000+counter,
            # /fork records the JSON payload and returns success.
            nonlocal alloc_port_counter
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/alloc_ports" in url:
                alloc_port_counter += 1
                resp.json.return_value = {
                    "status": "success",
                    "host": url.split("//")[1].split(":")[0],
                    "ports": [30000 + alloc_port_counter],
                }
            elif "/fork" in url:
                fork_calls.append(json)
                resp.json.return_value = {"status": "success"}
            return resp

        mock_async_client = AsyncMock()
        mock_async_client.post = mock_async_post

        with (
            patch.object(
                controller, "_get_async_client", return_value=mock_async_client
            ),
            patch.object(controller, "_async_fork_on_guard") as mock_fork,
            patch.object(controller, "_async_wait_for_service"),
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            mock_fork.side_effect = [
                ("10.0.0.1", 18081),  # router
                ("10.0.0.1", 18082),  # data proxy
                ("10.0.0.1", 18080),  # gateway
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=None,
            )

        # dp_size=1, nnodes_per_instance=2: total_workers = 2
        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.replicas == 2

        # Async client .post calls for inf server fork:
        # 1 rendezvous alloc (nnodes_per_instance > 1) + 2 node allocs + 2 forks = 5
        assert alloc_port_counter == 3  # 1 rendezvous + 2 per-node
        assert len(fork_calls) == 2  # 1 per node in the group

        # Verify fork payloads have correct worker_index and role
        assert fork_calls[0]["role"] == "inf-server"
        assert fork_calls[0]["worker_index"] == 0
        assert fork_calls[1]["role"] == "inf-server"
        assert fork_calls[1]["worker_index"] == 1

        cache_dirs = [fc["env"]["TRITON_CACHE_DIR"] for fc in fork_calls]
        cache_paths = [fc["env"]["TRITON_CACHE_PATH"] for fc in fork_calls]
        assert all(
            path.startswith("/tmp/controller-triton-dir/inf-server-")
            for path in cache_dirs
        )
        assert all(
            path.startswith("/tmp/controller-triton-path/inf-server-")
            for path in cache_paths
        )
        assert len(set(cache_dirs)) == 2
        assert len(set(cache_paths)) == 2

        # Verify dist_init_addr propagated to fork commands
        for fc in fork_calls:
            cmd_str = " ".join(fc["raw_cmd"])
            assert "--dist-init-addr" in cmd_str or "--dist_init_addr" in cmd_str

        # Only 1 data proxy (dp_size=1, on head worker only)
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1

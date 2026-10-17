# [MODULE DOC]
# PPO-family RL training entry point (PPOTrainer, exported as the trainer package's
# PPOTrainer). Builds actor, optional critic, reference and teacher models on
# fsdp/megatron/ archon backends, SGLang or vLLM rollout engines (train and eval),
# dataloaders (or an empty loader for online agent mode), weight-update transport,
# evaluator, saver, recover handler and stats logger, and runs the asynchronous rollout
# -> logp/values -> advantages -> PPO update -> weight sync loop. Handles GPU
# offload/onload when actor and rollout are colocated, v1 AWEX colocated weight
# handover, LoRA, proxy workers for agent workflows and both SPMD and single-controller
# modes. Logger "RLTrainer".
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    RolloutWorkflow,
    SaveLoadMeta,
    Scheduler,
    StepInfo,
    WeightUpdateMeta,
    WorkflowLike,
)
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    PPOCriticConfig,
    SchedulingStrategy,
    SchedulingStrategyType,
    SGLangConfig,
    TrainDatasetConfig,
    ValidDatasetConfig,
    vLLMConfig,
)
from areal.engine import RemoteSGLangEngine, RemotevLLMEngine
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    RolloutController,
    SlurmScheduler,
    current_platform,
)
from areal.infra.data_service import DataController
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.rdataset import RDataset
from areal.infra.utils.concurrent import call_maybe_async
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.dte import apply_dte_config_envvars
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)

if TYPE_CHECKING:
    from datasets import Dataset

    from areal.engine import (
        FSDPPPOActor,
        FSDPPPOCritic,
        MegatronPPOActor,
        MegatronPPOCritic,
    )
    from areal.experimental.engine.archon_engine import ArchonPPOActor, ArchonPPOCritic
    from areal.trainer.ppo.actor import PPOActorController
    from areal.trainer.ppo.critic import PPOCriticController

logger = logging.getLogger("RLTrainer")


class _EmptyDataLoader:
    # [CLASS DOC]
    # HELPER: Stand-in dataloader for online mode (no dataset). Infinite iterator
    # yielding lists of batch_size empty dicts; len() is steps_per_epoch so epoch-based
    # frequency controls work; state_dict/load_state_dict are no-ops for checkpoint
    # compatibility.
    """Minimal dataloader for online mode that yields empty dicts.

    Compatible with ``cycle_dataloader()`` and ``len()`` expectations.
    ``steps_per_epoch`` controls how many steps constitute one epoch,
    derived from ``total_train_steps // total_train_epochs`` to ensure
    epoch-frequency-gated components (Saver, RecoverHandler) behave correctly.
    """

    def __init__(self, batch_size: int = 1, steps_per_epoch: int = 1):
        # [FUNCTION DOC]
        # Stores batch_size (default 1) and steps_per_epoch (default 1).
        self.batch_size = batch_size
        self._steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        # [FUNCTION DOC]
        # Returns steps_per_epoch.
        return self._steps_per_epoch

    def __iter__(self):
        # [FUNCTION DOC]
        # Infinite generator yielding a list of batch_size empty dicts each time.
        while True:
            yield [{} for _ in range(self.batch_size)]

    def state_dict(self) -> dict:
        # [FUNCTION DOC]
        # Returns an empty dict (no state).
        return {}

    def load_state_dict(self, state_dict: dict) -> None:  # noqa: ARG002
        # [FUNCTION DOC]
        # No-op; ignores the given state.
        pass


class PPOTrainer:
    # [CLASS DOC]
    # Orchestrator of PPO/GRPO training. Owns config, tokenizer/processor, optional
    # scheduler, allocations, actor/critic/ref/teacher engines or controllers, rollout
    # and eval_rollout engines, dataloaders, optional DataController,
    # weight_update_meta, Evaluator, Saver, RecoverHandler, StatsLogger and offload
    # flags. Context manager that closes on exit.
    def __init__(
        self,
        config: PPOConfig,
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
    ):
        # [FUNCTION DOC]
        # Runs _init_impl; on any exception logs it, calls close() to tear down
        # partially created workers and re-raises.
        # KNOWN ISSUE: close() accesses attributes (saver, stats_logger, rollout,
        # eval_rollout, teacher, ref, critic, actor) without existence guards, so a
        # failure early in _init_impl makes close() raise AttributeError, which replaces
        # the original exception as the error seen by the caller (the original is only
        # chained).
        try:
            self._init_impl(config, train_dataset, valid_dataset)
        except Exception:
            logger.error(
                "PPOTrainer construction failed; tearing down partially "
                "created workers",
                exc_info=True,
            )
            self.close()
            raise

    def _init_impl(
        self,
        config: PPOConfig,
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Full construction. Reads RANK; in single-controller mode enables file
        # logging and creates the scheduler; applies DTE env vars; loads
        # tokenizer/processor; seeds RNGs; parses actor and rollout allocations; derives
        # offload flags (rollout offload when actor and rollout are colocated, actor
        # offload if that or actor.offload, critic/ref/teacher from their configs; both
        # rollout/actor flags cleared for v1 AWEX colocate); validates config; sets NCCL
        # and deterministic env vars on scheduling specs. Online mode
        # (rollout.agent.mode == "online") forbids a valid dataset; otherwise
        # train_dataset is required (ValueError). Creates actor, critic (if configured),
        # ref (only if kl_ctl > 0 and ref configured) and teacher allocation (deprecated
        # train-engine teacher warns). Builds the train dataloader: an _EmptyDataLoader
        # with steps_per_epoch = total_train_steps // total_train_epochs in online mode
        # (ValueError if steps unset or < 1), else a real loader, connecting RDatasets
        # to a new DataController in single-controller mode; same for the optional valid
        # loader. Creates FinetuneSpec, initializes actor/critic/ref (and train-type
        # teacher), saves initial LoRA weights, offloads the actor if needed, starts the
        # AWEX meta server and adapter for v1 AWEX colocate, initializes rollout and
        # (non-online) eval rollout and rollout-type teacher. Chooses
        # weight_update_meta: v2 -> disk (LoRA, keeping max_head_offpolicyness+2 adapter
        # versions) or awex; v1 -> disk, xccl (megatron or fsdp variant, with LoRA
        # fields) or awex; else ValueError. Connects actor to rollout, creates
        # Evaluator, Saver, RecoverHandler, StatsLogger, loads recover state (also
        # restoring inference weights), informs the staleness manager of the recovered
        # version, configures perf tracing and applies the initial offload policy. Side
        # effects: launches workers, servers and processes, loads models, file I/O.
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            # Set up file logging for controller process
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self._apply_dte_config_envvars()
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()
        self.data_controller: DataController | None = None
        self._train_rdataset: RDataset | None = None
        self._valid_rdataset: RDataset | None = None

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse per-engine allocations from config.
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
        self.rollout_alloc = ModelAllocation.from_str(
            config.rollout.backend, name="rollout"
        )
        self._should_offload_rollout = self._is_actor_rollout_colocated(config)
        self._should_offload_actor = (
            self._should_offload_rollout or config.actor.offload
        )
        self._should_offload_critic = (
            config.critic is not None and config.critic.offload
        )
        self._should_offload_ref = config.ref is not None and config.ref.offload
        self._should_offload_teacher = (
            config.teacher is not None and config.teacher.offload
        )
        # In colocate (awex) mode the GPU switch between rollout and training
        # is managed by the AWEX adapter (manual offload/onload + tagged SGLang
        # release), not by the TMS-based offload machinery below.
        if self._is_v1_awex_colocate(config):
            self._should_offload_rollout = False
            self._should_offload_actor = False

        # Validate config before proceeding with weight initialization
        self._validate_cfg()

        self._amend_xccl_weight_update_envvar()
        self._amend_deterministic_envvar()

        agent_cfg = config.rollout.agent
        self._online_mode = agent_cfg is not None and agent_cfg.mode == "online"

        if self._online_mode and config.valid_dataset is not None:
            raise ValueError(
                "valid_dataset must not be set when using online RL mode "
                "(agent.mode='online'). Online mode does not support "
                "validation datasets."
            )

        # -- Dataset loading --------------------------------------------------
        if not self._online_mode and train_dataset is None:
            raise ValueError(
                "train_dataset must be provided unless using online RL mode "
                "(agent.mode='online')."
            )

        # Create models: actor, critic, ref — each with its own allocation.
        self.actor = self._create_train_engine(config.actor, self.actor_alloc)
        self.critic = None
        if config.critic is not None:
            critic_alloc = ModelAllocation.from_str(
                config.critic.backend, name="critic"
            )
            self.critic = self._create_critic(config.critic, critic_alloc)
        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            ref_alloc = ModelAllocation.from_str(config.ref.backend, name="ref")
            self.ref = self._create_train_engine(config.ref, ref_alloc)

        self.teacher = None
        self.teacher_alloc = None
        if config.teacher is not None:
            if config.teacher.engine_type == "rollout":
                self.teacher_alloc = ModelAllocation.from_str(
                    config.teacher.rollout.backend, name="teacher"
                )
            else:
                assert config.teacher.train is not None
                self.teacher_alloc = ModelAllocation.from_str(
                    self.config.teacher.train.backend, name="teacher"
                )
                logger.warning(
                    "teacher.engine_type='train' uses legacy train-engine teacher path "
                    "and is deprecated; please migrate to engine_type='rollout'."
                )

        steps_per_epoch: int | None = None
        self.train_dataloader: StatefulDataLoader | _EmptyDataLoader
        if self._online_mode:
            if config.total_train_steps is None:
                raise ValueError(
                    "total_train_steps must be set for online mode. "
                    "Both total_train_epochs and total_train_steps are needed "
                    "to compute steps_per_epoch."
                )
            steps_per_epoch = config.total_train_steps // config.total_train_epochs
            if steps_per_epoch < 1:
                raise ValueError(
                    f"total_train_steps ({config.total_train_steps}) must be >= "
                    f"total_train_epochs ({config.total_train_epochs}) so that "
                    f"steps_per_epoch >= 1."
                )
            self.train_dataloader = _EmptyDataLoader(
                batch_size=config.train_dataset.batch_size,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            assert train_dataset is not None
            if is_single_controller() and isinstance(train_dataset, RDataset):
                ds_cfg = DataServiceConfig.from_dataset_config(
                    config.train_dataset, seed=config.seed
                )
                assert self.scheduler is not None
                controller = DataController(ds_cfg, self.scheduler)
                controller.initialize(
                    role="data", num_dataset_workers=ds_cfg.num_workers
                )
                self.data_controller = controller
                train_dataset.connect(
                    controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_train",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=config.train_dataset.shuffle,
                    drop_last=config.train_dataset.drop_last,
                )
                self._train_rdataset = train_dataset

            self.train_dataloader = self._create_dataloader(
                train_dataset,
                dataset_config=self.config.train_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        self.valid_dataloader: StatefulDataLoader | None = None
        if self.config.valid_dataset is not None and valid_dataset is not None:
            assert self.config.valid_dataset is not None
            if is_single_controller() and isinstance(valid_dataset, RDataset):
                assert self.data_controller is not None
                valid_dataset.connect(
                    self.data_controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_valid",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=self.config.valid_dataset.shuffle,
                    drop_last=self.config.valid_dataset.drop_last,
                )
                self._valid_rdataset = valid_dataset

            self.valid_dataloader = self._create_dataloader(
                valid_dataset,
                dataset_config=self.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        # -- FinetuneSpec -----------------------------------------------------
        if self._online_mode:
            assert steps_per_epoch is not None
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=steps_per_epoch * config.train_dataset.batch_size,
                train_batch_size=config.train_dataset.batch_size,
            )
        else:
            ft_spec = FinetuneSpec(
                total_train_epochs=config.total_train_epochs,
                dataset_size=len(self.train_dataloader)
                * config.train_dataset.batch_size,
                train_batch_size=config.train_dataset.batch_size,
            )

        # Initialize engines first — the scheduler must know about roles
        # before the data controller can colocate with them.
        engine_init_kwargs = {"addr": None, "ft_spec": ft_spec}
        self.actor.initialize(**engine_init_kwargs, role="actor")
        if self.critic is not None:
            self.critic.initialize(**engine_init_kwargs, role="critic")
        if self.ref is not None:
            self.ref.initialize(**engine_init_kwargs, role="ref")

        if (
            self.config.teacher is not None
            and self.config.teacher.engine_type == "train"
        ):
            assert self.config.teacher.train is not None
            self.teacher = self._create_train_engine(
                self.config.teacher.train, self.teacher_alloc
            )
            self.teacher.initialize(**engine_init_kwargs, role="teacher")

        # Save initial LoRA weights if enabled (for inference server pre-loading)
        initial_lora_path = self._save_initial_lora_weights()

        # Offload actor before rollout init so sglang can use the GPU memory
        if self._should_offload_actor:
            self._offload_model(self.actor, role="actor")

        # In colocate (awex) mode, offload training weights before SGLang starts
        # so that GPU memory is available for inference engine allocation.
        # Uses adapter-based manual offload (not TMS), so enable_offload is not required.
        self._awex_meta_server_addr: str | None = None
        if self._is_v1_awex_colocate(config):
            from awex.meta.meta_server import start_meta_server

            from areal.utils.network import gethostip

            host, port = start_meta_server()
            if host in ("0.0.0.0", ""):
                host = gethostip()
            self._awex_meta_server_addr = f"{host}:{port}"
            logger.info(
                "Started MetaServer on controller at %s",
                self._awex_meta_server_addr,
            )
            self.actor.init_awex_adapter(meta_server_addr=self._awex_meta_server_addr)
            self.actor.offload()

        # Initialize inference with LoRA path
        self.rollout = self._init_rollout(
            config.rollout, is_eval=False, lora_path=initial_lora_path
        )

        self.eval_rollout = None
        if not self._online_mode:
            self.eval_rollout = self._init_rollout(
                config.rollout, is_eval=True, lora_path=initial_lora_path
            )
        if (
            self.config.teacher is not None
            and self.config.teacher.engine_type == "rollout"
        ):
            self.teacher = self._init_teacher_rollout(self.config.teacher.rollout)

        # Proxy worker initialization (lazy, for AgentWorkflow support)
        self._proxy_started = False

        # Prepare weight update meta and connect to inference engine.
        # v2 controllers pick transport from use_lora: LoRA must go through
        # disk (P2P transports cannot carry PEFT-wrapped tensors); non-LoRA
        # uses awex. v1 keeps the legacy weight_update_mode dispatch.
        if self.config.actor._version == "v2":
            if config.actor.use_lora:
                disk_kwargs: dict[str, Any] = {
                    "experiment_name": config.experiment_name,
                    "trial_name": config.trial_name,
                    "file_root": config.cluster.fileroot,
                    "name": "default",
                    "clear_checkpoint_after_load": True,
                    "use_lora": config.actor.use_lora,
                    "lora_name": config.gconfig.lora_name,
                    "base_model_name": config.actor.path,
                    # Keep enough recent adapter versions for off-policy
                    # rollouts (max_head_offpolicyness) plus a safety margin;
                    # older versions are unloaded to bound sglang VRAM and
                    # avoid the adapter-accumulation hang.
                    "lora_keep_versions": config.rollout.max_head_offpolicyness + 2,
                }
                self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
            else:
                self.weight_update_meta = WeightUpdateMeta.from_awex()
        elif self.config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.actor.use_lora:
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                        # Keep enough recent adapter versions for off-policy
                        # rollouts (max_head_offpolicyness) plus a safety margin;
                        # older versions are unloaded to bound sglang VRAM and
                        # avoid the adapter-accumulation hang.
                        "lora_keep_versions": config.rollout.max_head_offpolicyness + 2,
                    }
                )
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif self.config.actor.weight_update_mode == "xccl":
            # NCCL/XCCL weight update (v1 only)
            xccl_kwargs: dict[str, Any] = {
                "gen_allocation": self.rollout_alloc,
            }

            if config.actor.use_lora:
                xccl_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "base_model_name": config.actor.path,
                    }
                )

            if self.actor_alloc.backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    **xccl_kwargs
                )
            else:
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        elif self.config.actor.weight_update_mode == "awex":
            self.weight_update_meta = WeightUpdateMeta.from_awex(
                meta_server_addr=self._awex_meta_server_addr,
            )
        else:
            raise ValueError(
                f"Invalid weight update mode: {self.config.actor.weight_update_mode}"
            )

        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Set up evaluation (skip in online mode)
        self.evaluator = Evaluator(config.evaluator, ft_spec)

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)

        # Set up statistics logging (wandb, tensoboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
            # Recompute placement instead of reusing _should_offload_rollout:
            # AWEX clears that flag because it drives the handover itself.
            colocated_rollout=self._is_actor_rollout_colocated(config),
        )

        # After recovery, sync the staleness manager so its capacity formula
        # stays bounded despite the version jumping from 0 to recovery_version.
        if self.recover_info is not None:
            recovery_version = self.recover_info.last_step_info.global_step + 1
            if is_single_controller():
                sm = self.rollout.staleness_manager
            else:
                sm = self.rollout.workflow_executor.staleness_manager
            if sm is not None:
                sm.on_version_recovered(recovery_version)

        self._config_perf_tracer()
        self._apply_initial_offload_policy()

    @staticmethod
    def _is_colocation(strategy: SchedulingStrategy | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: Static; True if the scheduling strategy type is colocation (enum or
        # string).
        if strategy is None:
            return False
        return strategy.type in (
            SchedulingStrategyType.colocation,
            SchedulingStrategyType.colocation.value,
            "colocation",
        )

    def _is_actor_rollout_colocated(self, config: PPOConfig) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the actor strategy colocates with target "rollout" or the
        # rollout strategy colocates with target "actor".
        actor_s = config.actor.scheduling_strategy
        rollout_s = config.rollout.scheduling_strategy
        return (self._is_colocation(actor_s) and actor_s.target == "rollout") or (
            self._is_colocation(rollout_s) and rollout_s.target == "actor"
        )

    def _is_v1_awex_colocate(self, config: PPOConfig) -> bool:
        # [FUNCTION DOC]
        # HELPER: True iff actor._version is "v1" and weight_update_mode is "awex" (v1
        # AWEX is always a colocated handover; scheduling strategy deliberately not
        # checked).
        """Whether this run is the v1 AWEX colocated actor-rollout setup.

        ``weight_update_mode`` alone is not enough: controller v2 selects AWEX
        from ``use_lora`` and never reads that field, so a v2 separation run may
        legitimately carry ``weight_update_mode="awex"`` and would otherwise
        take the v1 colocation handover.

        The scheduling strategy is deliberately not part of this check. v1 AWEX
        only exists colocated and runs opt in through ``weight_update_mode``
        while leaving actor and rollout on the default (separation) strategy;
        requiring colocation here skips the meta-server handoff, every training
        worker then starts its own server, and the run waits on ``infer_conf``
        forever.
        """
        return (
            config.actor._version == "v1" and config.actor.weight_update_mode == "awex"
        )

    def _onload_model(self, engine, role: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls engine.onload() inside timing and perf-trace scopes named after
        # the role.
        with (
            stats_tracker.record_timing(f"{role}_onload"),
            perf_tracer.trace_scope(
                f"train.{role}_onload",
                category=Category.IO,
            ),
        ):
            engine.onload()

    def _offload_model(self, engine, role: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls engine.offload() inside timing and perf-trace scopes named after
        # the role.
        with (
            stats_tracker.record_timing(f"{role}_offload"),
            perf_tracer.trace_scope(
                f"train.{role}_offload",
                category=Category.IO,
            ),
        ):
            engine.offload()

    def _offload_rollout(self, is_eval: bool = False):
        # [FUNCTION DOC]
        # HELPER: For the train or eval rollout (no-op if None): pause request
        # submission, pause generation (sync or async callable) and offload the
        # inference engine, each timed and traced.
        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        with (
            stats_tracker.record_timing("rollout_pause"),
            perf_tracer.trace_scope(
                "train.rollout_pause",
                category=Category.INSTR,
            ),
        ):
            rollout.pause()

        with (
            stats_tracker.record_timing("rollout_pause_generation"),
            perf_tracer.trace_scope(
                "train.rollout_pause_generation",
                category=Category.INSTR,
            ),
        ):
            call_maybe_async(rollout.pause_generation)

        with (
            stats_tracker.record_timing("rollout_offload"),
            perf_tracer.trace_scope(
                "train.rollout_offload",
                category=Category.IO,
            ),
        ):
            rollout.offload()

    def _onload_rollout(self, is_eval: bool = False) -> None:
        # [FUNCTION DOC]
        # HELPER: For the train or eval rollout (no-op if None): onload, continue
        # generation and resume, each attempted even if an earlier step failed; the
        # first exception encountered is re-raised at the end.
        cleanup_error: Exception | None = None

        rollout = self.rollout if not is_eval else self.eval_rollout
        if rollout is None:
            return

        try:
            with (
                stats_tracker.record_timing("rollout_onload"),
                perf_tracer.trace_scope(
                    "train.rollout_onload",
                    category=Category.IO,
                ),
            ):
                rollout.onload()
        except Exception as exc:  # noqa: BLE001
            cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_continue_generation"),
                perf_tracer.trace_scope(
                    "train.rollout_continue_generation",
                    category=Category.INSTR,
                ),
            ):
                call_maybe_async(rollout.continue_generation)
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        try:
            with (
                stats_tracker.record_timing("rollout_resume"),
                perf_tracer.trace_scope(
                    "train.rollout_resume",
                    category=Category.INSTR,
                ),
            ):
                rollout.resume()
        except Exception as exc:  # noqa: BLE001
            if cleanup_error is None:
                cleanup_error = exc

        if cleanup_error is not None:
            raise cleanup_error

    def _apply_initial_offload_policy(self) -> None:
        # [FUNCTION DOC]
        # HELPER: After construction offloads, according to the flags, rollout, ref,
        # critic, teacher and actor so the GPU starts free for the first step.
        if self._should_offload_rollout:
            self._offload_rollout()
        if self._should_offload_ref:
            self._offload_model(self.ref, role="ref")
        if self._should_offload_critic:
            self._offload_model(self.critic, role="critic")
        if self._should_offload_teacher:
            self._offload_model(self.teacher, role="teacher")
        if self._should_offload_actor:
            self._offload_model(self.actor, role="actor")

    def train(
        self,
        workflow: WorkflowLike | None = None,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        # [FUNCTION DOC]
        # Main RL loop. Resumes after the recovered step; total_epochs defaults to
        # config (ValueError if <= 0); loop runs to epochs * len(train_dataloader),
        # stopping at total_train_steps. workflow None is allowed only in online mode
        # (starts proxy workers); non-RolloutWorkflow workflows also start proxies. Per
        # step: (onload rollout if colocated) actor.prepare_batch collects rollouts with
        # group size n_samples, dynamic batch, reward normalization, incomplete-group
        # dropping and optional dynamic_filter_fn, then offload rollout; critic values
        # -> traj["values"]; ref logp -> traj["ref_logp"]; teacher logp plus rl/distill
        # loss weights; for v1 AWEX colocate, pause rollout, release SGLang KV cache and
        # weights and onload actor; onload actor if needed; recompute prox_logp when
        # config requires; compute advantages; wait for checkpoint staging; optional
        # CUDA memory profiling snapshots under the log dir; actor PPO update and LR
        # step; critic update and LR step (then offload critic); for AWEX save before
        # weight update; pause rollout, push weights with version step+1 and set
        # versions on all engines; save HF/recover checkpoints (non-AWEX); offload
        # actor; evaluate with the eval rollout (onloading/offloading it when
        # colocated); in single-controller mode clear cached batches on actor, critic,
        # ref and data controller; commit stats; resume rollout; save perf traces. Each
        # phase is timed and traced. Returns None.
        # KNOWN ISSUE: the teacher's cached batches are not cleared in single-controller
        # mode, unlike the other roles.
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        # Initialize proxy workers if not using RolloutWorkflow
        if workflow is None:
            agent_cfg = self.config.rollout.agent
            if agent_cfg is not None and agent_cfg.mode == "online":
                self._ensure_proxy_started()
            else:
                raise ValueError(
                    "workflow must be specified for train() unless "
                    "agent.mode='online' is configured. "
                    "Pass a RolloutWorkflow, AgentWorkflow, or callable."
                )
        elif self._requires_proxy_workflow(workflow):
            self._ensure_proxy_started()

        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            if self._should_offload_rollout:
                self._onload_rollout()
            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "train.rollout",
                    category=Category.COMPUTE,
                    args={
                        "global_step": global_step,
                        "epoch_step": step,
                    },
                ),
            ):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=dynamic_filter_fn,
                    group_size=config.gconfig.n_samples,
                    dynamic_bs=self.config.dynamic_bs,
                    reward_normalization=config.gconfig.reward_normalization,
                    drop_incomplete_group=config.gconfig.drop_incomplete_group,
                )
            if self._should_offload_rollout:
                self._offload_rollout()

            if self.critic is not None:
                if self._should_offload_critic:
                    self._onload_model(self.critic, role="critic")
                with (
                    stats_tracker.record_timing("critic_values"),
                    perf_tracer.trace_scope(
                        "train.compute_values",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    values = self.critic.compute_values(rollout_batch)
                    for traj, v in zip(rollout_batch, values):
                        traj["values"] = v
                    self.critic.get_device_stats().log("critic values")
                # Critic stays onloaded — offloaded after ppo_update below

            if self.ref is not None:
                if self._should_offload_ref:
                    self._onload_model(self.ref, role="ref")
                with (
                    stats_tracker.record_timing("ref_logp"),
                    perf_tracer.trace_scope(
                        "train.ref_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    ref_logps = self.ref.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, ref_logps):
                        traj["ref_logp"] = logp
                    self.ref.get_device_stats().log("ref logp")
                if self._should_offload_ref:
                    self._offload_model(self.ref, role="ref")

            if self.teacher is not None:
                if self._should_offload_teacher:
                    self._onload_model(self.teacher, role="teacher")
                with (
                    stats_tracker.record_timing("teacher_logp"),
                    perf_tracer.trace_scope(
                        "train.teacher_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    teacher_logps = self.teacher.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, teacher_logps):
                        traj["teacher_logp"] = logp
                        traj["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                        traj["distill_loss_weight"] = (
                            self.config.teacher.distill_loss_weight
                        )
                if self._should_offload_teacher:
                    self._offload_model(self.teacher, role="teacher")

            # In colocate (awex) mode: switch GPU from inference to training.
            # Release SGLang KV cache + weights to free GPU for actor.
            if self._is_v1_awex_colocate(self.config):
                logger.info("[AWEX] colocate: pausing rollout...")
                self.rollout.pause()
                logger.info("[AWEX] colocate: pause_generation_sync...")
                self.rollout.pause_generation_sync()
                logger.info("[AWEX] colocate: offload kv_cache...")
                self.rollout.offload(tags=["kv_cache"])
                logger.info("[AWEX] colocate: offload weights...")
                self.rollout.offload(tags=["weights"])
                logger.info("[AWEX] colocate: offload done, onloading actor...")
                self.actor.onload()

            if self._should_offload_actor:
                self._onload_model(self.actor, role="actor")
            if config.actor.should_compute_prox_logp():
                with (
                    stats_tracker.record_timing("recompute_logp"),
                    perf_tracer.trace_scope(
                        "train.recompute_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    prox_logps = self.actor.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, prox_logps):
                        traj["prox_logp"] = logp
                    self.actor.get_device_stats().log("recompute logp")

            with (
                stats_tracker.record_timing("compute_advantage"),
                perf_tracer.trace_scope(
                    "train.compute_advantage",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                adv_batch = self.actor.compute_advantages(rollout_batch)
                self.actor.get_device_stats().log("compute advantages")

            # Wait for async checkpoint staging to complete before modifying parameters
            self.saver.maybe_wait_for_staging()

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
            ):
                self.actor.start_memory_profile(config.memory_profiler.max_entries)

            with (
                stats_tracker.record_timing("train_step"),
                perf_tracer.trace_scope(
                    "train.ppo_update",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.ppo_update(adv_batch)
                self.actor.step_lr_scheduler()
                self.actor.get_device_stats().log("ppo update")

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
            ):
                log_dir = StatsLogger.get_log_path(config.stats_logger)
                snapshot_dir = os.path.join(
                    log_dir, "memory_snapshots", f"step_{global_step}"
                )
                os.makedirs(snapshot_dir, exist_ok=True)
                self.actor.stop_memory_profile(snapshot_dir)
                logger.info(f"Memory snapshots saved to {snapshot_dir}")

            if self.critic is not None:
                with (
                    stats_tracker.record_timing("critic_train_step"),
                    perf_tracer.trace_scope(
                        "train.critic_ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.critic.ppo_update(adv_batch)
                    self.critic.step_lr_scheduler()
                    self.critic.get_device_stats().log("ppo critic update")
                if self._should_offload_critic:
                    self._offload_model(self.critic, role="critic")

            # Save BEFORE update_weights. In AWEX colocate mode the
            # transfer ends with actor weights offloaded, so saving afterwards
            # would resume weights onto a card already crowded by the
            # fully-resumed rollout plus transfer staging leftovers and the HF
            # saver's TP coalesced all-gather transient can OOM. Here the
            # actor weights are still onloaded from ppo_update (no resume
            # needed) and MegatronEngine.save() drops the dead fp32 grad
            # buffers to fund the transient. Weights are identical on both
            # sides of the transfer, so the checkpoint content is unchanged.
            if self._is_v1_awex_colocate(config):
                self._save_training_state(
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            # Actor already onloaded; engine-internal _offload_aware_context
            # calls in update_weights/save are no-ops.

            with (
                stats_tracker.record_timing("update_weights"),
                perf_tracer.trace_scope(
                    "train.update_weights",
                    category=Category.COMM,
                    args={"global_step": global_step},
                ),
            ):
                # Use versioned path for weight updates
                new_version = global_step + 1
                versioned_meta = self.weight_update_meta.with_version(new_version)
                self.actor.update_weights(versioned_meta)

                self.actor.set_version(new_version)
                if self.critic is not None:
                    self.critic.set_version(new_version)
                self.rollout.set_version(new_version)
                if self.eval_rollout is not None:
                    self.eval_rollout.set_version(new_version)

            if not self._is_v1_awex_colocate(config):
                self._save_training_state(
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )

            # Offload actor before eval
            if self._should_offload_actor:
                self._offload_model(self.actor, role="actor")

            if self._should_offload_rollout:
                self._onload_rollout(is_eval=True)
            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )
            if self._should_offload_rollout:
                self._offload_rollout(is_eval=True)

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                # Each role runs in its own Python process with a
                # process-local ``_fetch_buffer``; one HTTP DELETE to the
                # storage owner clears ``_storage`` but not per-consumer
                # caches. Fan out ``clear_batches`` to every role that
                # localized the batch — see areal-project/AReaL#1209.
                # SPMD mode never populates ``_fetch_buffer`` (no RTensor
                # round-trip), so the fan-out is single-controller only.
                if is_single_controller():
                    self.actor.clear_batches(rollout_batch, adv_batch)
                    if self.critic is not None:
                        self.critic.clear_batches(rollout_batch, adv_batch)
                    if self.ref is not None:
                        self.ref.clear_batches(rollout_batch)
                    if self.data_controller is not None:
                        self.data_controller.clear_batches()

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            # Resume rollout
            self.rollout.resume()

            self._save_perf_tracer(step=global_step)

    def _save_training_state(
        self,
        *,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Runs _save_hf then _save_recover_checkpoint within timing/trace
        # scopes.
        with (
            stats_tracker.record_timing("save"),
            perf_tracer.trace_scope(
                "train.save",
                category=Category.IO,
                args={"global_step": global_step},
            ),
        ):
            self._save_hf(epoch=epoch, epoch_step=epoch_step, global_step=global_step)

        with (
            stats_tracker.record_timing("checkpoint_for_recover"),
            perf_tracer.trace_scope(
                "train.checkpoint",
                category=Category.IO,
                args={"global_step": global_step},
            ),
        ):
            self._save_recover_checkpoint(
                epoch=epoch,
                epoch_step=epoch_step,
                global_step=global_step,
            )

    def close(self):
        # [FUNCTION DOC]
        # Finalizes the saver, closes RDatasets and destroys the data controller
        # (guarded), closes the stats logger, destroys eval rollout, rollout, teacher,
        # ref, critic and actor, and force-saves the perf trace.
        self.saver.finalize()
        if hasattr(self, "_train_rdataset") and self._train_rdataset is not None:
            self._train_rdataset.close()
        if hasattr(self, "_valid_rdataset") and self._valid_rdataset is not None:
            self._valid_rdataset.close()
        if hasattr(self, "data_controller") and self.data_controller is not None:
            self.data_controller.destroy()
        self.stats_logger.close()
        if self.eval_rollout is not None:
            self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.teacher is not None:
            self.teacher.destroy()
        if self.ref is not None:
            self.ref.destroy()
        if self.critic is not None:
            self.critic.destroy()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def _config_perf_tracer(self):
        # [FUNCTION DOC]
        # HELPER: If perf tracing is configured, configures the local tracer (role
        # "master") and in single-controller mode the actor, critic, ref, rollout and
        # eval-rollout workers. The teacher is not configured.
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")

        if not is_single_controller():
            return

        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        if self.critic is not None:
            self.critic.config_perf_tracer(self.config.perf_tracer, role="critic")
        if self.ref is not None:
            self.ref.config_perf_tracer(self.config.perf_tracer, role="ref")
        self.rollout.config_perf_tracer(self.config.perf_tracer, role="rollout")
        if self.eval_rollout is not None:
            self.eval_rollout.config_perf_tracer(
                self.config.perf_tracer, role="eval-rollout"
            )

    def _save_perf_tracer(self, step: int):
        # [FUNCTION DOC]
        # HELPER: Saves perf traces for actor, ref, critic, eval rollout, rollout and
        # locally for the given step.
        # KNOWN ISSUE: unlike _config_perf_tracer (and DPOTrainer) it does not return
        # early when config.perf_tracer is None, so it issues save calls to every worker
        # each step even with tracing disabled.
        self.actor.save_perf_tracer(step=step)
        if self.ref is not None:
            self.ref.save_perf_tracer(step=step)
        if self.critic is not None:
            self.critic.save_perf_tracer(step=step)
        if self.eval_rollout is not None:
            self.eval_rollout.save_perf_tracer(step=step)
        self.rollout.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        # [FUNCTION DOC]
        # HELPER: Returns a Local/Ray/SlurmScheduler from config.scheduler.type; raises
        # NotImplementedError otherwise.
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _apply_dte_config_envvars(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Exports delta weight-transfer config as runtime env switches via
        # apply_dte_config_envvars.
        """Export delta weight-transfer config to worker runtime switches."""
        apply_dte_config_envvars(self.config)

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        # [FUNCTION DOC]
        # HELPER: Creates a StatefulDataLoader for the dataset sharded by rank/world
        # size using the default collate.
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
        )

    def _amend_xccl_weight_update_envvar(self):
        # [FUNCTION DOC]
        # HELPER: Single-controller with SGLang rollout only: sets NCCL_CUMEM_ENABLE=0
        # and NCCL_NVLS_ENABLE=0 in every actor scheduling spec (needed for NCCL weight
        # updates). In SPMD mode the launcher sets these.
        if not is_single_controller():
            # These environs are set by the launcher in the SPMD mode.
            return
        if self.rollout_alloc.backend != "sglang":
            return

        # Disable some environ for NCCL weight update.
        for spec in self.config.actor.scheduling_spec:
            spec.env_vars["NCCL_CUMEM_ENABLE"] = "0"
            spec.env_vars["NCCL_NVLS_ENABLE"] = "0"

    def _amend_deterministic_envvar(self):
        # [FUNCTION DOC]
        # HELPER: Single-controller only: for Megatron actor/critic/ref engines with
        # use_deterministic_algorithms, sets NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 in their
        # scheduling specs so TransformerEngine sees it at worker launch.
        if not is_single_controller():
            # This environ is set by the launcher in the SPMD mode.
            return

        engine_cfgs = [(self.config.actor, self.actor_alloc)]
        if self.config.critic is not None:
            engine_cfgs.append(
                (
                    self.config.critic,
                    ModelAllocation.from_str(self.config.critic.backend, name="critic"),
                )
            )
        if self.config.actor.kl_ctl > 0 and self.config.ref is not None:
            engine_cfgs.append(
                (
                    self.config.ref,
                    ModelAllocation.from_str(self.config.ref.backend, name="ref"),
                )
            )

        for engine_cfg, alloc in engine_cfgs:
            if alloc.backend != "megatron":
                continue
            if not engine_cfg.megatron.use_deterministic_algorithms:
                continue
            # TransformerEngine snapshots this env var at import or attention
            # module construction depending on version; exporting it at worker
            # launch is the only ordering safe for all of them.
            for spec in engine_cfg.scheduling_spec:
                spec.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> FSDPPPOActor | MegatronPPOActor | ArchonPPOActor | PPOActorController:
        # [FUNCTION DOC]
        # HELPER: Instantiates a PPO actor-type engine (used for actor, ref and train-
        # type teacher) for the allocation backend (FSDPPPOActor, MegatronPPOActor,
        # ArchonPPOActor; ValueError otherwise), as a controller in single-controller
        # mode or locally in SPMD, creates its process group and returns it.
        """Create a training engine (actor or ref) based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOActor

            actor_cls = FSDPPPOActor
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOActor

            actor_cls = MegatronPPOActor
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOActor

            actor_cls = ArchonPPOActor
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def _create_critic(
        self, critic_config: PPOCriticConfig, alloc: ModelAllocation
    ) -> FSDPPPOCritic | MegatronPPOCritic | ArchonPPOCritic | PPOCriticController:
        # [FUNCTION DOC]
        # HELPER: Same as _create_train_engine for critic classes (FSDP/Megatron/Archon
        # PPO critic).
        """Create a critic engine based on the allocation backend."""
        if alloc.backend == "fsdp":
            from areal.engine import FSDPPPOCritic

            critic_cls = FSDPPPOCritic
        elif alloc.backend == "megatron":
            from areal.engine import MegatronPPOCritic

            critic_cls = MegatronPPOCritic
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonPPOCritic

            critic_cls = ArchonPPOCritic
        else:
            raise ValueError(
                f"Invalid backend: {alloc.backend}, expected fsdp, megatron or archon"
            )
        if is_single_controller():
            critic = critic_cls.as_controller(critic_config, self.scheduler)
        else:
            critic = critic_cls(config=critic_config)
        critic.create_process_group(parallel_strategy=alloc.parallel)
        return critic

    def _init_rollout(
        self,
        rollout_config: InferenceEngineConfig,
        is_eval: bool = False,
        lora_path: str | None = None,
    ) -> InferenceEngine | RolloutController:
        # [FUNCTION DOC]
        # HELPER: Creates the train or eval inference engine. LoRA path in SPMD mode ->
        # ValueError. Works on a deep copy of the rollout config; eval disables
        # staleness limits, colocates with the rollout servers and requests 0 GPUs.
        # SGLang: enables routed-expert return if requested, registers the initial LoRA
        # adapter as "{lora_name}-v0", builds server args (plus AWEX colocate flags for
        # v1 AWEX). vLLM: rejects routed experts, registers LoRA modules, builds server
        # args. Other backends -> ValueError. SPMD: constructs and initializes the local
        # client engine. Single-controller: RolloutControllerV2 for v2 configs else the
        # engine's controller; eval reuses the train rollout's server_infos with role
        # "eval-rollout". Returns the engine/controller.
        # KNOWN ISSUE: mutates the shared self.config.sglang / self.config.vllm
        # (lora_paths, lora_modules, enable_return_routed_experts) rather than a copy.
        if lora_path is not None and not is_single_controller():
            raise ValueError(
                "LoRA is only supported in single-controller mode. "
                "Use `python3 train.py scheduler.type=local` instead of "
                "`python3 -m areal.infra.launcher.local`."
            )
        # Create a working copy of config
        config = deepcopy(rollout_config)
        if is_eval:
            # NOTE: eval does not have any offpolicyness control
            config.max_head_offpolicyness = int(1e12)
            # eval-rollout uses the same inference servers as rollout
            config.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.colocation, target="rollout"
            )
            for spec in config.scheduling_spec:
                spec.gpu = 0

        # Determine engine class and server args based on backend
        rollout_backend = self.rollout_alloc.backend
        if rollout_backend == "sglang":
            if self.config.rollout.return_routed_experts:
                self.config.sglang.enable_return_routed_experts = True
            if lora_path is not None and self.config.actor.use_lora:
                self.config.sglang.lora_paths = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
            if self._is_v1_awex_colocate(self.config):
                server_args["awex_colocate_mode"] = True
                server_args["awex_meta_server_addr"] = self._awex_meta_server_addr
        elif rollout_backend == "vllm":
            if self.config.rollout.return_routed_experts:
                raise ValueError(
                    "return_routed_experts is not supported with vLLM backend. Please disable return_routed_experts or switch to SGLang backend."
                )
            if lora_path is not None and self.config.actor.use_lora:
                self.config.vllm.lora_modules = [
                    f"{self.config.gconfig.lora_name}-v0={lora_path}"
                ]
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.rollout_alloc.parallel.tp_size,
                pp_size=self.rollout_alloc.parallel.pp_size,
            )
            # vLLM does not require LoRA paths during initialization.
            # LoRA is attached to generation requests.
        else:
            raise ValueError(
                f"Invalid backend: {rollout_backend}, expected sglang or vllm"
            )

        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine

        # Single-controller mode - no engine instantiation needed
        if config._version == "v2":
            controller = RolloutControllerV2(
                config=config, scheduler=cast(Scheduler, self.scheduler)
            )
        else:
            controller = engine_cls.as_controller(config, self.scheduler)
        init_kwargs = dict(
            role="rollout",
            server_args=server_args,
        )
        if is_eval:
            assert len(self.rollout.server_infos) > 0
            init_kwargs["server_infos"] = self.rollout.server_infos
            init_kwargs["role"] = "eval-rollout"
        controller.initialize(**init_kwargs)
        return controller

    def _init_teacher_rollout(
        self, rollout_config: InferenceEngineConfig
    ) -> InferenceEngine | RolloutController:
        # [FUNCTION DOC]
        # HELPER: Creates the teacher inference engine from teacher_alloc (RuntimeError
        # if unset): copies sglang/vllm config, swaps in teacher.path as the model (and
        # vLLM tokenizer path if unset), builds server args, and returns a local engine
        # (SPMD) or a controller initialized with role "teacher". ValueError for other
        # backends.
        if self.teacher_alloc is None:
            raise RuntimeError("teacher_alloc is not initialized")
        rollout_alloc = self.teacher_alloc
        config = deepcopy(rollout_config)
        if rollout_alloc.backend == "sglang":
            engine_cls = RemoteSGLangEngine
            teacher_sglang_cfg = deepcopy(self.config.sglang)
            if self.config.teacher is not None and self.config.teacher.path:
                teacher_sglang_cfg.model_path = self.config.teacher.path
            server_args = SGLangConfig.build_args(
                sglang_config=teacher_sglang_cfg,
                tp_size=rollout_alloc.parallel.tp_size,
                pp_size=rollout_alloc.parallel.pp_size,
                base_gpu_id=0,
            )
        elif rollout_alloc.backend == "vllm":
            engine_cls = RemotevLLMEngine
            teacher_vllm_cfg = deepcopy(self.config.vllm)
            if self.config.teacher is not None and self.config.teacher.path:
                teacher_vllm_cfg.model = self.config.teacher.path
                if not rollout_config.tokenizer_path:
                    config.tokenizer_path = self.config.teacher.path
            server_args = vLLMConfig.build_args(
                vllm_config=teacher_vllm_cfg,
                tp_size=rollout_alloc.parallel.tp_size,
                pp_size=rollout_alloc.parallel.pp_size,
            )
        else:
            raise ValueError(
                f"Invalid teacher rollout backend: {rollout_alloc.backend}, expected sglang or vllm"
            )
        if not is_single_controller():
            engine = engine_cls(config)
            engine.initialize(
                train_data_parallel_size=self.actor_alloc.parallel.dp_size
            )
            return engine
        controller = engine_cls.as_controller(config, self.scheduler)
        controller.initialize(role="teacher", server_args=server_args)
        return controller

    def _save_initial_lora_weights(self) -> str | None:
        # [FUNCTION DOC]
        # HELPER: If LoRA is enabled, saves the actor's initial adapter (HF format, no
        # optimizer, with tokenizer/processor) to <model save root>/initial_lora unless
        # adapter_config.json already exists there, and returns that path; returns None
        # without LoRA. Logs progress.
        """Save initial LoRA weights for inference server pre-loading.

        Returns path to saved LoRA weights, or None if LoRA is disabled.
        """
        if not self.config.actor.use_lora:
            return None

        path = os.path.join(
            Saver.get_model_save_root(
                self.config.experiment_name,
                self.config.trial_name,
                self.config.cluster.fileroot,
                name="actor",
            ),
            "initial_lora",
        )

        if os.path.exists(path) and os.path.isfile(
            os.path.join(path, "adapter_config.json")
        ):
            logger.info(f"Initial LoRA already exists at {path}, skipping save.")
            return path

        meta = SaveLoadMeta(
            path=path,
            weight_format="hf",
            with_optim=False,
            tokenizer=self.tokenizer,
            processor=self.processor,
            base_model_path=self.config.actor.path,
        )
        logger.info(f"Saving initial LoRA weights to {path}...")
        self.actor.save(meta=meta)
        logger.info(f"Initial LoRA saved: {os.listdir(path)}")

        return path

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Lets the saver export actor (and critic under name "critic") HF
        # checkpoints when due; in synchronous SPMD mode then barriers on the actor CPU
        # group and syncs the device.
        # Save as HF models for evaluation
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        if self.critic is not None:
            self.saver.save(
                self.critic,
                epoch,
                epoch_step,
                global_step,
                tokenizer=self.tokenizer,
                processor=self.processor,
                name="critic",
            )
        # Async mode: synchronization handled by AsyncCheckpointManager
        if not self.saver.is_async and not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Dumps a recover checkpoint (actor as "default", critic if present,
        # StepInfo, saver/evaluator/logger/dataloader state, tokenizer/processor) when
        # due; SPMD barrier and sync afterwards. Ref and teacher are not saved.
        # Save recoverable checkpoints
        to_save: dict = dict(default=self.actor)
        if self.critic is not None:
            to_save["critic"] = self.critic
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
            steps_per_epoch=len(self.train_dataloader),
        )
        self.recover_handler.dump(
            to_save,
            step_info,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: On the data-parallel head, submits every validation item to the eval
        # rollout with eval n_samples (no reward normalization, keep incomplete groups)
        # and waits for all results without timeout (stats recorded by the workflow);
        # SPMD barrier and sync.
        if self.actor.is_data_parallel_head():
            cnt = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self.config.eval_gconfig.n_samples,
                        is_eval=True,
                        reward_normalization=False,
                        drop_incomplete_group=False,
                    )
                    cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _evaluate(
        self,
        eval_workflow: WorkflowLike | None,
        eval_workflow_kwargs,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        # [FUNCTION DOC]
        # HELPER: No-op without eval rollout, valid dataloader or eval workflow;
        # otherwise lets the Evaluator decide by frequency whether to run _evaluate_fn;
        # SPMD barrier and sync.
        if (
            self.eval_rollout is None
            or self.valid_dataloader is None
            or eval_workflow is None
        ):
            return
        self.evaluator.evaluate(
            functools.partial(
                self._evaluate_fn,
                eval_workflow=eval_workflow,
                eval_workflow_kwargs=eval_workflow_kwargs,
            ),
            epoch,
            epoch_step,
            global_step,
        )
        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Merges exported stats from actor, rollout and eval rollout and commits
        # them to the stats logger; SPMD barrier and sync.
        # Upload statistics to the logger (e.g., wandb)
        stats = self.actor.export_stats()
        stats.update(self.rollout.export_stats())
        if self.eval_rollout is not None:
            stats.update(self.eval_rollout.export_stats())
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        if not is_single_controller():
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _validate_cfg(self):
        # [FUNCTION DOC]
        # HELPER: Fails fast (ValueError) before launching workers if: any offload is
        # needed but enable_offload is False; actor/rollout colocated without disk or
        # awex weight updates; v1 AWEX without Megatron actor or SGLang rollout; vLLM
        # with return_routed_experts; Megatron actor + LoRA + SGLang rollout; actor and
        # rollout controller versions differ.
        """validate config for incompatible settings before weight initialization, to avoid wasted resources on spawning workers and loading models."""
        rollout_backend = self.rollout_alloc.backend
        actor_backend = self.actor_alloc.backend
        requires_train_engine_offload = any(
            (
                self._should_offload_rollout,
                self._should_offload_actor,
                self._should_offload_critic,
                self._should_offload_ref,
                self._should_offload_teacher,
            )
        )

        if requires_train_engine_offload and not self.config.enable_offload:
            raise ValueError(
                "enable_offload must be True when colocation scheduling or train-engine "
                "offload is enabled. Please set enable_offload=True."
            )

        if self._is_actor_rollout_colocated(
            self.config
        ) and self.config.actor.weight_update_mode not in ("disk", "awex"):
            raise ValueError(
                "weight_update_mode must be 'disk' or 'awex' when colocation "
                "scheduling is enabled. Please set actor.weight_update_mode "
                "to one of them."
            )

        if self._is_v1_awex_colocate(self.config):
            if actor_backend != "megatron":
                raise ValueError(
                    "weight_update_mode='awex' requires Megatron actor training "
                    f"backend, got {actor_backend!r}."
                )
            if rollout_backend != "sglang":
                raise ValueError(
                    "weight_update_mode='awex' requires SGLang rollout backend, "
                    f"got {rollout_backend!r}."
                )

        if rollout_backend == "vllm" and self.config.rollout.return_routed_experts:
            raise ValueError(
                "return_routed_experts is only supported with SGLang backend. "
                "Please disable return_routed_experts or switch to SGLang backend."
            )
        if (
            actor_backend == "megatron"
            and self.config.actor.use_lora
            and rollout_backend == "sglang"
        ):
            raise ValueError(
                "Megatron actor with LoRA is not supported with SGLang rollout in "
                "RL trainer. Please use vLLM rollout backend, or disable LoRA, or "
                "switch actor backend from Megatron."
            )

        # Ensure actor and rollout controller versions match.
        actor_version = self.config.actor._version
        rollout_version = self.config.rollout._version
        if actor_version != rollout_version:
            raise ValueError(
                f"actor._version ('{actor_version}') and rollout._version "
                f"('{rollout_version}') must match. Both must be 'v1' or both 'v2'."
            )

    def _requires_proxy_workflow(self, workflow: WorkflowLike | None) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns False for None, RolloutWorkflow instances/subclasses, or
        # import strings resolving to them; True for anything else, including strings
        # whose import fails (fail-safe). Side effect: may import modules.
        """Check if workflow requires proxy workers (i.e., not a RolloutWorkflow).

        Returns True if:
        - Workflow is NOT a RolloutWorkflow instance
        - Workflow is NOT a RolloutWorkflow class
        - Workflow is a string that does NOT import to a RolloutWorkflow

        This enables any callable object with a compatible signature to work
        without requiring inheritance from AgentWorkflow.
        """
        # None workflow is handled separately in train()
        if workflow is None:
            return False

        # Direct RolloutWorkflow instances
        if isinstance(workflow, RolloutWorkflow):
            return False

        # RolloutWorkflow classes
        if isinstance(workflow, type) and issubclass(workflow, RolloutWorkflow):
            return False

        # String import paths
        if isinstance(workflow, str):
            from areal.utils.dynamic_import import import_from_string

            try:
                imported_obj = import_from_string(workflow)
            except (ValueError, ImportError, AttributeError):
                # If import fails, assume it needs proxy (fail-safe)
                return True

            # Check if imported object is RolloutWorkflow
            if isinstance(imported_obj, RolloutWorkflow):
                return False
            if isinstance(imported_obj, type) and issubclass(
                imported_obj, RolloutWorkflow
            ):
                return False

        # Everything else requires proxy workers
        return True

    def _ensure_proxy_started(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Idempotently starts proxy workers for agent workflows. SPMD mode ->
        # NotImplementedError. If the rollout is not a v1 RolloutController just marks
        # started. Otherwise starts proxies on rollout and eval rollout and, in online
        # mode, the proxy gateway (logging its address).
        """Lazily initialize proxy workers when agent workflows are used.

        This method is called before training when a non-RolloutWorkflow is detected
        or when online mode is configured. It creates proxy workers colocated with
        rollout workers to handle OpenAI-compatible API requests.

        In online mode, also starts the proxy gateway for external access.
        """
        if self._proxy_started:
            return

        # Only initialize proxy in single-controller mode with RolloutController
        if not is_single_controller():
            raise NotImplementedError("Proxy workers not supported in SPMD mode")

        if not isinstance(self.rollout, RolloutController):
            self._proxy_started = True
            return

        # v1 controller needs an explicit proxy launch call
        logger.info("Initializing proxy workers for AgentWorkflow support")
        self.rollout.start_proxy()
        if self.eval_rollout is not None:
            self.eval_rollout.start_proxy()

        # Start proxy gateway for online mode.
        agent_cfg = self.config.rollout.agent
        if agent_cfg is not None and agent_cfg.mode == "online":
            self.rollout.start_proxy_gateway()
            logger.info(
                "Proxy gateway available at %s",
                self.rollout.proxy_gateway_addr,
            )

        self._proxy_started = True

    def __enter__(self):
        # [FUNCTION DOC]
        # Returns self.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Logs any exception with traceback, calls close(), returns False so exceptions
        # propagate.
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        return False

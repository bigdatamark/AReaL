# [MODULE DOC]
# SFT trainer entry point. Defines SFTTrainer, which wires config, datasets, scheduler,
# actor engine (FSDP/Megatron/Archon, local SPMD or single-controller), dataloaders,
# evaluator, saver, recovery, stats logging and perf tracing, and runs the training loop
# calling actor.train_lm.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import FinetuneSpec, Scheduler, StepInfo
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    SFTConfig,
    TrainDatasetConfig,
    TrainEngineConfig,
    ValidDatasetConfig,
)
from areal.infra import (
    LocalScheduler,
    RayScheduler,
    SlurmScheduler,
    current_platform,
)
from areal.infra.data_service import DataController
from areal.infra.data_service.controller.config import DataServiceConfig
from areal.infra.data_service.rdataset import RDataset
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.data import (
    broadcast_tensor_container,
    collate_samples_to_list,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import (
    get_bool_env_var,
    is_single_controller,
    rank_in_env_filter,
)
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

if TYPE_CHECKING:
    from datasets import Dataset

    from areal.engine import FSDPLMEngine, MegatronLMEngine
    from areal.experimental.engine.archon_engine import ArchonLMEngine
    from areal.trainer.sft.lm_engine import LMController

logger = logging.getLogger("SFTTrainer")


class SFTTrainer:
    # [CLASS DOC]
    # Orchestrates supervised fine-tuning end to end. Works in SPMD mode (each rank runs
    # this; batches broadcast within model/context parallel groups) or single-controller
    # mode (this process drives remote workers through a scheduler). Usable as a context
    # manager that closes resources on exit.
    def __init__(
        self,
        config: SFTConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | None = None,
    ):
        # [FUNCTION DOC]
        # Builds the whole training setup. Inputs: config (SFTConfig), train_dataset,
        # optional valid_dataset. Steps: in single-controller mode sets up controller
        # file logging and a scheduler; loads HF processor/tokenizer; seeds RNG with key
        # "trainer<RANK>"; parses the actor allocation string and creates the actor
        # engine/controller; if the train dataset is a remote RDataset (single-
        # controller only) starts a DataController and connects the dataset to it;
        # builds the train dataloader sharded by actor DP rank/size; builds a
        # FinetuneSpec (dataset_size = number of batches times batch size) and
        # initializes the actor; optionally builds a validation dataloader (connecting a
        # valid RDataset to the same DataController); creates Evaluator, Saver,
        # RecoverHandler, StatsLogger; loads recover state (may restore
        # actor/dataloader/etc.) into self.recover_info; configures the perf tracer.
        # Side effects: process spawning via scheduler, file I/O, GPU model init.
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        self.scheduler = None
        if is_single_controller():
            self.scheduler = self._init_scheduler()
        self.data_controller: DataController | None = None
        self._train_rdataset: RDataset | None = None
        self._valid_rdataset: RDataset | None = None

        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")

        self.actor = self._create_actor(config.actor)

        if is_single_controller() and isinstance(train_dataset, RDataset):
            ds_cfg = DataServiceConfig.from_dataset_config(
                config.train_dataset, seed=config.seed
            )
            controller = DataController(ds_cfg, self.scheduler)
            controller.initialize(role="data", num_dataset_workers=ds_cfg.num_workers)
            self.data_controller = controller

            train_dataset.connect(
                controller,
                dataset_id=f"{config.experiment_name}_{config.trial_name}_train",
                tokenizer_or_processor_path=config.tokenizer_path,
                shuffle=config.train_dataset.shuffle,
                drop_last=config.train_dataset.drop_last,
            )
            self._train_rdataset = train_dataset

        self.train_dataloader: StatefulDataLoader = self._create_dataloader(
            train_dataset,
            dataset_config=config.train_dataset,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
        )

        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        self.actor.initialize(addr=None, ft_spec=ft_spec, role="actor")

        self.valid_dataloader: StatefulDataLoader | None = None
        if config.valid_dataset is not None and valid_dataset is not None:
            assert config.valid_dataset is not None
            if is_single_controller() and isinstance(valid_dataset, RDataset):
                assert self.data_controller is not None
                valid_dataset.connect(
                    self.data_controller,
                    dataset_id=f"{config.experiment_name}_{config.trial_name}_valid",
                    tokenizer_or_processor_path=config.tokenizer_path,
                    shuffle=config.valid_dataset.shuffle,
                    drop_last=config.valid_dataset.drop_last,
                )
                self._valid_rdataset = valid_dataset

            self.valid_dataloader = self._create_dataloader(
                valid_dataset,
                dataset_config=config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        self.evaluator = Evaluator(config.evaluator, ft_spec)
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)
        self.stats_logger = StatsLogger(config, ft_spec)
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
        )
        self._config_perf_tracer()

    def train(self):
        # [FUNCTION DOC]
        # Main training loop. Resumes from recover_info's next global step (else 0) and
        # iterates global steps up to epochs times steps_per_epoch, stopping early at
        # total_train_steps if set. Each step: loads/broadcasts a batch; waits for async
        # checkpoint staging; optionally starts a CUDA memory profile (only on ranks
        # allowed by env AREAL_MEMORY_PROFILER_RANKS); runs actor.train_lm(batch) and
        # steps the LR scheduler (logging device stats) inside a trace scope with the
        # torch profiler enabled (record_shapes on, profile_memory from env
        # AREAL_TORCH_PROFILER_PROFILE_MEMORY, default true); dumps memory snapshots to
        # <log_dir>/memory_snapshots/step_<n> if profiling; sets actor version to
        # step+1; saves HF checkpoint, recover checkpoint, runs evaluation (each timed
        # and traced); in single-controller mode clears cached batches on actor and
        # DataController; exports and commits stats; saves perf traces. Returns None.
        config = self.config
        rank = int(os.getenv("RANK", "0"))
        memory_profile_this_rank = rank_in_env_filter(
            "AREAL_MEMORY_PROFILER_RANKS", rank
        )
        torch_profiler_profile_memory = get_bool_env_var(
            "AREAL_TORCH_PROFILER_PROFILE_MEMORY", default="true"
        )
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )

        total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        global_step = 0
        data_generator = cycle_dataloader(self.train_dataloader)
        for global_step in range(start_step, max_steps):
            if (
                config.total_train_steps is not None
                and global_step >= config.total_train_steps
            ):
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            with (
                stats_tracker.record_timing("load_bcast"),
                perf_tracer.trace_scope(
                    "train.load_bcast",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                batch = self._load_bcast_from(data_generator)

            self.saver.maybe_wait_for_staging()

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
                and memory_profile_this_rank
            ):
                self.actor.start_memory_profile(config.memory_profiler.max_entries)

            with (
                stats_tracker.record_timing("train_step"),
                perf_tracer.trace_scope(
                    "train.sft_step",
                    category=Category.COMPUTE,
                    enable_profiler=True,
                    profiler_args={
                        "record_shapes": True,
                        "with_stack": False,
                        "profile_memory": torch_profiler_profile_memory,
                    },
                    args={"global_step": global_step},
                ),
            ):
                self.actor.train_lm(batch)
                self.actor.step_lr_scheduler()
                self.actor.get_device_stats().log("after train step")

            if (
                config.memory_profiler is not None
                and global_step in config.memory_profiler.profile_steps
                and memory_profile_this_rank
            ):
                log_dir = StatsLogger.get_log_path(config.stats_logger)
                snapshot_dir = os.path.join(
                    log_dir, "memory_snapshots", f"step_{global_step}"
                )
                os.makedirs(snapshot_dir, exist_ok=True)
                self.actor.stop_memory_profile(snapshot_dir)
                logger.info(f"Memory snapshots saved to {snapshot_dir}")

            self.actor.set_version(global_step + 1)

            with (
                stats_tracker.record_timing("save"),
                perf_tracer.trace_scope(
                    "train.save",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

            with (
                stats_tracker.record_timing("checkpoint_for_recover"),
                perf_tracer.trace_scope(
                    "train.checkpoint",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_recover_checkpoint(
                    epoch=epoch, epoch_step=step, global_step=global_step
                )

            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                # SPMD mode never populates ``_fetch_buffer`` (no RTensor
                # round-trip), so the fan-out is single-controller only.
                if is_single_controller():
                    self.actor.clear_batches(batch)
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

            self._save_perf_tracer(step=global_step)

    def close(self):
        # [FUNCTION DOC]
        # Tears down: finalizes the saver (flushes async checkpoints), closes connected
        # RDatasets, destroys the DataController, closes the stats logger, destroys the
        # actor and force-saves perf traces.
        # KNOWN ISSUE: unlike RWTrainer it accesses _train_rdataset etc. without hasattr
        # guards, so close on a partially constructed instance raises AttributeError.
        self.saver.finalize()
        if self._train_rdataset is not None:
            self._train_rdataset.close()
        if self._valid_rdataset is not None:
            self._valid_rdataset.close()
        if hasattr(self, "data_controller") and self.data_controller is not None:
            self.data_controller.destroy()
        self.stats_logger.close()
        self.actor.destroy()
        perf_tracer.save(force=True)

    def _config_perf_tracer(self):
        # [FUNCTION DOC]
        # HELPER: If a perf_tracer config exists, configures the local tracer as role
        # "master" with RANK from env, and in single-controller mode also configures the
        # actor's tracer.
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")
        if not is_single_controller():
            return
        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")

    def _save_perf_tracer(self, step: int):
        # [FUNCTION DOC]
        # HELPER: If perf tracing is configured, saves actor and local perf traces for
        # the step.
        if self.config.perf_tracer is None:
            return
        self.actor.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        # [FUNCTION DOC]
        # HELPER: Instantiates a Local/Ray/Slurm scheduler based on
        # config.scheduler.type with the full experiment config. Raises
        # NotImplementedError for unknown types.
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        elif cfg.type == "ray":
            return RayScheduler(exp_config=self.config)
        elif cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _create_dataloader(
        self,
        dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        # [FUNCTION DOC]
        # HELPER: Wraps create_dataloader for the dataset with the given DP rank/world
        # size and dataset config, using collate_samples_to_list as collate_fn. Returns
        # a StatefulDataLoader.
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
            collate_fn=collate_samples_to_list,
        )

    def _create_actor(
        self, actor_config: TrainEngineConfig
    ) -> FSDPLMEngine | MegatronLMEngine | ArchonLMEngine | LMController:
        # [FUNCTION DOC]
        # HELPER: Chooses the language-model engine class by backend ("fsdp",
        # "megatron", "archon"; lazily imported), else raises ValueError. In single-
        # controller mode builds a remote controller via as_controller(config,
        # scheduler), otherwise instantiates the engine locally. Creates its process
        # groups with the allocation's parallel strategy and returns it.
        if self.actor_alloc.backend == "fsdp":
            from areal.engine import FSDPLMEngine

            actor_cls = FSDPLMEngine
        elif self.actor_alloc.backend == "megatron":
            from areal.engine import MegatronLMEngine

            actor_cls = MegatronLMEngine
        elif self.actor_alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonLMEngine

            actor_cls = ArchonLMEngine
        else:
            raise ValueError(
                f"Invalid backend: {self.actor_alloc.backend}, "
                f"expected fsdp, megatron, or archon"
            )
        if is_single_controller():
            actor = actor_cls.as_controller(actor_config, self.scheduler)
        else:
            actor = actor_cls(config=actor_config)
        actor.create_process_group(parallel_strategy=self.actor_alloc.parallel)
        return actor

    def _load_bcast_from(self, data_generator):
        # [FUNCTION DOC]
        # HELPER: Pulls the next batch from the generator. In single-controller mode
        # returns it as is; otherwise moves tensors to the current device and broadcasts
        # from the DP head across the context+model parallel group (data is identical
        # within that group).
        batch = next(data_generator)

        if is_single_controller():
            return batch

        batch = tensor_container_to(batch, current_platform.current_device())
        batch = broadcast_tensor_container(
            batch,
            src_rank=self.actor.current_data_parallel_head(),
            group=self.actor.context_and_model_parallel_group,
        )
        return batch

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Saves an HF-format model (with tokenizer/processor) via Saver if due;
        # when not in async mode, barriers on the actor CPU group and synchronizes the
        # device.
        self.saver.save(
            self.actor,
            epoch,
            epoch_step,
            global_step,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        if not self.saver.is_async:
            dist.barrier(group=self.actor.cpu_group)
            current_platform.synchronize()

    def _save_recover_checkpoint(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Dumps a recoverable checkpoint (actor as "default", step info,
        # saver/evaluator/ stats logger/dataloader states, tokenizer/processor) via
        # RecoverHandler if due, then barriers on the CPU group and synchronizes the
        # device.
        to_save: dict = dict(default=self.actor)
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

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate_fn(self):
        # [FUNCTION DOC]
        # HELPER: Runs one full pass over the validation dataloader,
        # loading/broadcasting each batch and calling actor.evaluate_lm, then barrier +
        # device sync.
        # KNOWN ISSUE: assumes valid_dataloader is not None (only called via _evaluate
        # which checks).
        data_generator = cycle_dataloader(self.valid_dataloader, num_cycles=1)
        for _ in range(len(self.valid_dataloader)):
            data = self._load_bcast_from(data_generator)
            self.actor.evaluate_lm(data)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        # [FUNCTION DOC]
        # HELPER: If a validation dataloader exists, lets the Evaluator decide (by
        # frequency) whether to run _evaluate_fn, then barrier + device sync. Returns
        # None.
        if self.valid_dataloader is None:
            return
        self.evaluator.evaluate(
            self._evaluate_fn,
            epoch,
            epoch_step,
            global_step,
        )
        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _export_and_commit_stats(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Exports actor stats (reduced across ranks) and commits them to the
        # StatsLogger for the step, then barrier + device sync.
        stats = self.actor.export_stats()
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def __enter__(self):
        # [FUNCTION DOC]
        # Context-manager entry; returns self.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Context-manager exit: logs the exception with traceback if one occurred, calls
        # close(), and returns False so exceptions propagate.
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        return False

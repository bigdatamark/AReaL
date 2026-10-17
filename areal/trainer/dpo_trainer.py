# [MODULE DOC]
# DPO training entry point. Defines the collate function that turns preference items
# into chosen/rejected sequence dicts, and DPOTrainer, which builds an actor (trainable
# policy) and a reference model engine on the configured backend (fsdp, megatron or
# archon), the dataloaders, evaluator, saver, recover handler and stats logger, then
# runs the training loop. Works in SPMD mode (one trainer per rank, torch.distributed)
# or single-controller mode (one process that drives remote workers through a
# scheduler). Logger "DPOTrainer".
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import FinetuneSpec, Scheduler, StepInfo
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    DPOConfig,
    DPOEngineConfig,
    TrainDatasetConfig,
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
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import is_single_controller
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

if TYPE_CHECKING:
    from datasets import Dataset

    from areal.engine import FSDPDPOEngine, MegatronDPOEngine
    from areal.experimental.engine.archon_engine import ArchonDPOEngine
    from areal.trainer.dpo.dpo_engine import DPOController

logger = logging.getLogger("DPOTrainer")


def dpo_modeling_collate_fn(items):
    # [FUNCTION DOC]
    # Collate function for DPO dataloaders. For each item emits two dicts, chosen first
    # then rejected, each with input_ids [1, L] (list inputs converted to tensors), an
    # all-True bool attention_mask [1, L] and loss_mask [1, L] from {prefix}_loss_mask.
    # Returns a flat list of 2*len(items) dicts, so consecutive entries form a pair.
    # Raises KeyError if fields are missing.
    """Collate DPO items into per-sequence tensor dicts.

    Each dataset item has ``chosen_ids``, ``rejected_ids``, ``chosen_loss_mask``,
    and ``rejected_loss_mask``.
    This produces two dicts per item (chosen first, then rejected), each with
    ``[1, seqlen]`` tensors, ``attention_mask``, and ``loss_mask``.
    """
    result = []
    for item in items:
        for prefix in ("chosen", "rejected"):
            ids = item[f"{prefix}_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor(ids)
            seqlen = ids.shape[0]

            loss_mask = item[f"{prefix}_loss_mask"]
            if not torch.is_tensor(loss_mask):
                loss_mask = torch.tensor(loss_mask)

            result.append(
                {
                    "input_ids": ids.unsqueeze(0),
                    "attention_mask": torch.ones(1, seqlen, dtype=torch.bool),
                    "loss_mask": loss_mask.unsqueeze(0),
                }
            )
    return result


class DPOTrainer:
    # [CLASS DOC]
    # Orchestrates DPO training: owns config, tokenizer/processor, optional scheduler,
    # actor and ref engines (or their controllers), dataloaders, optional data-service
    # controller for remote RDatasets, Evaluator, Saver, RecoverHandler and StatsLogger.
    # Usable as a context manager that calls close() on exit.
    def __init__(
        self,
        config: DPOConfig,
        train_dataset: Dataset,
        valid_dataset: Dataset | None = None,
    ):
        # [FUNCTION DOC]
        # Builds all training components. Steps: read RANK env (default 0); in single-
        # controller mode enable file logging to the stats-logger path and create the
        # scheduler; load HF processor/tokenizer; seed RNGs with key "trainer{rank}";
        # parse actor and ref allocations from their backend strings and create both
        # engines (with process groups); remember config.ref.offload. If single-
        # controller and train_dataset is an RDataset, start a DataController (role
        # "data") and connect the dataset (and later the valid dataset) under ids
        # "{experiment}_{trial}_train/valid". Creates the train dataloader sharded by
        # the actor's DP rank/world size, a FinetuneSpec (dataset_size = n_batches *
        # batch_size), initializes actor (role "actor") and ref (role "ref"), builds the
        # optional valid dataloader only when both valid config and dataset are given,
        # creates Evaluator, Saver, RecoverHandler, StatsLogger, loads recover state
        # into actor/saver/evaluator/logger/ dataloader, and configures the perf tracer.
        # Side effects: process/worker launch, model loading, file I/O.
        # KNOWN ISSUE: _should_offload_ref is stored but never used in this class, so
        # ref offload is not applied by the trainer.
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

        # Set seed.
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")

        # Parse per-engine allocation.
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")

        self.actor = self._create_actor(config.actor)

        ref_alloc = ModelAllocation.from_str(config.ref.backend, name="ref")
        self.ref = self._create_ref(config.ref, ref_alloc)
        self._should_offload_ref = config.ref.offload

        if is_single_controller() and isinstance(train_dataset, RDataset):
            ds_cfg = DataServiceConfig.from_dataset_config(config.train_dataset)
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

        self.train_dataloader = self._create_dataloader(
            train_dataset,
            dataset_config=self.config.train_dataset,
            rank=self.actor.data_parallel_rank,
            world_size=self.actor.data_parallel_world_size,
        )

        ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        self.actor.initialize(addr=None, ft_spec=ft_spec, role="actor")
        self.ref.initialize(addr=None, ft_spec=ft_spec, role="ref")

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
                dataset_config=self.config.valid_dataset,
                rank=self.actor.data_parallel_rank,
                world_size=self.actor.data_parallel_world_size,
            )

        # Set up evaluation
        self.evaluator = Evaluator(config.evaluator, ft_spec)

        # Set up save as HF model
        self.saver = Saver(config.saver, ft_spec)
        self.recover_handler = RecoverHandler(config.recover, ft_spec)

        # Set up statistics logging (wandb, tensorboard, etc.)
        self.stats_logger = StatsLogger(config, ft_spec)

        # Set up checkpointing for recover
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
        # Main loop. Starts at the step after the recovered one (or 0); runs to
        # total_train_epochs * steps_per_epoch, stopping early at total_train_steps if
        # set. Each step (with timing and perf-trace scopes): load and broadcast a
        # batch; wait for async checkpoint staging; compute ref log-probs and attach
        # them as ref_logprobs per sequence (1-D outputs get a leading dim);
        # actor.train_dpo, step LR scheduler, log device stats; set actor version to
        # step+1; save HF model, recover checkpoint and run evaluation per their
        # frequency controllers; in single-controller mode clear cached batches on
        # actor, ref and data controller; export and commit stats; save perf trace.
        # Returns None.
        config = self.config
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

            # Wait for async checkpoint staging to complete before modifying parameters
            self.saver.maybe_wait_for_staging()

            with (
                stats_tracker.record_timing("ref_logp"),
                perf_tracer.trace_scope(
                    "train.ref_logp",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                ref_logps = self.ref.compute_logp(batch)
                for seq_dict, logp in zip(batch, ref_logps):
                    seq_dict["ref_logprobs"] = (
                        logp.unsqueeze(0) if logp.ndim == 1 else logp
                    )
                self.ref.get_device_stats().log("ref logp")

            with (
                stats_tracker.record_timing("train_step"),
                perf_tracer.trace_scope(
                    "train.dpo_step",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self.actor.train_dpo(batch)
                self.actor.step_lr_scheduler()
                self.actor.get_device_stats().log("after train step")

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
                    # ref DP heads also localized `batch` in compute_logp —
                    # drain their per-process fetch buffers. See
                    # areal-project/AReaL#1209.
                    if self.ref is not None:
                        self.ref.clear_batches(batch)
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
        # Releases resources: finalizes saver, closes train/valid RDatasets and destroys
        # data controller if present (hasattr guards allow partial init), closes stats
        # logger, destroys actor and ref engines, and force-saves the perf tracer.
        self.saver.finalize()
        if hasattr(self, "_train_rdataset") and self._train_rdataset is not None:
            self._train_rdataset.close()
        if hasattr(self, "_valid_rdataset") and self._valid_rdataset is not None:
            self._valid_rdataset.close()
        if hasattr(self, "data_controller") and self.data_controller is not None:
            self.data_controller.destroy()
        self.stats_logger.close()
        self.actor.destroy()
        self.ref.destroy()
        perf_tracer.save(force=True)

    def _config_perf_tracer(self):
        # [FUNCTION DOC]
        # HELPER: If config.perf_tracer is set, configures the local tracer for this
        # rank with role "master"; in single-controller mode also configures the actor
        # workers' tracers. The ref engine is not configured.
        rank = int(os.getenv("RANK", "0"))
        if self.config.perf_tracer is None:
            return
        perf_tracer.configure(self.config.perf_tracer, rank=rank, role="master")
        if not is_single_controller():
            return
        self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")

    def _save_perf_tracer(self, step: int):
        # [FUNCTION DOC]
        # HELPER: If perf tracing is enabled, saves actor worker traces and the local
        # trace for the given step.
        if self.config.perf_tracer is None:
            return
        self.actor.save_perf_tracer(step=step)
        perf_tracer.save(step=step)

    def _init_scheduler(self) -> Scheduler:
        # [FUNCTION DOC]
        # HELPER: Returns a Local/Ray/SlurmScheduler from config.scheduler.type; raises
        # NotImplementedError for any other type.
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
        dataset: Dataset,
        dataset_config: TrainDatasetConfig | ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        # [FUNCTION DOC]
        # HELPER: Creates a StatefulDataLoader via create_dataloader for the given
        # rank/world size and dataset config, using dpo_modeling_collate_fn.
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
            collate_fn=dpo_modeling_collate_fn,
        )

    def _create_actor(
        self, actor_config: DPOEngineConfig
    ) -> FSDPDPOEngine | MegatronDPOEngine | ArchonDPOEngine | DPOController:
        # [FUNCTION DOC]
        # HELPER: Chooses the DPO engine class from the actor allocation backend (fsdp
        # -> FSDPDPOEngine, megatron -> MegatronDPOEngine, archon -> ArchonDPOEngine;
        # lazy imports), raising ValueError otherwise. In single-controller mode wraps
        # it as a controller with the scheduler; else instantiates locally. Creates its
        # process group with the allocation's parallel strategy and returns it.
        if self.actor_alloc.backend == "fsdp":
            from areal.engine import FSDPDPOEngine

            actor_cls = FSDPDPOEngine
        elif self.actor_alloc.backend == "megatron":
            from areal.engine import MegatronDPOEngine

            actor_cls = MegatronDPOEngine
        elif self.actor_alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonDPOEngine

            actor_cls = ArchonDPOEngine
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

    def _create_ref(
        self, ref_config: DPOEngineConfig, alloc: ModelAllocation
    ) -> FSDPDPOEngine | MegatronDPOEngine | ArchonDPOEngine | DPOController:
        # [FUNCTION DOC]
        # HELPER: Same as _create_actor but for the reference model, using the given ref
        # config and allocation.
        if alloc.backend == "fsdp":
            from areal.engine import FSDPDPOEngine

            ref_cls = FSDPDPOEngine
        elif alloc.backend == "megatron":
            from areal.engine import MegatronDPOEngine

            ref_cls = MegatronDPOEngine
        elif alloc.backend == "archon":
            from areal.experimental.engine.archon_engine import ArchonDPOEngine

            ref_cls = ArchonDPOEngine
        else:
            raise ValueError(
                f"Invalid ref backend: {alloc.backend}, expected fsdp, megatron, or archon"
            )
        if is_single_controller():
            ref = ref_cls.as_controller(ref_config, self.scheduler)
        else:
            ref = ref_cls(config=ref_config)
        ref.create_process_group(parallel_strategy=alloc.parallel)
        return ref

    def _load_bcast_from(self, data_generator):
        # [FUNCTION DOC]
        # HELPER: Pulls the next batch from the generator. In single-controller mode
        # returns it as is; otherwise moves it to the current device and broadcasts it
        # from the DP head across the context+model parallel group so all ranks in that
        # group see identical data.
        batch = next(data_generator)

        if is_single_controller():
            return batch

        # NOTE: data are identical across model+context parallel group
        batch = tensor_container_to(batch, current_platform.current_device())
        batch = broadcast_tensor_container(
            batch,
            src_rank=self.actor.current_data_parallel_head(),
            group=self.actor.context_and_model_parallel_group,
        )
        return batch

    def _save_hf(self, epoch: int, epoch_step: int, global_step: int):
        # [FUNCTION DOC]
        # HELPER: Asks the saver to save the actor as an HF model (with
        # tokenizer/processor) if due. For synchronous savers, then does a barrier on
        # the actor CPU group and a device sync.
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
        # HELPER: Dumps a recover checkpoint (actor under key "default", StepInfo,
        # saver/evaluator/ logger/dataloader state, tokenizer/processor) via the recover
        # handler when due, then barrier and device sync. The ref model is not
        # checkpointed (frozen).
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
        # HELPER: One validation pass: iterates the valid dataloader once,
        # loading/broadcasting each batch, attaching ref log-probs and calling
        # actor.evaluate_dpo (stats recorded in the "dpo-eval" scope); then barrier and
        # device sync.
        data_generator = cycle_dataloader(self.valid_dataloader, num_cycles=1)
        for _ in range(len(self.valid_dataloader)):
            data = self._load_bcast_from(data_generator)
            ref_logps = self.ref.compute_logp(data)
            for seq_dict, logp in zip(data, ref_logps):
                seq_dict["ref_logprobs"] = logp.unsqueeze(0) if logp.ndim == 1 else logp
            self.actor.evaluate_dpo(data)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def _evaluate(
        self,
        epoch: int,
        epoch_step: int,
        global_step: int,
    ):
        # [FUNCTION DOC]
        # HELPER: No-op without a valid dataloader; otherwise lets the Evaluator decide
        # by frequency whether to run _evaluate_fn, then barrier and device sync.
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
        # HELPER: Exports the actor's aggregated stats and commits them to the stats
        # logger for this step, then barrier and device sync.
        stats = self.actor.export_stats()
        self.stats_logger.commit(epoch, epoch_step, global_step, stats)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()

    def __enter__(self):
        # [FUNCTION DOC]
        # Returns self for use in a with-statement.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Logs the exception (with traceback) if one occurred, calls close(), and
        # returns False so exceptions propagate.
        if exc_type is not None:
            logger.error(f"Training failed with exception: {exc_value}", exc_info=True)
        self.close()
        return False

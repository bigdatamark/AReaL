# [MODULE DOC]
# Fault-tolerance (recover) support for training loops: periodically saves DCP
# checkpoints of the train engine(s) together with a "recover_info" directory
# holding step counters and state dicts of the saver, evaluator, stats logger,
# dataloader and recover frequency controller; on restart restores all of them,
# reloads the checkpoint and re-syncs rollout engine weights (including the
# colocated AWEX pause/offload protocol). Layout lives under Saver's save root.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
import inspect
import json
import os
import pickle
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import AutoProcessor

from areal.api import (
    FinetuneSpec,
    InferenceEngine,
    SaveLoadMeta,
    StepInfo,
    TrainEngine,
    WeightUpdateMeta,
)
from areal.api.cli_args import RecoverConfig
from areal.infra import TrainController
from areal.utils import logging, timeutil
from areal.utils.evaluator import Evaluator
from areal.utils.saver import Saver

if TYPE_CHECKING:
    from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("Recover")


class InValidRecoverInfo(Exception):
    # [CLASS DOC]
    # Raised when a recover_info directory exists but cannot be fully loaded/parsed.
    pass


@dataclasses.dataclass
class RecoverInfo:
    # [CLASS DOC]
    # Dataclass snapshot of training progress: last completed StepInfo (resume starts
    # at its next()), state dicts of saver, evaluator, stats logger, checkpoint
    # frequency controller, and dataloader state (per-rank list when distributed).
    # Last step info is the counter of the saved checkpoint.
    # Recover will start from the next iteration, obtained by `last_step_info.next()`.
    last_step_info: StepInfo

    saver_info: dict
    evaluator_info: dict
    stats_logger_info: dict
    dataloader_info: dict | list[dict]
    checkpoint_info: dict

    def dump(self, dump_dir: str):
        # [FUNCTION DOC]
        # Writes the snapshot to dump_dir. When torch.distributed is initialized, all-gathers
        # every rank's dataloader state (collective; all ranks must call) and only rank 0
        # writes. Creates the directory and writes step_info.json, saver_info.json,
        # evaluator_info.json, stats_logger_info.json, checkpoint_info.json (JSON) and
        # dataloader_info.pkl (pickle).
        # Dumps the recover info to multiple files in `dump_dir`:
        # 1. step_info.json: contains the recover info
        # 2. *_info.json or *_info.pkl: contains other informantion required for recover.

        if dist.is_initialized():
            # Since dataloader state is different across distributed ranks,
            # we need to all gather the dataloader state from all ranks.
            # In this situation, saved dataloader_info is a list of states from all ranks.
            dataloader_info = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(dataloader_info, self.dataloader_info)

            # To avoid contention, do not dump on multiple ranks
            if dist.get_rank() != 0:
                return
        else:
            dataloader_info = self.dataloader_info

        os.makedirs(dump_dir, exist_ok=True)
        step_info_path = os.path.join(dump_dir, "step_info.json")
        with open(step_info_path, "w") as f:
            json.dump(dataclasses.asdict(self.last_step_info), f, indent=4)

        saver_info_path = os.path.join(dump_dir, "saver_info.json")
        with open(saver_info_path, "w") as f:
            json.dump(self.saver_info, f, indent=4)

        evaluator_info_path = os.path.join(dump_dir, "evaluator_info.json")
        with open(evaluator_info_path, "w") as f:
            json.dump(self.evaluator_info, f, indent=4)

        stats_logger_info_path = os.path.join(dump_dir, "stats_logger_info.json")
        with open(stats_logger_info_path, "w") as f:
            json.dump(self.stats_logger_info, f, indent=4)

        checkpoint_info_path = os.path.join(dump_dir, "checkpoint_info.json")
        with open(checkpoint_info_path, "w") as f:
            json.dump(self.checkpoint_info, f, indent=4)

        dataloader_info_path = os.path.join(dump_dir, "dataloader_info.pkl")
        with open(dataloader_info_path, "wb") as f:
            pickle.dump(dataloader_info, f)

    @classmethod
    def load(cls, load_dir: str):
        # [FUNCTION DOC]
        # Classmethod: raises FileNotFoundError if load_dir is missing; otherwise reads the
        # same six files and builds a RecoverInfo. A list dataloader state is indexed by the
        # current rank when distributed (asserts the world size matches). Any failure is
        # logged and re-raised as InValidRecoverInfo. Uses pickle (trusted input only).
        # Loads the recover info from multiple files in `load_dir`:
        if not os.path.exists(load_dir):
            raise FileNotFoundError(
                f"Recover info directory {load_dir} does not exist."
            )

        try:
            step_info_path = os.path.join(load_dir, "step_info.json")
            with open(step_info_path) as f:
                step_info_dict = json.load(f)
                last_step_info = StepInfo(**step_info_dict)

            evaluator_info_path = os.path.join(load_dir, "evaluator_info.json")
            with open(evaluator_info_path) as f:
                evaluator_info = json.load(f)

            saver_info_path = os.path.join(load_dir, "saver_info.json")
            with open(saver_info_path) as f:
                saver_info = json.load(f)

            stats_logger_info_path = os.path.join(load_dir, "stats_logger_info.json")
            with open(stats_logger_info_path) as f:
                stats_logger_info = json.load(f)

            checkpoint_info_path = os.path.join(load_dir, "checkpoint_info.json")
            with open(checkpoint_info_path) as f:
                checkpoint_info = json.load(f)

            dataloader_info_path = os.path.join(load_dir, "dataloader_info.pkl")
            with open(dataloader_info_path, "rb") as f:
                dataloader_info = pickle.load(f)
                if isinstance(dataloader_info, list):
                    # If dataloader_info a list, it means it is saved from a distributed run.
                    if dist.is_initialized():
                        # Loading dataloader states in a distributed context.
                        assert dist.get_world_size() == len(dataloader_info), (
                            f"Dataloader info list length {len(dataloader_info)} does not match "
                            f"the world size {dist.get_world_size()}."
                        )
                        dataloader_info = dataloader_info[dist.get_rank()]

            return cls(
                last_step_info=last_step_info,
                saver_info=saver_info,
                evaluator_info=evaluator_info,
                stats_logger_info=stats_logger_info,
                dataloader_info=dataloader_info,
                checkpoint_info=checkpoint_info,
            )
        except Exception as e:
            logger.error(f"Failed to load recover info from {load_dir}: {e}")
            raise InValidRecoverInfo(f"Invalid recover info in {load_dir}") from e


class RecoverHandler:
    # [CLASS DOC]
    # Drives recover checkpointing for a run: holds RecoverConfig, FinetuneSpec, the
    # last dumped StepInfo and an EpochStepTimeFreqCtl from config freq_epochs /
    # freq_steps / freq_secs. Engines may be a single engine/controller or a dict
    # name -> engine (one checkpoint per name). Not supported with
    # GatewayTrainController (v2).
    def __init__(self, config: RecoverConfig, ft_spec: FinetuneSpec):
        # [FUNCTION DOC]
        # Stores config and spec, initializes last_step_info to all -1 counters and builds
        # the frequency controller.
        self.config = config
        self.ft_spec = ft_spec
        self.last_step_info = StepInfo(
            epoch=-1,
            epoch_step=-1,
            global_step=-1,
            steps_per_epoch=ft_spec.steps_per_epoch,
        )
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
        )

    @staticmethod
    def recover_info_path(
        experiment_name: str,
        trial_name: str,
        fileroot: str,
    ):
        # [FUNCTION DOC]
        # Returns "<save_root>/recover_info" (creating the save root as a side effect).
        return os.path.join(
            Saver.get_save_root(experiment_name, trial_name, fileroot),
            "recover_info",
        )

    @staticmethod
    def _is_gateway_train_controller(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if engine, or any value of a dict of engines, is a
        # GatewayTrainController (imported lazily).
        from areal.v2.training_service.controller.controller import (
            GatewayTrainController,
        )

        if isinstance(engine, GatewayTrainController):
            return True
        if isinstance(engine, dict):
            return any(
                isinstance(controller, GatewayTrainController)
                for controller in engine.values()
            )
        return False

    def _ensure_recover_supported(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Raises NotImplementedError if a GatewayTrainController is involved.
        if self._is_gateway_train_controller(engine):
            raise NotImplementedError(
                "Recovery is not supported with GatewayTrainController "
                '(`_version="v2"`) yet. Disable `recover.mode` or use '
                '`_version="v1"`.'
            )

    @staticmethod
    def _normalize_recover_engines(
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
    ) -> dict[str, TrainEngine | TrainController]:
        # [FUNCTION DOC]
        # HELPER: Returns a dict as-is, else wraps a single engine as {"default": engine}.
        if isinstance(engine, dict):
            return engine
        return {"default": engine}

    @staticmethod
    def _should_run_awex_colocate_transfer(
        inference_engine: InferenceEngine | None,
        weight_update_meta: WeightUpdateMeta | None,
        colocated_rollout: bool,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: True iff an inference engine is given, the weight update meta's type is
        # "awex" and the rollout is colocated with the actor.
        """Whether recovery must drive the AWEX colocate pre-transfer sequence.

        The transport type alone is not enough: v2 selects AWEX for every
        non-LoRA run regardless of placement, so the caller has to state whether
        actor and rollout physically share devices.
        """
        return (
            inference_engine is not None
            and getattr(weight_update_meta, "type", None) == "awex"
            and colocated_rollout
        )

    @staticmethod
    def _require_colocate_rollout_protocol(
        inference_engine: InferenceEngine,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Verifies the inference engine has a callable pause_generation_sync and an
        # offload method accepting a "tags" parameter (unknown signatures are accepted).
        # Raises NotImplementedError listing missing capabilities.
        missing = []
        if not callable(getattr(inference_engine, "pause_generation_sync", None)):
            missing.append("pause_generation_sync()")

        offload = getattr(inference_engine, "offload", None)
        if not callable(offload):
            missing.append("offload(tags=...)")
        else:
            try:
                accepts_tags = "tags" in inspect.signature(offload).parameters
            except (TypeError, ValueError):
                accepts_tags = True
            if not accepts_tags:
                missing.append("offload(tags=...)")

        if missing:
            raise NotImplementedError(
                "Colocated AWEX recovery needs a rollout engine implementing "
                f"{', '.join(missing)}, which {type(inference_engine).__name__} "
                "does not provide. Disable `recover.mode` or run this "
                "configuration without actor-rollout colocation."
            )

    def dump(
        self,
        engine: TrainEngine
        | TrainController
        | dict[str, TrainEngine | TrainController],
        step_info: StepInfo,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        # [FUNCTION DOC]
        # No-op if mode is "disabled"/"off". Rejects unsupported controllers, ticks the
        # frequency controller (epoch tick when step_info is the epoch's last step) and
        # returns if not due. Otherwise saves a DCP checkpoint per engine, records
        # step_info as last_step_info, collects all component state dicts into a
        # RecoverInfo and dumps it to the recover_info path (distributed collective inside).
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        # currently only support recover on one engine
        if not self.freq_ctl.check(
            epochs=int(step_info.epoch_step == self.ft_spec.steps_per_epoch - 1),
            steps=1,
        ):
            return
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )
        for name, engine_ in normalized_engine.items():
            self._save_checkpoint(
                engine_,
                name=name,
                tokenizer=tokenizer,
                processor=processor,
                base_model_path=base_model_path,
            )

        self.last_step_info = step_info
        recover_info = RecoverInfo(
            last_step_info=self.last_step_info,
            saver_info=saver.state_dict(),
            evaluator_info=evaluator.state_dict(),
            stats_logger_info=stats_logger.state_dict(),
            dataloader_info=dataloader.state_dict(),
            checkpoint_info=self.freq_ctl.state_dict(),
        )

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        recover_info.dump(recover_info_path)

    def load(
        self,
        engine: TrainEngine | dict[str, TrainEngine] | TrainController,
        saver: Saver,
        evaluator: Evaluator,
        stats_logger: StatsLogger,
        dataloader: Any,
        inference_engine: InferenceEngine | None = None,
        weight_update_meta: WeightUpdateMeta | None = None,
        inference_engine_update_from: str = "default",
        colocated_rollout: bool = False,
    ) -> RecoverInfo | None:
        # [FUNCTION DOC]
        # No-op (returns None) if mode is disabled/off. Raises ValueError if an inference
        # engine is given without weight_update_meta. Loads RecoverInfo and restores saver,
        # frequency controller, evaluator, stats logger and dataloader state. Loads each
        # engine's DCP checkpoint (deferred for colocated AWEX until the rollout engine has
        # paused generation and offloaded kv_cache and weights, to avoid OOM). If an
        # inference engine is given: connects the engine named inference_engine_update_from
        # to it with meta versioned global_step + 1, pauses rollout, pushes weights
        # (always resuming afterwards), then sets both versions. Returns the RecoverInfo.
        # FileNotFoundError / InValidRecoverInfo are caught and logged as a warning,
        # returning None.
        # KNOWN ISSUE: a FileNotFoundError from checkpoint loading after component states
        # were already restored is also swallowed, leaving a partially recovered run.
        if self.config.mode in ("disabled", "off"):
            return
        self._ensure_recover_supported(engine)
        if inference_engine is not None and weight_update_meta is None:
            raise ValueError("Weight update meta is required for recovery.")

        # TODO(agent): GatewayTrainController is currently duck-typed and does
        # not satisfy this TrainController type check. Extend recovery to accept
        # controller-v2 instances (or make v2 inherit TrainController) before
        # relying on resumed runs with `_version="v2"`.
        normalized_engine: dict[str, TrainEngine | TrainController] = (
            self._normalize_recover_engines(engine)
        )

        recover_info_path = self.recover_info_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
        )
        logger.info(f"Loading recover info from {recover_info_path}")
        try:
            recover_info: RecoverInfo = RecoverInfo.load(recover_info_path)
            logger.info(f"Recovering from {recover_info.last_step_info.next()}.")
            saver.load_state_dict(recover_info.saver_info)
            self.freq_ctl.load_state_dict(recover_info.checkpoint_info)
            evaluator.load_state_dict(recover_info.evaluator_info)
            stats_logger.load_state_dict(recover_info.stats_logger_info)
            dataloader.load_state_dict(recover_info.dataloader_info)

            global_step = recover_info.last_step_info.global_step
            recovery_version = global_step + 1

            is_awex_colocate = self._should_run_awex_colocate_transfer(
                inference_engine=inference_engine,
                weight_update_meta=weight_update_meta,
                colocated_rollout=colocated_rollout,
            )
            if is_awex_colocate:
                self._require_colocate_rollout_protocol(inference_engine)

            if not is_awex_colocate:
                for name, engine_ in normalized_engine.items():
                    self._load_checkpoint(engine_, name=name)

            if inference_engine is not None:
                assert weight_update_meta is not None
                update_engine = normalized_engine[inference_engine_update_from]
                versioned_meta = weight_update_meta.with_version(recovery_version)
                update_engine.connect_engine(inference_engine, versioned_meta)
                inference_engine.pause()
                try:
                    # AWEX colocate transfer requires the full engine-level
                    # pause/offload protocol, not just the controller pause. The
                    # sglang plugin's patched event loop only drains the weight-
                    # update queue while scheduler._engine_paused is True (set by
                    # pause_generation), and the reader-side protocol expects the
                    # engine's kv/weights released before the writer publishes.
                    # Without this the recover-path transfer deadlocks: reader
                    # never consumes the queued version marker, writer blocks on
                    # weights_update_finished forever.
                    # Mirror of the trainer's pre-update sequence; the reverse
                    # side (kv_cache onload) happens inside update_weights.
                    if is_awex_colocate:
                        inference_engine.pause_generation_sync()
                        inference_engine.offload(tags=["kv_cache"])
                        inference_engine.offload(tags=["weights"])
                        # Load the actor checkpoint only after the colocated
                        # rollout engine has released its GPU memory; loading
                        # first would stack DCP weights/optimizer on top of the
                        # still-resident sglang allocation and risk OOM.
                        for name, engine_ in normalized_engine.items():
                            self._load_checkpoint(engine_, name=name)
                    update_engine.update_weights(versioned_meta)
                finally:
                    # Always resume: leaving rollout paused after a failed
                    # checkpoint load or transfer would hang every later step.
                    inference_engine.resume()
                update_engine.set_version(recovery_version)
                inference_engine.set_version(recovery_version)
            return recover_info
        except (FileNotFoundError, InValidRecoverInfo):
            logger.warning(
                f"Resume info not found at {recover_info_path}. "
                f"This should not be a resumed experiment!"
            )

    def _save_checkpoint(
        self,
        engine: TrainEngine,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        processor: AutoProcessor | None = None,
        base_model_path: str | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Saves engine state in DCP format to the recover checkpoint path for name,
        # with optimizer state unless config.no_save_optim; passes tokenizer, processor and
        # base_model_path through. Logs the path.
        path = Saver.get_recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name=name,
        )
        weight_format = "dcp"
        with_optim = not self.config.no_save_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=tokenizer,
            processor=processor,
            base_model_path=base_model_path,
        )
        engine.save(meta)
        logger.info(f"Saved recover checkpoint to {path} (with_optim={with_optim})")

    def _load_checkpoint(
        self,
        engine: TrainEngine | TrainController,
        name: str = "default",
        tokenizer: PreTrainedTokenizerFast | None = None,
        base_model_path: str | None = None,
    ):
        # [FUNCTION DOC]
        # HELPER: Loads the DCP checkpoint for name into engine (optimizer unless
        # config.no_load_optim). Raises FileNotFoundError if the path is missing.
        # KNOWN ISSUE: the path helper creates the directory, so the existence check
        # always passes; tokenizer and base_model_path parameters are ignored.
        path = Saver.get_recover_checkpoint_path(
            self.config.experiment_name,
            self.config.trial_name,
            self.config.fileroot,
            name=name,
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
        weight_format = "dcp"
        with_optim = not self.config.no_load_optim
        meta = SaveLoadMeta(
            path=path,
            weight_format=weight_format,
            with_optim=with_optim,
            tokenizer=None,
            processor=None,
            base_model_path=None,
        )
        engine.load(meta)


def check_if_auto_recover(config: RecoverConfig) -> bool:
    # [FUNCTION DOC]
    # Returns True if recover_info exists and loads, its last epoch is >= 0, and every
    # subdirectory of the save root has a recover checkpoint directory; logs and
    # returns False otherwise.
    # KNOWN ISSUE: get_recover_checkpoint_path creates the directory it returns, so the
    # per-model existence check always passes (and creates stray directories,
    # including under recover_info itself).
    # This method is called by check_if_recover to check if the experiment should
    # recover from a previous run when recovery is enabled ("on" or "auto" mode).
    experiment_name = config.experiment_name
    trial_name = config.trial_name
    fileroot = config.fileroot
    recover_info_path = RecoverHandler.recover_info_path(
        experiment_name, trial_name, fileroot
    )
    logger.info(f"Searching for recover info file in {recover_info_path}.")
    if os.path.exists(str(recover_info_path)):
        try:
            info = RecoverInfo.load(recover_info_path)
        except Exception as e:
            logger.warning(f"Failed to load recover info from {recover_info_path}: {e}")
            return False
        if info.last_step_info.epoch < 0:
            msg = (
                f"Recover checkpoint is not valid. "
                f"Expected last_step_info.epoch >= 0, "
                f"but found {info.last_step_info.epoch}"
            )
            logger.warning(msg)
            return False

        save_root = Saver.get_save_root(experiment_name, trial_name, fileroot)
        for name in os.listdir(save_root):
            if not os.path.isdir(os.path.join(save_root, name)):
                continue
            path = Saver.get_recover_checkpoint_path(
                experiment_name, trial_name, fileroot, name=name
            )
            if not os.path.exists(path):
                logger.warning(f"Recover checkpoint for model {name} does not exist.")
                return False
        return True
    logger.warning(f"Recover info not found at: {recover_info_path}")
    return False


def check_if_recover(config: RecoverConfig, _run_id: int) -> bool:
    # [FUNCTION DOC]
    # Returns False when config.mode is "disabled"/"off"; otherwise (on/auto) delegates
    # to check_if_auto_recover. _run_id is unused.
    """Check if the experiment should be a recover run.

    When recovery is enabled ('on' or 'auto'), this checks if valid recover
    info and checkpoints are available for automatic recovery.

    Args:
        config: Recovery configuration.
        _run_id: Unused. Kept for API compatibility.

    Returns:
        True if the experiment should recover from a previous run.
    """
    if config.mode in ("disabled", "off"):
        return False
    # Both "on" and "auto" use auto-recovery behavior
    return check_if_auto_recover(config)

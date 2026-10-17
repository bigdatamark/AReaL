# [MODULE DOC]
# Experiment metrics logger (rank 0 only) that fans training statistics out to
# Weights & Biases, SwanLab, Trackio and TensorBoard and pretty-prints them to the
# "StatsLogger" logger.
# SPDX-License-Identifier: Apache-2.0

import getpass
import os
import time
from dataclasses import asdict

import swanlab
import torch.distributed as dist
import trackio
import wandb
from tensorboardX import SummaryWriter

from areal.api import FinetuneSpec
from areal.api.cli_args import BaseExperimentConfig, StatsLoggerConfig
from areal.utils import logging
from areal.utils.printing import tabulate_stats
from areal.version import version_info

logger = logging.getLogger("StatsLogger", "system")


class StatsLogger:
    # [CLASS DOC]
    # Wraps experiment-tracking backends. Built from the full experiment config (uses
    # its stats_logger section) and a FinetuneSpec (for epoch/step totals in log lines).
    # Tracks the last committed step so logged steps stay monotonic across restarts.
    # All public methods are no-ops on non-zero ranks when torch.distributed is
    # initialized.
    def __init__(self, config: BaseExperimentConfig, ft_spec: FinetuneSpec):
        # [FUNCTION DOC]
        # Rejects a bare StatsLoggerConfig (ValueError, deprecated usage), stores configs,
        # calls init() and sets the last commit step to -1.
        if isinstance(config, StatsLoggerConfig):
            raise ValueError(
                "Passing config.stats_logger as the config is deprecated. "
                "Please pass the full config instead."
            )
        self.exp_config = config
        self.config = config.stats_logger
        self.ft_spec = ft_spec
        self.init()

        self._last_commit_step = -1

    def init(self):
        # [FUNCTION DOC]
        # Rank 0 only. Exports WANDB_BASE_URL / WANDB_API_KEY env vars if configured,
        # records a start time, logs into wandb unless disabled, builds the full experiment
        # config dict plus git/version info, and initializes wandb (project/name/group
        # defaults from experiment and trial names, id "<exp>_<trial>_<suffix>" where suffix
        # "timestamp" becomes the current time, resume allowed, dir from get_log_path).
        # Logs into SwanLab if enabled and always calls swanlab.init (mode may be
        # disabled); initializes Trackio if enabled; creates a TensorBoard SummaryWriter if
        # a tensorboard path is set. Network I/O and directory creation.
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.config.wandb.wandb_base_url:
            os.environ["WANDB_BASE_URL"] = self.config.wandb.wandb_base_url
        if self.config.wandb.wandb_api_key:
            os.environ["WANDB_API_KEY"] = self.config.wandb.wandb_api_key

        self.start_time = time.perf_counter()
        # wandb init, connect to remote wandb host
        if self.config.wandb.mode != "disabled":
            wandb.login()

        suffix = self.config.wandb.id_suffix
        if suffix == "timestamp":
            suffix = time.strftime("%Y_%m_%d_%H_%M_%S")

        exp_config_dict = asdict(self.exp_config)
        exp_config_dict["version_info"] = {
            "commit_id": version_info.commit,
            "branch": version_info.branch,
            "is_dirty": version_info.is_dirty,
            "version": version_info.full_version_with_dirty_description,
        }

        wandb.init(
            mode=self.config.wandb.mode,
            entity=self.config.wandb.entity,
            project=self.config.wandb.project or self.config.experiment_name,
            name=self.config.wandb.name or self.config.trial_name,
            job_type=self.config.wandb.job_type,
            group=self.config.wandb.group
            or f"{self.config.experiment_name}_{self.config.trial_name}",
            notes=self.config.wandb.notes,
            tags=self.config.wandb.tags,
            config=exp_config_dict,  # save all experiment config to wandb
            dir=self.get_log_path(self.config),
            force=True,
            id=f"{self.config.experiment_name}_{self.config.trial_name}_{suffix}",
            resume="allow",
        )

        swanlab_config = self.config.swanlab
        if swanlab_config.mode != "disabled":
            if swanlab_config.api_key:
                swanlab.login(swanlab_config.api_key)
            else:
                swanlab.login()

        swanlab_config = self.config.swanlab
        swanlab.init(
            project=swanlab_config.project or self.config.experiment_name,
            experiment_name=swanlab_config.name or self.config.trial_name + "_train",
            # NOTE: change from swanlab_config.config to log all experiment config, to be tested
            config=exp_config_dict,
            logdir=self.get_log_path(self.config),
            mode=swanlab_config.mode,
        )

        # trackio init
        self._trackio_enabled = False
        trackio_config = self.config.trackio
        if trackio_config.mode != "disabled":
            trackio.init(
                project=trackio_config.project or self.config.experiment_name,
                name=trackio_config.name or self.config.trial_name,
                config=exp_config_dict,
                space_id=trackio_config.space_id,
            )
            self._trackio_enabled = True

        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def state_dict(self):
        # [FUNCTION DOC]
        # Returns {"last_commit_step": ...} for checkpointing.
        return {
            "last_commit_step": self._last_commit_step,
        }

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC]
        # Restores last_commit_step from a state dict (KeyError if missing).
        self._last_commit_step = state_dict["last_commit_step"]

    def close(self):
        # [FUNCTION DOC]
        # Rank 0 only: logs total elapsed time and finishes wandb, swanlab, trackio (if
        # enabled) and closes the TensorBoard writer.
        # KNOWN ISSUE: start_time comes from time.perf_counter() but elapsed time is
        # computed with time.monotonic(), so the reported duration is meaningless.
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Training completes! Total time elapsed {time.monotonic() - self.start_time:.2f}."
        )
        wandb.finish()
        swanlab.finish()
        if getattr(self, "_trackio_enabled", False):
            trackio.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    def commit(self, epoch: int, step: int, global_step: int, data: dict | list[dict]):
        # [FUNCTION DOC]
        # Rank 0 only. Logs an epoch/step progress line, then for each stats dict (a single
        # dict is wrapped in a list) drops keys ending in "__count", prints the table and
        # logs it to every enabled backend at step log_step + i, where log_step is
        # max(global_step, last_commit_step + 1). Updates last_commit_step to the last
        # step used.
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        logger.info(
            f"Epoch {epoch + 1}/{self.ft_spec.total_train_epochs} "
            f"Step {step + 1}/{self.ft_spec.steps_per_epoch} "
            f"Train step {global_step + 1}/{self.ft_spec.total_train_steps} done."
        )
        if isinstance(data, dict):
            data = [data]
        log_step = max(global_step, self._last_commit_step + 1)
        for i, item in enumerate(data):
            # Filter out counter keys for scalar variables
            item = {k: v for k, v in item.items() if not k.endswith("__count")}

            logger.info(f"Stats ({i + 1}/{len(data)}):")
            self.print_stats(item)
            wandb.log(item, step=log_step + i)
            swanlab.log(item, step=log_step + i)
            if getattr(self, "_trackio_enabled", False):
                trackio.log(item, step=log_step + i)
            if self.summary_writer is not None:
                for key, val in item.items():
                    self.summary_writer.add_scalar(f"{key}", val, log_step + i)
        self._last_commit_step = log_step + len(data) - 1

    def print_stats(self, stats: dict[str, float]):
        # [FUNCTION DOC]
        # Logs the stats dict as a tabulated grid at INFO level.
        logger.info("\n" + tabulate_stats(stats))

    @staticmethod
    def get_log_path(
        config: StatsLoggerConfig | None = None,
        experiment_name: str | None = None,
        trial_name: str | None = None,
        fileroot: str | None = None,
    ) -> str:
        # [FUNCTION DOC]
        # Returns "<fileroot>/logs/<user>/<experiment>/<trial>", taking the three values from
        # config if given, else from the arguments; creates the directory. Raises
        # ValueError if any is missing.
        if config is not None:
            experiment_name = config.experiment_name
            trial_name = config.trial_name
            fileroot = config.fileroot
        if not fileroot or not experiment_name or not trial_name:
            raise ValueError(
                "fileroot, experiment_name, and trial_name must be provided."
            )
        path = f"{fileroot}/logs/{getpass.getuser()}/{experiment_name}/{trial_name}"
        os.makedirs(path, exist_ok=True)
        return path

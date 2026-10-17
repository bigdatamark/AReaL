# [MODULE DOC]
# Evaluation scheduling helper: decides when to run an evaluation callback based on
# epoch, step and wall-time frequencies.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from areal.api import FinetuneSpec
from areal.api.cli_args import EvaluatorConfig
from areal.utils import timeutil


class Evaluator:
    # [CLASS DOC]
    # Wraps an EpochStepTimeFreqCtl configured from EvaluatorConfig and checkpointable
    # via state_dict/load_state_dict.
    def __init__(self, config: EvaluatorConfig, ft_spec: FinetuneSpec):
        # [FUNCTION DOC]
        # Stores config and ft_spec (as attribute ft_sepc, misspelled) and builds the
        # frequency controller from freq_epochs/steps/secs, with eval_before_train as
        # the initial epoch value.
        self.config = config
        self.ft_sepc = ft_spec
        self.freq_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.freq_epochs,
            freq_step=config.freq_steps,
            freq_sec=config.freq_secs,
            initial_epoch_value=config.eval_before_train,
        )

    def state_dict(self):
        # [FUNCTION DOC]
        # Returns the frequency controller's state dict.
        return self.freq_ctl.state_dict()

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC]
        # Restores the frequency controller from state_dict.
        self.freq_ctl.load_state_dict(state_dict)

    def evaluate(
        self,
        evaluate_fn: Callable,
        epoch: int,
        step: int,
        global_step: int,
    ):
        # [FUNCTION DOC]
        # Calls evaluate_fn() if the frequency controller fires for this step (counts
        # one step, and one epoch when step is the last step of the epoch). epoch and
        # global_step are unused. Returns None.
        if not self.freq_ctl.check(
            epochs=int(step == self.ft_sepc.steps_per_epoch - 1), steps=1
        ):
            return
        evaluate_fn()

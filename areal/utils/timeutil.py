# [MODULE DOC]
# Timing/frequency utilities and simple scalar schedules: FrequencyControl (trigger
# by elapsed steps and/or seconds, optionally synchronized across distributed
# ranks), EpochStepTimeFreqCtl (OR of epoch/step/time triggers, used for saving and
# evaluation), and step-indexed value schedulers (constant, linear, exponential,
# cosine, chained). INFINITE_DURATION is ~1000 years in seconds.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import math
import threading
from abc import ABC
from datetime import datetime

import torch
import torch.distributed as dist

from areal.infra.platforms import current_platform

INFINITE_DURATION = 60 * 60 * 24 * 365 * 1000


class FrequencyControl:
    # [CLASS DOC]
    # Thread-safe trigger that fires when at least frequency_steps steps and/or
    # frequency_seconds seconds have passed since the last firing (both must hold if
    # both are set; never fires if neither is set, except an initial_value firing).
    # Uses wall-clock datetime and name-mangled private counters.
    """An utility to control the execution of code with a time or/and step
    frequency."""

    def __init__(
        self,
        frequency_seconds: int | None = None,
        frequency_steps: int | None = None,
        initial_value: bool = False,
        group: dist.ProcessGroup | None = None,
    ):
        # [FUNCTION DOC]
        # Stores frequencies and optional process group; initializes start/last times to
        # now, step counters to 0, intervals to None, the initial-fire flag and a lock.
        """Initialization method of FrequencyControl.
        Args:
            frequency_seconds: Minimal interval between two trigger.
            frequency_steps: Minimal number of steps between two triggers.
            initial_value: In true, the first call of check() returns True.
            group: If torch distributed is initialized, time intervals will be synchronized
                across processes in the specified distributed group. If group is None, the
                default process group will be used.

        NOTE:
            - If both frequency_seconds and frequency_steps are None, the checking will always return False except
             for the specified initial value.
            - If passed both, both frequency and steps conditions have to be met for check() to return True.
            - If one is passed, checking on the other condition will be ignored.
        """
        self.frequency_seconds = frequency_seconds
        self.frequency_steps = frequency_steps
        self.group = group
        self.__start_time = datetime.now()
        self.__steps = 0
        self.__last_time = datetime.now()
        self.__last_steps = 0
        self.__interval_seconds = self.__interval_steps = None
        self.__initial_value = initial_value
        self.__lock = threading.Lock()

    def state_dict(self):
        # [FUNCTION DOC]
        # Returns a dict of config and counters (times as POSIX timestamps). Note the key
        # "frequence_seconds" (misspelled) is used consistently with load_state_dict.
        return dict(
            frequence_seconds=self.frequency_seconds,
            frequency_steps=self.frequency_steps,
            start_time=self.__start_time.timestamp(),
            steps=self.__steps,
            last_time=self.__last_time.timestamp(),
            last_steps=self.__last_steps,
            interval_steps=self.__interval_steps,
            interval_seconds=self.__interval_seconds,
            initial_value=self.__initial_value,
        )

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC]
        # Restores all fields from state_dict (timestamps converted back to datetimes).
        # KeyError on missing keys.
        self.frequency_seconds = state_dict["frequence_seconds"]
        self.frequency_steps = state_dict["frequency_steps"]
        self.__start_time = datetime.fromtimestamp(state_dict["start_time"])
        self.__steps = state_dict["steps"]
        self.__last_time = datetime.fromtimestamp(state_dict["last_time"])
        self.__last_steps = state_dict["last_steps"]
        self.__interval_steps = state_dict["interval_steps"]
        self.__interval_seconds = state_dict["interval_seconds"]
        self.__initial_value = state_dict["initial_value"]

    @property
    def total_seconds(self):
        # [FUNCTION DOC]
        # Property: seconds elapsed since construction (or the restored start time).
        now = datetime.now()
        return (now - self.__start_time).total_seconds()

    @property
    def total_steps(self):
        # [FUNCTION DOC]
        # Property: cumulative steps counted by check().
        return self.__steps

    @property
    def interval_seconds(self):
        # [FUNCTION DOC]
        # Property: seconds between the last firing and the most recent check (None before
        # the first non-initial check).
        return self.__interval_seconds

    @property
    def interval_steps(self):
        # [FUNCTION DOC]
        # Property: steps between the last firing and the most recent check.
        return self.__interval_steps

    def check(self, steps=1):
        # [FUNCTION DOC]
        # Under the lock: adds steps to the counter. If the initial-value flag is set,
        # clears it, records now/steps as the last firing and returns True. Otherwise
        # computes interval seconds/steps; returns False if no frequency is configured.
        # For the time condition, when torch.distributed is initialized the interval is
        # all-reduced with MAX over the group (on the accelerator device; collective, all
        # ranks must call). Returns False if either configured condition is unmet;
        # otherwise records the firing and returns True.
        # KNOWN ISSUE: when frequency_seconds is set but torch.distributed is not
        # initialized, the local variable interval_seconds is never assigned and the
        # comparison raises UnboundLocalError.
        """Check whether frequency condition is met.
        Args:
            steps: number of step between this and the last call of check()

        Returns:
            flag: True if condition is met, False other wise
        """
        with self.__lock:
            now = datetime.now()
            self.__steps += steps

            if self.__initial_value:
                self.__last_time = now
                self.__last_steps = self.__steps
                self.__initial_value = False
                return True

            self.__interval_seconds = (now - self.__last_time).total_seconds()
            self.__interval_steps = self.__steps - self.__last_steps
            if self.frequency_steps is None and self.frequency_seconds is None:
                return False
            if self.frequency_seconds is not None:
                if dist.is_initialized():
                    interval_seconds = torch.tensor(
                        self.__interval_seconds, device=current_platform.device_type
                    )
                    dist.all_reduce(
                        interval_seconds, op=dist.ReduceOp.MAX, group=self.group
                    )
                    self.__interval_seconds = interval_seconds.item()
                # We pass time check if the interval second in any process
                # is larger than self.frequency_seconds.
                if interval_seconds < self.frequency_seconds:
                    return False
            if (
                self.frequency_steps is not None
                and self.__interval_steps < self.frequency_steps
            ):
                return False
            self.__last_time = now
            self.__last_steps = self.__steps

            return True

    def reset_time(self):
        # [FUNCTION DOC]
        # Resets the last-firing time to now (step counters unchanged; not locked).
        self.__last_time = datetime.now()


@dataclasses.dataclass
class EpochStepTimeFreqCtl:
    # [CLASS DOC]
    # Dataclass combining three FrequencyControls: per-epoch (step-counted with
    # freq_epoch, optional initial firing), per-step (freq_step) and time-based
    # (freq_sec, synchronized over group). None disables a criterion.
    freq_epoch: int | None
    freq_step: int | None
    freq_sec: int | None
    initial_epoch_value: bool = False
    group: dist.ProcessGroup | None = None

    def __post_init__(self):
        # [FUNCTION DOC]
        # Builds the epoch, step and time FrequencyControl instances.
        self.epoch_ctl = FrequencyControl(
            frequency_steps=self.freq_epoch,
            initial_value=self.initial_epoch_value,
        )
        self.step_ctl = FrequencyControl(frequency_steps=self.freq_step)
        self.time_ctl = FrequencyControl(
            frequency_seconds=self.freq_sec, group=self.group
        )

    def check(self, epochs: int, steps: int):
        # [FUNCTION DOC]
        # Advances all three controls (always evaluated, no short-circuit; epochs and
        # steps are increments) and returns True if any of them fires.
        x, y, z = (
            self.epoch_ctl.check(epochs),
            self.step_ctl.check(steps),
            self.time_ctl.check(),
        )
        return x or y or z

    def state_dict(self):
        # [FUNCTION DOC]
        # Returns {"epoch", "step", "time"} sub-state dicts.
        return dict(
            epoch=self.epoch_ctl.state_dict(),
            step=self.step_ctl.state_dict(),
            time=self.time_ctl.state_dict(),
        )

    def load_state_dict(self, state_dict):
        # [FUNCTION DOC]
        # Restores the three sub-controls from their state dicts.
        self.epoch_ctl.load_state_dict(state_dict["epoch"])
        self.step_ctl.load_state_dict(state_dict["step"])
        self.time_ctl.load_state_dict(state_dict["time"])


@dataclasses.dataclass
class Scheduler(ABC):
    # [CLASS DOC]
    # Abstract dataclass for a value schedule over steps [0, total_iters] starting at
    # init_value; subclasses implement _get.
    init_value: float
    total_iters: int

    def __post_init__(self):
        # [FUNCTION DOC]
        # Raises ValueError unless total_iters > 0.
        if self.total_iters <= 0:
            raise ValueError("total_iters should be a positive number.")

    def get(self, step: int) -> float:
        # [FUNCTION DOC]
        # Validates 0 <= step <= total_iters (ValueError otherwise) and returns _get(step).
        """Get the scheduled value at the current `step`."""
        if step < 0 or step > self.total_iters:
            raise ValueError(
                f"Scheduler step should be in the interval [0, {self.total_iters}]. Input {step}."
            )
        return self._get(step)

    def _get(self, step):
        # [FUNCTION DOC]
        # Abstract hook; raises NotImplementedError.
        raise NotImplementedError()

    @property
    def final_value(self):
        # [FUNCTION DOC]
        # Property: value at step total_iters.
        return self.get(step=self.total_iters)


@dataclasses.dataclass
class ConstantScheduler(Scheduler):
    # [CLASS DOC]
    # Schedule that always returns init_value.
    def _get(self, *args, **kwargs) -> float:
        # [FUNCTION DOC]
        # Returns init_value regardless of arguments.
        return self.init_value


@dataclasses.dataclass
class LinearScheduler(Scheduler):
    # [CLASS DOC]
    # Linear interpolation from init_value to end_value over total_iters.
    end_value: float

    def _get(self, step: int) -> float:
        # [FUNCTION DOC]
        # Returns init + (end - init) * step / total_iters.
        return (
            self.end_value - self.init_value
        ) / self.total_iters * step + self.init_value


@dataclasses.dataclass
class ExponentialScheduler(Scheduler):
    # [CLASS DOC]
    # Geometric decay: init_value * decay^step.
    decay: float

    def _get(self, step: int) -> float:
        # [FUNCTION DOC]
        # Returns init_value multiplied by decay raised to step.
        return self.init_value * self.decay**step


@dataclasses.dataclass
class CosineDecayScheduler(Scheduler):
    # [CLASS DOC]
    # Half-cosine decay from init_value to end_value (end must be smaller).
    end_value: float

    def __post_init__(self):
        # [FUNCTION DOC]
        # Runs base validation, then raises ValueError if end_value >= init_value.
        super().__post_init__()
        if self.end_value >= self.init_value:
            raise ValueError("end_value should be smaller than init_value!")

    def _get(self, step: int) -> float:
        # [FUNCTION DOC]
        # Returns end + (init - end) * 0.5 * (1 + cos(pi * step / total_iters)).
        delta = self.init_value - self.end_value
        return (
            delta * 0.5 * (1 + math.cos(math.pi / self.total_iters * step))
            + self.end_value
        )


@dataclasses.dataclass
class ChainedScheduler:
    # [CLASS DOC]
    # Concatenation of schedulers run back to back; adjacent schedules must join
    # continuously.
    schedulers: list[Scheduler]

    @property
    def total_iters(self):
        # [FUNCTION DOC]
        # Property: sum of the sub-schedulers' total_iters.
        return sum(x.total_iters for x in self.schedulers)

    @property
    def init_value(self):
        # [FUNCTION DOC]
        # Property: first scheduler's init_value.
        return self.schedulers[0].init_value

    @property
    def final_value(self):
        # [FUNCTION DOC]
        # Property: last scheduler's final value.
        return self.schedulers[-1].final_value

    def __post_init__(self):
        # [FUNCTION DOC]
        # Raises ValueError if any scheduler's value at step 0 differs by more than 1e-8
        # from the previous scheduler's final value.
        for i in range(len(self.schedulers) - 1):
            # Float point err 1e-8.
            if (
                abs(self.schedulers[i + 1].get(0) - self.schedulers[i].final_value)
                > 1e-8
            ):
                raise ValueError(
                    f"Values should be consecutive between "
                    f"the {i}-th ({type(self.schedulers[i])}) and "
                    f"the {i + 1}-th {type(self.schedulers[i + 1])} schedulers! "
                    f"End value is {self.schedulers[i].final_value} and the "
                    f"next init value is {self.schedulers[i + 1].get(0)}."
                )

    def get(self, step: int) -> float:
        # [FUNCTION DOC]
        # Walks the schedulers subtracting each total_iters until the step falls within one
        # and returns that scheduler's value at the local step.
        # KNOWN ISSUE: steps beyond the total fall through and return None; negative steps
        # raise from the first scheduler. A boundary step maps to the end of the earlier
        # scheduler (equal values by construction).
        for s in self.schedulers:
            if step > s.total_iters:
                step -= s.total_iters
            else:
                return s.get(step)

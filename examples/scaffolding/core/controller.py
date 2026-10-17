# [MODULE DOC]
# Vendored scaffolding controllers. A controller's process(tasks, **kwargs) is a
# generator: each yield hands either a list of Tasks (to be executed by workers matching
# each task's worker_tag) or a ParallelProcess (several sub-controller generators run
# concurrently) to the ScaffoldingLlm scheduler, which resumes the generator once the
# yielded work completes. Includes plain generation/chat/reward controllers plus
# majority-vote and best-of-N composites.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.controller

import copy
import logging
from abc import ABC
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch

from .math_utils import get_digit_majority_vote_result
from .task import ChatTask, GenerationTask, Task

logger = logging.getLogger(__name__)


class Controller(ABC):
    # [CLASS DOC]
    # Abstract base controller. Holds task_collections (name -> TaskCollection,
    # populated by the with_task_collection decorator). Subclasses implement process().
    task_collections: dict = {}

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes an empty per-instance task_collections dict.
        self.task_collections = {}

    def clone(self):
        # [FUNCTION DOC]
        # Returns a deep copy of this controller (used to give each parallel branch its
        # own state).
        return copy.deepcopy(self)

    def generate(self, prompt: str, **kwargs):
        # [FUNCTION DOC]
        # Generator convenience: wraps prompt in a GenerationTask, delegates to
        # process() (yielding its work items) and finally returns the task's
        # ScaffoldingOutput as the generator's return value.
        task = GenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()

    def process(self, tasks: list[Task], **kwargs):
        # [FUNCTION DOC]
        # Abstract generator hook; raises NotImplementedError. Subclasses yield lists of
        # tasks or ParallelProcess objects.
        raise NotImplementedError


class ParallelProcess:
    # [CLASS DOC]
    # Bundle of sub-generators to be run concurrently by the scheduler; yielded by
    # composite controllers.
    def __init__(
        self,
        controllers: list[Controller],
        tasks_list: list[list[Task]],
        kwargs_list: list[Mapping[str, Any]],
    ):
        # [FUNCTION DOC]
        # Zips controllers, tasks_list and kwargs_list and stores, in sub_gens, the (not
        # yet started) generator from each controller.process(tasks, **kwargs). Extra
        # elements of longer lists are silently dropped by zip.
        self.sub_gens = []
        for controller, tasks, kwargs in zip(controllers, tasks_list, kwargs_list):
            gen = controller.process(tasks, **kwargs)
            self.sub_gens.append(gen)


# Controller runs multiple generation tasks.
class NativeGenerationController(Controller):
    # [CLASS DOC]
    # Controller that marks tasks for execution by a "generation" worker and applies
    # default sampling parameters.
    class WorkerTag(Enum):
        # [CLASS DOC]
        # Enum of worker tags this controller emits; GENERATION = "generation".
        GENERATION = "generation"

    def __init__(self, sampling_params: dict = None, streaming: bool = False):
        # [FUNCTION DOC]
        # Inputs: sampling_params (dict of GenerationTask field defaults; keys not
        # present in GenerationTask annotations are logged as warnings and dropped; the
        # passed dict is mutated), streaming (sets streaming_output_flag on tasks).
        super().__init__()
        if sampling_params is None:
            sampling_params = {}
        for key, value in list(sampling_params.items()):
            if key not in GenerationTask.__annotations__:
                logger.warning(f"{key} is not a supported field for GenerationTask")
                sampling_params.pop(key)
        self.sampling_params = sampling_params
        self.streaming = streaming

    # [GenerationTask] -> [GenerationTask] | [ChatTask] -> [ChatTask]
    def process(self, tasks: list[Task], **kwargs):
        # [FUNCTION DOC]
        # For each task: sets worker_tag to GENERATION, fills every sampling param whose
        # current attribute is None, sets streaming_output_flag; then yields the whole
        # task list once.
        for task in tasks:
            task.worker_tag = self.WorkerTag.GENERATION
            for key, value in self.sampling_params.items():
                if getattr(task, key) is None:
                    setattr(task, key, value)

            task.streaming_output_flag = self.streaming

        yield tasks


class NativeChatController(NativeGenerationController):
    # [CLASS DOC]
    # Generation controller that converts incoming tasks into ChatTasks before dispatch.
    def __init__(self, sampling_params: dict = None, streaming: bool = False):
        # [FUNCTION DOC]
        # Same arguments as NativeGenerationController.
        super().__init__(sampling_params, streaming)

    def process(self, tasks: list[Task], **kwargs):
        # [FUNCTION DOC]
        # Creates a new ChatTask from each task's input_str and delegates to the parent
        # process with those chat tasks.
        # KNOWN ISSUE: Results are produced on the new ChatTask objects only; the
        # original tasks passed in are never updated, so callers reading the originals
        # (e.g. Controller.generate) see no output.
        chat_tasks = [ChatTask.create_from_prompt(task.input_str) for task in tasks]
        yield from super().process(chat_tasks, **kwargs)


class NativeRewardController(Controller):
    # [CLASS DOC]
    # Controller that marks tasks for a "reward" worker; subclasses/workers are expected
    # to populate self.scores.
    def __init__(self):
        # [FUNCTION DOC]
        # Sets scores to None.
        # KNOWN ISSUE: Does not call super().__init__(), so task_collections falls back
        # to the class-level dict shared by all controllers.
        self.scores = None

    class WorkerTag(Enum):
        # [CLASS DOC]
        # Enum of worker tags this controller emits; REWARD = "reward".
        REWARD = "reward"

    def process(self, tasks: list[Task], **kwargs):
        # [FUNCTION DOC]
        # Sets worker_tag REWARD on every task and yields the list once. (Creates an
        # unused GenerationTask that is immediately shadowed by the loop variable.)
        task = GenerationTask()
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        yield tasks


class MajorityVoteController(Controller):
    # [CLASS DOC]
    # Self-consistency controller: samples several generations in parallel and keeps the
    # result whose extracted digit answer is the majority.
    def __init__(self, generation_controller: Controller, default_sample_num: int = 1):
        # [FUNCTION DOC]
        # Inputs: generation_controller (template cloned per sample), default_sample_num
        # (minimum number of samples).
        super().__init__()
        self.generation_controller = generation_controller
        self.default_sample_num = default_sample_num

    def clone(self):
        # [FUNCTION DOC]
        # Returns a new MajorityVoteController with a cloned generation controller and
        # the same default_sample_num.
        generation_controller = self.generation_controller.clone()
        return MajorityVoteController(generation_controller, self.default_sample_num)

    def process(
        self,
        tasks: list[Task],
        sample_num: int = 1,
        generation_kwargs: dict = {},
        majority_vote_kwargs: dict = {},
    ):
        # [FUNCTION DOC]
        # Uses max(sample_num, default_sample_num) samples. Clones the generation
        # controller and deep-copies the tasks and generation_kwargs for each sample,
        # yields one ParallelProcess running them all, then calls majority_vote on the
        # per-sample task lists and copies the winning sample's first-task result into
        # tasks[0]. Asserts the majority answer is a str. Only the first task is
        # considered.
        # KNOWN ISSUE: Mutable default dict arguments.
        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [
            self.generation_controller.clone() for _ in range(sample_num)
        ]
        tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(generation_kwargs) for _ in range(sample_num)
        ]

        yield ParallelProcess(
            generation_controllers, tasks_list, generation_kwargs_list
        )

        majority_index, majority_answer = self.majority_vote(
            tasks_list, **majority_vote_kwargs
        )

        assert isinstance(majority_answer, str), "majority_vote failed"
        tasks[0].result = tasks_list[majority_index][0].result

    def majority_vote(
        self, candidates_tasks: list[list[Task]], **kwargs
    ) -> tuple[int, str]:
        # [FUNCTION DOC]
        # Collects output_str of the first task of each candidate and returns
        # get_digit_majority_vote_result(candidates): (index of a majority candidate,
        # majority answer).
        candidates = [tasks[0].output_str for tasks in candidates_tasks]
        return get_digit_majority_vote_result(candidates)


class BestOfNController(Controller):
    # [CLASS DOC]
    # Best-of-N controller: generates N candidates in parallel, scores them with a
    # reward controller and keeps the highest-scoring one.
    def __init__(
        self,
        generation_controller: Controller,
        reward_controller: Controller,
        default_sample_num: int = 4,
    ):
        # [FUNCTION DOC]
        # Inputs: generation_controller, reward_controller (must set .scores after
        # processing), default_sample_num (minimum N, default 4).
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.default_sample_num = default_sample_num

    def clone(self):
        # [FUNCTION DOC]
        # Returns a new BestOfNController with cloned generation and reward controllers.
        generation_controller = self.generation_controller.clone()
        reward_controller = self.reward_controller.clone()
        return BestOfNController(
            generation_controller, reward_controller, self.default_sample_num
        )

    def process(
        self,
        tasks: list[Task],
        sample_num: int = 4,
        generation_kwargs: dict = {},
        reward_kwargs: dict = {},
        select_best_kwargs: dict = {},
    ):
        # [FUNCTION DOC]
        # Asserts exactly one task. Uses N = max(sample_num, default_sample_num);
        # deep-copies the task N times and yields a ParallelProcess running the same
        # generation controller instance on each copy; then delegates (yield from) to
        # the reward controller on all candidates. Asserts reward_controller.scores is
        # set, logs each candidate with its score at INFO, picks the best via
        # select_best and copies its result into the original task.
        # KNOWN ISSUE: The generation controller is not cloned per branch (unlike
        # MajorityVoteController), so stateful controllers would share state; mutable
        # default dict arguments.
        assert len(tasks) == 1, "BestOfNController only supports one task"
        task = tasks[0]

        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [self.generation_controller for _ in range(sample_num)]
        generation_kwargs_list = [generation_kwargs for _ in range(sample_num)]
        generation_tasks = [copy.deepcopy(task) for _ in range(sample_num)]

        yield ParallelProcess(
            generation_controllers,
            [[t] for t in generation_tasks],
            generation_kwargs_list,
        )

        yield from self.reward_controller.process(generation_tasks, **reward_kwargs)

        assert self.reward_controller.scores is not None
        reward_values = self.reward_controller.scores

        for i, gen_task, reward_value in zip(
            range(sample_num), generation_tasks, reward_values
        ):
            logger.info(f"[output {i}, score {reward_value}]:\n{gen_task.output_str}")

        best_task, best_idx = self.select_best(
            generation_tasks, reward_values, **select_best_kwargs
        )
        task.result = best_task.result

    def select_best(self, tasks: list[Task], reward_values, **kwargs) -> Task:
        # [FUNCTION DOC]
        # Returns (task with max reward, its index) using torch argmax over
        # reward_values (ties -> first).
        # KNOWN ISSUE: Return annotation says Task but a (Task, int) tuple is returned.
        max_index = torch.argmax(torch.tensor(reward_values)).item()
        return tasks[max_index], max_index

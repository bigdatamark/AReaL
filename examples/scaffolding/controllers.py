# [MODULE DOC]
# AReaL-specific scaffolding controllers and task collections that turn scaffolding
# rollouts into RL training data: verifiable-reward and LLM-judge reward controllers, a
# generate-then-reward pipeline (PipelineTrajectoryMaker), a multi-turn
# chat-with-reflection controller, ChatTracer (records each ChatTask turn as an
# InteractionWithTokenLogpReward), and TraceTrajectoryMaker (runs a rollout controller
# under ChatTracer and rewards the traced interactions). Rewards are written onto
# interaction objects that are exported as the ScaffoldingOutput data.
"""
RLVR Controllers for Scaffolding Framework.

This module provides controllers for RLVR (Reinforcement Learning with Verifiable
Rewards) that integrate with the scaffolding framework.

Key Components:
- RLVRRewardController: Controller that processes reward computation
- LLMJudgeController: Controller that uses an LLM to judge answer correctness
- PipelineTrajectoryMaker: Controller that composes generation and reward pipelines
- MultiTurnChatController: Controller for multi-turn chat with reflection
- ChatTracer: TaskCollection for tracing multi-turn chat conversations
- TraceTrajectoryMaker: Controller that traces ChatTask objects during rollout
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from enum import Enum
from typing import Any

from areal.api.reward_api import AsyncRewardWrapper
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging

from ._compat import (
    ChatTask,
    Controller,
    GenerationTask,
    NativeGenerationController,
    RoleMessage,
    Task,
    TaskCollection,
    UserMessage,
    with_task_collection,
)
from .task import (
    ChatRewardTask,
    RLVRRewardTask,
    TraceGenerationTask,
)

logger = logging.getLogger("RLVRControllers")


class RLVRRewardController(Controller):
    # [CLASS DOC]
    # Reward controller for verifiable rewards. Computes reward_fn(prompt_str,
    # completion_str, input_tokens, output_tokens, **task_data) for RLVRRewardTask (sets
    # task.reward and interaction.reward) or GenerationTask (stores in
    # customized_result_fields["reward"]), collecting all values in self.scores. Exposes
    # a REWARD worker tag but computes locally.
    """Controller for computing RLVR (verifiable) rewards.

    This controller processes RLVRRewardTask objects and computes rewards
    using a provided reward function. The reward function should verify
    whether the generated answer is correct.

    The reward computation follows the pattern from RLVRWorkflow._compute_rewards:
    1. Decode output tokens to string (if needed)
    2. Call reward_fn(prompt_str, completion_str, input_tokens, output_tokens, **task_data)
    3. Store the reward in the task and update the interaction object

    Parameters
    ----------
    reward_fn : Callable
        The reward function that takes (prompt, completions, prompt_ids, completion_ids, **data)
        and returns a reward value (typically 0 or 1 for verifiable rewards).

    Example
    -------
    ```python
    from areal.reward.gsm8k import gsm8k_reward_fn

    reward_controller = RLVRRewardController(gsm8k_reward_fn)
    ```
    """

    class WorkerTag(Enum):
        # [CLASS DOC]
        # Enum with the worker tag REWARD = "rlvr_reward".
        """Worker tag for reward computation."""

        REWARD = "rlvr_reward"

    def __init__(self, reward_fn: Callable[..., Any]):
        # [FUNCTION DOC]
        # Stores reward_fn, wraps it in AsyncRewardWrapper for the async path, and sets
        # scores to None.
        """Initialize the RLVR reward controller.

        Parameters
        ----------
        reward_fn : Callable
            The reward function for verifying answers.
        """
        super().__init__()
        self.reward_fn = reward_fn
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        self.scores: list[float] | None = None

    def __deepcopy__(self, memo):
        # [FUNCTION DOC]
        # Deep-copy hook: returns a fresh RLVRRewardController with the same reward_fn
        # (avoids copying the non-copyable wrapper state) and registers it in memo.
        # Other state (scores, task_collections) is not copied.
        """Create a new RLVRRewardController with the same reward function.

        ``AsyncRewardWrapper`` contains ``ProcessPoolExecutor`` and
        ``threading.Lock`` which cannot be deep-copied.  Instead of
        copying those objects, we create a fresh controller that shares
        the same underlying executor pool via ``AsyncRewardWrapper``'s
        class-level state.
        """
        new = RLVRRewardController(self.reward_fn)
        memo[id(self)] = new
        return new

    def process(self, tasks: list[Task], **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator. Tags every task with REWARD, resets scores, computes rewards
        # synchronously (blocking the caller, possibly the scheduler's event loop) for
        # RLVRRewardTask and GenerationTask instances as described in the class doc
        # (kwargs forwarded to the generation-task path), then yields the task list
        # once. Other task types are skipped (no score appended). Reward function
        # exceptions propagate.
        """Process reward tasks and compute rewards.

        This method computes rewards for each task using the reward function.
        The rewards are stored in:
        1. task.reward - the computed reward value
        2. task.interaction.reward - if an interaction object is provided
        3. self.scores - list of all computed rewards

        Parameters
        ----------
        tasks : list[Task]
            List of RLVRRewardTask objects to process.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        list[Task]
            The processed tasks with rewards computed.
        """
        # Mark tasks with worker tag (for potential worker-based execution)
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        # Compute rewards synchronously
        # Note: For async execution, this would be handled by a worker
        self.scores = []
        for task in tasks:
            if isinstance(task, RLVRRewardTask):
                reward = self._compute_reward(task)
                task.reward = reward
                self.scores.append(reward)

                # Update the interaction object if provided
                if task.interaction is not None:
                    task.interaction.reward = reward
            elif isinstance(task, GenerationTask):
                # For generation tasks, compute reward from customized fields
                reward = self._compute_reward_from_generation_task(task, **kwargs)
                task.customized_result_fields["reward"] = reward
                self.scores.append(reward)

        yield tasks

    def _compute_reward(self, task: RLVRRewardTask) -> float:
        # [FUNCTION DOC]
        # HELPER: Calls reward_fn with the task's prompt, completion, token lists and
        # task_data kwargs; returns the result as float.
        """Compute reward for an RLVR reward task.

        Parameters
        ----------
        task : RLVRRewardTask
            The reward task containing prompt, completion, and task data.

        Returns
        -------
        float
            The computed reward value.
        """
        reward = self.reward_fn(
            task.prompt_str,
            task.completion_str,
            task.input_tokens,
            task.output_tokens,
            **task.task_data,
        )
        return float(reward)

    def _compute_reward_from_generation_task(
        self, task: GenerationTask, **kwargs
    ) -> float:
        # [FUNCTION DOC]
        # HELPER: Calls reward_fn with kwargs["prompt_str"] (default task.input_str or
        # ""), output_str or "", copies of input/output tokens and kwargs["task_data"]
        # (default {}) as kwargs; returns float.
        """Compute reward from a generation task.

        Parameters
        ----------
        task : GenerationTask
            The completed generation task.
        **kwargs
            Should contain 'task_data' with ground truth.

        Returns
        -------
        float
            The computed reward value.
        """
        task_data = kwargs.get("task_data", {})
        prompt_str = kwargs.get("prompt_str", task.input_str or "")

        reward = self.reward_fn(
            prompt_str,
            task.output_str or "",
            list(task.input_tokens or []),
            list(task.output_tokens or []),
            **task_data,
        )
        return float(reward)

    async def aprocess(self, tasks: list[Task], **kwargs) -> Any:
        # [FUNCTION DOC]
        # Async variant of process (a coroutine, not a generator): tags tasks, resets
        # scores, awaits AsyncRewardWrapper for each RLVRRewardTask, sets task.reward /
        # interaction.reward, and returns the task list. GenerationTasks are ignored
        # here.
        """Process reward tasks asynchronously.

        This method computes rewards asynchronously using AsyncRewardWrapper.

        Parameters
        ----------
        tasks : list[Task]
            List of RLVRRewardTask objects to process.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        list[Task]
            The processed tasks with rewards computed.
        """
        # Mark tasks with worker tag
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        # Compute rewards asynchronously
        self.scores = []
        for task in tasks:
            if isinstance(task, RLVRRewardTask):
                reward = await self._acompute_reward(task)
                task.reward = reward
                self.scores.append(reward)

                # Update the interaction object if provided
                if task.interaction is not None:
                    task.interaction.reward = reward

        return tasks

    async def _acompute_reward(self, task: RLVRRewardTask) -> float:
        # [FUNCTION DOC]
        # HELPER: Awaits async_reward_fn with the same arguments as _compute_reward;
        # returns float.
        """Compute reward asynchronously for an RLVR reward task.

        Parameters
        ----------
        task : RLVRRewardTask
            The reward task containing prompt, completion, and task data.

        Returns
        -------
        float
            The computed reward value.
        """
        reward = await self.async_reward_fn(
            task.prompt_str,
            task.completion_str,
            task.input_tokens,
            task.output_tokens,
            **task.task_data,
        )
        return float(reward)


class ChatTracer(TaskCollection):
    # [CLASS DOC]
    # TaskCollection that records every completed ChatTask turn. After each yield, for
    # ChatTasks that carry a non-None completion attribute (set by the worker), builds
    # an InteractionWithTokenLogpReward and stores it in an InteractionCache keyed by
    # completion.id. get_trace_results exports them with reward discounting.
    """TaskCollection for tracing multi-turn chat conversations.

    This class traces ChatTask objects during the controller's process execution.
    A multi-turn conversation uses the same ChatTask object across multiple yields,
    allowing us to track the evolution of the conversation.

    The tracer:
    1. In `before_yield`: Records the state of ChatTask before worker execution
    2. In `after_yield`: Captures the new messages added by the worker and creates
       InteractionWithTokenLogpReward objects

    The traced results can be exported via `get_trace_results()`, which returns
    a dict[str, InteractionWithTokenLogpReward] similar to client.py's export_interactions.

    Parameters
    ----------
    reward_discount : float
        Discount factor for backward reward propagation across turns.
    export_style : str
        Export style for interactions: 'concat' (tree structure) or 'individual'.

    Example
    -------
    ```python
    tracer = ChatTracer(reward_discount=0.9, export_style="individual")
    # Used via with_task_collection decorator or TraceTrajectoryMaker
    ```
    """

    def __init__(
        self,
        reward_discount: float = 1.0,
        export_style: str = "individual",
    ):
        # [FUNCTION DOC]
        # Inputs: reward_discount (factor for backward reward propagation across turns
        # at export, default 1.0), export_style ("individual" or "concat"). Creates an
        # empty InteractionCache.
        super().__init__()
        self.reward_discount = reward_discount
        self.export_style = export_style

        # Cache for storing interactions, similar to InteractionCache in client.py
        self._cache = InteractionCache()

    def before_yield(self, tasks: list[Task]):
        # [FUNCTION DOC]
        # No-op hook before workers run.
        """Called before tasks are yielded to workers.

        Parameters
        ----------
        tasks : list[Task]
            List of tasks about to be yielded.
        """
        pass

    def after_yield(self, tasks: list[Task]):
        # [FUNCTION DOC]
        # For each ChatTask with a completion, creates an interaction (see
        # _create_interaction_from_chat_task) and caches it under completion.id (later
        # turns of the same task produce new ids, so each turn is a separate entry).
        """Called after tasks return from workers.

        Creates InteractionWithTokenLogpReward objects for each ChatTask.
        Uses task.completion.id as the interaction ID.

        Parameters
        ----------
        tasks : list[Task]
            List of tasks that have been processed by workers.
        """
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue

            # Skip tasks without a completion (e.g. not yet processed by worker)
            if not hasattr(task, "completion") or task.completion is None:
                continue

            interaction = self._create_interaction_from_chat_task(task)
            # Use completion.id as the interaction key
            completion_id = task.completion.id
            self._cache[completion_id] = interaction

    def _create_interaction_from_chat_task(
        self,
        task: ChatTask,
    ) -> InteractionWithTokenLogpReward:
        # [FUNCTION DOC]
        # HELPER: Builds an InteractionWithTokenLogpReward from a ChatTask: messages
        # converted to dicts, ModelResponse with copies of input/output tokens, zero
        # logprobs and version -1 per output token, reward None, empty
        # output_message_list, the task's completion, and chat_template_type =
        # export_style. Imports ModelResponse lazily.
        # KNOWN ISSUE: chat_template_type receives the export style
        # ("individual"/"concat") rather than a chat template type such as
        # "hf"/"concat"; placeholder logprobs/versions are not real policy values.
        """Create an InteractionWithTokenLogpReward from a ChatTask.

        Parameters
        ----------
        task : ChatTask
            The ChatTask. Must contain a `completion` attribute
            with the ChatCompletion object.

        Returns
        -------
        InteractionWithTokenLogpReward
            The interaction object capturing this turn.
        """
        from areal.api.io_struct import ModelResponse

        # Extract all messages
        messages = [
            msg.to_dict() if hasattr(msg, "to_dict") else msg for msg in task.messages
        ]

        # Create ModelResponse from task data
        input_tokens = list(task.input_tokens or [])
        output_tokens = list(task.output_tokens or [])

        model_response = ModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[-1] * len(output_tokens),
        )

        # Get completion from task (ChatTask will contain the ChatCompletion)
        completion = task.completion

        interaction = InteractionWithTokenLogpReward(
            model_response=model_response,
            reward=None,
            messages=messages,
            output_message_list=[],
            completion=completion,
            chat_template_type=self.export_style,
        )

        return interaction

    def get_trace_results(self) -> dict[str, InteractionWithTokenLogpReward]:
        # [FUNCTION DOC]
        # Returns {} if nothing was traced; otherwise the cache's export_interactions
        # with this tracer's export_style and reward_discount (which may apply
        # discounting/tree linking as implemented by InteractionCache).
        """Export traced interactions.

        Returns the traced interactions in the specified export style.
        Applies reward discount before export if configured.

        Returns
        -------
        dict[str, InteractionWithTokenLogpReward]
            Dictionary mapping interaction IDs to their data.

        See Also
        --------
        client.py : export_interactions method for similar functionality
        """
        if len(self._cache) == 0:
            return {}

        return self._cache.export_interactions(
            style=self.export_style,
            reward_discount=self.reward_discount,
        )

    def clear(self) -> None:
        # [FUNCTION DOC]
        # Empties the interaction cache.
        """Clear all traced data."""
        self._cache.clear()


class PipelineTrajectoryMaker(Controller):
    # [CLASS DOC]
    # Controller running generation then local reward computation for each task,
    # attaching interaction objects. Per-episode inputs task_data, prompt_str and
    # input_tokens are attributes set before generate().
    """Controller that composes generation and reward controllers into a pipeline.

    This controller orchestrates the full RLVR pipeline:
    1. Run generation via the generation controller
    2. Compute rewards via the reward controller
    3. Assemble results into InteractionWithTokenLogpReward objects

    Parameters
    ----------
    generation_controller : Controller
        The controller for text generation (e.g., NativeGenerationController).
    reward_controller : RLVRRewardController
        The controller for reward computation.
    task_data : dict[str, Any]
        Task data containing ground truth (e.g., "answer" field) for reward computation.
    prompt_str : str
        The prompt string used for generation.

    Example
    -------
    ```python
    from examples.scaffolding._compat import NativeGenerationController

    gen_controller = NativeGenerationController()
    reward_controller = RLVRRewardController(gsm8k_reward_fn)
    trajectory_maker = PipelineTrajectoryMaker(
        gen_controller,
        reward_controller,
        task_data={"answer": "42"},
        prompt_str="What is the answer?",
    )
    ```
    """

    def __init__(
        self,
        generation_controller: Controller,
        reward_controller: RLVRRewardController,
        task_data: dict[str, Any] | None = None,
        prompt_str: str = "",
        input_tokens: list[int] | None = None,
    ):
        # [FUNCTION DOC]
        # Stores the generation and reward controllers and per-episode task_data
        # (default {}), prompt_str (default "") and input_tokens (default []).
        """Initialize the pipeline trajectory maker.

        Parameters
        ----------
        generation_controller : Controller
            The generation controller.
        reward_controller : RLVRRewardController
            The reward controller.
        task_data : dict[str, Any], optional
            Task data containing ground truth for reward computation.
        prompt_str : str, optional
            The prompt string used for generation.
        input_tokens : list[int], optional
            The tokenized input IDs for the prompt.
        """
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.task_data = task_data if task_data is not None else {}
        self.prompt_str = prompt_str
        self.input_tokens = input_tokens if input_tokens is not None else []

    def process(self, tasks: list[Task], **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator: delegates generation to generation_controller.process (yielding its
        # task lists). Then, for each GenerationTask, builds an interaction (key
        # "task_<index>") and an RLVRRewardTask (prompt_str falls back to
        # task.input_str, task_data from self), drives the reward controller's generator
        # to completion locally without yielding, and stores the shared interactions
        # dict in every task's customized_result_fields["interactions"].
        """Process tasks through the generation and reward pipeline.

        Yields task lists only for generation (worker execution). Reward
        computation is done locally without yielding to workers.
        Interactions are stored in ``task.customized_result_fields["interactions"]``
        for retrieval in ``generate()``.

        Parameters
        ----------
        tasks : list[Task]
            List of generation tasks to process.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        list[Task]
            Task lists for worker execution (generation only).
        """
        # Step 1: Run generation (yields task lists for worker execution)
        yield from self.generation_controller.process(tasks, **kwargs)

        # Step 2: Create reward tasks and compute rewards locally
        reward_tasks = []
        interactions = {}

        for i, task in enumerate(tasks):
            if isinstance(task, GenerationTask):
                # Create interaction object
                interaction = self._create_interaction_from_task(task)
                task_id = f"task_{i}"
                interactions[task_id] = interaction

                # Create reward task using constructor-provided task_data and prompt_str
                reward_task = RLVRRewardTask.create_from_generation_task(
                    gen_task=task,
                    prompt_str=self.prompt_str or task.input_str or "",
                    task_data=self.task_data,
                    interaction=interaction,
                )
                reward_tasks.append(reward_task)

        # Compute rewards locally (no yield to workers)
        for _ in self.reward_controller.process(reward_tasks, **kwargs):
            pass

        # Store interactions on tasks for retrieval in generate()
        for task in tasks:
            task.customized_result_fields["interactions"] = interactions

    def generate(self, prompt: str, **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator overriding Controller.generate: creates a GenerationTask from
        # prompt, sets input_tokens from self if any, resets stop to None so sampling
        # params apply, runs process, and returns the task's ScaffoldingOutput with data
        # replaced by the interactions dict.
        """Generate with the full pipeline and return interactions in output.

        Overrides the base ``Controller.generate()`` to:
        1. Set ``input_tokens`` on the task so ``to_tensor_dict()`` works.
        2. Reset ``stop`` to ``None`` so ``NativeGenerationController`` can
           set it from ``sampling_params``.
        3. Return a ``ScaffoldingOutput`` with interactions in ``data``.

        Parameters
        ----------
        prompt : str
            The input prompt string.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        ScaffoldingOutput
            Output with ``data`` containing the interactions dict.
        """

        task = GenerationTask.create_from_prompt(prompt)
        if self.input_tokens:
            task.input_tokens = self.input_tokens
        # Reset stop to None so NativeGenerationController can set from sampling_params
        task.stop = None

        yield from self.process([task], **kwargs)

        output = task.create_scaffolding_output()
        output.data = task.customized_result_fields.get("interactions")
        return output

    def _create_interaction_from_task(
        self, task: GenerationTask
    ) -> InteractionWithTokenLogpReward:
        # [FUNCTION DOC]
        # HELPER: Builds an InteractionWithTokenLogpReward (reward None) whose
        # ModelResponse holds copies of the task's input/output tokens and
        # output_logprobs / output_versions from customized_result_fields, defaulting to
        # zeros and -1 per output token when absent.
        """Create an InteractionWithTokenLogpReward from a generation task.

        Parameters
        ----------
        task : GenerationTask
            The completed generation task.

        Returns
        -------
        InteractionWithTokenLogpReward
            The interaction object with model response data.
        """
        from areal.api.io_struct import ModelResponse

        # Build ModelResponse from task data
        input_tokens = list(task.input_tokens or [])
        output_tokens = list(task.output_tokens or [])
        output_logprobs = task.customized_result_fields.get("output_logprobs", [])
        output_versions = task.customized_result_fields.get("output_versions", [])

        # Create ModelResponse
        model_response = ModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=list(output_logprobs)
            if output_logprobs
            else [0.0] * len(output_tokens),
            output_versions=list(output_versions)
            if output_versions
            else [-1] * len(output_tokens),
        )

        # Create interaction
        interaction = InteractionWithTokenLogpReward(
            model_response=model_response,
            reward=None,  # Will be set by reward controller
        )

        return interaction


class MultiTurnChatController(Controller):
    # [CLASS DOC]
    # Multi-turn chat controller: runs up to max_turns generations on one ChatTask built
    # from self.messages, appending a reflection user message between turns. Per-episode
    # messages and input_tokens are set as attributes before generate (the scheduler
    # clones the controller per request).
    """Controller for multi-turn chat with reflection between turns.

    Handles the chat loop: for each turn, yields a ChatTask to the worker
    for generation, then appends a reflection message for non-final turns.

    Per-episode data (``messages``, ``input_tokens``) should be set before
    calling ``generate()`` or ``process()``. ``ScaffoldingLlm`` deep-copies
    the controller via ``clone()`` so each request gets its own copy.

    Parameters
    ----------
    generation_controller : Controller
        The controller for text generation (e.g., NativeGenerationController).
    max_turns : int
        Maximum number of chat turns per episode.
    reflection_message : str
        Message appended after each non-final turn to prompt retry.
    tokenizer : Any
        Tokenizer for encoding the final output to token IDs.
    messages : list[dict], optional
        The original chat messages to create the ChatTask from (set per-episode).
    input_tokens : list[int], optional
        The tokenized input IDs for the original prompt (set per-episode).
    """

    def __init__(
        self,
        generation_controller: Controller,
        max_turns: int = 2,
        reflection_message: str = "",
        tokenizer: Any = None,
        messages: list[dict] | None = None,
        input_tokens: list[int] | None = None,
    ):
        # [FUNCTION DOC]
        # Stores generation_controller, max_turns (default 2), reflection_message,
        # tokenizer (for encoding the final output), messages (default []) and
        # input_tokens (default []).
        super().__init__()
        self.generation_controller = generation_controller
        self.max_turns = max_turns
        self.reflection_message = reflection_message
        self.tokenizer = tokenizer
        self.messages = messages if messages is not None else []
        self.input_tokens = input_tokens if input_tokens is not None else []

    def process(self, tasks: list[Task], **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator ignoring incoming tasks: builds a ChatTask from self.messages (dicts
        # -> RoleMessage, role/content only), clears stop so sampling params apply,
        # attaches input_tokens if set, then for each of max_turns turns delegates one
        # generation to generation_controller.process and, except after the last turn,
        # appends a UserMessage with reflection_message. Always runs all turns (no early
        # stop on correctness); results stay on the internal ChatTask.
        """Run multi-turn chat generation.

        Creates a ChatTask from the stored messages and yields it to the
        worker for each turn. Between turns, appends a reflection message.

        Parameters
        ----------
        tasks : list[Task]
            Ignored; the ChatTask is created from ``self.messages``.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        list[Task]
            Task lists for worker execution (one per turn).
        """
        role_messages = [RoleMessage.from_dict(m) for m in self.messages]
        chat_task = ChatTask.create_from_messages(role_messages)
        # Reset stop so NativeGenerationController can set from sampling_params
        chat_task.stop = None
        if self.input_tokens:
            chat_task.input_tokens = self.input_tokens

        for turn in range(self.max_turns):
            yield from self.generation_controller.process([chat_task], **kwargs)

            if turn < self.max_turns - 1:
                chat_task.add_message(UserMessage(self.reflection_message))


@with_task_collection("chat_tracer", ChatTracer)
class TraceTrajectoryMaker(Controller):
    # [CLASS DOC]
    # Controller (decorated with with_task_collection, so each instance has a ChatTracer
    # under "chat_tracer" observing every task list its process yields, including
    # ParallelProcess branches) that runs a rollout controller, then rewards each traced
    # interaction with a reward controller and returns the interactions.
    """Controller that traces ChatTask objects during rollout using ChatTracer.

    This controller uses the @with_task_collection decorator to automatically
    apply ChatTracer's before_yield and after_yield hooks around each yield
    in the rollout controller's process execution.

    A multi-turn conversation uses the same ChatTask object, which is traced
    across all yields. The trace results are stored in the TraceGenerationTask
    after processing.

    Parameters
    ----------
    rollout_controller : Controller
        The controller for rollout (e.g., a chat or agent controller).
    reward_controller : Controller
        The controller for computing rewards on traced interactions.

    Example
    -------
    ```python
    from examples.scaffolding._compat import NativeGenerationController

    chat_controller = SomeChatController()
    reward_controller = RLVRRewardController(gsm8k_reward_fn)

    trace_maker = TraceTrajectoryMaker(
        rollout_controller=chat_controller,
        reward_controller=reward_controller,
    )

    # Process tasks
    result = trace_maker.generate(prompt)

    # Or use process directly
    task = TraceGenerationTask.create_from_prompt(prompt)
    for _ in trace_maker.process([task]):
        pass
    trace_results = task.trace_results
    ```
    """

    def __init__(
        self,
        rollout_controller: Controller,
        reward_controller: Controller,
    ):
        # [FUNCTION DOC]
        # Stores the rollout controller and reward controller (the decorator then
        # registers the ChatTracer).
        """Initialize the trace trajectory maker.

        Parameters
        ----------
        rollout_controller : Controller
            The controller for rollout execution.
        reward_controller : Controller
            The controller for reward computation.
        """
        super().__init__()
        self.rollout_controller = rollout_controller
        self.reward_controller = reward_controller

    def process(self, tasks: list[Task], **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator. Takes tasks[0]; unwraps generation_task if it is a
        # TraceGenerationTask. Delegates to rollout_controller.process on it (ChatTracer
        # records each turn). Exports trace results, wraps each interaction in a
        # ChatRewardTask, and if any, delegates to reward_controller.process (allowing
        # judge LLM calls to be yielded), then copies non-None task rewards onto the
        # interactions. Finally stores trace_results on the TraceGenerationTask. Only
        # the first task is processed.
        """Process tasks through the rollout and reward pipeline with tracing.

        This method:
        1. Extracts the generation_task from the TraceGenerationTask
        2. Runs the rollout_controller.process() with ChatTracer tracing
        3. Gets trace results from the ChatTracer
        4. Creates ChatRewardTask objects for each traced interaction
        5. Runs the reward_controller.process() to compute rewards
        6. Stores the trace results in the original task

        Parameters
        ----------
        tasks : list[Task]
            List of TraceGenerationTask objects to process.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        Any
            Results from the controllers.
        """
        # Get the generation task from the first TraceGenerationTask
        task = tasks[0]
        if isinstance(task, TraceGenerationTask):
            generation_task = task.generation_task
        else:
            generation_task = task

        # Run rollout with tracing (ChatTracer hooks applied via decorator)
        yield from self.rollout_controller.process([generation_task], **kwargs)

        # Get trace results from the ChatTracer (registered via decorator)
        chat_tracer = self.task_collections["chat_tracer"]
        trace_results = chat_tracer.get_trace_results()

        # Create reward tasks for each traced interaction
        reward_tasks = [
            ChatRewardTask.create_from_trace_result(interaction_id, interaction)
            for interaction_id, interaction in trace_results.items()
        ]

        # Run reward computation — yield from so that controllers like
        # LLMJudgeController can send tasks to workers via yield.
        if reward_tasks:
            yield from self.reward_controller.process(reward_tasks, **kwargs)

            # Update trace_results with computed rewards
            for reward_task in reward_tasks:
                if (
                    reward_task.interaction is not None
                    and reward_task.reward is not None
                ):
                    reward_task.interaction.reward = reward_task.reward

        # Store trace results in the original task
        if isinstance(task, TraceGenerationTask):
            task.trace_results = trace_results

    def generate(self, prompt: str, **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator overriding Controller.generate: wraps prompt in a
        # TraceGenerationTask, runs process and returns its ScaffoldingOutput (data =
        # trace results dict).
        """Generate with tracing from a prompt string.

        Parameters
        ----------
        prompt : str
            The input prompt.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        ScaffoldingOutput
            Output with trace results in ``data``.
        """
        task = TraceGenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()


def _parse_judge_result(raw_response: str) -> float:
    # [FUNCTION DOC]
    # HELPER: Parses a judge reply into a binary reward. Takes the text after the last
    # "```json" and before the next "```", trying json.loads then ast.literal_eval; if
    # both fail, falls back to substring checks for '"judgement": "incorrect"' then
    # '"judgement": "correct"'; otherwise logs a WARNING and treats as unknown. Returns
    # 1.0 only if the parsed object has judgement == "correct", else 0.0.
    # KNOWN ISSUE: If parsing yields a non-container value (e.g. a number), the
    # membership test raises TypeError.
    """Parse the LLM judge response and extract a binary reward.

    The judge is expected to return a JSON block with a ``"judgement"`` field
    set to ``"correct"`` or ``"incorrect"``.

    Parameters
    ----------
    raw_response : str
        Raw text response from the judge LLM.

    Returns
    -------
    float
        1.0 if the judgement is ``"correct"``, 0.0 otherwise.
    """
    mbe = None
    for parse_fn in [json.loads, ast.literal_eval]:
        try:
            mbe = parse_fn(raw_response.split("```json")[-1].split("```")[0].strip())
            break
        except Exception:
            pass
    if mbe is None and '"judgement": "incorrect"' in raw_response:
        mbe = {"judgement": "incorrect"}
    if mbe is None and '"judgement": "correct"' in raw_response:
        mbe = {"judgement": "correct"}
    if mbe is None:
        logger.warning("Unknown judge result. Raw response: %s", raw_response)
        mbe = {"judgement": "unknown"}
    return float("judgement" in mbe and mbe["judgement"] == "correct")


_JUDGE_PROMPT_TEMPLATE = (
    "You are an evaluation assistant. Please determine if the predicted answer "
    "is equivalent to the labeled answer.\n"
    "You should first give your rationale for the judgement, and then give your "
    "judgement result (i.e., correct or incorrect).\n\n"
    "\n"
    "question: {question}\n"
    "ground truth answers: {gt_answer}\n"
    "pred_answer: {pred_answer}\n\n"
    "Did the model give an answer **equivalent** to the labeled answer? \n\n"
    "The output should in the following json format:\n"
    "```json\n"
    "{{\n"
    '    "rationale": "your rationale for the judgement, as a text",\n'
    "    \"judgement\": \"your judgement result, can only be 'correct' or 'incorrect'\n"
    "}}\n"
    "```\n"
    "Your output:"
)


class LLMJudgeController(Controller):
    # [CLASS DOC]
    # Reward controller that asks an LLM judge (sent as ChatTasks to the GENERATION
    # worker) whether each prediction is equivalent to the ground truth. Per-episode
    # question/answer for traced chat rewards come from self.task_data, set before
    # generate.
    """Controller that uses an LLM to judge answer correctness.

    Instead of using a deterministic reward function, this controller sends
    a judge prompt to the same (or a separate) LLM worker and parses the
    response to determine whether the predicted answer is correct.

    This is the scaffolding-framework equivalent of
    ``MultiTurnReactAgent.calc_reward_with_llm_judge`` from the
    tongyi_deepresearch example.

    Parameters
    ----------
    judge_prompt_template : str, optional
        The prompt template for the judge.  Must contain ``{question}``,
        ``{gt_answer}``, and ``{pred_answer}`` placeholders.
    max_pred_chars : int
        Maximum characters of the predicted answer to include in the
        judge prompt (to avoid exceeding context limits).
    max_judge_tokens : int
        Maximum tokens for the judge LLM response.
    """

    class WorkerTag(Enum):
        # [CLASS DOC]
        # Enum with JUDGE = "llm_judge" (declared but judge tasks actually use the
        # GENERATION tag).
        JUDGE = "llm_judge"

    def __init__(
        self,
        judge_prompt_template: str | None = None,
        max_pred_chars: int = 200,
        max_judge_tokens: int = 8192,
    ):
        # [FUNCTION DOC]
        # Inputs: judge_prompt_template (format string with {question}, {gt_answer},
        # {pred_answer}; defaults to the module template), max_pred_chars (prediction
        # truncation, default 200), max_judge_tokens (default 8192). Initializes scores
        # None and task_data {}.
        super().__init__()
        self.judge_prompt_template = judge_prompt_template or _JUDGE_PROMPT_TEMPLATE
        self.max_pred_chars = max_pred_chars
        self.max_judge_tokens = max_judge_tokens
        self.scores: list[float] | None = None
        # Per-episode data set before generate(); deep-copied via clone()
        self.task_data: dict[str, Any] = {}

    def _build_judge_prompt(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
    ) -> str:
        # [FUNCTION DOC]
        # HELPER: Formats the judge template with question, ground truth and the
        # prediction truncated to max_pred_chars.
        """Format the judge prompt with the given data.

        Parameters
        ----------
        question : str
            The original question.
        ground_truth : str
            The ground-truth answer.
        prediction : str
            The model's predicted answer (truncated to ``max_pred_chars``).

        Returns
        -------
        str
            The formatted judge prompt.
        """
        return self.judge_prompt_template.format(
            question=question,
            gt_answer=ground_truth,
            pred_answer=prediction[: self.max_pred_chars],
        )

    def _extract_answer_and_data(self, task: Task) -> tuple[str, str, str] | None:
        # [FUNCTION DOC]
        # HELPER: Returns (question, ground_truth, prediction) or None for unsupported
        # task types. RLVRRewardTask: from its task_data (list answers -> first element)
        # and the <answer> tag content of completion_str (whole completion if absent).
        # ChatRewardTask: question/answer from self.task_data, prediction from the
        # interaction's completion first-choice content, narrowed to the <answer> tag if
        # present. GenerationTask: question/answer from customized_result_fields,
        # prediction = output_str.
        """Extract (question, ground_truth, prediction) from a task.

        Supports ``RLVRRewardTask``, ``ChatRewardTask``, and
        ``GenerationTask`` with ``customized_result_fields``.

        For ``ChatRewardTask``, falls back to ``self.task_data`` for
        question/answer and extracts the prediction from the last
        assistant message in the traced interaction.

        Returns ``None`` if the task type is not recognised.
        """
        if isinstance(task, RLVRRewardTask):
            question = task.task_data.get("question", "")
            gt = task.task_data.get("answer", "")
            if isinstance(gt, list):
                gt = str(gt[0]) if gt else ""
            # Extract from <answer> tags if present, else use full completion
            match = re.search(r"<answer>(.*?)</answer>", task.completion_str, re.DOTALL)
            pred = match.group(1).strip() if match else task.completion_str
            return question, str(gt), pred

        if isinstance(task, ChatRewardTask):
            # Use per-episode task_data for question/answer
            question = self.task_data.get("question", "")
            gt = self.task_data.get("answer", "")
            if isinstance(gt, list):
                gt = str(gt[0]) if gt else ""
            # Extract prediction from the interaction's completion
            pred = ""
            if task.interaction is not None:
                completion = getattr(task.interaction, "completion", None)
                if completion is not None:
                    pred = completion.choices[0].message.content or ""
                # Extract from <answer> tags if present
                match = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
                if match:
                    pred = match.group(1).strip()
            return question, str(gt), pred

        if isinstance(task, GenerationTask):
            data = task.customized_result_fields
            question = data.get("question", "")
            gt = data.get("answer", "")
            if isinstance(gt, list):
                gt = str(gt[0]) if gt else ""
            pred = task.output_str or ""
            return question, str(gt), pred

        return None

    def process(self, tasks: list[Task], **kwargs) -> Any:
        # [FUNCTION DOC]
        # Generator. Resets scores; for each task with extractable data where question
        # or ground truth is non-empty, builds a single-user-message judge ChatTask
        # (GENERATION tag, max_tokens = max_judge_tokens, temperature 1.0, stop None).
        # Yields all judge tasks as one batch (if any). Then for every input task
        # computes reward = parsed judge verdict from the last message if the worker
        # appended a reply, else 0.0 (also 0.0 for skipped tasks), stores it on
        # task.reward and task.interaction.reward (reward tasks) or
        # customized_result_fields["reward"] (generation tasks), and appends it to
        # scores.
        """Process tasks by sending judge prompts to the LLM worker.

        For each task, builds a judge prompt and yields a ``ChatTask`` to
        the worker.  After the worker responds, parses the judge result
        and stores the reward.

        Parameters
        ----------
        tasks : list[Task]
            Tasks to compute rewards for.
        **kwargs
            Additional keyword arguments.

        Yields
        ------
        list[Task]
            ChatTask lists sent to the worker for judge LLM calls.
        """
        self.scores = []

        # Build judge ChatTasks for each input task
        # judge_map: input task index -> index in judge_chat_tasks list
        judge_chat_tasks: list[ChatTask] = []
        judge_map: dict[int, int] = {}

        for i, task in enumerate(tasks):
            extracted = self._extract_answer_and_data(task)
            if extracted is None:
                continue
            question, gt, pred = extracted
            if not question and not gt:
                continue

            judge_prompt = self._build_judge_prompt(question, gt, pred)
            judge_messages = [
                RoleMessage.from_dict({"role": "user", "content": judge_prompt})
            ]
            chat_task = ChatTask.create_from_messages(judge_messages)
            chat_task.worker_tag = NativeGenerationController.WorkerTag.GENERATION
            chat_task.max_tokens = self.max_judge_tokens
            chat_task.temperature = 1.0
            chat_task.stop = None
            judge_map[i] = len(judge_chat_tasks)
            judge_chat_tasks.append(chat_task)

        # Yield all judge tasks to the worker in one batch
        if judge_chat_tasks:
            yield judge_chat_tasks

        # Parse responses and assign rewards
        for i, task in enumerate(tasks):
            reward = 0.0
            if i in judge_map:
                jt = judge_chat_tasks[judge_map[i]]
                # The worker appends an AssistantMessage after the user message
                if jt.messages and len(jt.messages) > 1:
                    judge_response = jt.messages[-1].content or ""
                else:
                    judge_response = ""
                reward = _parse_judge_result(judge_response)

            # Store reward on the original task
            if isinstance(task, (RLVRRewardTask, ChatRewardTask)):
                task.reward = reward
                if task.interaction is not None:
                    task.interaction.reward = reward
            elif isinstance(task, GenerationTask):
                task.customized_result_fields["reward"] = reward

            self.scores.append(reward)

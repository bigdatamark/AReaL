# [MODULE DOC]
# Tool-Integrated Reasoning (TIR) rollout workflow: generates in segments, stopping at
# tool start/end markers, executes detected tool calls via ToolManager, appends tool
# output
# (loss-masked) to the context and repeats up to max_turns tool calls, then computes the
# reward. Imports prompts and tool_manager as top-level modules (examples/tir must be on
# sys.path).
import ast
import copy
import re
import uuid
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    dataclass,
    field,
)
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import apply_chat_template

from prompts import ANSWER, SYSTEM_PROMPT, TORL_PROMPT  # isort: skip
from tool_manager import ToolCallStatus, ToolManager  # isort: skip

logger = logging.getLogger("TIR workflow")


@dataclass
class TIRConfig:
    # [CLASS DOC]
    # TIR options: max_turns tool calls (2), max_length total token budget for prompt
    # plus
    # completion (3000), tool_timeout seconds (30), enable_tools (semicolon-separated
    # tool
    # names, "python"), is_chat_model (False: use the ToRL plain prompt; True: chat
    # template).
    max_turns: int = field(default=2)
    max_length: int = field(default=3000)
    tool_timeout: float = field(default=30)
    enable_tools: str = field(default="python")
    is_chat_model: bool = field(default=False)


@dataclass
class TIRGRPOConfig(GRPOConfig):
    # [CLASS DOC]
    # GRPOConfig extension with a tir: TIRConfig field.
    tir: TIRConfig = field(default_factory=TIRConfig)


class TIRWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # RolloutWorkflow implementing multi-turn tool-integrated generation. Holds a
    # default
    # ToolManager (for markers and fallback) but builds a fresh one per episode.
    """Tool-Integrated Reasoning Workflow for multi-turn tool calling."""

    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        tir_config: TIRConfig,
        enable_thinking: bool = False,
    ):
        # [FUNCTION DOC]
        # Loads the tokenizer and resolves reward_fn from strings if needed, derives
        # gconfig with
        # stop/pad ids, stores tool and TIR settings, creates a ToolManager (non-debug),
        # wraps
        # reward_fn in AsyncRewardWrapper and caches all tool start/end markers
        # (logged).
        super().__init__()
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            reward_fn = import_from_string(reward_fn)
        self.reward_fn = reward_fn
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer
        self.tool_timeout = tir_config.tool_timeout
        self.enable_tools = tir_config.enable_tools
        self.tool_manager = ToolManager(
            self.tool_timeout, self.enable_tools, debug_mode=False
        )
        self.is_chat_model = tir_config.is_chat_model
        self.max_turns = tir_config.max_turns
        self.max_length = tir_config.max_length
        self.enable_thinking = enable_thinking
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        self.start_markers = self.tool_manager.get_all_start_markers()
        self.end_markers = self.tool_manager.get_all_end_markers()

        logger.info(
            f"start markers: {self.start_markers}, end markers {self.end_markers}"
        )

    def _build_tool_manager(self) -> ToolManager:
        # [FUNCTION DOC]
        # HELPER: Returns a new non-debug ToolManager with the workflow's timeout and
        # enabled
        # tools.
        return ToolManager(self.tool_timeout, self.enable_tools, debug_mode=False)

    @staticmethod
    def _process_tool_result(tool_result) -> str:
        # [FUNCTION DOC]
        # HELPER: Tries to literal-eval the tool result as a (result, status) pair; if
        # that fails
        # treats the whole text as the result with status "Done". Returns the result (or
        # "Error: <status>" when status is not "Done") wrapped in an output-fenced block
        # with
        # surrounding newlines.
        try:
            run_result, run_status = ast.literal_eval(tool_result)
        except Exception:
            run_result = tool_result
            run_status = "Done"
        res = run_result if run_status == "Done" else f"Error: {run_status}"
        return f"\n```output\n{res}\n```\n"

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Builds a per-episode tool manager, replaces or inserts the system message with
        # SYSTEM_PROMPT filled with tool descriptions (mutating data["messages"] in
        # place), and
        # tokenizes the prompt either via the chat template (chat models) or as
        # TORL_PROMPT
        # around messages[1] content. Returns _multi_round_response's tensor dict.
        # Always cleans up
        # the tool manager (important for remote sandboxes).
        """Run a complete TIR inference episode.
        :param engine: The inference engine.
        :param data: The input data.
        :return: The output tensor dict.
        """
        tool_manager = self._build_tool_manager()
        # Daytona-backed tools can allocate remote sandboxes per episode, so
        # cleanup must run even when generation or reward computation fails.
        try:
            # Initialize conversation history
            messages = data["messages"]

            # Add system prompt with tool usage instructions
            system_prompt = SYSTEM_PROMPT.format(
                tool_descriptions=tool_manager.get_tool_descriptions_prompt()
            )
            if messages[0]["role"] == "system":
                messages[0]["content"] = system_prompt
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})

            # Prepare input
            if self.is_chat_model:
                input_ids = apply_chat_template(
                    self.tokenizer,
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            else:
                input_ids = self.tokenizer.encode(
                    TORL_PROMPT.format(prompt=messages[1]["content"]),
                    add_special_tokens=False,
                )

            # Run single trajectory
            return await self._multi_round_response(
                engine, input_ids, data, tool_manager
            )
        finally:
            await tool_manager.acleanup()

    async def _multi_round_response(self, engine, prompt_ids, data, tool_manager=None):
        # [FUNCTION DOC]
        # Core loop. Tracks context ids, full sequence, logprobs, loss mask (1 for
        # generated, 0
        # for prompt/tool tokens), versions (-1 for non-generated) and the completion
        # string.
        # While turn <= max_turns and context is under max_length: generates with stop
        # markers
        # depending on state; stops on pad/eos as last token or when the boxed answer
        # pattern
        # appears in the latest segment. When waiting and the segment ends with a start
        # marker,
        # records the marker position and switches to waiting for an end marker. When a
        # tool call
        # is open and generation stopped, executes the text from the marker; NOT_FOUND
        # continues
        # generating, otherwise increments turn and counters (success requires SUCCESS
        # and no
        # "Error" in output), appends the formatted tool output tokens and resets the
        # state.
        # Then awaits the reward with prompt/completion strings and ids plus tool_using,
        # tool_status (the call count) and all data fields, logs tool_call_count and
        # tool_success_count to the stat scope, and returns batch-1 tensors input_ids,
        # logprobs,
        # loss_mask, versions, attention_mask (all truncated to max_length) and rewards.
        # KNOWN ISSUE: generation length is not capped by the remaining budget (max_len
        # is not
        # used in _generate_response), so the final sequence may be silently truncated;
        # and the
        # reward is computed on the untruncated completion.
        if tool_manager is None:
            tool_manager = self.tool_manager

        prompt_str = self.tokenizer.decode(prompt_ids)
        completions_str = ""
        has_tool = False
        tool_call_count = 0
        tool_success_count = 0
        stop_reason = None
        max_len = self.max_length
        turn = 0
        # State flag for each episode: whether waiting for tool start marker
        waiting_for_tool_start = True
        tool_start_idx = -1

        # initialize seq, logprobs, loss_mask, versions
        context_ids = copy.deepcopy(prompt_ids)
        seq = copy.deepcopy(prompt_ids)
        logprobs = [0.0] * len(context_ids)
        loss_mask = [0] * len(context_ids)
        versions = [-1] * len(context_ids)
        output_ids = []

        while turn <= self.max_turns:
            if len(context_ids) >= max_len:
                break

            # Generate response
            resp, stop_reason = await self._generate_response(
                engine, context_ids, max_len, waiting_for_tool_start
            )

            context_ids.extend(resp.output_tokens)
            seq.extend(resp.output_tokens)
            logprobs.extend(resp.output_logprobs)
            loss_mask.extend([1] * resp.output_len)
            versions.extend(resp.output_versions)

            cur_completions_str = self.tokenizer.decode(resp.output_tokens)
            completions_str += cur_completions_str
            output_ids.extend(resp.output_tokens)

            # End token, truncate
            if context_ids[-1] in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ]:
                break

            # If answer appears, truncate immediately
            if re.search(ANSWER, cur_completions_str):
                break

            # State transition logic: detect if tool start marker is encountered
            if waiting_for_tool_start and stop_reason == "stop":
                # Check if tool start marker is detected
                tool_start_marker = self._detect_tool_start_marker(cur_completions_str)
                if tool_start_marker:
                    waiting_for_tool_start = False
                    tool_start_idx = len(completions_str) - len(tool_start_marker)
                    # Continue generating until tool end marker
                    continue

            # If tool call is detected, execute tool call
            if (
                not waiting_for_tool_start
                and stop_reason == "stop"
                and tool_start_idx != -1
            ):
                tool_results, tool_status = await self._execute_tools(
                    completions_str[tool_start_idx:], tool_manager
                )
                if tool_status == ToolCallStatus.NOT_FOUND:
                    # No match found, continue generating until next tool end marker
                    continue
                turn += 1
                has_tool = True
                tool_call_count += 1  # Increment tool call count
                if (
                    tool_status == ToolCallStatus.SUCCESS
                    and "Error" not in tool_results
                ):
                    tool_success_count += 1
                tool_results = self._process_tool_result(tool_results)
                # Append tool response token IDs
                tool_rsp_token_ids = self.tokenizer.encode(
                    tool_results, add_special_tokens=False
                )
                # Concatenate to seq
                # Build tool mask
                context_ids.extend(tool_rsp_token_ids)
                seq.extend(tool_rsp_token_ids)
                logprobs.extend([0.0] * len(tool_rsp_token_ids))
                loss_mask.extend([0] * len(tool_rsp_token_ids))
                versions.extend([-1] * len(tool_rsp_token_ids))
                completions_str += tool_results

                # After tool execution completes, reset state flag to prepare for next tool call detection
                waiting_for_tool_start = True

        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            prompt_ids,
            output_ids,
            tool_using=has_tool,
            tool_status=tool_call_count,
            **data,
        )

        # Record tool call count to stats_tracker
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            tool_call_count=tool_call_count, tool_success_count=tool_success_count
        )

        res = dict(
            input_ids=torch.tensor(seq[:max_len]).unsqueeze(0),
            logprobs=torch.tensor(logprobs[:max_len]).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask[:max_len]).unsqueeze(0),
            versions=torch.tensor(versions[:max_len]).unsqueeze(0),
            attention_mask=torch.ones(len(seq[:max_len]), dtype=torch.bool).unsqueeze(
                0
            ),
            rewards=torch.tensor([float(reward)]),
        )
        return res

    async def _generate_response(
        self,
        engine: InferenceEngine,
        input_ids: list[int],
        max_len: int,
        waiting_for_tool_start: bool,
    ) -> tuple[ModelResponse, str]:
        # [FUNCTION DOC]
        # HELPER: Requests one generation from the engine with a fresh rid, n_samples=1
        # and stop
        # strings equal to the start markers (when waiting for a tool) or end markers.
        # Returns
        # (ModelResponse, stop_reason). max_len is ignored.
        """Generate response with tool call detection support"""

        # Select stop condition based on state flag
        if waiting_for_tool_start:
            # When waiting for tool start marker, use start_markers to stop
            stop_markers = [marker for marker in self.start_markers]
        else:
            # Tool start detected, use end_markers to stop
            stop_markers = [marker for marker in self.end_markers]

        # Set generation config, add tool call stop tokens
        gconfig = self.gconfig.new(
            n_samples=1, stop=[marker for marker in stop_markers]
        )

        # Generate response
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )

        resp = await engine.agenerate(req)
        return resp, resp.stop_reason

    def _detect_tool_start_marker(self, text: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the first start marker that text ends with, else None.
        """Detect if text ends with tool start marker"""
        for marker in self.start_markers:
            if text.endswith(marker):
                return marker
        return None

    async def _execute_tools(
        self, response: str, tool_manager=None
    ) -> tuple[str, ToolCallStatus]:
        # [FUNCTION DOC]
        # HELPER: Awaits aexecute_tool_call on the given (or default) tool manager for
        # the text
        # and returns (result, status).
        """Execute tool call"""
        if tool_manager is None:
            tool_manager = self.tool_manager
        return await tool_manager.aexecute_tool_call(response)

# [MODULE DOC]
# Multi-turn ReAct research agent (Tongyi DeepResearch style) driven through AReaL's
# ArealOpenAI client: the model alternates reasoning with tool_call XML invocations of
# the
# search and visit tools until it emits an answer, and an LLM judge scores the final
# answer. Imports prompt/tools relatively or, when run directly, via sys.path. OBS_START
# /
# OBS_END delimit tool responses; MAX_LLM_CALL_PER_RUN (env, default 100) is unused
# here.
import datetime
import json
import os
import sys
import time
from pathlib import Path

import json5
from pydantic import BaseModel
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import Message
from transformers import PreTrainedTokenizer

from areal.experimental.openai import ArealOpenAI
from areal.utils import logging

try:
    from .prompt import SYSTEM_PROMPT
    from .tool_search import Search
    from .tool_visit import Visit
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from prompt import SYSTEM_PROMPT
    from tool_search import Search
    from tool_visit import Visit


logger = logging.getLogger("Tongyi-DeepResearch react agent")

OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"

MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", 100))


def today_date():
    # [FUNCTION DOC]
    # HELPER: Returns today's local date formatted YYYY-MM-DD.
    return datetime.date.today().strftime("%Y-%m-%d")


def parse_judge_result(raw_response):
    # [FUNCTION DOC]
    # HELPER: Converts a judge LLM reply to a 0/1 score. Takes the text after the last
    # json
    # fence (up to the next fence), tries JSON then Python literal parsing (warning on
    # each
    # failure), falls back to substring checks for an incorrect / correct judgement,
    # else
    # "unknown" with a warning. Returns 1.0 only if judgement == "correct".
    # KNOWN ISSUE: a successfully parsed non-dict (e.g. a list or string) makes the
    # membership
    # test misbehave or the indexing raise.
    # parse results
    import ast
    import json

    mbe = None
    for parse_fn in [json.loads, ast.literal_eval]:
        try:
            mbe = parse_fn(raw_response.split("```json")[-1].split("```")[0].strip())
            break
        except Exception:
            logger.warning(f"Error parsing judge result with {parse_fn}.")
    if mbe is None and '"judgement": "incorrect"' in raw_response:
        mbe = dict(judgement="incorrect")
    if mbe is None and '"judgement": "correct"' in raw_response:
        mbe = dict(judgement="correct")
    if mbe is None:
        logger.warning(f"Unknown judge result. Raw response: {raw_response}")
        mbe = dict(judgement="unknown")
    score = float("judgement" in mbe and mbe["judgement"] == "correct")
    return score


class MultiTurnReactAgent(FnCallAgent):
    # [CLASS DOC]
    # qwen-agent FnCallAgent subclass (base initializer not called) holding the
    # tokenizer,
    # per-turn and total token budgets (finishing is forced at 80% of the total), max
    # LLM
    # calls, a judge client, and Visit (using the judge client as summarizer) and Search
    # tool
    # instances keyed by name.
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_tokens_per_turn: int = 10000,
        max_llm_calls_per_run: int = 100,
        max_total_tokens: int = 32768,
        judge_client: ArealOpenAI | None = None,
    ):
        # [FUNCTION DOC]
        # Stores tokenizer, max_tokens_per_turn (10000), max_llm_calls_per_run (100),
        # max_total_tokens (32768) and its 80% soft limit, judge_client, and builds the
        # tool list
        # and name-to-tool map.
        self.tokenizer = tokenizer
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_llm_calls_per_run = max_llm_calls_per_run
        self.max_total_tokens = max_total_tokens
        self.max_total_tokens_before_finishing = int(max_total_tokens * 0.8)
        self.judge_client = judge_client
        self.tool_class = [Visit(summary_client=self.judge_client), Search()]
        self.tool_map = {tool.name: tool for tool in self.tool_class}

    def count_tokens(self, messages):
        # [FUNCTION DOC]
        # Approximates prompt length: renders each message (pydantic models dumped to
        # dicts) in
        # ChatML form, appends an assistant header, tokenizes and returns the token
        # count.
        message_strs = []
        for msg in messages:
            if isinstance(msg, BaseModel):
                msg = msg.model_dump()
                assert "role" in msg and "content" in msg
            message_strs.append(
                f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            )
        message_strs.append("<|im_start|>assistant\n")
        prompt_token_ids = self.tokenizer.encode("".join(message_strs))
        return len(prompt_token_ids)

    async def call_server(
        self, client: ArealOpenAI, messages: list[dict], max_attempts: int = 100
    ) -> str:
        # [FUNCTION DOC]
        # HELPER: Calls client.chat.completions.create with the messages, temperature
        # 1.0, stop
        # at the tool_response tag and max_completion_tokens per turn. Returns a
        # (completion, message) tuple. RuntimeError is logged and retried; after
        # max_attempts
        # raises RuntimeError.
        # KNOWN ISSUE: the attempts counter is never incremented, so persistent
        # RuntimeErrors loop
        # forever; the annotation says str but a tuple is returned.
        attempts = 0
        while attempts < max_attempts:
            try:
                completion = await client.chat.completions.create(
                    messages=messages,
                    temperature=1.0,
                    stop=["\n<tool_response>", "<tool_response>"],
                    max_completion_tokens=self.max_tokens_per_turn,
                )
                message = completion.choices[0].message
                assert message, "Error: LLM response is empty."
                return completion, message
            except RuntimeError as e:
                logger.warning(
                    f"RuntimeError during LLM call_server at attempt {attempts}: {e}"
                )
                continue
        raise RuntimeError(
            f"Failed to get response from LLM after {max_attempts} attempts."
        )

    async def run_agent(
        self, data, client: ArealOpenAI, save_path: str | None = None
    ) -> list[list[Message]]:
        # [FUNCTION DOC]
        # Runs the ReAct loop for data (requires qid, question, answer). Starts with the
        # system
        # prompt plus today's date and the question. Each round (until LLM calls are
        # exhausted)
        # calls the model, appends the message, and if it contains a tool_call block
        # parses it
        # with JSON5, executes the tool (counting search/visit in stats) or produces an
        # error
        # text, and appends the result wrapped in tool_response tags as a user message.
        # Breaks on
        # an answer block. When calls run out without an answer, appends a final
        # "provide your
        # answer" instruction. If the token count exceeds the soft limit, appends a
        # context-limit instruction, makes one more call and returns immediately with
        # the
        # prediction and termination reason. After a 150-minute wall clock limit returns
        # early.
        # Otherwise extracts the prediction from the answer block (or "No answer
        # found."), sets
        # termination, optionally dumps the result without completions to save_path as
        # JSON, and
        # returns a dict with question, answer, messages, prediction, termination,
        # completions
        # (all raw completions) and stats (turns, num_search, num_access).
        # KNOWN ISSUE: the time-limit early return omits completions and stats
        # (make_trajectory
        # would raise KeyError); None message content raises TypeError; return
        # annotation says a
        # list of message lists but a dict is returned; with zero allowed calls content
        # is
        # undefined.
        start_time = time.time()
        data["qid"]
        question = data["question"]
        answer = data["answer"]
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        stats = dict(
            turns=0,
            num_search=0,
            num_access=0,
        )
        num_llm_calls_available = self.max_llm_calls_per_run
        completions = []
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = "No answer found after 2h30mins"
                termination = "No answer found after 2h30mins"
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                }
                return result
            round += 1
            stats["turns"] += 1
            num_llm_calls_available -= 1
            completion, message = await self.call_server(client, messages)
            content = message.content
            completions.append(completion)
            messages.append(message)
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_call = content.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("arguments", {})
                    result = await self.custom_call_tool(tool_name, tool_args)
                    if tool_name == "search":
                        stats["num_search"] += 1
                    elif tool_name == "visit":
                        stats["num_access"] += 1
                except Exception as e:
                    result = f'Error: {e} Tool call must be a valid json contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            if "<answer>" in content and "</answer>" in content:
                termination = "answer"
                break
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                messages.append(
                    {
                        "role": "user",
                        "content": "Sorry, the number of llm calls exceeds the limit. You should stop making tool calls and, "
                        "based on all the information above, think again and provide what you consider the most likely answer "
                        "in the following format:<think>your final thinking</think>\n<answer>your answer</answer>",
                    }
                )

            max_tokens = self.max_total_tokens_before_finishing
            token_count = self.count_tokens(messages)
            logger.debug(
                f"QID {data['qid']} Round: {round}, token count: {token_count}"
            )

            if token_count > max_tokens:
                logger.debug(
                    f"QID {data['qid']} Token quantity exceeds the limit: {token_count} > {max_tokens}"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "You have now reached the maximum context length you can handle. "
                        "You should stop making tool calls and, based on all the information above, "
                        "think again and provide what you consider the most likely answer in the following format:"
                        "<think>your final thinking</think>\n<answer>your answer</answer>",
                    }
                )
                completion, message = await self.call_server(client, messages)
                completions.append(completion)
                content = message.content
                messages.append(message)
                if "<answer>" in content and "</answer>" in content:
                    prediction = content.split("<answer>")[1].split("</answer>")[0]
                    termination = "generate an answer as token limit reached"
                else:
                    prediction = content
                    termination = (
                        "format error: generate an answer as token limit reached"
                    )
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "completions": completions,
                    "stats": stats,
                }
                token_count = self.count_tokens(messages)
                if token_count > self.max_total_tokens:
                    logger.warning(
                        f"Warning: total token count {token_count} exceeds the hard limit {self.max_total_tokens}."
                    )
                return result

        if "<answer>" in content:
            prediction = content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"
        result = {
            "question": question,
            "answer": answer,
            "messages": [
                m.model_dump() if isinstance(m, BaseModel) else m for m in messages
            ],
            "prediction": prediction,
            "termination": termination,
            "completions": completions,  # final completion
            "stats": stats,
        }
        if save_path:
            to_dump = dict(**result)
            to_dump.pop("completions")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(to_dump, f, ensure_ascii=False, indent=4)
            logger.debug(f"Result dumped to {save_path}")
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        # [FUNCTION DOC]
        # HELPER: If tool_name is known, sets tool_args["params"] to tool_args itself (a
        # self-reference) and awaits the tool's call with it, returning the result
        # string;
        # otherwise returns a not-found error string.
        if tool_name in self.tool_map:
            tool_args["params"] = tool_args
            raw_result = await self.tool_map[tool_name].call(tool_args, **kwargs)
            result = raw_result
            return result
        else:
            return f"Error: Tool {tool_name} not found"

    async def calc_reward_with_llm_judge(
        self,
        result: dict[str, str],
    ):
        # [FUNCTION DOC]
        # Builds a judge prompt with the question, ground truth (single-element lists
        # unwrapped)
        # and the first 200 chars of the prediction, asking for JSON with rationale and
        # judgement;
        # calls the judge client (temperature 1.0, 8192 max tokens, store=False) and
        # returns
        # parse_judge_result of the reply. Any exception logs a warning and returns 0.0.
        # Compute reward with LLM-as-Judge
        # judge_client = ArealOpenAI(engine=rollout_engine, tokenizer=tokenizer)
        judge_prompt_template = (
            "You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.\n"
            "You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).\n\n"
            "\n"
            "question: {question}\n"
            "ground truth answers: {gt_answer}\n"
            "pred_answer: {pred_answer}\n\n"
            "Did the model give an answer **equivalent** to the labeled answer? \n\nThe output should in the following json format:\n"
            "```json\n"
            "{{\n"
            """    "rationale": "your rationale for the judgement, as a text",\n"""
            """    "judgement": "your judgement result, can only be 'correct' or 'incorrect'\n"""
            "}}\n"
            "```\n"
            "Your output:"
        )
        pred_answer = result["prediction"]
        ground_truth = result["answer"]
        if isinstance(ground_truth, list) and len(ground_truth) == 1:
            ground_truth = str(ground_truth[0])
        judge_prompt = judge_prompt_template.format(
            question=result["question"],
            gt_answer=str(ground_truth),
            pred_answer=pred_answer[:200],
        )
        try:
            judge_completion = await self.judge_client.chat.completions.create(
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=1.0,
                max_completion_tokens=8192,
                store=False,
            )
            judge_response = judge_completion.choices[0].message.content
            reward = parse_judge_result(judge_response)
        except Exception as e:
            logger.warning(f"Error in calling LLM judge: {e}")
            reward = 0.0
        return reward

    async def make_trajectory(
        self,
        data: dict[str, str],
        client: ArealOpenAI,
        save_path: str | None = None,
    ) -> dict:
        # [FUNCTION DOC]
        # Runs run_agent, computes the judge reward, assigns it to the last completion
        # via
        # client.set_reward (so the ArealOpenAI client can export rewarded
        # interactions), and
        # returns the stats dict.
        result = await self.run_agent(data, client, save_path=save_path)
        reward = await self.calc_reward_with_llm_judge(result)
        completions = result["completions"]
        last_completion = completions[-1]
        client.set_reward(last_completion.id, reward)
        stats = result["stats"]
        return stats

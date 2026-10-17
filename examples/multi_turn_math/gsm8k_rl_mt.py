# [MODULE DOC]
# Multi-turn GSM8K RL example: an agent answers via an ArealOpenAI client, gets a reward
# each turn, and on a wrong answer is asked to retry, up to max_turns. Defines the
# reward function, the agent, the RolloutWorkflow and a training entry point.
import sys
from collections.abc import Callable

from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizerFast

from examples.multi_turn_math.config import MultiTurnGRPOConfig

from areal import PPOTrainer, workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.openai import ArealOpenAI
from areal.reward import get_math_verify_worker
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer


def gsm8k_reward_fn(result, answer):
    # [FUNCTION DOC]
    # Reward function: returns the math-verify score of str(result) against str(answer);
    # any exception yields 0.0.
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


class MultiTurnMathAgent:
    # [CLASS DOC]
    # Agent that runs a retry-on-failure chat loop for one math problem, rewarding each
    # turn.
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        reward_fn: Callable[[str, str], float | int],
        max_turns: int = 2,
    ):
        # [FUNCTION DOC]
        # Stores gconfig and max_turns and wraps reward_fn in AsyncRewardWrapper so it
        # can be awaited.
        self.gconfig = gconfig
        self.max_turns = max_turns
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        # [FUNCTION DOC]
        # Runs up to max_turns chat completions starting from a copy of
        # data["messages"]. After each response, appends the assistant message, computes
        # the reward against data["answer"], and records it on the client via
        # set_reward(response.id). Stops when reward == 1; otherwise appends a fixed
        # user message asking to re-check and retry. Returns the last turn's reward.
        # KNOWN ISSUE: If max_turns is 0 the loop never runs and returning reward raises
        # UnboundLocalError.
        messages = data["messages"].copy()
        for _ in range(self.max_turns):
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                **self.gconfig.to_openai_args_dict(),
            )
            message = response.choices[0].message
            messages.append(message)
            reward = await self.async_reward_fn(
                result=message.content, answer=data["answer"]
            )
            client.set_reward(response.id, reward)
            if reward == 1:
                break
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                        "Please carefully read the original question, check the previous errors, and try to answer it again.",
                    }
                )
        return reward


class MultiturnRLVRWorkflow(RolloutWorkflow):
    # [CLASS DOC]
    # RolloutWorkflow running MultiTurnMathAgent per episode and exporting the reward-
    # annotated interactions.
    def __init__(
        self,
        reward_fn: Callable[[str, str], float | int] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        export_style: str = "concat",
        max_turns: int = 2,
    ):
        # [FUNCTION DOC]
        # Accepts reward_fn and tokenizer either as objects or as dotted path /
        # tokenizer path strings (resolved lazily). Validates export_style in
        # {"individual", "concat"} (ValueError otherwise) and maps it to chat template
        # type "concat" or "hf". Builds the agent with gconfig forced to n_samples=1.
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(reward_fn, str):
            from areal.utils.dynamic_import import import_from_string

            reward_fn = import_from_string(reward_fn)
        self.tokenizer = tokenizer
        self.export_style = export_style
        if export_style not in ["individual", "concat"]:
            raise ValueError(f"Invalid export style: {export_style}")
        self.chat_template_type = "concat" if export_style == "concat" else "hf"

        # Search hyper-parameters
        self.agent = MultiTurnMathAgent(
            gconfig=gconfig.new(n_samples=1),
            reward_fn=reward_fn,
            max_turns=max_turns,
        )

    async def arun_episode(self, engine, data):
        # [FUNCTION DOC]
        # Creates an ArealOpenAI client over the inference engine, runs the agent on
        # data, logs the reward scalar to the current workflow stat scope, applies a 0.9
        # per-turn reward discount, and returns the exported interactions (dict keyed by
        # completion id) in the configured style.
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type=self.chat_template_type,
        )

        # Collect single trajectory
        reward = await self.agent.run_agent(
            data=data,
            client=client,
        )
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=0.9)
        completions_with_reward = client.export_interactions(style=self.export_style)
        return completions_with_reward


def main(args):
    # [FUNCTION DOC]
    # Entry point: loads MultiTurnGRPOConfig, tokenizer and train/test datasets; reads
    # max_turns from agent_run_args; trains with MultiturnRLVRWorkflow, using a gconfig
    # with temperature 0.6 and n_samples=1 for evaluation.
    config, _ = load_expr_config(args, MultiTurnGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    max_turns = config.agent_run_args.get("max_turns", 2)

    workflow_kwargs = dict(
        reward_fn="examples.multi_turn_math.gsm8k_rl_mt.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        export_style=config.export_style,
        max_turns=max_turns,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.multi_turn_math.gsm8k_rl_mt.MultiturnRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.multi_turn_math.gsm8k_rl_mt.MultiturnRLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])

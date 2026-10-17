# [MODULE DOC]
# Reward function for GSM8K math word problems using the shared MathVerifyWorker.
# SPDX-License-Identifier: Apache-2.0

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("GSM8KReward")


def gsm8k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    # [FUNCTION DOC]
    # Reward function with the standard AReaL signature (only completions and answer are
    # used). Returns the math-verify singleton's 1.0/0.0 verdict comparing
    # str(completions) against str(answer); any exception is logged as a warning and
    # yields 0.0.
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in gsm8k_reward_fn", exc_info=True)
        return 0.0

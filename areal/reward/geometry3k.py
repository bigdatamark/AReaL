# [MODULE DOC]
# Reward function for the Geometry3K dataset: extracts a bracketed (or last numeric)
# answer from the completion and ground truth and compares them with the shared
# MathVerifyWorker.
# SPDX-License-Identifier: Apache-2.0

import re

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("Geometry3KReward")


def extract_answer(pred_str, use_last_number=True):
    # [FUNCTION DOC]
    # HELPER: Returns the content of the last "[...]" group in pred_str. If none and
    # use_last_number (default True), returns the last (optionally signed, decimal)
    # number found after removing commas. Returns "" if nothing is found.
    matches = re.findall(r"\[([^\]]+)\]", pred_str)
    if matches:
        return matches[-1]

    # Fallback to last number if no bracket format found
    if use_last_number:
        pattern = r"-?\d*\.?\d+"
        nums = re.findall(pattern, pred_str.replace(",", ""))
        if nums:
            return nums[-1]

    return ""


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    # [FUNCTION DOC]
    # Reward function with the standard AReaL signature (only completions and answer are
    # used). Extracts the answer from str(completions) and from str(answer) (falling
    # back to the raw answer string), returns 0.0 if either is empty, else the 1.0/0.0
    # result of the math-verify singleton. Exceptions are logged as warnings and give
    # 0.0.
    try:
        sol = extract_answer(str(completions))
        ans = extract_answer(str(answer)) or str(answer)

        if not sol or not ans:
            return 0.0

        worker = get_math_verify_worker()
        return worker.verify(sol, ans)
    except Exception:
        logger.warning("Exception in geometry3k_reward_fn", exc_info=True)
        return 0.0

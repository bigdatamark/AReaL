# [MODULE DOC]
# Reward function for the CLEVR-Count-70K visual counting dataset: the model must output
# its count inside square brackets, and reward is exact string match against the answer.
# SPDX-License-Identifier: Apache-2.0

import re

from areal.utils import logging

logger = logging.getLogger("CLEVR70KReward")


def extract_answer(pred_str, data_name, use_last_number=True):
    # [FUNCTION DOC]
    # HELPER: Returns the last bracketed numeric token (digits and dots inside "[...]")
    # found in pred_str, or "" if none. data_name and use_last_number are accepted but
    # unused.
    match = re.findall(r"\[([0-9\.]+)\]", pred_str)
    if match:
        return match[-1]

    return ""


def clevr_count_70k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    # [FUNCTION DOC]
    # Reward function with the standard AReaL signature (prompt, completions,
    # prompt_ids, completion_ids, answer, **kwargs; only completions and answer are
    # used). Extracts the bracketed answer from str(completions), compares stripped
    # string to str(answer) and returns 1.0 on exact match else 0.0 (also 0.0 if either
    # is empty). Logs completion and answer at INFO on correct answers; any exception is
    # logged as a warning and yields 0.0.
    try:
        sol = extract_answer(str(completions), data_name="")  # str number
        ans = str(answer)

        if not sol or not ans:
            return 0.0

        is_correct = sol.strip() == ans.strip()
        if is_correct:
            logger.info(f"completions: {completions}, answer: {answer}")
        return float(is_correct)
    except Exception:
        logger.warning("Exception in clevr_count_70k_reward_fn", exc_info=True)
        return 0.0

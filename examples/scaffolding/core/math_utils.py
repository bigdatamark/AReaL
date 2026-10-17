# [MODULE DOC]
# HELPER module (vendored): answer extraction from math solutions (regex or LaTeX
# \boxed{...}) and majority voting over extracted answers.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.math_utils

import re


def extract_answer_with_regex(
    string: str, extract_regex: str = r"The final answer is (.+)$"
):
    # [FUNCTION DOC]
    # HELPER: Returns capture group 1 of the first re.search match of extract_regex in
    # string (default matches "The final answer is X" at end of string), or None if no
    # match.
    match = re.search(extract_regex, string)
    if match:
        return match.group(1)
    return None


def extract_answer_from_boxed(string: str):
    # [FUNCTION DOC]
    # HELPER: Returns the content of the last \boxed{...} in string, matching nested
    # braces by counting from the \boxed position to the brace that closes the first
    # opened one. Returns None if "\boxed" is absent, braces are unbalanced, or the
    # extracted span does not literally start with "\boxed{" (e.g. "\boxed 5").
    # KNOWN ISSUE: The "\fbox" fallback is unreachable because the function returns
    # early when "\boxed" is not in the string.
    """Extract Answer String from \\boxed expression or based on regex"""

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


def get_majority_result(
    results: list,
    result_extractor=lambda x: x,
    result_validator=lambda x: True,
):
    # [FUNCTION DOC]
    # HELPER: Applies result_extractor to each result, keeps non-None answers for which
    # result_validator returns exactly True, and returns (index in the original list of
    # the first result whose extracted answer equals the most frequent answer, that
    # answer). Ties go to the answer first counted. Returns (None, None) if no valid
    # answers.
    extract_answers = [result_extractor(result) for result in results]
    valid_answers = [
        result
        for result in extract_answers
        if result is not None and result_validator(result) is True
    ]
    if len(valid_answers) == 0:
        return None, None

    answer_counts = {}
    for answer in valid_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    majority_answer = max(answer_counts, key=answer_counts.get)
    majority_index = next(
        filter(lambda x: x[1] == majority_answer, enumerate(extract_answers))
    )[0]
    return majority_index, majority_answer


def get_digit_majority_vote_result(results: list[str]) -> str:
    # [FUNCTION DOC]
    # Majority vote over solution strings using \boxed{} extraction and keeping only
    # pure-digit answers. Returns (index, answer) or (0, None) when no valid answer
    # exists.
    # KNOWN ISSUE: Annotated to return str but returns a tuple.
    def is_digit(result: str):
        # [FUNCTION DOC]
        # HELPER: True if the string consists only of digits (str.isdigit).
        return result.isdigit()

    index, extract_answer = get_majority_result(
        results, result_extractor=extract_answer_from_boxed, result_validator=is_digit
    )
    return (index, extract_answer) if extract_answer else (0, None)

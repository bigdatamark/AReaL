# [MODULE DOC]
# Reward scoring for the Countdown arithmetic task: extracts the last
# <answer>...</answer> equation from a completion, checks it uses exactly the given
# numbers, evaluates it safely and compares to the target.
import random
import re


def extract_solution(solution_str):
    # [FUNCTION DOC]
    # HELPER: Strips text up to the first 'Assistant:' or '<|im_start|>assistant'
    # marker, keeps only the last line, and returns the content of the last
    # <answer>...</answer> tag (stripped) or None.
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        final_answer = matches[-1].strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    # [FUNCTION DOC]
    # HELPER: Returns True if the multiset of integers found in the equation equals
    # available_numbers (each used exactly once), False otherwise or on error.
    # KNOWN ISSUE: digits are matched with an integer pattern, so decimals like '1.5'
    # count as numbers 1 and 5.
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except Exception:
        return False


def evaluate_equation(equation_str):
    # [FUNCTION DOC]
    # HELPER: Evaluates the equation with eval() after checking it contains only digits,
    # + - * / ( ) . and whitespace, with builtins disabled; returns the result or None
    # on any error.
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    # [FUNCTION DOC]
    # Scores a solution: 0 if no equation found, format_score (default 0.1) if numbers
    # invalid, unevaluable or the result differs from target (tolerance 1e-5), score
    # (default 1.0) if correct. ground_truth must have 'target' and 'numbers'. method is
    # unused. Randomly (1/64) prints debug info to stdout.
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print("No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except Exception:
        if do_print:
            print("Error evaluating equation")
        return format_score

# [PACKAGE DOC]
# Reward package. Provides MathVerifyWorker (a thread-safe, timeout-guarded wrapper over
# the math_verify library for comparing a model answer to ground truth), a lazily-
# created process-wide singleton accessor, and lazy (PEP 562 module __getattr__) exports
# of the dataset reward functions gsm8k_reward_fn, geometry3k_reward_fn and
# clevr_count_70k_reward_fn so importing the package does not import every reward
# module.
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures

from math_verify.grader import verify as math_verify_verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

from areal.utils import logging

logger = logging.getLogger("RewardUtils")


class MathVerifyWorker:
    # [CLASS DOC]
    # Holds gold and prediction extraction configs (both: ExprExtractionConfig with the
    # configured anchor setting plus LatexExtractionConfig), the float rounding
    # precision and the per-call timeout. verify() returns 1.0 for a match and 0.0
    # otherwise, never raising.
    """Thin wrapper over math_verify with configurable extraction/precision.

    Uses ``parse()`` + ``verify()`` directly instead of ``math_metric()``
    so that signal-based timeouts can be disabled (``parsing_timeout=None``,
    ``timeout_seconds=None``). This avoids ``signal.alarm()`` which only
    works in the main thread. A thread-safe timeout is enforced via
    ``concurrent.futures`` instead.

    Args:
        try_extract_without_anchor: When False, only answers with explicit anchors
            (e.g., "answer = 1", "final answer = 1") are matched. When True,
            any numeric string in the text may be extracted.
        precision: Number of significant digits that must match.
        timeout: Thread-safe timeout in seconds for the entire verify call
            (parsing + comparison). ``None`` disables the timeout.

    Notes:
        Tune these knobs based on dataset format and model output style.
    """

    def __init__(
        self,
        try_extract_without_anchor=True,
        precision: int = 6,
        timeout: float | None = 5.0,
    ):
        # [FUNCTION DOC]
        # Stores extraction targets built from try_extract_without_anchor (default
        # True), precision (significant digits used as float_rounding, default 6) and
        # timeout (seconds, default 5.0; None disables the timeout). No side effects.
        self.gold_extraction_target = (
            ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
            LatexExtractionConfig(),
        )
        self.pred_extraction_target = (
            ExprExtractionConfig(try_extract_without_anchor=try_extract_without_anchor),
            LatexExtractionConfig(),
        )
        self.precision = precision
        self.timeout = timeout

    def _verify_impl(self, response: str, ground_truth: str) -> float:
        # [FUNCTION DOC]
        # HELPER: Parses ground_truth and response with math_verify.parse (no signal
        # timeouts), returns 0.0 if either parse is empty, else 1.0 if
        # math_verify.verify judges them equivalent with float rounding = precision,
        # otherwise 0.0. May raise on library errors.
        """Core verification logic without timeout wrapper."""
        gold_parsed = parse(
            ground_truth,
            extraction_config=self.gold_extraction_target,
            parsing_timeout=None,
        )
        pred_parsed = parse(
            response,
            extraction_config=self.pred_extraction_target,
            parsing_timeout=None,
        )
        if not gold_parsed or not pred_parsed:
            return 0.0
        result = math_verify_verify(
            gold_parsed,
            pred_parsed,
            float_rounding=self.precision,
            timeout_seconds=None,
        )
        return 1.0 if result else 0.0

    def verify(self, response: str, ground_truth: str) -> float:
        # [FUNCTION DOC]
        # Public verification entry point. If timeout is None runs _verify_impl
        # directly; otherwise runs it in a single-worker ThreadPoolExecutor and waits up
        # to timeout seconds. On timeout or any exception logs a warning and returns
        # 0.0.
        # KNOWN ISSUE: on timeout the executor context manager's exit waits for the
        # worker thread to finish (shutdown(wait=True)), so a hung verification still
        # blocks the caller beyond the timeout.
        try:
            if self.timeout is None:
                return self._verify_impl(response, ground_truth)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._verify_impl, response, ground_truth)
                return future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"Timeout ({self.timeout}s) in MathVerifyWorker.verify for "
                f"response={response!r} and ground_truth={ground_truth!r}",
            )
            return 0.0
        except Exception:
            logger.warning(
                f"Exception in MathVerifyWorker.verify for response={response} and ground_truth={ground_truth}",
                exc_info=True,
            )
            return 0.0


_MATH_VERIFY_WORKER: MathVerifyWorker | None = None


def get_math_verify_worker() -> MathVerifyWorker:
    # [FUNCTION DOC]
    # Returns the module-global MathVerifyWorker singleton, creating it with default
    # settings on first call (not lock-protected; concurrent first calls may create
    # duplicates, which is harmless).
    global _MATH_VERIFY_WORKER
    if _MATH_VERIFY_WORKER is None:
        _MATH_VERIFY_WORKER = MathVerifyWorker()
    return _MATH_VERIFY_WORKER


__all__ = [
    "MathVerifyWorker",
    "get_math_verify_worker",
    "gsm8k_reward_fn",
    "geometry3k_reward_fn",
    "clevr_count_70k_reward_fn",
]


_LAZY_IMPORTS = {
    "gsm8k_reward_fn": "areal.reward.gsm8k",
    "geometry3k_reward_fn": "areal.reward.geometry3k",
    "clevr_count_70k_reward_fn": "areal.reward.clevr_count_70k",
}


def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module-level lazy attribute hook. For names in the lazy-import table, imports the
    # backing module, fetches the attribute, caches it in module globals and returns it.
    # Raises AttributeError for any other name.
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # [FUNCTION DOC]
    # Returns the list of public names (__all__) for dir() on the package.
    return list(__all__)

# [MODULE DOC]
# HELPER: Thin re-export of get_dataset_path and get_model_path from
# areal.utils.testing_utils (local path if present, else HF Hub fallback) for tests
# importing tests.utils. Note: importing testing_utils may trigger model downloads.
"""Test utilities.

Re-exports common test utilities from areal.utils.testing_utils for use
by test files that import from tests.utils.
"""

from areal.utils.testing_utils import (
    get_dataset_path,
    get_model_path,
)

__all__ = ["get_dataset_path", "get_model_path"]

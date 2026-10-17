# [MODULE DOC]
# torch_memory_saver (TMS) helpers. Exposes `torch_memory_saver`: the real TMS
# singleton if the package is installed, else a no-op mock with the same
# disable/pause/resume interface. Also builds the LD_PRELOAD environment needed to
# enable TMS in child processes and checks whether TMS is enabled.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os
from contextlib import nullcontext

try:
    from torch_memory_saver import torch_memory_saver
except ImportError:

    class MockTorchMemorySaver:
        # [CLASS DOC]
        # HELPER: Fallback used when torch_memory_saver cannot be imported; every method is
        # a no-op.
        def disable(self):
            # [FUNCTION DOC]
            # Returns a nullcontext so it can be used as a context manager like the real API.
            return nullcontext()

        def pause(self):
            # [FUNCTION DOC]
            # No-op stand-in for pausing (offloading) tracked memory.
            pass

        def resume(self):
            # [FUNCTION DOC]
            # No-op stand-in for resuming (reloading) tracked memory.
            pass

    torch_memory_saver = MockTorchMemorySaver()


def get_tms_env_vars() -> dict[str, str]:
    # [FUNCTION DOC]
    # Locates the TMS LD_PRELOAD hook library
    # (torch_memory_saver_hook_mode_preload.abi3.so) in the parent directory of the
    # installed torch_memory_saver package and returns env vars LD_PRELOAD=<path>,
    # TMS_INIT_ENABLE=1 and TMS_INIT_ENABLE_CPU_BACKUP=1. Raises ImportError if the
    # package is missing and RuntimeError if the .so file does not exist.
    """Get environment variables for torch_memory_saver (TMS)."""
    import torch_memory_saver as tms_pkg

    # Locate the LD_PRELOAD shared library
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(tms_pkg.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    env_vars = {
        "LD_PRELOAD": dynlib_path,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    return env_vars


def is_tms_enabled() -> bool:
    # [FUNCTION DOC]
    # Returns True iff env var TMS_INIT_ENABLE equals "1".
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"

# [MODULE DOC]
# HELPER: CUDA compute-capability (SM version) detection used to choose FP8 code paths
# (Hopper vs Blackwell).
# SPDX-License-Identifier: Apache-2.0

# CUDA device detection utilities

import torch


def get_device_sm() -> int:
    # [FUNCTION DOC]
    # Return the current CUDA device's SM version as major*10+minor (e.g. 90 Hopper, 100
    # Blackwell), or 0 when CUDA is unavailable. Queries torch.cuda for the current
    # device.
    """Get device SM version (e.g., 90 for Hopper, 100 for Blackwell)."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def is_hopper() -> bool:
    # [FUNCTION DOC]
    # Return True if the device SM is in [90, 100) (Hopper); False otherwise or on any
    # exception.
    """Check if device is Hopper (SM90-99)."""
    try:
        sm = get_device_sm()
        return 90 <= sm < 100
    except Exception:
        return False


def is_blackwell() -> bool:
    # [FUNCTION DOC]
    # Return True if the device SM is >= 100 (Blackwell or newer); False otherwise or on
    # any exception.
    """Check if device is Blackwell (SM100+)."""
    try:
        return get_device_sm() >= 100
    except Exception:
        return False


def is_sm90_or_above() -> bool:
    # [FUNCTION DOC]
    # Return True if the device SM is >= 90 (Hopper or newer); False otherwise or on any
    # exception.
    """Check if device is SM90 or above (Hopper+)."""
    try:
        return get_device_sm() >= 90
    except Exception:
        return False

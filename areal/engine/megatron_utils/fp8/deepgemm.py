# [MODULE DOC]
# DeepGEMM capability detection (adapted from SGLang). Exposes cached boolean probes for
# whether DeepGEMM JIT is usable (SM90+ and deep_gemm importable), whether it runs on
# Blackwell, and whether UE8M0 scales are required, plus lazy module attributes
# ENABLE_JIT_DEEPGEMM / DEEPGEMM_BLACKWELL / DEEPGEMM_SCALE_UE8M0 resolved through a
# module-level __getattr__.
# SPDX-License-Identifier: Apache-2.0

# DeepGEMM detection and configuration
# Adapted from SGLang sglang/srt/layers/deep_gemm_wrapper/configurer.py

import functools


@functools.cache
def _compute_enable_deep_gemm() -> bool:
    # [FUNCTION DOC]
    # HELPER: Cached (functools.cache) probe: True only if the device SM >= 90 and the
    # deep_gemm package imports. Any error in SM detection or ImportError yields False.
    # Side effect: may import deep_gemm and query CUDA once per process.
    """Check if DeepGEMM JIT is enabled.

    DeepGEMM requires:
    1. SM90+ GPU (Hopper or newer)
    2. deep_gemm package installed
    """
    try:
        from areal.engine.megatron_utils.fp8.cuda import get_device_sm

        sm_version = get_device_sm()
        if sm_version < 90:
            return False
    except Exception:
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return True


@functools.cache
def _is_deepgemm_blackwell() -> bool:
    # [FUNCTION DOC]
    # HELPER: Cached probe: True if DeepGEMM is enabled and the device is Blackwell
    # (SM100+); False on any exception.
    """Check if DeepGEMM is enabled on Blackwell GPU."""
    if not _compute_enable_deep_gemm():
        return False
    try:
        from areal.engine.megatron_utils.fp8.cuda import is_blackwell

        return is_blackwell()
    except Exception:
        return False


@functools.cache
def _is_deepgemm_scale_ue8m0() -> bool:
    # [FUNCTION DOC]
    # HELPER: Cached probe: whether DeepGEMM uses UE8M0 (power-of-two) scales; identical
    # to the Blackwell probe.
    """Check if DeepGEMM uses UE8M0 scale format (Blackwell only)."""
    return _is_deepgemm_blackwell()


# Public accessors - lazy evaluated on first access
def get_enable_jit_deepgemm() -> bool:
    # [FUNCTION DOC]
    # Public accessor returning the cached DeepGEMM-JIT-enabled flag.
    """Get whether DeepGEMM JIT is enabled."""
    return _compute_enable_deep_gemm()


def get_deepgemm_blackwell() -> bool:
    # [FUNCTION DOC]
    # Public accessor returning the cached DeepGEMM-on-Blackwell flag.
    """Get whether DeepGEMM is enabled on Blackwell GPU."""
    return _is_deepgemm_blackwell()


def get_deepgemm_scale_ue8m0() -> bool:
    # [FUNCTION DOC]
    # Public accessor returning the cached DeepGEMM UE8M0-scale flag.
    """Get whether DeepGEMM uses UE8M0 scale format."""
    return _is_deepgemm_scale_ue8m0()


# Backward-compatible module-level constants (lazy properties via __getattr__)
# These are kept for backward compatibility but now use lazy initialization
def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module-level __getattr__ (PEP 562) mapping the legacy constant names
    # ENABLE_JIT_DEEPGEMM, DEEPGEMM_BLACKWELL and DEEPGEMM_SCALE_UE8M0 to the
    # corresponding accessors, evaluated on access. Raises AttributeError for any other
    # name.
    """Lazy initialization for module-level constants."""
    if name == "ENABLE_JIT_DEEPGEMM":
        return get_enable_jit_deepgemm()
    elif name == "DEEPGEMM_BLACKWELL":
        return get_deepgemm_blackwell()
    elif name == "DEEPGEMM_SCALE_UE8M0":
        return get_deepgemm_scale_ue8m0()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def should_deepgemm_weight_requant_ue8m0(
    weight_block_size: list[int] | tuple[int, int] | None,
) -> bool:
    # [FUNCTION DOC]
    # Return True when FP8 weights should be requantized to UE8M0 scale format at load
    # time: DeepGEMM JIT enabled, UE8M0 scales in use (Blackwell), and weight_block_size
    # (list/tuple of ints or None) is not None.
    """Should we requant fp8 weights into UE8M0 format when loading the model.

    UE8M0 format is used by DeepGEMM on Blackwell GPUs for optimal performance.

    From SGLang sglang/srt/model_loader/utils.py.

    Args:
        weight_block_size: Block size for weight quantization, e.g. [128, 128]

    Returns:
        True if UE8M0 requantization should be applied
    """
    return (
        get_enable_jit_deepgemm()
        and get_deepgemm_scale_ue8m0()
        and weight_block_size is not None
    )

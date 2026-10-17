# [MODULE DOC]
# Validation helpers for the attn_impl config option: accepts builtin HF attention
# backends or Hugging Face kernels hub repo IDs.
# SPDX-License-Identifier: Apache-2.0

from transformers.integrations.hub_kernels import is_kernel

BUILTIN_ATTN_IMPLS = (
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flex_attention",
)


def is_valid_attn_impl(attn_impl: str) -> bool:
    # [FUNCTION DOC]
    # Returns True if attn_impl is one of the builtin backend names or transformers'
    # is_kernel recognises it as valid kernels-hub syntax.
    """Return whether ``attn_impl`` is a builtin backend or valid HF kernels syntax."""
    return attn_impl in BUILTIN_ATTN_IMPLS or is_kernel(attn_impl)


def get_attn_impl_validation_error(attn_impl: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the standard error message string listing builtin backends and the
    # expected kernels repo-ID format, embedding the offending value.
    """Build the canonical validation message for invalid attention implementations."""
    return (
        "attn_impl must be one of "
        f"{BUILTIN_ATTN_IMPLS} or a Hugging Face kernels repo ID "
        "formatted as org/repo[@revision][:entrypoint] "
        "(optionally prefixed with wrapper|), "
        f"got '{attn_impl}'."
    )

# [PACKAGE DOC]
# Package for profiling helpers used by the AReaL profiling tools (profile_fsdp,
# profile_archon, profile_engines). Re-exports generate_random_seq_lens from the utils
# submodule.
# SPDX-License-Identifier: Apache-2.0

"""Profiling utilities for AReaL."""

from areal.tools.profiling_utils.utils import generate_random_seq_lens

__all__ = [
    "generate_random_seq_lens",
]

# [MODULE DOC]
# HELPER module/CLI for profiling scripts: resolves which ranks to profile. Supports
# aliases (pp_rank0, pp-rank0, auto_pp, auto-pp) that expand to the global rank 0 of
# every pipeline-parallel stage, derived from the actor backend allocation string in an
# experiment YAML config.
# SPDX-License-Identifier: Apache-2.0

"""Rank selection helpers for profile scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from areal.api.alloc_mode import ModelAllocation

_PP_RANK0_ALIASES = frozenset({"pp_rank0", "pp-rank0", "auto_pp", "auto-pp"})


def actor_backend_from_config(config_path: Path) -> str:
    # [FUNCTION DOC]
    # Reads a YAML config file and returns actor.backend, falling back to top-level
    # allocation_mode, as a string. Raises ValueError if neither is set. An empty YAML
    # is treated as an empty dict.
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    actor_config = config.get("actor") or {}
    backend = actor_config.get("backend") or config.get("allocation_mode")
    if not backend:
        raise ValueError(
            f"Could not find actor.backend or allocation_mode in {config_path}"
        )
    return str(backend)


def pp_rank0_ranks_from_backend(backend: str) -> str:
    # [FUNCTION DOC]
    # Parses an allocation string with ModelAllocation and returns a comma-separated
    # list of the first rank of each pipeline stage, i.e. multiples of world_size /
    # pp_size for pp ranks 0..pp_size-1 (assumes PP is the outermost rank dimension).
    # Raises ValueError if world_size is not divisible by pp_size.
    allocation = ModelAllocation.from_str(backend)
    pp_size = allocation.parallel.pipeline_parallel_size
    world_size = allocation.parallel.world_size
    if world_size % pp_size != 0:
        raise ValueError(
            f"World size {world_size} must be divisible by pp size {pp_size}"
        )
    pp_stage_stride = world_size // pp_size
    return ",".join(str(pp_rank * pp_stage_stride) for pp_rank in range(pp_size))


def pp_rank0_ranks_from_config(config_path: Path) -> str:
    # [FUNCTION DOC]
    # Composes actor_backend_from_config and pp_rank0_ranks_from_backend for a config
    # path.
    return pp_rank0_ranks_from_backend(actor_backend_from_config(config_path))


def resolve_profile_ranks(profile_ranks: str, config_path: Path) -> str:
    # [FUNCTION DOC]
    # Returns profile_ranks unchanged unless it is a PP-rank0 alias, in which case
    # returns the ranks computed from the config.
    if profile_ranks not in _PP_RANK0_ALIASES:
        return profile_ranks
    return pp_rank0_ranks_from_config(config_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # [FUNCTION DOC]
    # HELPER: Parses required --config (Path) and --profile-ranks CLI args.
    parser = argparse.ArgumentParser(description="Resolve profile rank selection.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile-ranks", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # [FUNCTION DOC]
    # CLI entry: prints the resolved rank list and returns exit code 0.
    args = _parse_args(argv)
    print(resolve_profile_ranks(args.profile_ranks, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

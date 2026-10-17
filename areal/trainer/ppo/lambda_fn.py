# [MODULE DOC]
# Per-sample GAE lambda functions and resolution. A lambda function receives a
# GAELambdaContext (per-trajectory lengths, [bs] tensors) plus keyword hyperparameters
# and returns a float32 [bs] lambda tensor. Provides constant, VAPO length-adaptive and
# relative-position lambdas, and a resolver turning a config value (number or dotted
# import path) into a callable. GAELambdaFn is an alias for any callable returning a
# tensor.
# SPDX-License-Identifier: Apache-2.0

"""Built-in functions and resolution helpers for per-sample GAE lambda."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import TypedDict

import torch

from areal.utils.dynamic_import import import_from_string

__all__ = [
    "GAELambdaContext",
    "GAELambdaFn",
    "constant_gae_lambda",
    "relative_position_gae_lambda",
    "resolve_gae_lambda_fn",
    "vapo_length_adaptive_gae",
]


class GAELambdaContext(TypedDict):
    # [CLASS DOC]
    # TypedDict of [bs] tensors: effective_token_lengths (active loss tokens per
    # sample), turn_counts (active turns per sample, zeros if turn metadata is missing)
    # and timestep_lengths (whichever of the two matches the configured GAE timestep
    # unit).
    """Batch-local trajectory lengths available to a GAE lambda function."""

    effective_token_lengths: torch.Tensor
    turn_counts: torch.Tensor
    timestep_lengths: torch.Tensor


GAELambdaFn = Callable[..., torch.Tensor]


def constant_gae_lambda(
    context: GAELambdaContext,
    *,
    value: float,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Returns a float32 tensor shaped like context["timestep_lengths"] filled with value
    # (keyword-only). No validation of value's range.
    """Return one constant lambda value for every local trajectory."""
    return torch.full_like(
        context["timestep_lengths"],
        fill_value=value,
        dtype=torch.float32,
    )


def vapo_length_adaptive_gae(
    context: GAELambdaContext,
    *,
    alpha: float,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # VAPO length-adaptive lambda: per sample lambda = max(0, 1 - 1/(alpha*L)) where L =
    # timestep_lengths clamped to >=1 for the division; samples with L == 0 get 0.
    # Raises ValueError unless alpha is a positive int/float (bools rejected). Returns
    # float32 [bs].
    r"""Compute VAPO's length-adaptive ``lambda = 1 - 1 / (alpha * length)``.

    ``length`` is selected by ``gae_timestep_unit`` before this function is called:
    effective generated-token count in token mode and effective turn count in turn
    mode. Empty and sufficiently short trajectories receive lambda zero.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or alpha <= 0:
        raise ValueError(f"alpha must be a positive number, got {alpha!r}")

    lengths = context["timestep_lengths"]
    lengths_float = lengths.to(dtype=torch.float32)
    safe_lengths = lengths_float.clamp_min(1.0)
    adaptive_lambda = 1.0 - 1.0 / (float(alpha) * safe_lengths)
    adaptive_lambda = adaptive_lambda.clamp_min(0.0)
    return torch.where(lengths > 0, adaptive_lambda, torch.zeros_like(adaptive_lambda))


def relative_position_gae_lambda(
    context: GAELambdaContext,
    *,
    q: float,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Returns per-sample lambda = q ** (1/(L-1)) for L >= 2 (computed as exp(log q /
    # (L-1))), 1.0 for L == 1, 0.0 for L == 0, where L = timestep_lengths; makes a
    # terminal reward's retained fraction depend only on relative position. q is
    # keyword-only and must be a finite non-bool number in (0, 1], else ValueError.
    # Returns float32 [bs].
    r"""Keep terminal-reward decay invariant at equal relative positions.

    For a trajectory with ``L >= 2`` effective timesteps, this function returns
    ``lambda = q ** (1 / (L - 1))``. With discount one, a terminal outcome reward
    therefore retains

    ``lambda ** (L - k) = q ** ((L - k) / (L - 1))``

    at one-based timestep ``k``. The first timestep retains exactly ``q``, the
    final timestep retains one, and every equal endpoint-normalized position
    ``(k - 1) / (L - 1)`` has the same retention for every trajectory length.

    ``q`` is the retained fraction, not the fraction removed. A one-timestep
    trajectory has no transition over which lambda can decay and therefore uses
    lambda one. An empty trajectory uses lambda zero.
    """
    if (
        isinstance(q, bool)
        or not isinstance(q, int | float)
        or not math.isfinite(q)
        or not 0 < q <= 1
    ):
        raise ValueError(f"q must be a finite number in (0, 1], got {q!r}")

    lengths = context["timestep_lengths"]
    decay_steps = (lengths - 1).clamp_min(1).to(dtype=torch.float32)
    step_lambdas = torch.exp(math.log(float(q)) / decay_steps)
    return torch.where(
        lengths > 1,
        step_lambdas,
        torch.where(
            lengths == 1,
            torch.ones_like(step_lambdas),
            torch.zeros_like(step_lambdas),
        ),
    )


def resolve_gae_lambda_fn(gae_lambda: float | str) -> tuple[GAELambdaFn, bool]:
    # [FUNCTION DOC]
    # Converts a config gae_lambda into (callable, is_custom). Number (int/float, not
    # bool) -> (partial of constant_gae_lambda with that value, False). Non-empty string
    # -> dynamically imported object via import_from_string (side effect: imports a
    # module), returned with True; TypeError if the import is not callable. Bools, empty
    # strings and other types raise TypeError. Import errors propagate.
    """Resolve a static value or dotted function path to a worker-local callable.

    Returns the callable and whether it came from a custom import path so callers can
    keep the static-lambda fast path while resolving custom values per batch.
    """
    if isinstance(gae_lambda, bool):
        raise TypeError("gae_lambda must be a float or dotted function path, not bool")
    if isinstance(gae_lambda, int | float):
        return (
            functools.partial(constant_gae_lambda, value=float(gae_lambda)),
            False,
        )
    if not isinstance(gae_lambda, str) or not gae_lambda:
        raise TypeError(
            "gae_lambda must be a float or a non-empty dotted function path, "
            f"got {gae_lambda!r}"
        )

    fn = import_from_string(gae_lambda)
    if not callable(fn):
        raise TypeError(f"Imported gae_lambda object is not callable: {gae_lambda}")
    return fn, True

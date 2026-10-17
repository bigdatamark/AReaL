# [MODULE DOC]
# Environment-variable helpers: string lookup with fallback names, boolean parsing with
# one-time warnings for unrecognized values, mode checks (CI, single-controller vs
# SPMD), and rank filtering from comma/range env lists. Uses module-level sets to dedupe
# warnings.
# SPDX-License-Identifier: Apache-2.0

import os

from areal.utils import logging

logger = logging.getLogger("EnvironUtils")

_warned_bool_env_var_keys = set()
_warned_rank_env_var_values = set()


def get_env_var(
    name: str,
    default: str | None = None,
    *,
    fallback_names: tuple[str, ...] = (),
    allow_empty: bool = False,
) -> str | None:
    # [FUNCTION DOC]
    # Returns the first value among name and fallback_names that is set (and non-blank
    # unless allow_empty), else default. Reads os.environ only.
    """Read an environment variable with ordered fallback names.

    Empty values are skipped by default so a legacy name can supply the value.
    Set ``allow_empty=True`` when an explicitly empty value has domain meaning.
    """
    for candidate in (name, *fallback_names):
        value = os.getenv(candidate)
        if value is None:
            continue
        if allow_empty or value.strip() != "":
            return value
    return default


def get_bool_env_var(
    name: str,
    default: str = "false",
    *,
    fallback_names: tuple[str, ...] = (),
    truthy_values: tuple[str, ...] = ("true", "1"),
    falsy_values: tuple[str, ...] = ("false", "0"),
    strip_value: bool = False,
) -> bool:
    # [FUNCTION DOC]
    # Reads a boolean env var via get_env_var (empty explicit values allowed only when
    # there are no fallbacks), optionally strips, lowercases, and returns True iff it is
    # in truthy_values. Unrecognized values log a warning and count as False. Default
    # "false". Asserts the resolved value is not None (fails if default=None and unset).
    # KNOWN ISSUE: the warning dedupe set is keyed by the value, not the variable name,
    # so the same bad value in different variables warns only once.
    """Read a boolean environment variable.

    The default accepted values remain backward compatible. Callers that
    historically accepted additional spellings can opt in through
    ``truthy_values`` and ``falsy_values``.
    """
    value = get_env_var(
        name,
        default,
        fallback_names=fallback_names,
        # Preserve the original single-name behavior: an explicitly empty
        # boolean is invalid rather than silently replaced by the default.
        allow_empty=not fallback_names,
    )
    assert value is not None
    value = (value.strip() if strip_value else value).lower()

    if (value not in truthy_values) and (value not in falsy_values):
        if value not in _warned_bool_env_var_keys:
            logger.warning(
                f"get_bool_env_var({name}) see non-understandable value={value} and treat as false"
            )
        _warned_bool_env_var_keys.add(value)

    return value in truthy_values


def is_in_ci():
    # [FUNCTION DOC]
    # True if env AREAL_IS_IN_CI is truthy.
    return get_bool_env_var("AREAL_IS_IN_CI")


def is_single_controller():
    # [FUNCTION DOC]
    # True unless env AREAL_SPMD_MODE is truthy (single-controller is the default mode).
    return not get_bool_env_var("AREAL_SPMD_MODE")


def rank_in_env_filter(name: str, rank: int) -> bool:
    # [FUNCTION DOC]
    # Parses env var name as a rank filter: unset/blank or "all" selects every rank;
    # otherwise comma-separated ints and inclusive "a-b" ranges. Invalid parts (non-int,
    # reversed range) are ignored with a once-per-(name, value) warning. Returns whether
    # rank is selected.
    """Return whether rank is selected by a comma/range env filter.

    Empty or unset values mean all ranks. Accepted examples: ``0``, ``0,2``,
    ``0-3``, and ``all``.
    """

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return True

    value = value.strip().lower()
    if value == "all":
        return True

    selected: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    raise ValueError
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))
        except ValueError:
            warn_key = (name, value)
            if warn_key not in _warned_rank_env_var_values:
                logger.warning(
                    "rank_in_env_filter(%s) got invalid value=%s and will ignore part=%s",
                    name,
                    value,
                    part,
                )
                _warned_rank_env_var_values.add(warn_key)

    return rank in selected

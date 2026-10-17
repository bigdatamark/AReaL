# [MODULE DOC]
# Parses the FP8 block size from a HuggingFace-style quantization config dict.
# SPDX-License-Identifier: Apache-2.0


def get_block_size_from_config(
    quantization_config: dict[str, int | str | list[str]] | None,
    default: int = 128,
    strict: bool = False,
) -> int:
    # [FUNCTION DOC]
    # Return the FP8 quantization block size (int). Inputs: quantization_config (dict or
    # None) possibly holding "weight_block_size"; default (int, 128) returned when the
    # config or key is absent or (non-strict) malformed; strict (bool) - when True,
    # raise ValueError if weight_block_size is not a 2-element list/tuple or its two
    # entries differ. Otherwise returns the first element of weight_block_size (non-
    # strict mode does not check squareness). Pure function.
    """Extract block size from quantization configuration.

    Args:
        quantization_config: Quantization configuration dict, may contain "weight_block_size"
        default: Default block size to return if not specified in config or if format is invalid
        strict: If True, enforce that weight_block_size must be a square matrix [N, N].
                If False, only check that it's a list/tuple of length 2 and use the first element.
                If format is invalid and strict=False, returns default instead of raising.

    Returns:
        Block size as an integer.

    Raises:
        ValueError: If strict=True and weight_block_size format is invalid or not a square matrix.

    Examples:
        >>> config = {"weight_block_size": [128, 128]}
        >>> get_block_size_from_config(config)
        128
        >>> get_block_size_from_config(config, strict=True)
        128
        >>> get_block_size_from_config(None)
        128
    """
    if quantization_config is None:
        return default

    weight_block_size = quantization_config.get("weight_block_size", None)
    if weight_block_size is None:
        return default

    if not isinstance(weight_block_size, (list, tuple)) or len(weight_block_size) != 2:
        if strict:
            raise ValueError(
                f"weight_block_size must be a list/tuple of length 2, got {weight_block_size}"
            )
        return default

    if strict:
        if weight_block_size[0] != weight_block_size[1]:
            raise ValueError(
                f"weight_block_size must be a square matrix for hardware efficiency and simplicity, "
                f"got {weight_block_size}."
            )

    return weight_block_size[0]

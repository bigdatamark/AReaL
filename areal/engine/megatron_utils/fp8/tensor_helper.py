# [MODULE DOC]
# Tensor-subclass helper that lets Megatron<->HF weight conversion code treat a
# blockwise FP8 weight (fp8 data plus per-block scale_inv) like a plain tensor: shape
# ops (view, reshape, chunk, split, cat, slicing, flatten) are applied to the data and
# mirrored on the scale tensor. Also converts to/from TransformerEngine
# Float8BlockwiseQTensor and to plain PyTorch float8 + scale pairs.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from megatron.core.fp8_utils import is_float8tensor

from areal.infra.platforms import current_platform


class FP8BlockwiseTensorHelper(torch.Tensor):
    # [CLASS DOC]
    # Wrapper tensor subclass (via _make_wrapper_subclass, no storage of its own) whose
    # apparent shape/dtype/device is the FP8 data's. Attributes: _rowwise_data
    # (float8_e4m3fn view of the data), _rowwise_scale_inv (per-block multiplicative
    # scales, same rank as data, last two dims divided by block size, earlier dims
    # unchanged), _block_size (int). Supported aten ops via __torch_dispatch__ are only
    # cat/split/chunk; everything else raises NotImplementedError. Every op returns a
    # new helper; nothing mutates in place except to_te_fp8_inplace on the target.
    """A helper wrapper tensor that maps operations on data to operations on both data and scale_inv.

    This is a customized helper class for blockwise FP8 quantization, not a general-purpose tensor.
    It allows conversion functions to work with FP8 tensors as if they were regular tensors,
    while automatically handling the corresponding scale_inv transformations.

    - scale_inv shape mirrors data shape, but each dimension is divided by block_size
    - When data is [M, K], scale_inv is [ceil(M/block), ceil(K/block)]
    - When data is reshaped to [A, B, C], scale_inv is reshaped to [A, ceil(B/block), ceil(C/block)]
      (assuming the reshape is compatible with block boundaries)
    - Operations like chunk, split, cat on data apply to scale_inv on the SAME dimension
    """

    @staticmethod
    def __new__(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        block_size: int = 128,
    ):
        # [FUNCTION DOC]
        # Construct the wrapper: reinterpret rowwise_data as float8_e4m3fn (view, no
        # copy), create the subclass with that shape/dtype/device and requires_grad
        # False, and store data, scale_inv (as is) and block_size (default 128).
        # Create a tensor subclass that wraps rowwise_data
        rowwise_data = rowwise_data.view(torch.float8_e4m3fn)
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            rowwise_data.shape,
            dtype=rowwise_data.dtype,
            device=rowwise_data.device,
            requires_grad=False,
        )
        obj._rowwise_data = rowwise_data
        obj._rowwise_scale_inv = rowwise_scale_inv
        obj._block_size = block_size
        return obj

    def __repr__(self) -> str:
        # [FUNCTION DOC]
        # Return a debug string with data, scale_inv, both shapes and block size.
        return f"FP8BlockwiseTensorHelper(data={self._rowwise_data}\nscale_inv={self._rowwise_scale_inv}\ndata_shape={self.shape}, scale_shape={self._rowwise_scale_inv.shape}, block_size={self._block_size})"

    def _ceil_div(self, a: int, b: int) -> int:
        # [FUNCTION DOC]
        # HELPER: Integer ceiling division of a by b.
        return (a + b - 1) // b

    def _map_data_dim_to_scale_dim(self, data_dim_size: int) -> int:
        # [FUNCTION DOC]
        # HELPER: Number of scale entries covering a data dimension of the given size
        # (ceil divide by block size).
        """Map a data dimension size to the corresponding scale dimension size."""
        return self._ceil_div(data_dim_size, self._block_size)

    def _compute_scale_shape(
        self, old_data_shape: tuple, new_data_shape: tuple, old_scale_shape: tuple
    ) -> tuple:
        # [FUNCTION DOC]
        # HELPER: Compute the scale_inv shape matching a reshape of the data. Asserts
        # the last data dim is unchanged. If the rank is unchanged, asserts the shapes
        # are identical and returns old_scale_shape. Otherwise maps each new data dim:
        # the last two dims become ceil(dim / block_size), earlier dims are kept as is;
        # asserts the element count of the new scale shape equals the old one. Returns
        # the tuple.
        # KNOWN ISSUE: same-rank reshapes to a different shape (e.g. [A, B] -> [B, A])
        # are rejected by assertion rather than supported, and splitting a block-dim
        # into leading dims only works when the block boundaries line up (checked only
        # via total element count).
        """Compute scale_inv shape after a view/reshape operation.

        - When data dimension is divided by block_size to get scale dimension
        - Reshapes should preserve this relationship

        For example:
        - data: [4096, 4096] -> scale: [32, 32]
        - After view to [8, 512, 4096]:
          - scale should be [8, 4, 32]
          - 8 is a factor that doesn't need scaling (it's a "grouping" dimension)
        """
        assert old_data_shape[-1] == new_data_shape[-1], (
            "last dimension of old_data_shape and new_data_shape must be the same"
        )
        # Same number of dimensions
        if len(old_data_shape) == len(new_data_shape):
            assert old_data_shape == new_data_shape, (
                "old_data_shape and new_data_shape must be the same"
            )
            return old_scale_shape

        # For view to dimensions like view(A, B, C, K)
        # Try to infer scale shape
        new_scale_shape = []

        for i, data_dim in enumerate(new_data_shape):
            if i == len(new_data_shape) - 1 or i == len(new_data_shape) - 2:
                # Last two dimensions
                scale_dim = self._map_data_dim_to_scale_dim(data_dim)
                new_scale_shape.append(scale_dim)
            else:
                new_scale_shape.append(data_dim)

        assert math.prod(new_scale_shape) == math.prod(old_scale_shape), (
            f"product of new_scale_shape {new_scale_shape} and old_scale_shape {old_scale_shape} must be the same"
        )
        return tuple(new_scale_shape)

    def chunk(self, chunks: int, dim: int = 0):
        # [FUNCTION DOC]
        # Split data and scale_inv into `chunks` pieces along the same dim (default 0)
        # and return a tuple of helpers. Asserts data and scale ranks match. Note that
        # chunking a block dim splits data and scales independently, so it is only
        # consistent when chunk sizes are block aligned.
        """Chunk operation: split both data and scale_inv along the same dimension."""
        assert self._rowwise_data.ndim == self._rowwise_scale_inv.ndim, (
            "data and scale_inv must have the same number of dimensions"
        )
        data_chunks = self._rowwise_data.chunk(chunks, dim=dim)
        scale_inv_chunks = self._rowwise_scale_inv.chunk(chunks, dim=dim)

        return tuple(
            FP8BlockwiseTensorHelper(data, scale_inv, self._block_size)
            for data, scale_inv in zip(data_chunks, scale_inv_chunks)
        )

    def split(self, split_size_or_sections, dim: int = 0):
        # [FUNCTION DOC]
        # Split data and scale_inv with the same split_size_or_sections along dim
        # (default 0); returns a tuple of helpers. Asserts equal ranks and that dim is
        # not one of the last two dims (dim < ndim-2 for non-negative or dim < -2 for
        # negative).
        """Split operation: split both data and scale_inv along the same dimension."""
        assert self._rowwise_data.ndim == self._rowwise_scale_inv.ndim, (
            "data and scale_inv must have the same number of dimensions"
        )
        # Do not split on last two dims
        assert dim < self._rowwise_data.ndim - 2 or dim < -2, (
            "do not split on last two dims"
        )

        data_splits = list(self._rowwise_data.split(split_size_or_sections, dim=dim))
        scale_inv_splits = list(
            self._rowwise_scale_inv.split(split_size_or_sections, dim=dim)
        )

        return tuple(
            FP8BlockwiseTensorHelper(data, scale_inv, self._block_size)
            for data, scale_inv in zip(data_splits, scale_inv_splits)
        )

    def view(self, *shape):
        # [FUNCTION DOC]
        # View both tensors: views the data to *shape, derives the scale shape with
        # _compute_scale_shape, views scale_inv to it, returns a new helper. Raises as
        # the underlying view/assertions do.
        """View operation: reshape both data and scale_inv.

        When data is reshaped, scale_inv needs to be reshaped correspondingly.
        From hf_load.py pattern:
        - data: view(num_groups, -1, head_dim, hidden) with shape like [8, 16, 128, 4096]
        - scale: view(num_groups, -1, head_dim/block, hidden/block) with shape like [8, 16, 1, 32]
        """
        old_data_shape = self._rowwise_data.shape
        new_data = self._rowwise_data.view(*shape)
        new_data_shape = new_data.shape

        new_scale_shape = self._compute_scale_shape(
            old_data_shape, new_data_shape, self._rowwise_scale_inv.shape
        )
        new_scale = self._rowwise_scale_inv.view(*new_scale_shape)

        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    def reshape(self, *shape):
        # [FUNCTION DOC]
        # Same as view but uses reshape (may copy) for both data and scale_inv.
        """Reshape operation: same as view but allows non-contiguous tensors."""
        old_data_shape = self._rowwise_data.shape
        new_data = self._rowwise_data.reshape(*shape)
        new_data_shape = new_data.shape

        new_scale_shape = self._compute_scale_shape(
            old_data_shape, new_data_shape, self._rowwise_scale_inv.shape
        )
        new_scale = self._rowwise_scale_inv.reshape(*new_scale_shape)
        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    def __getitem__(self, indices):
        # [FUNCTION DOC]
        # Index data and the scale tensor consistently and return a new helper. A single
        # slice or int indexes dim 0: for <=2D tensors dim 0 is a block dim so the slice
        # bounds are converted to block indices (start floor, stop ceil; an int keeps a
        # one-row scale slice); for >2D it is applied to scale as is (int index keeps a
        # length-1 dim on the scale while the data loses the dim). A tuple maps each
        # slice/int per dim, converting indices on the last two dims to block
        # coordinates. Other index types (tensors, Ellipsis, None) raise
        # NotImplementedError. Slice steps are ignored for scales.
        # KNOWN ISSUE: int indexing leaves scale rank different from data rank (data
        # drops a dim, scale keeps a size-1 dim in the single-int case, and for tuple
        # ints in the last two dims the scale drops a dim while still representing a
        # whole block), so subsequent chunk/split/cat rank asserts may fail; and for >2D
        # tensors the "last two dims" test uses the original rank.
        """Indexing operation: slice data and scale_inv accordingly.

        Only the last two dimensions' indices need to be divided by block_size.
        """
        new_data = self._rowwise_data[indices]
        ndim = self._rowwise_data.ndim

        if isinstance(indices, slice):
            # Single slice: indexing first dimension
            # Check if dim 0 is one of the last two dimensions
            if ndim <= 2:
                start = indices.start if indices.start is not None else 0
                stop = indices.stop if indices.stop is not None else self.shape[0]
                scale_start = start // self._block_size
                scale_stop = self._ceil_div(stop, self._block_size)
                new_scale = self._rowwise_scale_inv[scale_start:scale_stop]
            else:
                # >2D tensor: first dimension doesn't need block_size division
                new_scale = self._rowwise_scale_inv[indices]
        elif isinstance(indices, int):
            # Single int: indexing first dimension
            if ndim <= 2:
                scale_idx = indices // self._block_size
                new_scale = self._rowwise_scale_inv[scale_idx : scale_idx + 1]
            else:
                new_scale = self._rowwise_scale_inv[indices : indices + 1]
        elif isinstance(indices, tuple):
            # Multiple dimensions indexing
            scale_indices = []

            for dim_idx, index in enumerate(indices):
                # Check if this dimension is one of the last two (ndim-2 or ndim-1)
                is_last_two_dims = dim_idx >= ndim - 2

                if isinstance(index, slice):
                    if is_last_two_dims:
                        start = index.start if index.start is not None else 0
                        # Get the actual stop from the corresponding dimension size
                        dim_size = (
                            self.shape[dim_idx] if dim_idx < ndim else self.shape[-1]
                        )
                        stop = index.stop if index.stop is not None else dim_size
                        scale_start = start // self._block_size
                        scale_stop = self._ceil_div(stop, self._block_size)
                        scale_indices.append(slice(scale_start, scale_stop))
                    else:
                        # Not in last two dims, use index as-is
                        scale_indices.append(index)
                elif isinstance(index, int):
                    if is_last_two_dims:
                        scale_idx = index // self._block_size
                        scale_indices.append(scale_idx)
                    else:
                        scale_indices.append(index)
                else:
                    raise NotImplementedError(
                        f"indexing with {type(index)} is not supported"
                    )
            new_scale = self._rowwise_scale_inv[tuple(scale_indices)]
        else:
            raise NotImplementedError(f"indexing with {type(indices)} is not supported")

        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    @staticmethod
    def cat(tensors, dim: int = 0):
        # [FUNCTION DOC]
        # Concatenate a non-empty sequence of helpers along dim (default 0) by
        # concatenating data and scale_inv along the same dim; uses the first tensor's
        # block size. Raises ValueError when empty or any element is not a helper;
        # AssertionError on rank mismatches.
        """Concatenate FP8BlockwiseTensorHelper instances along specified dimension.

        Both data and scale_inv are concatenated along the same dimension.
        """
        if not tensors:
            raise ValueError("cat expects at least one tensor")
        if not all(isinstance(t, FP8BlockwiseTensorHelper) for t in tensors):
            raise ValueError("All tensors must be FP8BlockwiseTensorHelper instances")

        # Check that all tensors have matching dimensions
        first_ndim = tensors[0]._rowwise_data.ndim
        first_scale_ndim = tensors[0]._rowwise_scale_inv.ndim
        assert first_ndim == first_scale_ndim, (
            "data and scale_inv must have the same number of dimensions"
        )
        assert all(t._rowwise_data.ndim == first_ndim for t in tensors), (
            "All tensors must have the same number of dimensions"
        )
        assert all(t._rowwise_scale_inv.ndim == first_scale_ndim for t in tensors), (
            "All scale_inv tensors must have the same number of dimensions"
        )

        block_size = tensors[0]._block_size
        data_tensors = [t._rowwise_data for t in tensors]
        scale_tensors = [t._rowwise_scale_inv for t in tensors]

        new_data = torch.cat(data_tensors, dim=dim)
        new_scale = torch.cat(scale_tensors, dim=dim)

        return FP8BlockwiseTensorHelper(new_data, new_scale, block_size)

    def contiguous(self):
        # [FUNCTION DOC]
        # Return a helper whose data and scale_inv are both made contiguous.
        """Make both data and scale_inv contiguous."""
        return FP8BlockwiseTensorHelper(
            self._rowwise_data.contiguous(),
            self._rowwise_scale_inv.contiguous(),
            self._block_size,
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1):
        # [FUNCTION DOC]
        # Flatten data and scale_inv over the same dim range (default all) and return a
        # new helper. Only meaningful when the flattened dims are not block dims.
        """Flatten operation."""
        new_data = self._rowwise_data.flatten(start_dim, end_dim)
        new_scale = self._rowwise_scale_inv.flatten(start_dim, end_dim)
        return FP8BlockwiseTensorHelper(new_data, new_scale, self._block_size)

    def to_pytorch_fp8(
        self, scale_inv_dtype: torch.dtype = torch.bfloat16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC]
        # Convert to a plain (float8_e4m3fn data, scale_inv) pair: data viewed as
        # float8, scale cast to scale_inv_dtype (default bfloat16) and trimmed/cloned to
        # [ceil(M/block), ceil(K/block)] where M is the first data dim and K the last,
        # dropping any TE padding.
        # KNOWN ISSUE: assumes 2D data; for >2D tensors M is taken from dim 0 and only
        # the first two scale dims are sliced, which is wrong.
        """Convert FP8BlockwiseTensorHelper to PyTorch float8 tensor.

        This function extracts the data and scale_inv from this FP8BlockwiseTensorHelper
        and converts them to PyTorch FP8 format

        Returns:
            Tuple of (torch_fp8_tensor, scale_inv) where:
            - torch_fp8_tensor: PyTorch float8 tensor (torch.float8_e4m3fn)
            - scale_inv: Inverse scale tensor (1/scale) with compact blockwise shape
                         [M/block_size, K/block_size] without padding
        """
        torch_fp8_tensor = self._rowwise_data.view(torch.float8_e4m3fn)

        # Extract rowwise_scale_inv and remove padding if needed
        # FP8BlockwiseTensorHelper's scale_inv should already be compact, but we verify and trim padding
        rowwise_scale_inv = self._rowwise_scale_inv.to(scale_inv_dtype)

        # Calculate the actual (unpadded) scale_inv shape from the data shape
        data_shape = torch_fp8_tensor.shape
        M = data_shape[0] if len(data_shape) >= 1 else 1
        K = data_shape[-1] if len(data_shape) >= 1 else 1

        # For 2D tensor (M, K), scale_inv should be (M // block_size, K // block_size)
        actual_scale_rows = (M + self._block_size - 1) // self._block_size
        actual_scale_cols = (K + self._block_size - 1) // self._block_size

        # Extract only the valid (non-padded) portion
        scale_inv = rowwise_scale_inv[:actual_scale_rows, :actual_scale_cols].clone()

        return torch_fp8_tensor, scale_inv

    @classmethod
    def from_te(
        cls,
        tensor: torch.Tensor,
        block_size: int = 128,
    ) -> "FP8BlockwiseTensorHelper":
        # [FUNCTION DOC]
        # Build a helper from a TransformerEngine Float8BlockwiseQTensor by cloning its
        # _rowwise_data and _rowwise_scale_inv (scale may include TE padding). Raises
        # ValueError if not a TE float8 tensor, RuntimeError if the rowwise attributes
        # are missing.
        """Convert Transformer Engine Float8BlockwiseQTensor to FP8BlockwiseTensorHelper.

        Args:
            tensor: Transformer Engine Float8BlockwiseQTensor to convert
            block_size: Block size (defaults to 128).

        Returns:
            FP8BlockwiseTensorHelper instance

        Raises:
            ValueError: If tensor is not a Transformer Engine Float8Tensor
            RuntimeError: If tensor doesn't have required attributes
        """
        if not is_float8tensor(tensor):
            raise ValueError("tensor must be a Transformer Engine Float8Tensor")

        if not (
            hasattr(tensor, "_rowwise_data") and hasattr(tensor, "_rowwise_scale_inv")
        ):
            raise RuntimeError(
                "tensor must be a Float8BlockwiseQTensor with _rowwise_data and _rowwise_scale_inv"
            )

        rowwise_data = tensor._rowwise_data.clone()
        rowwise_scale_inv = tensor._rowwise_scale_inv.clone()

        return cls(rowwise_data, rowwise_scale_inv, block_size)

    def to_te_fp8_inplace(self, target_te_tensor: torch.Tensor) -> None:
        # [FUNCTION DOC]
        # Copy this helper's data (viewed as uint8) and 2D scale_inv into an existing TE
        # Float8BlockwiseQTensor in place (scale copied into the top-left region of the
        # possibly padded TE scale buffer), synchronize the device via current_platform,
        # then regenerate TE's columnwise (transposed) copy on the target's CUDA device.
        # Raises ValueError if target is not a TE float8 tensor and RuntimeError if it
        # lacks rowwise buffers; asserts matching data shape and 2D scales. Returns
        # None.
        """Convert FP8BlockwiseTensorHelper to Transformer Engine Float8BlockwiseQTensor format inplace.

        This function copies the data and scale_inv from this FP8BlockwiseTensorHelper
        to an existing TE Float8BlockwiseQTensor.

        Args:
            target_te_tensor: Target TranformerEngine Float8BlockwiseQTensor to copy into
        """

        # Validate target tensor
        if not is_float8tensor(target_te_tensor):
            raise ValueError(
                "target_te_tensor must be a Transformer Engine Float8Tensor"
            )

        # For Float8BlockwiseQTensor, copy rowwise_data and rowwise_scale_inv
        if hasattr(target_te_tensor, "_rowwise_data") and hasattr(
            target_te_tensor, "_rowwise_scale_inv"
        ):
            assert self._rowwise_data.shape == target_te_tensor._rowwise_data.shape
            # rowwise_data is stored in uint8 format
            target_te_tensor._rowwise_data.copy_(
                self._rowwise_data.view(torch.uint8), non_blocking=False
            )
            scale_inv_shape = self._rowwise_scale_inv.shape
            assert len(scale_inv_shape) == 2
            target_te_tensor._rowwise_scale_inv[
                : scale_inv_shape[0], : scale_inv_shape[1]
            ].copy_(self._rowwise_scale_inv, non_blocking=False)
            current_platform.synchronize()
            with torch.cuda.device(target_te_tensor.device):
                target_te_tensor._create_columnwise()
        else:
            raise RuntimeError(
                "target_te_tensor must be a Transformer Engine Float8BlockwiseQTensor"
            )

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC]
        # Torch dispatch hook: routes aten cat (all inputs must be helpers, else
        # RuntimeError), split and chunk (when the first arg is a helper) to the
        # corresponding helper methods, reading dims/sizes from positional or keyword
        # args. Any other op raises NotImplementedError.
        """Intercept torch operations and handle them appropriately."""
        if kwargs is None:
            kwargs = {}

        # Handle torch.cat
        if func is torch.ops.aten.cat.default:
            tensors = args[0] if args else kwargs.get("tensors", [])
            if tensors and all(
                isinstance(t, FP8BlockwiseTensorHelper) for t in tensors
            ):
                dim = kwargs.get("dim", 0) if not args or len(args) < 2 else args[1]
                return FP8BlockwiseTensorHelper.cat(tensors, dim=dim)
            else:
                raise RuntimeError(f"cat delegate failed with tensors {tensors}")

        # Handle torch.split
        if func is torch.ops.aten.split.default:
            tensor = args[0]
            if isinstance(tensor, FP8BlockwiseTensorHelper):
                split_size = (
                    args[1] if len(args) > 1 else kwargs.get("split_size_or_sections")
                )
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                return tensor.split(split_size, dim=dim)

        # Handle torch.chunk
        if func is torch.ops.aten.chunk.default:
            tensor = args[0]
            if isinstance(tensor, FP8BlockwiseTensorHelper):
                chunks = args[1] if len(args) > 1 else kwargs.get("chunks")
                dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
                return tensor.chunk(chunks, dim=dim)

        raise NotImplementedError(f"operation {func} is not supported")


def convert_fp8_helper_to_pytorch_fp8(
    named_tensors: list[tuple[str, torch.Tensor | FP8BlockwiseTensorHelper]],
) -> list[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC]
    # Flatten a list of (name, tensor-or-helper) into plain tensors: each helper becomes
    # (name, float8 data) followed by (name + "_scale_inv", bfloat16 scale) using
    # to_pytorch_fp8; regular tensors pass through. Returns a new list.
    """Convert FP8BlockwiseTensorHelper instances in named tensors to PyTorch FP8 format.

    This function processes a list of (name, tensor) pairs and converts any
    FP8BlockwiseTensorHelper instances to PyTorch FP8 format by splitting them into
    weight and scale_inv tensors.

    Args:
        named_tensors: List of (name, tensor) pairs. Tensors can be regular tensors
                      or FP8BlockwiseTensorHelper instances.

    Returns:
        List of (name, tensor) pairs where FP8BlockwiseTensorHelper instances are
        converted to (weight_name, weight) and (weight_name_scale_inv, scale_inv) pairs.
    """
    converted = []
    for name, tensor in named_tensors:
        if isinstance(tensor, FP8BlockwiseTensorHelper):
            weight, scale_inv = tensor.to_pytorch_fp8()
            converted.append((name, weight))
            converted.append((f"{name}_scale_inv", scale_inv))
        else:
            converted.append((name, tensor))
    return converted

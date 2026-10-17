# [MODULE DOC]
# Vision SP Shard: splits vision-encoder (ViT) work across Ulysses sequence-parallel
# ranks for Qwen2-VL / Qwen2.5-VL / Qwen3-VL. Whole images are assigned contiguously to
# SP ranks, each rank encodes only its images, and the embeddings are all-gathered so
# every rank has the full vision embedding sequence. A custom autograd Function
# all-reduces (sum) gradients in backward and slices out each rank's portion. Naming
# uses "dp" for the ranks sharing vision work, which are in fact the SP group ranks.
# SPDX-License-Identifier: Apache-2.0

"""
Vision SP Shard: distribute vision encoder work across Ulysses SP ranks.

In VLMs with Ulysses Sequence Parallelism, each SP rank holds a disjoint
slice of the text sequence but must still produce the *full* set of vision
embeddings.  This module avoids redundant ViT computation by assigning
whole images (not sub-image patches) to individual SP ranks, running the
vision encoder locally, and then all-gathering the results so every rank
sees the complete embedding sequence.

Forward:  each rank encodes its assigned images → all_gather → full embeddings
Backward: all_reduce(SUM) recovers complete gradients → slice by assignment
"""

import importlib

import torch
import torch.distributed as dist
from torch.autograd import Function

from areal.models.fsdp.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)
from areal.utils import logging

logger = logging.getLogger("VisionSPShard")


def _get_image_patch_counts(grid_thw: torch.Tensor) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Given grid_thw [num_images, 3] (t, h, w patch grid per image), returns a
    # Python list of t*h*w per image (empty list for an empty tensor). Calls tolist
    # (host sync if on GPU).
    """Return [t*h*w for each image] from a [num_images, 3] grid_thw tensor."""
    if grid_thw.numel() == 0:
        return []
    return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()


def _get_image_embedding_counts(
    grid_thw: torch.Tensor, spatial_merge_size: int = 1
) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Like _get_image_patch_counts but returns per-image output-embedding counts
    # after spatial merging: t * (h // merge) * (w // merge); identical to patch counts
    # when spatial_merge_size is 1 (default).
    """Return per-image embedding counts after spatial merging: t * (h/merge) * (w/merge)."""
    if grid_thw.numel() == 0:
        return []

    if spatial_merge_size == 1:
        return (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

    t = grid_thw[:, 0]
    h = grid_thw[:, 1] // spatial_merge_size
    w = grid_thw[:, 2] // spatial_merge_size
    return (t * h * w).tolist()


def _assign_images_to_dp_ranks(
    patch_counts: list[int],
    dp_size: int,
) -> tuple[list[list[int]], list[int]]:
    # [FUNCTION DOC]
    # HELPER: Greedy contiguous bin packing of images (in order) onto dp_size ranks. For
    # each rank, target = remaining patches / remaining ranks; it takes images in order
    # (at least one when allowed) until its load reaches the target, capped so that
    # enough images remain to give each later rank one. When there are fewer images than
    # ranks, earlier ranks get none. Returns (per-rank lists of image indices, per-rank
    # patch loads). With zero images returns empty lists and zero loads.
    """Assign whole images to DP ranks via greedy contiguous bin-packing."""
    num_images = len(patch_counts)
    if num_images == 0:
        return [[] for _ in range(dp_size)], [0] * dp_size

    image_assignments = [[] for _ in range(dp_size)]
    rank_loads = [0] * dp_size

    remaining_patches = sum(patch_counts)
    img_idx = 0
    for rank in range(dp_size):
        remaining_ranks = dp_size - rank
        remaining_images = num_images - img_idx

        if remaining_images <= 0:
            break

        # Dynamic target: distribute remaining patches evenly among remaining ranks
        target = remaining_patches / remaining_ranks

        # Must leave at least 1 image for each remaining rank
        max_images = remaining_images - (remaining_ranks - 1)

        # Greedily add images until we reach the target load or hit the max
        count = 0
        while img_idx < num_images and count < max_images:
            image_assignments[rank].append(img_idx)
            rank_loads[rank] += patch_counts[img_idx]
            img_idx += 1
            count += 1

            # Stop early once we've reached the target (always take at least 1)
            if rank_loads[rank] >= target:
                break

        remaining_patches -= rank_loads[rank]

    return image_assignments, rank_loads


def _prepare_local_vision_inputs(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    image_assignments: list[list[int]],
    dp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    # [FUNCTION DOC]
    # HELPER: Selects this rank's slice of the flattened patch tensor pixel_values (rows
    # are patches of all images concatenated) and grid_thw rows for the rank's
    # contiguous image indices, using cumulative patch-count offsets. Returns
    # (local_pixel_values, local_grid_thw, local_indices); for a rank with no images
    # returns empty tensors of matching trailing dim/dtype/device and an empty list.
    # Relies on assignments being contiguous; uses .item().
    """Extract pixel values and grid_thw for this DP rank's assigned images."""
    local_indices = image_assignments[dp_rank]

    if len(local_indices) == 0:
        return (
            torch.empty(
                (0, pixel_values.shape[1]) if pixel_values.dim() > 1 else (0,),
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            ),
            torch.empty((0, 3), dtype=grid_thw.dtype, device=grid_thw.device),
            [],
        )

    # local_indices are contiguous (e.g. [2, 3, 4]), so use tensor slicing
    first_img_idx = local_indices[0]
    last_img_idx = local_indices[-1]

    # Compute patch offsets using cumsum
    patch_counts = _get_image_patch_counts(grid_thw)
    patch_counts_tensor = torch.tensor(
        patch_counts, device=grid_thw.device, dtype=torch.long
    )
    offsets = torch.cat(
        (
            torch.tensor([0], device=grid_thw.device, dtype=torch.long),
            torch.cumsum(patch_counts_tensor, dim=0),
        )
    )

    start_patch = offsets[first_img_idx].item()
    end_patch = offsets[last_img_idx + 1].item()

    local_pixel_values = pixel_values[start_patch:end_patch]
    local_grid_thw = grid_thw[first_img_idx : last_img_idx + 1]

    return local_pixel_values, local_grid_thw, local_indices


class GatherVisionEmbeddings(Function):
    # [CLASS DOC]
    # Autograd Function performing a variable-length all-gather of vision embeddings
    # over a process group. Forward pads each rank's [n_r, hidden] embeddings to the max
    # count, all-gathers, strips padding and concatenates in rank order. Backward sums
    # the incoming gradient across ranks (each rank only has gradient for vision tokens
    # in its own sequence shard) and returns this rank's contiguous slice.
    """All-gather vision embeddings across DP ranks with gradient support.

    Forward: all_gather + remove padding + concat.
    Backward: all_reduce(SUM) aggregates partial gradients, then slice by assignment.
    """

    @staticmethod
    def forward(
        ctx,
        local_embeddings: torch.Tensor,
        dp_group,
        all_counts: list[int],
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Saves group size/rank, group and all_counts (list of embedding counts per
        # rank, in rank order) on ctx. Returns the input unchanged when the group has
        # one rank or all counts are zero. Otherwise zero-pads local_embeddings to
        # max(all_counts) rows (hidden size from dim 1, or 1 for 1D input), all-gathers
        # padded tensors over dp_group, trims each to its count and concatenates.
        # Returns [sum(all_counts), hidden]. Collective call.
        # KNOWN ISSUE: 1D inputs are padded with a 2D zero tensor, which would fail the
        # concat; in practice inputs are 2D.
        dp_size = dist.get_world_size(dp_group)
        dp_rank = dist.get_rank(dp_group)
        ctx.dp_size = dp_size
        ctx.dp_group = dp_group
        ctx.all_counts = all_counts
        ctx.dp_rank = dp_rank

        if dp_size == 1:
            return local_embeddings

        max_count = max(all_counts) if all_counts else 0

        if max_count == 0:
            return local_embeddings

        hidden_size = local_embeddings.shape[1] if local_embeddings.dim() > 1 else 1

        # Pad to same length for all_gather
        if local_embeddings.shape[0] < max_count:
            pad_size = max_count - local_embeddings.shape[0]
            padding = torch.zeros(
                (pad_size, hidden_size),
                dtype=local_embeddings.dtype,
                device=local_embeddings.device,
            )
            local_padded = torch.cat([local_embeddings, padding], dim=0)
        else:
            local_padded = local_embeddings.contiguous()

        # All-gather
        gathered = [torch.empty_like(local_padded) for _ in range(dp_size)]
        dist.all_gather(gathered, local_padded, group=dp_group)

        # Remove padding and concat (no reordering needed - contiguous assignment)
        result_chunks = [gathered[r][: all_counts[r]] for r in range(dp_size)]
        result = torch.cat(result_chunks, dim=0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Returns grad unchanged for a single-rank group. Otherwise all-reduces (SUM, in
        # place on a contiguous copy/view) grad_output over the group and returns the
        # slice [sum(counts before rank), +count of rank) as the gradient for
        # local_embeddings, plus None for the group and counts arguments. Collective
        # call.
        # KNOWN ISSUE: forward returns early when all counts are zero but backward still
        # performs the all_reduce and slicing; consistent across ranks, so harmless.
        dp_size = ctx.dp_size

        if dp_size == 1:
            return grad_output, None, None

        all_counts = ctx.all_counts
        dp_rank = ctx.dp_rank
        dp_group = ctx.dp_group

        # all_reduce(SUM) aggregates partial gradients from all SP ranks:
        # each rank only has non-zero grad for vision tokens in its sequence shard.
        # NCCL all_reduce requires contiguous tensors — defensive guard.
        grad = grad_output.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=dp_group)

        # Extract gradients for this rank's images (contiguous slice)
        start = sum(all_counts[:dp_rank])
        end = start + all_counts[dp_rank]
        local_grad = grad[start:end]

        return local_grad, None, None


def _gather_vision_embeddings(
    local_embeddings: torch.Tensor,
    dp_group,
    all_counts: list[int],
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns local_embeddings unchanged if dp_group is None or has world size
    # 1; otherwise applies GatherVisionEmbeddings (autograd-aware gather).
    """All-gather vision embeddings from all DP ranks with gradient support."""
    if dp_group is None or dist.get_world_size(dp_group) == 1:
        return local_embeddings

    return GatherVisionEmbeddings.apply(local_embeddings, dp_group, all_counts)


def _unpack_deepstack(
    model,
    local_embeddings: torch.Tensor | tuple,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
    # [FUNCTION DOC]
    # HELPER: For Qwen3-VL style models (having deepstack_merger_list) splits the vision
    # forward output into (main embeddings, list of deepstack embeddings). If the output
    # is a tuple, returns its first two elements; if it is a plain tensor (rank with no
    # images), returns it together with one empty [0, hidden] requires-grad tensor per
    # deepstack merger (dtype/device from hidden_states). Returns None for models
    # without deepstack.
    """Unpack Qwen3-VL deepstack from forward output, or return None."""
    if not hasattr(model, "deepstack_merger_list"):
        return None

    if isinstance(local_embeddings, tuple):
        return local_embeddings[0], local_embeddings[1]

    # Empty rank: create matching empty deepstack tensors
    num_deepstack = len(model.deepstack_merger_list)
    h = local_embeddings.shape[1]
    deepstack = [
        torch.empty(
            (0, h), dtype=hidden_states.dtype, device=hidden_states.device
        ).requires_grad_()
        for _ in range(num_deepstack)
    ]
    return local_embeddings, deepstack


def create_dp_vision_forward(original_forward):
    # [FUNCTION DOC]
    # Factory wrapping a vision transformer's original forward(self, hidden_states,
    # grid_thw, **kwargs) into an SP-sharded forward. Returns the wrapper function.
    """Wrap VisionTransformer.forward for Vision SP Shard (shard across SP ranks).

    Strategy:
    1. Distribute whole images to SP ranks (not patches within images)
    2. Each rank processes its assigned images independently
    3. All-gather embeddings at the end (contiguous assignment, no reordering)

    Gradient correctness: after all-gather in forward, each SP rank's inputs_embeds
    contains vision tokens from ALL images. But Ulysses gives each rank only its
    sequence shard. In backward, each rank only has non-zero gradient for vision
    tokens in its own shard. The all_reduce(SUM) in GatherVisionEmbeddings.backward
    aggregates partial gradients from all ranks, recovering the complete gradient.
    """

    def dp_vision_forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Sharded vision forward. Falls back to original forward when Ulysses SP size <=
        # 1. Otherwise copies grid_thw to CPU, computes per-image patch counts
        # (asserting they sum to hidden_states rows) and post-merge embedding counts
        # (self.spatial_merge_size, default 1), assigns images to SP ranks, extracts
        # this rank's patches and grid (grid moved back to the original device) and runs
        # original forward on them. A rank with no images creates an empty [0, hidden]
        # requires-grad tensor, hidden from config out_hidden_size or hidden_size
        # (RuntimeError if neither exists). Unpacks deepstack outputs if present,
        # computes every rank's embedding count locally, all-gathers main embeddings
        # (asserting the total equals expected) and each deepstack tensor. Returns the
        # full embeddings, or (embeddings, list of deepstack embeddings) for deepstack
        # models. All SP ranks must call it together (collectives).
        sp_group = get_ulysses_sequence_parallel_group()
        sp_size = get_ulysses_sequence_parallel_world_size(sp_group)
        sp_rank = get_ulysses_sequence_parallel_rank(sp_group)

        if sp_size <= 1:
            return original_forward(self, hidden_states, grid_thw, **kwargs)

        # Move grid_thw to CPU once to avoid repeated GPU→CPU syncs in
        # metadata helpers (grid_thw is a tiny [num_images, 3] tensor).
        grid_thw_cpu = grid_thw.cpu()

        # Step 1: Get image assignment based on patch counts
        patch_counts = _get_image_patch_counts(grid_thw_cpu)
        total_patches = sum(patch_counts)

        assert hidden_states.shape[0] == total_patches, (
            f"[Vision SP Shard] Input patch count mismatch: "
            f"hidden_states.shape[0]={hidden_states.shape[0]}, "
            f"sum(grid_thw products)={total_patches}, "
            f"grid_thw.shape={grid_thw.shape}"
        )

        spatial_merge_size = getattr(self, "spatial_merge_size", 1)

        # Calculate embedding counts (after merger) for gather verification
        embedding_counts = _get_image_embedding_counts(grid_thw_cpu, spatial_merge_size)
        total_embeddings = sum(embedding_counts)

        image_assignments, _ = _assign_images_to_dp_ranks(patch_counts, sp_size)

        # Step 2: Extract local inputs (use CPU grid_thw to avoid GPU→CPU syncs
        # in metadata helpers; move local_grid_thw back to GPU for original_forward)
        local_pixels, local_grid_thw, local_indices = _prepare_local_vision_inputs(
            hidden_states, grid_thw_cpu, image_assignments, sp_rank
        )
        local_grid_thw = local_grid_thw.to(grid_thw.device)

        # Step 3: Process local images
        if local_pixels.shape[0] > 0:
            local_embeddings = original_forward(
                self, local_pixels, local_grid_thw, **kwargs
            )
        else:
            # This rank has no images, create empty tensor with correct hidden size
            cfg = getattr(self, "config", None)
            hidden_size = getattr(cfg, "out_hidden_size", None) or getattr(
                cfg, "hidden_size", None
            )
            if hidden_size is None:
                raise RuntimeError(
                    f"Cannot determine hidden_size from config. "
                    f"Model type: {type(self).__name__}"
                )

            local_embeddings = torch.empty(
                (0, hidden_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            # Empty rank must participate in autograd for backward all_reduce
            local_embeddings.requires_grad_()

        # Step 4: Unpack deepstack if present (Qwen3-VL returns (embeddings, list[Tensor]))
        local_deepstack = _unpack_deepstack(self, local_embeddings, hidden_states)
        if local_deepstack is not None:
            local_embeddings = local_deepstack[0]

        # Step 5: All-gather (contiguous assignment, no reordering needed)
        # Compute per-rank embedding counts locally (grid_thw is replicated on all ranks)
        all_counts = [
            sum(embedding_counts[i] for i in image_assignments[r])
            for r in range(sp_size)
        ]
        all_embeddings = _gather_vision_embeddings(
            local_embeddings, sp_group, all_counts
        )

        assert all_embeddings.shape[0] == total_embeddings, (
            f"[Vision SP Shard] Output embedding count mismatch: "
            f"all_embeddings.shape[0]={all_embeddings.shape[0]}, "
            f"expected={total_embeddings}"
        )

        # Step 6: All-gather deepstack embeddings (all ranks must participate)
        if local_deepstack is not None:
            gathered_deepstack = [
                _gather_vision_embeddings(ds, sp_group, all_counts)
                for ds in local_deepstack[1]
            ]
            return all_embeddings, gathered_deepstack

        return all_embeddings

    return dp_vision_forward


def _patch_vision_class(cls, class_name: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Replaces cls.forward with the SP-sharded wrapper unless the class is
    # already marked _vision_sp_shard_patched; sets the marker and logs INFO.
    # Class-level mutation.
    """Patch a single VisionTransformer class for Vision SP Shard, with idempotency guard."""
    if getattr(cls, "_vision_sp_shard_patched", False):
        return
    original = cls.forward
    cls.forward = create_dp_vision_forward(original)
    cls._vision_sp_shard_patched = True
    logger.info(f"[Vision SP Shard] Patched {class_name}.forward")


# Registry of supported VisionTransformer classes for Vision SP Shard patching.
# To add a new model: append (module_path, class_name).
_VISION_CLASSES = [
    (
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
        "Qwen2VisionTransformerPretrainedModel",
    ),
    (
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "Qwen2_5_VisionTransformerPretrainedModel",
    ),
    (
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "Qwen3VLVisionModel",
    ),
]


def apply_vision_sp_shard_patch():
    # [FUNCTION DOC]
    # Imports each registered HF vision transformer class (Qwen2-VL, Qwen2.5-VL,
    # Qwen3-VL) and patches its forward idempotently, silently skipping classes that
    # fail to import or are missing. Logs a warning if none were patched. Returns None;
    # process-global side effect.
    """Apply Vision SP Shard monkey patch to supported VisionTransformer classes.

    Patches the class-level forward method. Works whether called before or
    after model instantiation (Python MRO resolves at call time).
    Safe to call multiple times — each class is only patched once.
    """
    patched = []
    for module_path, class_name in _VISION_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            _patch_vision_class(cls, class_name)
            patched.append(class_name)
        except (ImportError, AttributeError):
            pass

    if not patched:
        logger.warning(
            "[Vision SP Shard] No VisionTransformer classes found to patch. "
            "Check that your transformers version supports "
            "Qwen2-VL, Qwen2.5-VL, or Qwen3-VL."
        )

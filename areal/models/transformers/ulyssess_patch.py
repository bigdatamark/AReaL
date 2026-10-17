# [MODULE DOC]
# Ulysses sequence-parallelism monkey patches for HuggingFace transformers models
# (adapted from verl). For text-only models it replaces transformers' internal
# _flash_attention_forward with an all-to-all wrapped version; for supported VLMs
# (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) it replaces the attention class forward, wraps the
# text model forward to slice inputs_embeds (and Qwen3-VL deepstack visual data) along
# the sequence dimension per SP rank, and optionally enables vision-tower SP sharding.
# SPDX-License-Identifier: Apache-2.0

# Adapted from verl

import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import PreTrainedModel

from areal.models.fsdp.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from areal.utils import logging

logger = logging.getLogger("UlyssesPatch")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Repeats KV heads n_rep times along the head dim. Input [batch, seq,
    # kv_heads, head_dim]; output [batch, seq, kv_heads*n_rep, head_dim] (each head
    # repeated contiguously, like repeat_interleave on dim 2). Returns the input
    # unchanged when n_rep is 1. Note: head dim is axis 2 here, unlike HF's repeat_kv
    # (axis 1).
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep).
    The hidden states go from (batch, seqlen, num_key_value_heads, head_dim)
    to (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: Drop-in replacement for transformers' _flash_attention_forward with
    # tensors in [1, S/sp, heads, head_dim] layout. When SP size > 1: if there are fewer
    # KV heads than the SP size, repeats K/V by sp // kv_heads (at least 1) so they can
    # be scattered, then all-to-all gathers sequence/scatters heads for Q, K and V;
    # calls the original flash attention with remaining args; finally all-to-alls back
    # to [1, S/sp, heads, head_dim]. Returns the attention output. Collective
    # communication when sp > 1.
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        repeats = max(ulysses_sp_size // key_states.size(2), 1)
        key_states = repeat_kv(key_states, repeats)
        value_states = repeat_kv(value_states, repeats)

        # (1, total_seqlen / sp_size, num_heads, head_dim)
        # -> (1, total_seqlen, num_heads / sp_size, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

    # (1, total_seqlen, num_heads / sp_size, head_dim)
    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        *args,
        **kwargs,
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen, num_heads / sp_size, head_dim)
        # -> (1, total_seqlen / sp_size, num_heads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


# NOTE: For vision models, inputs_embeds will be sliced instead of input_ids.
def patch_vlm_for_ulysses_input_slicing(model_class: type):
    # [FUNCTION DOC]
    # Patches model_class.forward (class-level, affects all instances) with a wrapper
    # that slices inputs_embeds along the sequence dim for this Ulysses rank before
    # calling the original forward. Logs an INFO message. Vision models are sliced on
    # inputs_embeds rather than input_ids.
    def _create_ulysses_wrapped_decoder_forward(original_forward):
        # [FUNCTION DOC]
        # HELPER: Closure factory returning the wrapped forward for original_forward.
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Wrapped text-model forward. Slices only when inputs_embeds is
            # given, SP size > 1 and the instance flag _needs_initial_slice (default
            # True) is set. Then: slices inputs_embeds on dim 1 (no padding); if
            # visual_pos_masks is given slices it likewise; if deepstack_visual_embeds
            # is also given, computes this rank's [start, end) sequence window (seq_len
            # // sp per rank), counts visual tokens before start and before end across
            # the whole batch, and slices each deepstack embedding tensor to that token
            # range (or to an empty slice when the shard has no visual tokens, keeping
            # grad flow). Sets _needs_initial_slice False during the call (to avoid
            # re-slicing on nested calls) and restores it in a finally block. Calls the
            # original forward with modified kwargs and returns its result. Uses .item()
            # (GPU-CPU sync).
            # KNOWN ISSUE: the window computation assumes the sequence length is
            # divisible by the SP size and that visual tokens are counted across the
            # flattened batch in order, which is only correct for batch size 1 / packed
            # inputs.
            inputs_embeds = kwargs.get("inputs_embeds")
            visual_pos_masks = kwargs.get("visual_pos_masks")
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
            call_kwargs = kwargs.copy()

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

            slice_now = (
                inputs_embeds is not None
                and current_ulysses_sp_size > 1
                and getattr(self, "_needs_initial_slice", True)
            )
            if slice_now:
                call_kwargs["inputs_embeds"] = slice_input_tensor(
                    inputs_embeds, dim=1, padding=False
                )
                # Slice visual_pos_masks and deepstack_visual_embeds for Qwen3-VL models (adapted from verl)
                if visual_pos_masks is not None:
                    original_visual_mask = visual_pos_masks
                    sliced_visual_mask = slice_input_tensor(
                        visual_pos_masks, dim=1, padding=False
                    )
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask

                    if deepstack_visual_embeds is not None:
                        sliced_embeds = []

                        num_visual_before = original_visual_mask.sum().item()
                        num_visual_in_shard = sliced_visual_mask.sum().item()

                        if num_visual_in_shard > 0 and num_visual_before > 0:
                            # Calculate which visual embeddings belong to this shard
                            # We need to find the offset of visual tokens in this shard

                            rank = get_ulysses_sequence_parallel_rank()
                            seq_len = original_visual_mask.shape[1]
                            local_seq_len = seq_len // current_ulysses_sp_size
                            start_idx = rank * local_seq_len
                            end_idx = start_idx + local_seq_len

                            # Get total visual tokens before and up to the end of the shard's sequence slice
                            # This correctly handles batches by summing across all samples
                            visual_start = (
                                original_visual_mask[:, :start_idx].sum().item()
                                if start_idx > 0
                                else 0
                            )
                            visual_end = original_visual_mask[:, :end_idx].sum().item()

                            # Slice each tensor in deepstack_visual_embeds
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[visual_start:visual_end])
                        else:
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow
                            for embed in deepstack_visual_embeds:
                                sliced_embeds.append(embed[:0])
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds
                self._needs_initial_slice = False
            try:
                return original_forward(self, *args, **call_kwargs)
            finally:
                if slice_now:
                    self._needs_initial_slice = True

        return ulysses_wrapped_decoder_forward

    original_forward = model_class.forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)
    model_class.forward = wrapped_forward
    logger.info(f"Patched {model_class.__name__}.forward")


def apply_monkey_patch(
    model: PreTrainedModel,
    ulysses_sp_size: int = 1,
    shard_vision_across_sp: bool = False,
):
    # [FUNCTION DOC]
    # Entry point that applies Ulysses SP patches for a model. Reads attention and KV
    # head counts from model.config (or model.config.text_config). Raises ValueError if
    # heads are not divisible by ulysses_sp_size, or if KV heads and sp size are not
    # divisible one way or the other. Returns immediately if ulysses_sp_size <= 1 (after
    # validation). For model_type in qwen2_5_vl/qwen2_vl/qwen3_vl: imports the HF
    # modeling module, replaces the attention class's forward with the matching areal
    # ulysses_flash_attn_forward, patches the text model class for input slicing, and if
    # shard_vision_across_sp applies the vision SP shard patch. Otherwise replaces
    # transformers.integrations.flash_attention._flash_attention_forward globally. All
    # patches are process-global class/module mutations; logs INFO messages.
    try:
        num_attention_heads, num_key_value_heads = (
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
        )
    except AttributeError:
        num_attention_heads, num_key_value_heads = (
            model.config.text_config.num_attention_heads,
            model.config.text_config.num_key_value_heads,
        )

    if num_attention_heads % ulysses_sp_size != 0:
        raise ValueError(
            f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"
        )
    if (
        num_key_value_heads % ulysses_sp_size != 0
        and ulysses_sp_size % num_key_value_heads != 0
    ):
        raise ValueError(
            f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "
            f"{ulysses_sp_size} or vice versa. Upon ulysses_sp_size % num_key_value_heads == 0,"
            "kv heads are repeated to ensure correctness."
        )

    vl_model_mappings = {
        "qwen2_5_vl": {
            "module": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
            "attn_class": "Qwen2_5_VLAttention",
            "model_class": "Qwen2_5_VLTextModel",
            "patch_module": "areal.models.transformers.qwen2_vl",
            "patch_attn_func": "ulysses_flash_attn_forward",
        },
        "qwen2_vl": {
            "module": "transformers.models.qwen2_vl.modeling_qwen2_vl",
            "attn_class": "Qwen2VLAttention",
            "model_class": "Qwen2VLTextModel",
            "patch_module": "areal.models.transformers.qwen2_vl",
            "patch_attn_func": "ulysses_flash_attn_forward",
        },
        "qwen3_vl": {
            "module": "transformers.models.qwen3_vl.modeling_qwen3_vl",
            "attn_class": "Qwen3VLTextAttention",
            "model_class": "Qwen3VLTextModel",
            "patch_module": "areal.models.transformers.qwen3_vl",
            "patch_attn_func": "ulysses_flash_attn_forward",
        },
    }

    if ulysses_sp_size <= 1:
        return

    if model.config.model_type in vl_model_mappings:
        # NOTE: The following code segment will patch only TextModel's attention and forward methods
        mapping = vl_model_mappings[model.config.model_type]
        module_name = mapping["module"]
        attn_class_name = mapping["attn_class"]
        model_class_name = mapping["model_class"]
        patch_module_name = mapping["patch_module"]
        patch_attn_func_name = mapping["patch_attn_func"]

        module = __import__(
            module_name,
            fromlist=[attn_class_name, model_class_name],
        )
        getattr(module, attn_class_name)
        attn_class = getattr(module, attn_class_name)
        model_class = getattr(module, model_class_name)

        patch_module = __import__(
            patch_module_name,
            fromlist=[patch_attn_func_name],
        )
        patch_attn_func = getattr(patch_module, patch_attn_func_name)

        attn_class.forward = patch_attn_func
        logger.info(
            f"Patched {module_name}.{attn_class_name}.forward using {patch_module_name}.{patch_attn_func_name} and {model_class_name}.forward"
        )

        patch_vlm_for_ulysses_input_slicing(model_class)
        logger.info(f"Patched {model_class_name}.forward")

        if shard_vision_across_sp:
            from areal.models.transformers.vision_sp_shard import (
                apply_vision_sp_shard_patch,
            )

            apply_vision_sp_shard_patch()
    else:
        from transformers.integrations import flash_attention

        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        logger.info(
            "Patched transformers.integrations.flash_attention._flash_attention_forward"
        )

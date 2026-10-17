# [MODULE DOC]
# HF <-> Archon state-dict conversion for Qwen3.5 dense/MoE hybrid models. Extends
# the Qwen3 scheme with GatedDeltaNet (linear_attn.*) keys including bare A_log/dt_bias
# parameters, MoE shared-expert and shared-expert-gate keys, and support for composite
# VLM checkpoints whose text weights live under "model.language_model." (detected from
# the HF safetensors index and normalized to/from the text-only "model." namespace).
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.tensor import DTensor

from areal.experimental.models.archon.base import BaseStateDictAdapter
from areal.experimental.models.archon.moe_weight_converter import (
    MoEConversionState,
    MoEWeightConverter,
)

if TYPE_CHECKING:
    from transformers import PretrainedConfig


class Qwen3_5StateDictAdapter(BaseStateDictAdapter):
    # [CLASS DOC]
    # BaseStateDictAdapter for Qwen3.5. State: composite prefix and whether the source
    # checkpoint uses it, from_hf_map/to_hf_map key templates, MoE detection
    # (moe_enabled, num_experts, read from text_config when present), weight-tying flag,
    # expert key templates and a MoEWeightConverter/MoEConversionState for DTensor
    # experts.
    """State dict adapter for Qwen3.5 Dense and MoE models.

    Handles:
    - Key name mapping between HF and Archon conventions
    - Dual-pattern mapping: full_attention (self_attn.*) and
      linear_attention (linear_attn.*) keys in a single map
    - Bare parameters (A_log, dt_bias) without .weight suffix
    - MoE expert weights: HF uses list of 2D weights, Archon uses 3D weights
    - DTensor-aware distributed checkpoint support for MoE
    - No weight permutation needed
    """

    def __init__(
        self, model_config: PretrainedConfig, hf_assets_path: str | None = None
    ):
        # [FUNCTION DOC]
        # Calls base init (which loads fqn_to_index_mapping from hf_assets_path),
        # detects
        # the composite namespace, builds the key maps (rotary inv_freq -> None),
        # determines
        # num_experts from num_experts / num_local_experts / moe_args.num_experts of the
        # text
        # config, enables MoE when > 1 or moe_enabled is True (ValueError if enabled but
        # count
        # unknown), reads tie_word_embeddings from the top-level config and creates the
        # MoE
        # converter/state for MoE models.
        # KNOWN ISSUE: tie_word_embeddings is read from the top-level config only, not
        # from
        # text_config, unlike the MoE settings.
        super().__init__(model_config, hf_assets_path)
        self._composite_text_prefix = "model.language_model."
        self._use_composite_text_namespace = self._detect_composite_text_namespace()

        # HuggingFace -> Archon key mapping
        # Use {} as placeholder for layer numbers and expert ids
        self.from_hf_map = {
            # Embedding
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # ---- full_attention layers (self_attn) ----
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Skip rotary_emb (computed at runtime)
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # ---- linear_attention layers (GatedDeltaNet) ----
            "model.layers.{}.linear_attn.in_proj_qkv.weight": "layers.{}.linear_attn.in_proj_qkv.weight",
            "model.layers.{}.linear_attn.in_proj_z.weight": "layers.{}.linear_attn.in_proj_z.weight",
            "model.layers.{}.linear_attn.in_proj_a.weight": "layers.{}.linear_attn.in_proj_a.weight",
            "model.layers.{}.linear_attn.in_proj_b.weight": "layers.{}.linear_attn.in_proj_b.weight",
            "model.layers.{}.linear_attn.conv1d.weight": "layers.{}.linear_attn.conv1d.weight",
            "model.layers.{}.linear_attn.out_proj.weight": "layers.{}.linear_attn.out_proj.weight",
            "model.layers.{}.linear_attn.norm.weight": "layers.{}.linear_attn.norm.weight",
            # Bare parameters (no .weight suffix)
            "model.layers.{}.linear_attn.A_log": "layers.{}.linear_attn.A_log",
            "model.layers.{}.linear_attn.dt_bias": "layers.{}.linear_attn.dt_bias",
            # ---- Common per-layer ----
            # LayerNorm
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # Dense MLP
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE experts (HF: 2D per expert, Archon: 3D combined)
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            # MoE router
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            # MoE shared expert
            "model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            # MoE shared expert gate (sigmoid gate for shared expert output)
            "model.layers.{}.mlp.shared_expert_gate.weight": "layers.{}.moe.shared_expert_gate.weight",
            # Final norm and output
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Build reverse mapping (Archon -> HF), excluding None values and MoE experts
        self.to_hf_map = {}
        for hf_key, archon_key in self.from_hf_map.items():
            if archon_key is not None:
                self.to_hf_map[archon_key] = hf_key

        # MoE configuration — drill into text_config for composite VLM configs
        moe_cfg = (
            model_config.text_config
            if hasattr(model_config, "text_config")
            else model_config
        )
        num_experts = getattr(moe_cfg, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(moe_cfg, "num_local_experts", None)
        if num_experts is None:
            moe_args = getattr(moe_cfg, "moe_args", None)
            if moe_args is not None:
                num_experts = getattr(moe_args, "num_experts", None)
        moe_enabled_flag = getattr(moe_cfg, "moe_enabled", None)
        self.moe_enabled = (num_experts is not None and num_experts > 1) or (
            moe_enabled_flag is True
        )
        if self.moe_enabled:
            if num_experts is None:
                raise ValueError(
                    "moe_enabled is True but num_experts could not be determined from "
                    "model_config. Expected one of: num_experts, num_local_experts, or "
                    "moe_args.num_experts"
                )
            self.num_experts = num_experts

        # Weight tying configuration
        self.enable_weight_tying = getattr(model_config, "tie_word_embeddings", False)

        # HF abstract key templates for MoE expert weights
        self._hf_expert_abstract_keys = {
            "layers.{}.moe.experts.w1": "model.layers.{}.mlp.experts.{}.gate_proj.weight",
            "layers.{}.moe.experts.w2": "model.layers.{}.mlp.experts.{}.down_proj.weight",
            "layers.{}.moe.experts.w3": "model.layers.{}.mlp.experts.{}.up_proj.weight",
        }

        # Composition: create MoE weight converter for DTensor-aware conversion
        self._moe_converter = MoEWeightConverter() if self.moe_enabled else None
        self._moe_state = MoEConversionState() if self.moe_enabled else None

    def to_hf(self, archon_state_dict: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Archon -> HF: skips output.weight under weight tying, splits expert 3D weights
        # into
        # per-expert HF entries (DTensor-aware), renames other keys (into composite
        # namespace if
        # detected), dropping unmapped keys. Returns a new dict.
        # KNOWN ISSUE: expert keys are always emitted in the text-only "model.layers..."
        # namespace even when the composite namespace is in use; critic score.weight is
        # dropped.
        """Convert Archon state dict to HuggingFace format.

        Main transformations:
        1. Key renaming: Archon names -> HF names
        2. MoE: 3D weights (num_experts, out, in) -> list of 2D weights
        3. Weight tying: skip output.weight if enabled
        """
        hf_state_dict = {}

        for key, value in archon_state_dict.items():
            # Skip output.weight when weight tying is enabled
            if self.enable_weight_tying and key == "output.weight":
                continue

            if "moe.experts.w" in key:
                if isinstance(value, DTensor):
                    hf_pairs = self._split_moe_experts_distributed(key, value)
                else:
                    hf_pairs = self._split_moe_experts(key, value)
                hf_state_dict.update(hf_pairs)
            else:
                hf_key = self._convert_key_to_hf(key)
                if hf_key is not None:
                    hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HF -> Archon: under weight tying with no lm_head, aliases lm_head to the
        # embedding
        # found by _find_hf_embed_tokens_key (on a shallow copy). Expert keys (MoE only)
        # are
        # handled as fused 3D, per-expert DTensor (accumulated through the converter) or
        # plain
        # per-expert tensors (preallocated 3D buffer). Other keys are normalized from
        # the
        # composite namespace and renamed; unmapped keys dropped. Returns the Archon
        # dict.
        """Convert HuggingFace state dict to Archon format.

        Main transformations:
        1. Key renaming: HF names -> Archon names
        2. MoE: list of 2D weights -> 3D weights (num_experts, out, in)
        3. Weight tying: copy embed_tokens to lm_head if missing
        """
        # Handle weight tying
        if self.enable_weight_tying and "lm_head.weight" not in hf_state_dict:
            embed_key = self._find_hf_embed_tokens_key(hf_state_dict)
            if embed_key is not None:
                hf_state_dict = dict(hf_state_dict)
                hf_state_dict["lm_head.weight"] = hf_state_dict[embed_key]

        state_dict = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, Any]]] = {}
        expert_buffer: dict[str, tuple[torch.Tensor, int]] = {}

        for key, value in hf_state_dict.items():
            if ".mlp.experts." in key and self.moe_enabled:
                # Try fused 3D format first (newer HF Qwen3-MoE format
                # where experts are stored as gate_up_proj / down_proj)
                fused = self._parse_fused_expert_key(key)
                if fused is not None:
                    layer_id, proj_type = fused
                    self._handle_fused_expert_weight(
                        layer_id, proj_type, value, state_dict
                    )
                    continue

                # Per-expert 2D format (older HF format)
                parsed = self._parse_expert_key(key)
                if parsed is None:
                    continue
                layer_id, expert_id, archon_abstract_key = parsed

                if isinstance(value, DTensor):
                    if layer_id not in expert_weights_by_layer:
                        expert_weights_by_layer[layer_id] = {}
                    if archon_abstract_key not in expert_weights_by_layer[layer_id]:
                        expert_weights_by_layer[layer_id][archon_abstract_key] = {}

                    expert_weights_by_layer[layer_id][archon_abstract_key][
                        expert_id
                    ] = value

                    assert (
                        self._moe_converter is not None and self._moe_state is not None
                    ), "MoE converter not initialized for MoE model"
                    result = self._moe_converter.concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        archon_abstract_key,
                        layer_id,
                        value.device_mesh,
                        self._moe_state,
                    )
                    if result is not None:
                        archon_key = archon_abstract_key.format(layer_id)
                        state_dict[archon_key] = result
                else:
                    self._collect_expert_weight(key, value, expert_buffer, state_dict)
            else:
                archon_key = self._convert_key_from_hf(key)
                if archon_key is not None:
                    state_dict[archon_key] = value

        return state_dict

    def convert_single_to_hf(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC]
        # Converts one Archon parameter for incremental weight sync: strips activation
        # checkpoint and torch.compile wrapper segments, splits expert weights
        # (gathering
        # DTensors), otherwise renames. Returns a list of (hf_name, tensor), empty if
        # unmapped.
        """Convert a single Archon (name, tensor) pair to HuggingFace format.

        Used for incremental weight updates.
        """
        # Strip activation checkpoint wrapper prefix if present
        name = name.replace("._checkpoint_wrapped_module", "")
        # Strip torch.compile wrapper prefix if present
        name = name.replace("._orig_mod", "")

        if "moe.experts.w" in name:
            hf_dict = self._split_moe_experts(name, tensor)
            return list(hf_dict.items())
        else:
            hf_key = self._convert_key_to_hf(name)
            if hf_key is not None:
                return [(hf_key, tensor)]
            return []

    def _convert_key_to_hf(self, archon_key: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Exact or layer-templated lookup in to_hf_map, then remapped into the
        # composite namespace when needed. Returns None if unmapped.
        """Convert an Archon key to HuggingFace key."""
        if archon_key in self.to_hf_map:
            return self._maybe_composite_hf_key(self.to_hf_map[archon_key])

        match = re.search(r"layers\.(\d+)\.", archon_key)
        if match:
            layer_num = match.group(1)
            abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", archon_key)
            if abstract_key in self.to_hf_map:
                hf_abstract = self.to_hf_map[abstract_key]
                return self._maybe_composite_hf_key(
                    hf_abstract.replace("{}", layer_num, 1)
                )

        return None

    def _convert_key_from_hf(self, hf_key: str) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Normalizes composite keys to "model.", then exact or layer-templated
        # lookup
        # in from_hf_map. Returns None for unmapped or None-mapped keys.
        """Convert a HuggingFace key to Archon key."""
        hf_key = self._normalize_hf_text_key(hf_key)
        if hf_key in self.from_hf_map:
            result = self.from_hf_map[hf_key]
            return result if result is not None else None

        match = re.search(r"layers\.(\d+)\.", hf_key)
        if match:
            layer_num = match.group(1)
            abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", hf_key)
            if abstract_key in self.from_hf_map:
                archon_abstract = self.from_hf_map[abstract_key]
                if archon_abstract is None:
                    return None
                return archon_abstract.replace("{}", layer_num, 1)

        return None

    def _parse_expert_key(self, hf_key: str) -> tuple[str, int, str] | None:
        # [FUNCTION DOC]
        # HELPER: After namespace normalization, regex-parses a per-expert key into
        # (layer_id str, expert_id int, Archon abstract key w1/w3/w2). None if no match.
        """Parse HF expert key into (layer_id, expert_id, archon_abstract_key)."""
        hf_key = self._normalize_hf_text_key(hf_key)
        match = re.match(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight",
            hf_key,
        )
        if not match:
            return None

        layer_id = match.group(1)
        expert_id = int(match.group(2))
        proj_type = match.group(3)

        proj_map = {
            "gate_proj": "layers.{}.moe.experts.w1",
            "up_proj": "layers.{}.moe.experts.w3",
            "down_proj": "layers.{}.moe.experts.w2",
        }
        archon_abstract_key = proj_map[proj_type]

        return layer_id, expert_id, archon_abstract_key

    # -- Fused expert format support (newer HF checkpoints) --

    _FUSED_EXPERT_RE = re.compile(
        r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)(?:\.weight)?$"
    )

    def _parse_fused_expert_key(self, hf_key: str) -> tuple[str, str] | None:
        # [FUNCTION DOC]
        # HELPER: After normalization, matches fused gate_up_proj/down_proj expert keys;
        # returns (layer_id, proj_type) or None.
        """Parse fused HF expert key into (layer_id, proj_type).

        Matches keys from newer HuggingFace format where expert weights
        are stored as 3D tensors (no per-expert index):
            model.layers.0.mlp.experts.gate_up_proj  (fused gate+up)
            model.layers.0.mlp.experts.down_proj
        """
        hf_key = self._normalize_hf_text_key(hf_key)
        match = self._FUSED_EXPERT_RE.match(hf_key)
        if not match:
            return None
        return match.group(1), match.group(2)

    def _normalize_hf_text_key(self, hf_key: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Rewrites "model.language_model.X" to "model.X"; other keys unchanged.
        """Normalize composite Qwen3.5 text keys to text-only namespace."""
        if hf_key.startswith(self._composite_text_prefix):
            return "model." + hf_key.removeprefix(self._composite_text_prefix)
        return hf_key

    def _maybe_composite_hf_key(self, hf_key: str) -> str:
        # [FUNCTION DOC]
        # HELPER: When the composite namespace is in use, rewrites "model.X" to
        # "model.language_model.X"; otherwise returns the key unchanged.
        """Map text-only HF keys into composite text namespace when needed."""
        if self._use_composite_text_namespace and hf_key.startswith("model."):
            return self._composite_text_prefix + hf_key.removeprefix("model.")
        return hf_key

    def _detect_composite_text_namespace(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if any key in the base-class fqn_to_index_mapping (from the HF
        # index
        # file) starts with the composite prefix; False if no mapping is loaded.
        """Detect whether source HF assets use composite Qwen3.5 text keys."""
        if not self.fqn_to_index_mapping:
            return False
        return any(
            key.startswith(self._composite_text_prefix)
            for key in self.fqn_to_index_mapping
        )

    def _find_hf_embed_tokens_key(self, hf_state_dict: dict[str, Any]) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the first embedding key present in hf_state_dict among the
        # namespace-adjusted, text-only and composite variants, or None.
        """Find the embedding weight key in text-only or composite checkpoints."""
        for key in (
            self._maybe_composite_hf_key("model.embed_tokens.weight"),
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ):
            if key in hf_state_dict:
                return key
        return None

    def _handle_fused_expert_weight(
        self,
        layer_id: str,
        proj_type: str,
        value: torch.Tensor,
        state_dict: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Gathers DTensor to full; gate_up_proj is split in half along dim 1
        # into w1
        # (gate) and w3 (up) contiguous tensors; down_proj stored as w2. Writes into
        # state_dict.
        """Handle fused 3D expert weights from newer HF format.

        Args:
            layer_id: Layer index as string.
            proj_type: 'gate_up_proj' or 'down_proj'.
            value: 3D tensor.  gate_up_proj has shape
                (num_experts, 2 * moe_inter_dim, dim); down_proj has shape
                (num_experts, dim, moe_inter_dim).
            state_dict: Output state dict to populate.
        """
        if isinstance(value, DTensor):
            value = value.full_tensor()

        if proj_type == "gate_up_proj":
            # Split fused gate+up into w1 (gate) and w3 (up) along dim 1
            half = value.shape[1] // 2
            state_dict[f"layers.{layer_id}.moe.experts.w1"] = value[
                :, :half, :
            ].contiguous()
            state_dict[f"layers.{layer_id}.moe.experts.w3"] = value[
                :, half:, :
            ].contiguous()
        elif proj_type == "down_proj":
            state_dict[f"layers.{layer_id}.moe.experts.w2"] = value

    def _split_moe_experts_distributed(
        self, archon_key: str, weight_3d: DTensor
    ) -> dict[str, DTensor]:
        # [FUNCTION DOC]
        # HELPER: Delegates a 3D DTensor expert weight to the converter's
        # split_expert_weights_dtensor using the HF expert template; {} if unrecognized.
        """Split 3D DTensor expert weight into 2D DTensors."""
        match = re.search(r"layers\.(\d+)\.", archon_key)
        if not match:
            return {}
        layer_id = match.group(1)

        archon_abstract_key = re.sub(r"layers\.\d+\.", "layers.{}.", archon_key)
        hf_abstract_key = self._hf_expert_abstract_keys.get(archon_abstract_key)
        if hf_abstract_key is None:
            return {}

        assert self._moe_converter is not None and self._moe_state is not None, (
            "MoE converter not initialized for MoE model"
        )
        return self._moe_converter.split_expert_weights_dtensor(
            hf_abstract_key, archon_abstract_key, layer_id, weight_3d, self._moe_state
        )

    def _split_moe_experts(
        self, archon_key: str, weight_3d: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Gathers DTensor to full and unbinds dim 0 into per-expert 2D views
        # keyed
        # "model.layers.L.mlp.experts.E.{gate|down|up}_proj.weight"; {} if unrecognized.
        """Split 3D expert weight into HuggingFace format 2D weights."""
        if isinstance(weight_3d, DTensor):
            weight_3d = weight_3d.full_tensor()

        match = re.search(r"layers\.(\d+)\.", archon_key)
        if not match:
            return {}
        layer_num = match.group(1)

        if ".w1" in archon_key:
            hf_proj = "gate_proj"
        elif ".w2" in archon_key:
            hf_proj = "down_proj"
        elif ".w3" in archon_key:
            hf_proj = "up_proj"
        else:
            return {}

        expert_weights = torch.unbind(weight_3d, dim=0)

        result = {}
        for expert_id, expert_weight in enumerate(expert_weights):
            hf_key = (
                f"model.layers.{layer_num}.mlp.experts.{expert_id}.{hf_proj}.weight"
            )
            result[hf_key] = expert_weight

        return result

    def _collect_expert_weight(
        self,
        hf_key: str,
        weight_2d: torch.Tensor,
        buffer: dict[str, tuple[torch.Tensor, int]],
        state_dict: dict[str, Any],
    ):
        # [FUNCTION DOC]
        # HELPER: Copies one per-expert 2D weight into a lazily preallocated
        # [num_experts, out, in] buffer; when all experts have arrived, moves it into
        # state_dict and drops the buffer entry.
        """Collect expert weights and merge into 3D when complete."""
        parsed = self._parse_expert_key(hf_key)
        if parsed is None:
            return

        layer_num, expert_num, archon_abstract_key = parsed
        archon_key = archon_abstract_key.format(layer_num)

        if archon_key not in buffer:
            weight_3d = torch.empty(
                self.num_experts,
                weight_2d.shape[0],
                weight_2d.shape[1],
                dtype=weight_2d.dtype,
                device=weight_2d.device,
            )
            buffer[archon_key] = (weight_3d, 0)

        weight_3d, count = buffer[archon_key]
        weight_3d[expert_num].copy_(weight_2d)
        buffer[archon_key] = (weight_3d, count + 1)

        if buffer[archon_key][1] == self.num_experts:
            state_dict[archon_key] = buffer[archon_key][0]
            del buffer[archon_key]

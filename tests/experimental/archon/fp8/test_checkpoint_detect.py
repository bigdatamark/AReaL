# [MODULE DOC]
# Unit tests for the private _detect_fp8_checkpoint helper, which decides whether an HF
# checkpoint directory holds blockwise FP8 weights by looking for "weight_scale_inv"
# keys in either model.safetensors.index.json or a single model.safetensors file. Uses
# pytest tmp_path; CPU only.
import json

import torch

from areal.experimental.models.archon.fp8_checkpoint import (
    _detect_fp8_checkpoint,
)


class TestDetectFP8Checkpoint:
    # [CLASS DOC]
    # Groups detection cases: index with/without scale keys, missing index, single-file
    # FP8 and BF16 safetensors, and an empty weight map.
    def test_fp8_with_scale_inv_keys(self, tmp_path):
        # [FUNCTION DOC]
        # Writes an index whose weight_map contains a weight and its weight_scale_inv
        # entry; asserts detection returns True.
        index = {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.layers.0.self_attn.q_proj.weight_scale_inv": "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        assert _detect_fp8_checkpoint(str(tmp_path)) is True

    def test_bf16_checkpoint(self, tmp_path):
        # [FUNCTION DOC]
        # Writes an index with only plain weight keys; asserts detection returns False.
        index = {
            "weight_map": {
                "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001.safetensors",
            }
        }
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

    def test_no_index_file(self, tmp_path):
        # [FUNCTION DOC]
        # Empty directory (no index, no safetensors); asserts detection returns False.
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

    def test_single_file_fp8_checkpoint(self, tmp_path):
        # [FUNCTION DOC]
        # Saves a single model.safetensors with an FP8 e4m3 weight and a float32 [2,4]
        # scale_inv tensor; asserts detection returns True without an index file.
        from safetensors.torch import save_file

        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(
                256, 512, dtype=torch.float8_e4m3fn
            ),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(
                2, 4, dtype=torch.float32
            ),
        }
        save_file(tensors, tmp_path / "model.safetensors")
        assert _detect_fp8_checkpoint(str(tmp_path)) is True

    def test_single_file_bf16_checkpoint(self, tmp_path):
        # [FUNCTION DOC]
        # Saves a single model.safetensors with only a BF16 weight; asserts detection
        # returns False.
        from safetensors.torch import save_file

        tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                256, 512, dtype=torch.bfloat16
            )
        }
        save_file(tensors, tmp_path / "model.safetensors")
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

    def test_empty_weight_map(self, tmp_path):
        # [FUNCTION DOC]
        # Index file with an empty weight_map; asserts detection returns False.
        index = {"weight_map": {}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        assert _detect_fp8_checkpoint(str(tmp_path)) is False

# [MODULE DOC]
# Tests for multi-node SGLang launch argument and command generation via
# SGLangConfig.build_args / build_cmd (nnodes, node_rank, dist_init_addr), with SGLang
# package version checks patched out.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SGLang multi-node CLI generation."""

from __future__ import annotations

from unittest.mock import patch

from areal.api.cli_args import SGLangConfig


class TestSGLangMultiNode:
    # [CLASS DOC]
    # Mirror of the vLLM multi-node tests for the SGLang backend.
    """Mirror of TestVLLMMultiNode for the SGLang backend."""

    def _build_args(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Calls SGLangConfig.build_args with defaults (model test-model, tp 8,
        # GPU 0)
        # overridden by kwargs, with version checks patched to pass; returns the args
        # dict.
        """Helper that patches sglang version checks away."""
        defaults = dict(
            sglang_config=SGLangConfig(model_path="test-model"),
            tp_size=8,
            base_gpu_id=0,
        )
        defaults.update(kwargs)
        with (
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            return SGLangConfig.build_args(**defaults)

    def _build_cmd(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Same as _build_args but calls SGLangConfig.build_cmd and returns the
        # command
        # list.
        """Helper that patches sglang version checks away."""
        defaults = dict(
            sglang_config=SGLangConfig(model_path="test-model"),
            tp_size=8,
            base_gpu_id=0,
        )
        defaults.update(kwargs)
        with (
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            return SGLangConfig.build_cmd(**defaults)

    def test_build_args_single_node_defaults(self):
        # [FUNCTION DOC]
        # Defaults give nnodes 1, node_rank 0 and no dist_init_addr.
        """Single-node (default) should have nnodes=1, node_rank=0."""
        args = self._build_args()
        assert args["nnodes"] == 1
        assert args["node_rank"] == 0
        assert args.get("dist_init_addr") is None

    def test_build_args_multi_node_head(self):
        # [FUNCTION DOC]
        # Head node (rank 0) of 2 nodes gets nnodes 2, node_rank 0 and the
        # dist_init_addr.
        """Head node (rank 0) with n_nodes > 1 should set nnodes and dist_init_addr."""
        args = self._build_args(
            tp_size=16,
            n_nodes=2,
            node_rank=0,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert args["dist_init_addr"] == "10.0.0.1:29500"

    def test_build_args_multi_node_worker(self):
        # [FUNCTION DOC]
        # Worker node (rank 1) gets nnodes 2, node_rank 1 and the dist_init_addr.
        """Worker node (rank > 0) should set nnodes and node_rank."""
        args = self._build_args(
            tp_size=16,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 1
        assert args["dist_init_addr"] == "10.0.0.1:29500"

    def test_build_args_multi_node_no_dist_init_addr(self):
        # [FUNCTION DOC]
        # Multi-node without dist_init_addr leaves it None.
        """Multi-node without dist_init_addr should have dist_init_addr=None."""
        args = self._build_args(
            tp_size=16,
            n_nodes=2,
            node_rank=0,
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert args.get("dist_init_addr") is None

    def test_build_cmd_multi_node_produces_flags(self):
        # [FUNCTION DOC]
        # Multi-node build_cmd contains --nnodes, --node-rank and --dist-init-addr
        # flags.
        """build_cmd with multi-node should produce CLI flags for nnodes and node-rank."""
        cmd = self._build_cmd(
            tp_size=16,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        cmd_str = " ".join(cmd)
        assert "--nnodes" in cmd_str
        assert "--node-rank" in cmd_str
        assert "--dist-init-addr" in cmd_str

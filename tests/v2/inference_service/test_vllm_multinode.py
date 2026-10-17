# [MODULE DOC]
# Tests for multi-node vLLM launch argument and command generation via
# vLLMConfig.build_args / build_cmd (nnodes, node_rank, headless, master addr/port split
# from dist_init_addr).
"""Tests for vLLM multi-node CLI generation."""

from __future__ import annotations

from areal.api.cli_args import vLLMConfig


class TestVLLMMultiNode:
    # [CLASS DOC]
    # Tests of vLLMConfig multi-node flag generation.
    def test_build_args_single_node_no_extra_flags(self):
        # [FUNCTION DOC]
        # Single-node has no nnodes, node_rank, headless or master address/port keys.
        """Single-node (default) should not add nnodes/node_rank/headless."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(cfg, tp_size=8, pp_size=1)
        assert "nnodes" not in args
        assert "node_rank" not in args
        assert "headless" not in args
        assert "master_addr" not in args
        assert "master_port" not in args

    def test_build_args_multi_node_head(self):
        # [FUNCTION DOC]
        # Head node gets nnodes/node_rank and master addr/port split from dist_init_addr
        # but no
        # headless flag.
        """Head node (rank 0) with n_nodes > 1 should add nnodes/node_rank but NOT headless."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=0,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert "headless" not in args
        assert args["master_addr"] == "10.0.0.1"
        assert args["master_port"] == "29500"

    def test_build_args_multi_node_worker(self):
        # [FUNCTION DOC]
        # Worker node (rank > 0) additionally gets headless True.
        """Worker node (rank > 0) should add headless=True."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 1
        assert args["headless"] is True
        assert args["master_addr"] == "10.0.0.1"
        assert args["master_port"] == "29500"

    def test_build_args_multi_node_no_dist_init_addr(self):
        # [FUNCTION DOC]
        # Without dist_init_addr no master_addr/master_port is added.
        """Multi-node without dist_init_addr should not add master_addr/master_port."""
        cfg = vLLMConfig(model="test-model")
        args = vLLMConfig.build_args(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=0,
        )
        assert args["nnodes"] == 2
        assert args["node_rank"] == 0
        assert "master_addr" not in args
        assert "master_port" not in args

    def test_build_cmd_multi_node_produces_flags(self):
        # [FUNCTION DOC]
        # Multi-node worker build_cmd includes --nnodes, --node-rank, --headless,
        # --master-addr
        # and --master-port.
        """build_cmd with multi-node should produce CLI flags for nnodes and node-rank."""
        cfg = vLLMConfig(model="test-model")
        cmd = vLLMConfig.build_cmd(
            cfg,
            tp_size=16,
            pp_size=1,
            n_nodes=2,
            node_rank=1,
            dist_init_addr="10.0.0.1:29500",
        )
        cmd_str = " ".join(cmd)
        assert "--nnodes" in cmd_str
        assert "--node-rank" in cmd_str
        assert "--headless" in cmd_str
        assert "--master-addr" in cmd_str
        assert "--master-port" in cmd_str

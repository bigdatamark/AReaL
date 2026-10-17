# [MODULE DOC]
# CPU-only unit tests for SGLang pipeline-parallel (PP) weight-update support: the per-
# PP-rank NCCL group payload built by SGLangBackend.build_init_weights_group_request
# (PP=1 legacy single group vs. PP>1 per-rank groups whose names end in _<digit>),
# allocation-mode parsing with a PP dimension, the pp_bridge module, pp_size threading
# into SGLangConfig and rl_trainer, and Archon's per-PP-stage weight sync (group
# creation, all-gather participation, train/gen PP-size validation) using fakes and
# monkeypatching.
"""Unit tests for sglang PP pipeline parallelism support.

Tests the per-PP-rank NCCL group creation logic in SGLangBackend and related
allocation mode parsing without requiring actual GPU hardware or a running
sglang server.

Covers two scenarios:
  1. PP=1 (original / backward compatible)
  2. PP>1 with per-PP-rank groups (group name ends with _{digit})

Also tests allocation mode parsing with PP dimension and pp_bridge module imports.
"""

import pytest

from areal.api.alloc_mode import (
    AllocationValidationError,
    ModelAllocation,
)
from areal.api.io_struct import WeightUpdateMeta
from areal.engine.sglang_remote import SGLangBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(tp=1, pp=1, dp=1, group_name="update_weight_group_0"):
    # [FUNCTION DOC]
    # HELPER: Builds an xccl WeightUpdateMeta whose gen_allocation is parsed from
    # "sglang:d{dp}p{pp}t{tp}", with master address 127.0.0.1, port 12345 and the given
    # NCCL group name (default "update_weight_group_0"). Returns the meta.
    """Build a WeightUpdateMeta with the given parallel dimensions."""
    meta = WeightUpdateMeta(type="xccl")
    meta.gen_allocation = ModelAllocation.from_str(f"sglang:d{dp}p{pp}t{tp}")
    meta.nccl_master_address = "127.0.0.1"
    meta.nccl_master_port = 12345
    meta.nccl_group_name = group_name
    return meta


# ===================================================================== #
#  Scenario 1: PP=1 (backward compatible, single group)                 #
# ===================================================================== #


class TestPP1BackwardCompatible:
    # [CLASS DOC]
    # PP=1 cases: one NCCL group spanning all generation workers plus the trainer;
    # payload has world_size = dp*tp + 1, rank_offset = 1 + server_idx*tp and no
    # pp_rank.
    """PP=1 should use original behavior: single NCCL group, no pp_rank."""

    def test_pp1_tp2_dp2_server0(self):
        # [FUNCTION DOC]
        # dp2 tp2, server 0: asserts world_size 5, rank_offset 1, no pp_rank.
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # world_size = total_gen_workers + 1 = 2*1*2 + 1 = 5
        assert req.payload["world_size"] == 5
        # rank_offset = 1 + server_idx * tp_size = 1 + 0*2 = 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp1_tp2_dp2_server1(self):
        # [FUNCTION DOC]
        # dp2 tp2, server 1: asserts rank_offset 3, world_size 5, no pp_rank.
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2)
        req = backend.build_init_weights_group_request("addr", 1, meta)
        # rank_offset = 1 + 1*2 = 3
        assert req.payload["rank_offset"] == 3
        assert req.payload["world_size"] == 5
        assert "pp_rank" not in req.payload

    def test_pp1_tp1_dp1(self):
        # [FUNCTION DOC]
        # dp1 tp1: asserts world_size 2, rank_offset 1, no pp_rank.
        """Simplest case: single GPU inference."""
        backend = SGLangBackend()
        meta = _make_meta(tp=1, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 2  # 1 + 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp1_tp4_dp1(self):
        # [FUNCTION DOC]
        # dp1 tp4: asserts world_size 5, rank_offset 1, no pp_rank.
        backend = SGLangBackend()
        meta = _make_meta(tp=4, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 5  # 4 + 1
        assert req.payload["rank_offset"] == 1
        assert "pp_rank" not in req.payload

    def test_pp1_group_name_preserved(self):
        # [FUNCTION DOC]
        # Asserts a custom group name is passed through unchanged.
        """Group name from meta should be passed through unchanged."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1, group_name="my_custom_group")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["group_name"] == "my_custom_group"

    def test_pp1_endpoint(self):
        # [FUNCTION DOC]
        # Asserts the request endpoint is /init_weights_update_group.
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.endpoint == "/init_weights_update_group"

    def test_pp1_master_address_and_port(self):
        # [FUNCTION DOC]
        # Asserts master_port in the payload is the string "12345".
        # KNOWN ISSUE: name mentions master address but only the port is asserted.
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1)
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["master_port"] == str(12345)


# ===================================================================== #
#  Scenario 2: PP>1 with per-PP-rank groups                             #
# ===================================================================== #


class TestPerPPRankGroups:
    # [CLASS DOC]
    # PP>1 cases: one group per PP rank; pp_rank is parsed from the group name suffix
    # and world_size = n_servers*tp + 1 where n_servers = gen world size / (tp*pp).
    """PP>1, group name ends with _{digit} -> per-PP-rank groups.

    All three training engines (Megatron, FSDP, Archon) use per-PP-rank
    group naming (``update_weight_group_{pp_rank}``) when PP>1.
    """

    def test_pp2_tp2_dp1_rank0(self):
        # [FUNCTION DOC]
        # pp2 tp2 dp1, group _0: asserts world_size 3, rank_offset 1, pp_rank 0.
        """PP=2, TP=2, DP=1: per-PP-rank group for pp_rank=0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # n_servers = world_size / (tp * pp) = 4 / (2*2) = 1
        # per-PP world_size = n_servers * tp + 1 = 1*2 + 1 = 3
        assert req.payload["world_size"] == 3
        # rank_offset = 1 + server_idx * tp = 1 + 0*2 = 1
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 0

    def test_pp2_tp2_dp1_rank1(self):
        # [FUNCTION DOC]
        # pp2 tp2 dp1, group _1: asserts world_size 3, rank_offset 1, pp_rank 1.
        """PP=2, TP=2, DP=1: per-PP-rank group for pp_rank=1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 3
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 1

    def test_dp2_pp2_tp2_server0(self):
        # [FUNCTION DOC]
        # dp2 pp2 tp2, server 0: asserts world_size 5, rank_offset 1, pp_rank 0.
        """DP=2, PP=2, TP=2: 8 total inference GPUs, server 0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr0", 0, meta)
        # n_servers = 8 / (2*2) = 2
        # per-PP world_size = 2*2 + 1 = 5
        assert req.payload["world_size"] == 5
        assert req.payload["rank_offset"] == 1  # 1 + 0*2
        assert req.payload["pp_rank"] == 0

    def test_dp2_pp2_tp2_server1(self):
        # [FUNCTION DOC]
        # dp2 pp2 tp2, server 1: asserts world_size 5, rank_offset 3, pp_rank 0.
        """DP=2, PP=2, TP=2: 8 total inference GPUs, server 1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr1", 1, meta)
        assert req.payload["world_size"] == 5
        assert req.payload["rank_offset"] == 3  # 1 + 1*2
        assert req.payload["pp_rank"] == 0

    def test_dp2_pp2_tp2_rank1_server0(self):
        # [FUNCTION DOC]
        # dp2 pp2 tp2, group _1, server 0: asserts world_size 5, rank_offset 1, pp_rank
        # 1.
        """DP=2, PP=2, TP=2: pp_rank=1, server 0."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=2, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr0", 0, meta)
        assert req.payload["world_size"] == 5
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 1

    def test_pp4_tp2_dp1_rank3(self):
        # [FUNCTION DOC]
        # pp4 tp2 dp1, group _3: asserts world_size 3 and pp_rank 3.
        """PP=4 with higher pp_rank to verify general extraction."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=4, dp=1, group_name="update_weight_group_3")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # n_servers = 8 / (2*4) = 1
        # per-PP world = 1*2 + 1 = 3
        assert req.payload["world_size"] == 3
        assert req.payload["pp_rank"] == 3

    def test_group_name_with_pp_rank_preserved(self):
        # [FUNCTION DOC]
        # Asserts the full group name (with rank suffix) is kept in the payload.
        """The full group name should be preserved in payload."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["group_name"] == "update_weight_group_0"


# ===================================================================== #
#  Group name parsing edge cases                                        #
# ===================================================================== #


class TestGroupNameParsing:
    # [CLASS DOC]
    # Edge cases for extracting pp_rank from group names.
    """Test that pp_rank extraction from group name handles edge cases."""

    def test_sequential_pp_ranks(self):
        # [FUNCTION DOC]
        # For suffixes 0, 1, 5, 10 with PP=2 asserts pp_rank equals the suffix.
        # KNOWN ISSUE: suffixes 5 and 10 exceed pp_size=2; the test confirms no range
        # validation is done, which may mask misconfigured group names.
        """All pp_rank values from 0..N should be correctly extracted."""
        backend = SGLangBackend()
        for pp_rank in [0, 1, 5, 10]:
            meta = _make_meta(
                tp=1, pp=2, dp=1, group_name=f"update_weight_group_{pp_rank}"
            )
            req = backend.build_init_weights_group_request("addr", 0, meta)
            assert req.payload["pp_rank"] == pp_rank

    def test_group_name_digit_suffix_only_triggers_when_pp_gt_1(self):
        # [FUNCTION DOC]
        # Asserts PP=1 never adds pp_rank even if the group name has a digit suffix.
        """Even with digit suffix, PP=1 should use Scenario 1 (no pp_rank)."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        # PP=1 -> Scenario 1, no pp_rank regardless of group name
        assert "pp_rank" not in req.payload


# ===================================================================== #
#  Allocation mode parsing with PP dimension                            #
# ===================================================================== #


class TestAllocationModeParsing:
    # [CLASS DOC]
    # Tests ModelAllocation.from_str parsing of d/p/t dimensions.
    """Test that sglang allocation mode correctly parses the PP dimension."""

    def test_sglang_with_pp(self):
        # [FUNCTION DOC]
        # Asserts sglang:d2p2t2 parses to pp 2, tp 2, dp 2, world 8.
        alloc = ModelAllocation.from_str("sglang:d2p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2
        assert alloc.parallel.dp_size == 2
        assert alloc.parallel.world_size == 8

    def test_sglang_without_pp(self):
        # [FUNCTION DOC]
        # Asserts sglang:d4t2 defaults pp to 1.
        alloc = ModelAllocation.from_str("sglang:d4t2")
        assert alloc.parallel.pp_size == 1
        assert alloc.parallel.tp_size == 2
        assert alloc.parallel.dp_size == 4

    def test_sglang_pp_only(self):
        # [FUNCTION DOC]
        # Asserts sglang:p2t2 parses pp 2, tp 2.
        alloc = ModelAllocation.from_str("sglang:p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2

    def test_megatron_with_pp(self):
        # [FUNCTION DOC]
        # Asserts megatron:d2p2t2 parses to pp 2, tp 2, dp 2, world 8.
        alloc = ModelAllocation.from_str("megatron:d2p2t2")
        assert alloc.parallel.pp_size == 2
        assert alloc.parallel.tp_size == 2
        assert alloc.parallel.dp_size == 2
        assert alloc.parallel.world_size == 8

    def test_fsdp_with_pp(self):
        # [FUNCTION DOC]
        # Asserts fsdp with p2 raises AllocationValidationError "FSDP backend only
        # supports".
        with pytest.raises(
            AllocationValidationError, match="FSDP backend only supports"
        ):
            ModelAllocation.from_str("fsdp:d2p2t2")

    def test_world_size_computation(self):
        # [FUNCTION DOC]
        # Asserts world_size = dp*pp*tp for sglang:d3p2t4.
        """world_size = dp * pp * tp."""
        alloc = ModelAllocation.from_str("sglang:d3p2t4")
        assert alloc.parallel.world_size == 3 * 2 * 4


# ===================================================================== #
#  Backward compatibility per engine type                               #
# ===================================================================== #


class TestBackwardCompatibilityPerEngine:
    # [CLASS DOC]
    # Checks each engine's group naming maps to the right scenario (PP=1 legacy vs per-
    # PP-rank).
    """Verify that each engine type's group naming convention maps to the
    correct scenario in build_init_weights_group_request."""

    def test_megatron_pp1_uses_scenario1(self):
        # [FUNCTION DOC]
        # Megatron PP=1 with suffix _0: asserts no pp_rank and world_size 5.
        """Megatron with PP=1: group_name='update_weight_group_0' but PP=1
        means Scenario 1 (no pp_rank in payload)."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert "pp_rank" not in req.payload
        assert req.payload["world_size"] == 5  # 4 + 1

    def test_megatron_pp2_uses_scenario2(self):
        # [FUNCTION DOC]
        # Megatron PP=2 with _0: asserts pp_rank 0 and world_size 3.
        """Megatron with PP=2: group_name='update_weight_group_0' and PP>1
        triggers per-PP-rank path."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["pp_rank"] == 0
        assert req.payload["world_size"] == 3  # 1*2 + 1

    def test_fsdp_pp1_uses_scenario1(self):
        # [FUNCTION DOC]
        # FSDP PP=1 with unsuffixed name: asserts no pp_rank and world_size 5.
        """FSDP with PP=1: group_name='update_weight_group', Scenario 1."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=1, dp=2, group_name="update_weight_group")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert "pp_rank" not in req.payload
        assert req.payload["world_size"] == 5  # 4 + 1

    def test_fsdp_pp2_per_pp_rank_groups(self):
        # [FUNCTION DOC]
        # FSDP PP=2 with _0: asserts pp_rank 0 and world_size 3.
        """FSDP with PP=2: per-PP-rank group names like 'update_weight_group_0',
        same as Megatron. Uses per-PP-rank path."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["pp_rank"] == 0
        assert req.payload["world_size"] == 3

    def test_archon_pp2_per_pp_rank_groups(self):
        # [FUNCTION DOC]
        # Archon PP=2 with _1: asserts pp_rank 1 and world_size 3.
        """Archon with PP=2: per-PP-rank group names like 'update_weight_group_1'."""
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["pp_rank"] == 1
        assert req.payload["world_size"] == 3


# ===================================================================== #
#  PPSchedulerBridge module importability and helpers                    #
# ===================================================================== #


class TestPPBridgeModule:
    # [CLASS DOC]
    # Import and basic behaviour tests for areal.v2.inference_service.sglang.pp_bridge.
    """Test the pp_bridge module can be imported and has expected symbols."""

    def test_pp_bridge_class_exists(self):
        # [FUNCTION DOC]
        # Asserts PPSchedulerBridge is importable.
        from areal.v2.inference_service.sglang.pp_bridge import (
            PPSchedulerBridge,
        )

        assert PPSchedulerBridge is not None

    def test_extract_pp_rank_from_group_name(self):
        # [FUNCTION DOC]
        # Asserts _extract_pp_rank_from_group_name returns the trailing integer for
        # update_weight_group_N and None for names without a numeric suffix.
        from areal.v2.inference_service.sglang.pp_bridge import (
            _extract_pp_rank_from_group_name,
        )

        assert _extract_pp_rank_from_group_name("update_weight_group_0") == 0
        assert _extract_pp_rank_from_group_name("update_weight_group_3") == 3
        assert _extract_pp_rank_from_group_name("update_weight_group_10") == 10
        assert _extract_pp_rank_from_group_name("my_custom_group") is None
        assert _extract_pp_rank_from_group_name("update_weight_group") is None

    def test_pp_bridge_bind_is_callable(self):
        # [FUNCTION DOC]
        # Asserts PPSchedulerBridge has a callable bind attribute.
        from areal.v2.inference_service.sglang.pp_bridge import (
            PPSchedulerBridge,
        )

        assert callable(getattr(PPSchedulerBridge, "bind", None))

    def test_pp_bridge_noop_when_pp1(self):
        # [FUNCTION DOC]
        # Constructs PPSchedulerBridge with a fake scheduler and server args pp_size=1
        # and asserts bind() does not raise (no-op).
        """PPSchedulerBridge.bind() should be a no-op when pp_size <= 1."""
        from areal.v2.inference_service.sglang.pp_bridge import (
            PPSchedulerBridge,
        )

        class FakeServerArgs:
            # [CLASS DOC]
            # HELPER: Fake server args with pp_size 1.
            pp_size = 1

        class FakeScheduler:
            # [CLASS DOC]
            # HELPER: Empty fake scheduler.
            pass

        bridge = PPSchedulerBridge(FakeScheduler(), FakeServerArgs())
        # Should not raise
        bridge.bind()


@pytest.mark.skipif(
    not __import__("areal.utils.pkg_version", fromlist=["is_available"]).is_available(
        "sglang"
    ),
    reason="sglang package not installed",
)
class TestLocalLaunchPPSizeThreading:
    # [CLASS DOC]
    # Skipped if sglang is not installed. Verifies pp_size is threaded through
    # SGLangConfig.build_args/build_cmd and rl_trainer so local launches honour PP.
    """``rl_trainer.py`` and the local controller path
    must thread ``pp_size`` through to ``SGLangConfig.build_args`` /
    ``SGLangConfig.build_cmd``; otherwise ``scheduler.type=local`` workflows
    silently fall back to PP=1.
    """

    def test_build_args_accepts_pp_size(self):
        # [FUNCTION DOC]
        # Asserts build_args with tp 2 / pp 2 returns both pp_size and tp_size 2.
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(model_path="/tmp/ignored")
        args = SGLangConfig.build_args(
            sglang_config=cfg,
            tp_size=2,
            pp_size=2,
            base_gpu_id=0,
        )
        assert args.get("pp_size") == 2
        assert args.get("tp_size") == 2

    def test_build_args_pp_size_default_is_one(self):
        # [FUNCTION DOC]
        # Asserts build_args without pp_size omits the key (backward compat) and keeps
        # tp_size.
        """Backward compat: omitting pp_size should not inject the key."""
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(model_path="/tmp/ignored")
        args = SGLangConfig.build_args(
            sglang_config=cfg,
            tp_size=2,
            base_gpu_id=0,
        )
        assert "pp_size" not in args
        assert args.get("tp_size") == 2

    def test_build_cmd_propagates_pp_size(self):
        # [FUNCTION DOC]
        # Asserts build_cmd with pp_size 2 includes a pp-size / pp_size flag in the
        # command text.
        from areal.api.cli_args import SGLangConfig

        cfg = SGLangConfig(model_path="/tmp/ignored")
        cmd = SGLangConfig.build_cmd(
            sglang_config=cfg,
            tp_size=2,
            pp_size=2,
            base_gpu_id=0,
        )
        joined = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
        assert ("pp-size" in joined) or ("pp_size" in joined)

    def test_rl_trainer_sglang_branch_passes_pp_size(self):
        # [FUNCTION DOC]
        # Source-level check: reads areal.trainer.rl_trainer source, finds the first
        # "SGLangConfig.build_args(" call and asserts "pp_size=" appears within the next
        # 800 chars.
        """Source-level regression: the SGLang branch in rl_trainer.py must
        pass ``pp_size=...parallel.pp_size`` into ``SGLangConfig.build_args``.
        """
        import inspect

        import areal.trainer.rl_trainer as mod

        src = inspect.getsource(mod)
        idx = src.find("SGLangConfig.build_args(")
        assert idx >= 0, "SGLangConfig.build_args(...) call not found"
        window = src[idx : idx + 800]
        assert "pp_size=" in window, (
            "rl_trainer.py must thread pp_size into SGLangConfig.build_args(); "
            "found:\n" + window
        )


class TestPPDPAttentionRankMath:
    # [CLASS DOC]
    # Checks per-PP world size and rank offset when PP>1 is combined with DP (multiple
    # servers): each stage group has dp*tp workers plus one trainer.
    """``build_init_weights_group_request`` must produce
    correct rank math when SGLang PP>1 is combined with DP-attention
    (``sglang:d>1p>1t*``). Each PP stage's group must contain
    ``dp_size * tp_size`` inference workers + 1 trainer.
    """

    def test_pp_with_dp_attention_per_pp_world_size(self):
        # [FUNCTION DOC]
        # dp2 tp4 pp2 group _0 server 0: asserts world_size 9, rank_offset 1, pp_rank 0.
        backend = SGLangBackend()
        # sglang:d2t4p2 -> n_servers=2, tp=4, pp=2
        meta = _make_meta(tp=4, pp=2, dp=2, group_name="update_weight_group_0")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 2 * 4 + 1  # n_servers*tp+1
        assert req.payload["rank_offset"] == 1  # server_idx=0
        assert req.payload["pp_rank"] == 0

    def test_pp_with_dp_attention_second_server(self):
        # [FUNCTION DOC]
        # dp2 tp4 pp2 group _1 server 1: asserts world_size 9, rank_offset 5, pp_rank 1.
        backend = SGLangBackend()
        meta = _make_meta(tp=4, pp=2, dp=2, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr", 1, meta)
        assert req.payload["world_size"] == 2 * 4 + 1
        # server_idx=1 -> rank_offset = 1 + 1*tp = 5
        assert req.payload["rank_offset"] == 1 + 1 * 4
        assert req.payload["pp_rank"] == 1

    def test_pp_without_dp_still_works(self):
        # [FUNCTION DOC]
        # dp1 tp2 pp2 group _1: asserts world_size 3, rank_offset 1, pp_rank 1.
        backend = SGLangBackend()
        meta = _make_meta(tp=2, pp=2, dp=1, group_name="update_weight_group_1")
        req = backend.build_init_weights_group_request("addr", 0, meta)
        assert req.payload["world_size"] == 1 * 2 + 1
        assert req.payload["rank_offset"] == 1
        assert req.payload["pp_rank"] == 1


class TestArchonPerStageInit:
    # [CLASS DOC]
    # Tests Archon _init_per_pp_weight_update_groups: each PP-stage head creates only
    # its own update_weight_group_<train_pp_rank>; non-heads create none but record all
    # names.
    """Archon's ``_init_per_pp_weight_update_groups`` must
    follow the Megatron pattern — each PP-stage head creates ONLY its own
    ``update_weight_group_{train_pp_rank}`` group, not enumerate all
    gen_pp_size groups from a single rank.
    """

    def test_per_stage_head_creates_only_own_group(self, monkeypatch):
        # [FUNCTION DOC]
        # Monkeypatches init_custom_process_group (records kwargs), find_free_ports and
        # gethostip; runs init for a head at train_pp_rank 1 with gen meta dp2 pp2 tp2.
        # Asserts exactly one group "update_weight_group_1" with world_size 5 and rank
        # 0, and state group_names/group_name set to that single name.
        """Stage head with train_pp_rank=k creates only group_k (one port,
        one group, world_size = per_pp_world_size + 1).
        """
        import areal.experimental.engine.archon_weight_sync as aws

        class FakeLogger:
            # [CLASS DOC]
            # HELPER: Logger stub with no-op info/debug.
            def info(self, *a, **k):
                # [FUNCTION DOC]
                # HELPER: No-op; accepts and ignores any arguments.
                pass

            def debug(self, *a, **k):
                # [FUNCTION DOC]
                # HELPER: No-op; accepts and ignores any arguments.
                pass

        class FakeFut:
            # [CLASS DOC]
            # HELPER: Completed-future stub.
            def result(self):
                # [FUNCTION DOC]
                # HELPER: Returns None (an already-completed future result).
                return None

        class FakeRollout:
            # [CLASS DOC]
            # HELPER: Rollout engine stub.
            def init_weights_update_group(self, meta):
                # [FUNCTION DOC]
                # HELPER: Ignores meta and returns a FakeFut.
                return FakeFut()

        class FakeLock:
            # [CLASS DOC]
            # HELPER: No-op lock usable as a context manager.
            def __enter__(self):
                # [FUNCTION DOC]
                # HELPER: Context-manager enter; returns self.
                return self

            def __exit__(self, *a):
                # [FUNCTION DOC]
                # HELPER: Context-manager exit; returns False so exceptions propagate.
                return False

        created = []

        def fake_init_pg(**kwargs):
            # [FUNCTION DOC]
            # HELPER: Records keyword arguments into created and returns a dummy
            # process-group object.
            created.append(kwargs)
            return object()

        def fake_free_ports(n):
            # [FUNCTION DOC]
            # HELPER: Returns n consecutive fake ports starting at 40000.
            return [40000 + i for i in range(n)]

        monkeypatch.setattr(aws, "init_custom_process_group", fake_init_pg)
        monkeypatch.setattr(aws, "find_free_ports", fake_free_ports)
        monkeypatch.setattr(aws, "gethostip", lambda: "127.0.0.1")

        class FakeParallelDims:
            # [CLASS DOC]
            # HELPER: Holds the training pp size as attribute pp.
            def __init__(self, pp):
                # [FUNCTION DOC]
                # HELPER: Stores pp.
                self.pp = pp

        class FakeEngine:
            # [CLASS DOC]
            # HELPER: Minimal Archon engine stub exposing logger, rollout_engine,
            # engine_lock, parallel_dims, PP-head check and pipeline_parallel_rank.
            def __init__(self, train_pp_rank, train_pp_size=2):
                # [FUNCTION DOC]
                # HELPER: Stores train pp rank and builds stub collaborators with
                # pp=train_pp_size (default 2).
                self._pp_rank = train_pp_rank
                self.logger = FakeLogger()
                self.rollout_engine = FakeRollout()
                self.engine_lock = FakeLock()
                self.parallel_dims = FakeParallelDims(pp=train_pp_size)

            def is_pipeline_parallel_head(self):
                # [FUNCTION DOC]
                # HELPER: Always True (this rank is a PP-stage head).
                return True

            @property
            def pipeline_parallel_rank(self):
                # [FUNCTION DOC]
                # HELPER: Returns the configured training PP rank.
                return self._pp_rank

        # gen_pp_size=2, gen_world_size=8 -> per_pp_world_size=4
        gen_pp_size = 2
        meta = _make_meta(tp=2, pp=gen_pp_size, dp=2)

        # Stage head at train_pp_rank=1 should ONLY create group_1
        engine = FakeEngine(train_pp_rank=1, train_pp_size=gen_pp_size)
        state = aws.WeightSyncState(pp_rank=1)
        aws._init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)

        assert len(created) == 1, f"expected 1 group, got {len(created)}"
        assert created[0]["group_name"] == "update_weight_group_1"
        assert created[0]["world_size"] == 4 + 1
        assert created[0]["rank"] == 0
        assert state.group_names == ["update_weight_group_1"]
        assert state.group_name == "update_weight_group_1"

    def test_non_pp_head_creates_no_group(self, monkeypatch):
        # [FUNCTION DOC]
        # For a non-head rank asserts no process group is created but state.group_names
        # still lists both per-rank names for later cleanup.
        """Non-PP-head ranks (dp>0/tp>0/cp>0) must not create any NCCL group."""
        import areal.experimental.engine.archon_weight_sync as aws

        created = []
        monkeypatch.setattr(
            aws,
            "init_custom_process_group",
            lambda **kw: created.append(kw) or object(),
        )
        monkeypatch.setattr(aws, "find_free_ports", lambda n: list(range(n)))
        monkeypatch.setattr(aws, "gethostip", lambda: "127.0.0.1")

        class FakeParallelDims:
            # [CLASS DOC]
            # HELPER: Holds pp size.
            def __init__(self, pp):
                # [FUNCTION DOC]
                # HELPER: Stores pp.
                self.pp = pp

        class FakeEngine:
            # [CLASS DOC]
            # HELPER: Engine stub that is not a PP head, PP rank 0, pp size 2, no-op
            # logger.
            logger = type(
                "L", (), {"info": lambda *a, **k: None, "debug": lambda *a, **k: None}
            )()
            parallel_dims = FakeParallelDims(pp=2)

            def is_pipeline_parallel_head(self):
                # [FUNCTION DOC]
                # HELPER: Returns False.
                return False

            @property
            def pipeline_parallel_rank(self):
                # [FUNCTION DOC]
                # HELPER: Returns 0.
                return 0

        gen_pp_size = 2
        meta = _make_meta(tp=2, pp=gen_pp_size, dp=2)
        state = aws.WeightSyncState(pp_rank=0)
        aws._init_per_pp_weight_update_groups(state, meta, FakeEngine(), gen_pp_size)

        assert created == []
        # State still records expected group names so destroy/cleanup can
        # iterate without special-casing this rank.
        assert state.group_names == [
            "update_weight_group_0",
            "update_weight_group_1",
        ]


class TestArchonPerStageAllGatherParticipation:
    # [CLASS DOC]
    # Regression tests that Archon _update_weights_per_stage calls _get_full_tensor for
    # every parameter on all ranks (needed for FSDP all-gather participation), while
    # only heads broadcast.
    """Regression: ``_update_weights_per_stage`` must call ``_get_full_tensor``
    on every rank (head and non-head) so the FSDP DTensor all-gather over the
    ``dp_shard * cp`` mesh can complete. Skipping it on non-PP-head ranks
    would deadlock the all-gather.
    """

    def _make_engine_and_state(self, is_head: bool, gen_pp_size: int = 2):
        # [FUNCTION DOC]
        # HELPER: Builds fakes for _update_weights_per_stage: an engine stub (logger,
        # lock, rollout whose update_weights_from_distributed returns a completed
        # future, no state_dict_adapter, two parameters w0/w1 holding CPU tensors) and a
        # WeightSyncState for pp_rank 0 that, for a head, has one group with name,
        # address and port filled in. Returns (engine, state, module).
        import areal.experimental.engine.archon_weight_sync as aws

        class FakeLogger:
            # [CLASS DOC]
            # HELPER: Logger stub with no-op info/debug.
            def info(self, *a, **k):
                # [FUNCTION DOC]
                # HELPER: No-op; accepts and ignores any arguments.
                pass

            def debug(self, *a, **k):
                # [FUNCTION DOC]
                # HELPER: No-op; accepts and ignores any arguments.
                pass

        class FakeLock:
            # [CLASS DOC]
            # HELPER: No-op lock context manager.
            def __enter__(self):
                # [FUNCTION DOC]
                # HELPER: Context-manager enter; returns self.
                return self

            def __exit__(self, *a):
                # [FUNCTION DOC]
                # HELPER: Context-manager exit; returns False so exceptions propagate.
                return False

        class FakeRollout:
            # [CLASS DOC]
            # HELPER: Rollout stub for distributed weight updates.
            def update_weights_from_distributed(self, meta, specs):
                # [FUNCTION DOC]
                # HELPER: Ignores arguments and returns a completed-future stub.
                class F:
                    # [CLASS DOC]
                    # HELPER: Completed-future stub.
                    def result(self_inner):
                        # [FUNCTION DOC]
                        # HELPER: Returns None (an already-completed future result).
                        return None

                return F()

        # Fake parameter object that records each call to _get_full_tensor.
        # We model it as a plain torch.Tensor (non-DTensor branch) so the
        # function returns it untouched but the call is still observable.
        import torch

        class FakeParam:
            # [CLASS DOC]
            # HELPER: Fake parameter carrying a tag and a 1-element CPU tensor as data.
            def __init__(self, tag):
                # [FUNCTION DOC]
                # HELPER: Stores tag and a zero tensor.
                self.tag = tag
                # plain CUDA-less tensor; _get_full_tensor's else branch
                # only does device-type checks, which a CPU tensor satisfies
                # without triggering all-gather (we are simulating the
                # call-count, not the all-gather itself).
                self.data = torch.zeros(1)

        class FakeEngine:
            # [CLASS DOC]
            # HELPER: Engine stub with configurable PP-head status yielding two named
            # parameters.
            logger = FakeLogger()
            engine_lock = FakeLock()
            rollout_engine = FakeRollout()
            state_dict_adapter = None

            def __init__(self, is_head):
                # [FUNCTION DOC]
                # HELPER: Stores whether this rank is a head.
                self._is_head = is_head

            def is_pipeline_parallel_head(self):
                # [FUNCTION DOC]
                # HELPER: Returns the configured head flag.
                return self._is_head

            def _get_model_name_parameters(self):
                # [FUNCTION DOC]
                # HELPER: Generator yielding ("w0", FakeParam) and ("w1", FakeParam).
                yield "w0", FakeParam("w0")
                yield "w1", FakeParam("w1")

        state = aws.WeightSyncState(pp_rank=0)
        if is_head:
            state.groups = [object()]
            state.group_names = ["update_weight_group_0"]
            state.master_addrs = ["127.0.0.1"]
            state.master_ports = [40000]
        return FakeEngine(is_head), state, aws

    def test_head_calls_get_full_tensor_for_each_param(self, monkeypatch):
        # [FUNCTION DOC]
        # Head rank: wraps _get_full_tensor to record tags, stubs the bucket broadcast,
        # sets a 1MB chunk size and asserts both params were gathered in order.
        engine, state, aws = self._make_engine_and_state(is_head=True)
        calls = []
        real = aws._get_full_tensor
        monkeypatch.setattr(
            aws,
            "_get_full_tensor",
            lambda p: (calls.append(p.tag), real(p))[1],
        )
        # Stub broadcast so we don't actually try to NCCL-broadcast.
        monkeypatch.setattr(
            aws,
            "_update_bucket_weights_multi_group",
            lambda *a, **k: None,
        )
        meta = _make_meta(tp=2, pp=2, dp=1)
        meta.weight_chunked_mem_mb = 1
        aws._update_weights_per_stage(state, meta, engine)
        assert calls == ["w0", "w1"], (
            f"PP-head must call _get_full_tensor for each param, got {calls}"
        )

    def test_non_head_still_calls_get_full_tensor(self, monkeypatch):
        # [FUNCTION DOC]
        # Non-head rank: asserts _get_full_tensor is still called for w0 and w1 and that
        # the bucket broadcast is never invoked.
        """Non-PP-head ranks (dp>0/tp>0/cp>0) must STILL iterate every param
        and call ``_get_full_tensor`` so the FSDP all-gather collective
        completes. Otherwise the head will hang on the all-gather.
        """
        engine, state, aws = self._make_engine_and_state(is_head=False)
        calls = []
        real = aws._get_full_tensor
        monkeypatch.setattr(
            aws,
            "_get_full_tensor",
            lambda p: (calls.append(p.tag), real(p))[1],
        )
        broadcast_calls = []
        monkeypatch.setattr(
            aws,
            "_update_bucket_weights_multi_group",
            lambda *a, **k: broadcast_calls.append(a),
        )
        meta = _make_meta(tp=2, pp=2, dp=1)
        meta.weight_chunked_mem_mb = 1
        aws._update_weights_per_stage(state, meta, engine)
        # Critical: every param must be visited so the FSDP all-gather
        # mesh has full participation.
        assert calls == ["w0", "w1"], (
            f"Non-PP-head must STILL call _get_full_tensor for every param "
            f"(FSDP all-gather participation); got {calls}"
        )
        # Non-head must NOT broadcast.
        assert broadcast_calls == [], (
            f"Non-PP-head must not broadcast; got {broadcast_calls}"
        )


class TestArchonPPSizeMismatchValidation:
    # [CLASS DOC]
    # Tests that _init_per_pp_weight_update_groups raises ValueError ("train_pp_size ==
    # gen_pp_size") on any rank when training and generation PP sizes differ, and passes
    # when they match.
    """when ``train_pp_size != gen_pp_size`` the per-PP-rank
    sync silently deadlocks (training heads create groups sglang never joins,
    or sglang stages have no training source). The init function MUST
    fail-fast with a clear ValueError on EVERY rank before any group-name
    bookkeeping.
    """

    class _FakeParallelDims:
        # [CLASS DOC]
        # HELPER: Holds pp size.
        def __init__(self, pp):
            # [FUNCTION DOC]
            # HELPER: Stores pp.
            self.pp = pp

    class _FakeEngine:
        # [CLASS DOC]
        # HELPER: Engine stub with configurable train pp size, pp rank and head flag;
        # no-op logger.
        def __init__(self, train_pp_size, train_pp_rank, is_head):
            # [FUNCTION DOC]
            # HELPER: Stores parallel_dims (pp=train_pp_size), pp rank, head flag and a
            # no-op logger.
            self.parallel_dims = TestArchonPPSizeMismatchValidation._FakeParallelDims(
                pp=train_pp_size
            )
            self._pp_rank = train_pp_rank
            self._is_head = is_head
            self.logger = type(
                "L",
                (),
                {"info": lambda *a, **k: None, "debug": lambda *a, **k: None},
            )()

        def is_pipeline_parallel_head(self):
            # [FUNCTION DOC]
            # HELPER: Returns the configured head flag.
            return self._is_head

        @property
        def pipeline_parallel_rank(self):
            # [FUNCTION DOC]
            # HELPER: Returns the configured pp rank.
            return self._pp_rank

    def test_train_lt_gen_raises_on_head(self):
        # [FUNCTION DOC]
        # Head with train pp 2 vs gen pp 4: asserts ValueError.
        """train_pp_size=2, gen_pp_size=4: sglang stages 2,3 have no training
        source; head must raise immediately.
        """
        import areal.experimental.engine.archon_weight_sync as aws

        gen_pp_size = 4
        meta = _make_meta(tp=2, pp=gen_pp_size, dp=1)
        engine = self._FakeEngine(train_pp_size=2, train_pp_rank=0, is_head=True)
        state = aws.WeightSyncState(pp_rank=0)
        with pytest.raises(ValueError, match="train_pp_size == gen_pp_size"):
            aws._init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)

    def test_train_gt_gen_raises_on_head(self):
        # [FUNCTION DOC]
        # Head with train pp 4 vs gen pp 2: asserts ValueError.
        """train_pp_size=4, gen_pp_size=2: training heads with rank 2,3 would
        create groups sglang never joins; must fail-fast.
        """
        import areal.experimental.engine.archon_weight_sync as aws

        gen_pp_size = 2
        meta = _make_meta(tp=2, pp=gen_pp_size, dp=1)
        engine = self._FakeEngine(train_pp_size=4, train_pp_rank=0, is_head=True)
        state = aws.WeightSyncState(pp_rank=0)
        with pytest.raises(ValueError, match="train_pp_size == gen_pp_size"):
            aws._init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)

    def test_mismatch_raises_on_non_head(self):
        # [FUNCTION DOC]
        # Non-head with train pp 2 vs gen pp 4: asserts ValueError too.
        """Non-PP-head ranks must ALSO raise; otherwise they silently record
        placeholder names while heads error out, leaving the world in an
        inconsistent state.
        """
        import areal.experimental.engine.archon_weight_sync as aws

        gen_pp_size = 4
        meta = _make_meta(tp=2, pp=gen_pp_size, dp=1)
        engine = self._FakeEngine(train_pp_size=2, train_pp_rank=0, is_head=False)
        state = aws.WeightSyncState(pp_rank=0)
        with pytest.raises(ValueError, match="train_pp_size == gen_pp_size"):
            aws._init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)

    def test_match_passes_validation(self, monkeypatch):
        # [FUNCTION DOC]
        # Matching pp sizes (2) on a head with patched group creation: asserts no error
        # and group_names is ["update_weight_group_0"].
        """train_pp_size == gen_pp_size: validation passes; head proceeds to
        normal group creation.
        """
        import areal.experimental.engine.archon_weight_sync as aws

        monkeypatch.setattr(aws, "init_custom_process_group", lambda **kw: object())
        monkeypatch.setattr(aws, "find_free_ports", lambda n: [40000])
        monkeypatch.setattr(aws, "gethostip", lambda: "127.0.0.1")

        class FakeFut:
            # [CLASS DOC]
            # HELPER: Completed-future stub.
            def result(self):
                # [FUNCTION DOC]
                # HELPER: Returns None (an already-completed future result).
                return None

        class FakeRollout:
            # [CLASS DOC]
            # HELPER: Rollout stub.
            def init_weights_update_group(self, m):
                # [FUNCTION DOC]
                # HELPER: Returns a FakeFut.
                return FakeFut()

        class FakeLock:
            # [CLASS DOC]
            # HELPER: No-op lock context manager.
            def __enter__(self):
                # [FUNCTION DOC]
                # HELPER: Context-manager enter; returns self.
                return self

            def __exit__(self, *a):
                # [FUNCTION DOC]
                # HELPER: Context-manager exit; returns False so exceptions propagate.
                return False

        gen_pp_size = 2
        meta = _make_meta(tp=2, pp=gen_pp_size, dp=2)
        engine = self._FakeEngine(
            train_pp_size=gen_pp_size, train_pp_rank=0, is_head=True
        )
        engine.rollout_engine = FakeRollout()
        engine.engine_lock = FakeLock()
        state = aws.WeightSyncState(pp_rank=0)
        # Should not raise.
        aws._init_per_pp_weight_update_groups(state, meta, engine, gen_pp_size)
        assert state.group_names == ["update_weight_group_0"]

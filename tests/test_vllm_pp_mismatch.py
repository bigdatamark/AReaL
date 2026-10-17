# [MODULE DOC]
# Tests that vLLM rollout weight-sync uses a single flat NCCL group (so trainer and
# vLLM pipeline-parallel sizes may differ), while the Archon dispatcher still enforces
# train_pp_size == gen_pp_size for SGLang PP>1. No GPU or server; network helpers are
# monkeypatched.
"""Unit tests for differing training/inference pipeline-parallel sizes with vLLM.

The per-PP-rank weight-sync path (one NCCL group per inference PP stage) is
specific to SGLang: its rollout-side ``build_init_weights_group_request`` forms
one group per PP stage and therefore requires ``train_pp_size == gen_pp_size``.
vLLM instead joins a single flat group spanning every inference worker, so a
Megatron/Archon actor may train with a different PP size than a vLLM rollout
(e.g. ``megatron:d1p4t2`` + ``vllm:d1p2t4``).

These tests lock in that behavior:

  1. ``VLLMBackend`` builds a single flat group regardless of its own PP size
     (no ``pp_rank`` in the payload, ``world_size`` counts all workers).
  2. The Archon weight-sync dispatcher routes vLLM (any PP size) to the
     single-group path and does NOT enforce ``train_pp_size == gen_pp_size``.
  3. The same dispatcher still enforces the 1:1 mapping for SGLang PP>1, so the
     SGLang guarantees are unchanged.

Runs without GPUs or a live inference server.
"""

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.api.io_struct import WeightUpdateMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(backend, tp=1, pp=1, dp=1, group_name="update_weight_group_0"):
    # [FUNCTION DOC]
    # HELPER: Returns an "xccl" WeightUpdateMeta whose gen_allocation is parsed from
    # "<backend>:d<dp>p<pp>t<tp>", with fixed master address 127.0.0.1, port 12345 and the
    # given group_name.
    """Build a WeightUpdateMeta whose rollout allocation uses ``backend``."""
    meta = WeightUpdateMeta(type="xccl")
    meta.gen_allocation = ModelAllocation.from_str(f"{backend}:d{dp}p{pp}t{tp}")
    meta.nccl_master_address = "127.0.0.1"
    meta.nccl_master_port = 12345
    meta.nccl_group_name = group_name
    return meta


# ===================================================================== #
#  vLLM rollout backend: single flat group regardless of PP             #
# ===================================================================== #


class TestVLLMBackendFlatGroup:
    # [CLASS DOC]
    # Checks VLLMBackend.build_init_weights_group_request world_size/rank_offset math.
    """vLLM joins one group spanning all inference workers; its internal PP is
    just more flat workers and never appears as a per-PP group."""

    def test_pp2_single_flat_group_server0(self):
        # [FUNCTION DOC]
        # vllm tp4 pp2 dp1, server 0: asserts world_size 9, rank_offset 1, no pp_rank key.
        from areal.engine.vllm_remote import VLLMBackend

        meta = _make_meta("vllm", tp=4, pp=2, dp=1)
        req = VLLMBackend().build_init_weights_group_request("addr", 0, meta)
        # world_size = dp*tp*pp + 1 = 1*4*2 + 1 = 9 (all workers + trainer)
        assert req.payload["world_size"] == 9
        # rank_offset = 1 + server_idx * tp * pp = 1 + 0 = 1
        assert req.payload["rank_offset"] == 1
        # vLLM has no per-PP path: pp_rank must never be emitted.
        assert "pp_rank" not in req.payload

    def test_pp2_flat_group_server_offset(self):
        # [FUNCTION DOC]
        # vllm tp4 pp2 dp2, server 1: asserts world_size 17, rank_offset 9, no pp_rank key.
        from areal.engine.vllm_remote import VLLMBackend

        meta = _make_meta("vllm", tp=4, pp=2, dp=2)
        req = VLLMBackend().build_init_weights_group_request("addr", 1, meta)
        # world_size = 2*4*2 + 1 = 17
        assert req.payload["world_size"] == 17
        # rank_offset = 1 + server_idx * tp * pp = 1 + 1*8 = 9
        assert req.payload["rank_offset"] == 9
        assert "pp_rank" not in req.payload


# ===================================================================== #
#  Archon dispatcher: backend-gated routing                             #
# ===================================================================== #


class _FakeParallelDims:
    # [CLASS DOC]
    # HELPER: Stub of engine parallel_dims exposing only the pp size.
    def __init__(self, pp):
        # [FUNCTION DOC]
        # Stores pp.
        self.pp = pp


class _FakeEngine:
    # [CLASS DOC]
    # HELPER: Fake engine exposing parallel_dims, PP head/rank and a no-op logger.
    """Minimal ArchonEngine stand-in for the weight-sync dispatcher."""

    def __init__(self, train_pp_size, train_pp_rank=0, is_head=False):
        # [FUNCTION DOC]
        # Stores fake PP size/rank/head flag and builds a logger whose info/debug do nothing.
        self.parallel_dims = _FakeParallelDims(pp=train_pp_size)
        self._pp_rank = train_pp_rank
        self._is_head = is_head
        self.logger = type(
            "L", (), {"info": lambda *a, **k: None, "debug": lambda *a, **k: None}
        )()

    def is_pipeline_parallel_head(self):
        # [FUNCTION DOC]
        # Returns the configured is_head flag.
        return self._is_head

    @property
    def pipeline_parallel_rank(self):
        # [FUNCTION DOC]
        # Property returning the configured PP rank.
        return self._pp_rank


class TestArchonDispatcherBackendRouting:
    # [CLASS DOC]
    # Checks Archon init_weight_update_group routing by rollout backend.
    """``init_weight_update_group`` must select the per-PP-rank path only for
    SGLang PP>1, and fall back to the single-group path for vLLM."""

    def test_vllm_pp_mismatch_does_not_raise(self, monkeypatch):
        # [FUNCTION DOC]
        # vllm pp2 rollout with train pp4, non-head: asserts init_weight_update_group succeeds,
        # marks initialized, records no per-PP groups and creates no group.
        """megatron:p4 actor + vllm:p2 rollout: differing PP sizes are valid.

        The dispatcher must route to the single-group path and skip the
        SGLang-only ``train_pp_size == gen_pp_size`` check.
        """
        import areal.experimental.engine.archon_weight_sync as aws

        monkeypatch.setattr(aws, "find_free_ports", lambda n: [40000])
        monkeypatch.setattr(aws, "gethostip", lambda: "127.0.0.1")

        meta = _make_meta("vllm", tp=4, pp=2, dp=1)
        engine = _FakeEngine(train_pp_size=4, is_head=False)
        state = aws.WeightSyncState(pp_rank=0)

        # Must NOT raise despite train_pp_size (4) != gen_pp_size (2).
        aws.init_weight_update_group(state, meta, engine)

        assert state.group_initialized is True
        # Single-group path => no per-PP groups recorded.
        assert state.groups == []
        assert state.group is None  # non-head does not create the group

    def test_sglang_pp_mismatch_still_raises(self, monkeypatch):
        # [FUNCTION DOC]
        # sglang pp2 with train pp4: asserts ValueError matching "train_pp_size == gen_pp_size".
        """SGLang PP>1 must still enforce the 1:1 training/inference mapping."""
        import areal.experimental.engine.archon_weight_sync as aws

        monkeypatch.setattr(aws, "find_free_ports", lambda n: [40000])
        monkeypatch.setattr(aws, "gethostip", lambda: "127.0.0.1")

        meta = _make_meta("sglang", tp=2, pp=2, dp=1)
        engine = _FakeEngine(train_pp_size=4, is_head=False)
        state = aws.WeightSyncState(pp_rank=0)

        with pytest.raises(ValueError, match="train_pp_size == gen_pp_size"):
            aws.init_weight_update_group(state, meta, engine)

    def test_vllm_pp_matched_uses_single_group(self, monkeypatch):
        # [FUNCTION DOC]
        # vllm with matching PP sizes: asserts the single-group path is still used.
        """Routing is gated on backend, not PP size: vLLM with matching PP
        sizes still takes the single-group path (never per-PP)."""
        import areal.experimental.engine.archon_weight_sync as aws

        monkeypatch.setattr(aws, "find_free_ports", lambda n: [40000])
        monkeypatch.setattr(aws, "gethostip", lambda: "127.0.0.1")

        meta = _make_meta("vllm", tp=2, pp=2, dp=1)
        engine = _FakeEngine(train_pp_size=2, is_head=False)
        state = aws.WeightSyncState(pp_rank=0)

        aws.init_weight_update_group(state, meta, engine)

        assert state.group_initialized is True
        assert state.groups == []

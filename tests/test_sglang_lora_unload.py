# [MODULE DOC]
# Regression tests for SGLangBackend.build_disk_weight_update_requests with LoRA:
# loading adapter version N must also emit a best-effort unload of version N -
# lora_keep_versions so old adapters do not accumulate; keep=0 and non-LoRA updates are
# unchanged.
"""Tests for SGLang LoRA disk weight-update request building.

Regression: each train step loads `lora-<name>-v{N}` via /load_lora_adapter but
never unloads old versions, so sglang accumulates adapters until it hangs
(~130 steps, VRAM creeps up). The builder must emit a best-effort unload of the
version that falls outside the retention window (kept for off-policy rollouts).
"""

from areal.api.io_struct import WeightUpdateMeta
from areal.engine.sglang_remote import SGLangBackend


def _lora_meta(version, keep, lora_name="lora-gsm8k", path="/tmp/wu"):
    # [FUNCTION DOC]
    # HELPER: Builds a disk-type WeightUpdateMeta with use_lora True, the given version,
    # lora_keep_versions=keep, lora_name and path (defaults "lora-gsm8k", "/tmp/wu").
    return WeightUpdateMeta(
        type="disk",
        path=path,
        use_lora=True,
        lora_name=lora_name,
        lora_keep_versions=keep,
        version=version,
    )


def _triples(reqs):
    # [FUNCTION DOC]
    # HELPER: Maps a request bundle to a list of (endpoint, payload lora_name or None,
    # best_effort) tuples.
    """(endpoint, lora_name, best_effort) for each request."""
    return [
        (r.endpoint, r.payload.get("lora_name"), r.best_effort) for r in reqs.requests
    ]


def test_disk_update_unloads_stale_version_beyond_window():
    # [FUNCTION DOC]
    # Version 10, keep 4: asserts a best-effort load of lora-gsm8k-v10 and exactly one
    # best-effort unload of lora-gsm8k-v6.
    # KNOWN ISSUE: asserts the load request is best_effort True; if loads are meant to
    # be mandatory this assertion encodes that loads are best-effort too, which may be
    # unintended.
    """load v{N} also emits a best-effort unload of v{N - keep}."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(10, keep=4))
    triples = _triples(reqs)
    assert ("/load_lora_adapter", "lora-gsm8k-v10", True) in triples
    unloads = [t for t in triples if t[0] == "/unload_lora_adapter"]
    assert unloads == [("/unload_lora_adapter", "lora-gsm8k-v6", True)]


def test_disk_update_no_unload_within_window():
    # [FUNCTION DOC]
    # Version 2, keep 4: asserts a load but no unload request.
    """No unload when version - keep < 0 (nothing stale yet)."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(2, keep=4))
    endpoints = [r.endpoint for r in reqs.requests]
    assert "/unload_lora_adapter" not in endpoints
    assert "/load_lora_adapter" in endpoints


def test_disk_update_keep_zero_disables_unload():
    # [FUNCTION DOC]
    # keep 0: asserts only the load request is produced.
    """lora_keep_versions=0 keeps the old behaviour (load only)."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(100, keep=0))
    assert [r.endpoint for r in reqs.requests] == ["/load_lora_adapter"]


def test_disk_update_unload_is_best_effort():
    # [FUNCTION DOC]
    # Asserts the unload request has best_effort True.
    """The unload request must be best-effort (stale adapter may be gone)."""
    reqs = SGLangBackend().build_disk_weight_update_requests(_lora_meta(10, keep=4))
    unload = next(r for r in reqs.requests if r.endpoint == "/unload_lora_adapter")
    assert unload.best_effort is True


def test_disk_update_full_model_unchanged():
    # [FUNCTION DOC]
    # Non-LoRA disk update: asserts only /update_weights_from_disk is requested.
    """Non-LoRA disk update path is untouched."""
    meta = WeightUpdateMeta(type="disk", path="/tmp/wu", use_lora=False)
    reqs = SGLangBackend().build_disk_weight_update_requests(meta)
    assert [r.endpoint for r in reqs.requests] == ["/update_weights_from_disk"]

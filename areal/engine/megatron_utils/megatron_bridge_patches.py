# [MODULE DOC]
# Import-time monkey patches for megatron-bridge bugs not yet in a release. Importing
# this module applies all patches once (idempotent via a class-level sentinel) and logs
# through the "MegatronBridgePatches" logger. Each patch silently does nothing if the
# patched megatron-bridge class cannot be imported.
# SPDX-License-Identifier: Apache-2.0

"""Runtime patches for megatron-bridge bugs not yet in a released version.

Each patch is keyed to an upstream PR. Patches are not version-gated; instead
each one's hot path becomes a no-op once the upstream fix is present (the patch
checks for the missing attribute/behavior before acting), and an idempotency
sentinel prevents double-application. Apply patches at import time via
``_apply_patches_on_import()`` at module bottom.
"""

from __future__ import annotations

import areal.utils.logging as logging

logger = logging.getLogger("MegatronBridgePatches")


def _patch_qwen3vl_pr3143_word_embeddings() -> None:
    # [FUNCTION DOC]
    # HELPER: Workaround for megatron-bridge PR #3143. If Qwen3VLGPTModel is importable
    # and not already patched, replaces its _postprocess with a wrapper that, when the
    # instance's "embedding" is a plain closure lacking word_embeddings (the sequence-
    # parallel MTP shadow embedding), scans the closure cells for an object with
    # word_embeddings and attaches that attribute to the closure, then calls the
    # original _postprocess. Sets the _areal_pr3143_applied sentinel on the class and
    # logs at INFO. Returns None; global class mutation.
    """megatron-bridge PR #3143: expose word_embeddings on MTP shadow embedding.

    Bug (issue #3112 / PR #3143): in ``Qwen3VLGPTModel.forward``, when
    ``mtp_process and sequence_parallel`` are both True, ``self.embedding`` is
    temporarily replaced with a plain closure ``_sp_scatter_embedding``. The
    closure lacks the ``word_embeddings`` attribute that
    ``shared_embedding_or_output_weight()`` accesses during ``_postprocess``
    when ``share_embeddings_and_output_weights=True`` — typical for the
    smaller Qwen3.5 dense models (0.8B/2B/4B).

    Failure mode:
        ``AttributeError: 'function' object has no attribute 'word_embeddings'``

    Affected versions: megatron-bridge 0.4.0 and 0.4.1. Fixed on ``main``
    by commit 20749b09 (PR #3143) but not in any non-alpha release yet.

    Strategy: wrap ``Qwen3VLGPTModel._postprocess`` so it lazily restores
    ``word_embeddings`` on the shadow embedding by inspecting its closure.
    Closure-based recovery is non-invasive — we don't touch ``forward``
    itself (~70 LoC method).
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
            Qwen3VLGPTModel,
        )
    except ImportError:
        return

    if getattr(Qwen3VLGPTModel, "_areal_pr3143_applied", False):
        return

    _orig_postprocess = Qwen3VLGPTModel._postprocess

    def _patched_postprocess(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Replacement _postprocess: lazily restores word_embeddings on the
        # shadow embedding closure (skipping empty cells) before delegating all args to
        # the original method and returning its result.
        emb = self.__dict__.get("embedding")
        # Only intervene when the shadow closure is currently installed and
        # lacks the expected attribute.
        if (
            callable(emb)
            and not hasattr(emb, "word_embeddings")
            and emb.__closure__ is not None
        ):
            for cell in emb.__closure__:
                try:
                    target = cell.cell_contents
                except ValueError:
                    continue
                if hasattr(target, "word_embeddings"):
                    emb.word_embeddings = target.word_embeddings
                    break
        return _orig_postprocess(self, *args, **kwargs)

    Qwen3VLGPTModel._postprocess = _patched_postprocess
    Qwen3VLGPTModel._areal_pr3143_applied = True
    logger.info(
        "Applied megatron-bridge PR #3143 workaround: "
        "Qwen3VLGPTModel shadow embedding word_embeddings restoration."
    )


def _apply_patches_on_import() -> None:
    # [FUNCTION DOC]
    # HELPER: Apply every patch in this module; called once at module import.
    _patch_qwen3vl_pr3143_word_embeddings()


_apply_patches_on_import()

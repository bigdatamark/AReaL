# [PACKAGE DOC]
# FSDP2 utility package for FSDPEngine: wrapping a HF model with fully_shard, loading a
# full (rank-0) state dict into a sharded model, a cosine LR schedule with warmup and
# min LR, and re-exports of the per-layer optimizer wrapper/kernels. Importing raises
# ModuleNotFoundError on torch < 2.6.0 (needs the private _fully_shard module).
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
)
from transformers import PreTrainedModel

from areal.engine.fsdp_utils.optimizer import (
    AdamKernel,
    OptimKernel,
    PerLayerOptimWrapper,
)
from areal.infra.platforms import current_platform
from areal.utils import logging, pkg_version

if pkg_version.is_version_greater_or_equal("torch", "2.6.0"):
    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard
else:
    raise ModuleNotFoundError(
        "Current PyTorch version < 2.6.0 is not supported for FSDPEngine."
    )


__all__ = [
    "apply_fsdp2",
    "fsdp2_load_full_state_dict",
    "get_cosine_schedule_with_warmup",
    "PerLayerOptimWrapper",
    "OptimKernel",
    "AdamKernel",
]


logger = logging.getLogger("FSDPEngine")


# Adapted from verl
@contextmanager
def maybe_patch_fsdp_module(model):
    # [FUNCTION DOC]
    # HELPER: Context manager. If the model is an ABC instance (e.g. sequence-
    # classification models), temporarily replaces torch's private FSDPModule class with
    # a subclass that also inherits ABC so fully_shard's dynamic class creation does not
    # hit a metaclass conflict; always restores the original on exit. Side effect:
    # mutates a torch module global for the duration.
    orig_fsdp_module = fully_shard_module.FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        # [CLASS DOC]
        # HELPER: Empty class combining ABC with the original FSDPModule to make
        # metaclasses compatible.
        pass

    try:
        if isinstance(model, ABC):
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module


# Adapted from verl
def apply_fsdp2(model, fsdp_kwargs, wrap_policy):
    # [FUNCTION DOC]
    # Shards a HF model with FSDP2 in place. Determines layer class names to wrap from
    # wrap_policy.transformer_layer_cls_to_wrap (str or iterable) or, if empty/None,
    # from the model's _no_split_modules. Calls fully_shard with fsdp_kwargs on each
    # module whose class name matches, and on every nn.Embedding when word embeddings
    # are not tied, then shards the root model (inside maybe_patch_fsdp_module). Returns
    # None. Raises AssertionError if no wrap classes are found.
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, (
        "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
    )

    def _normalize_wrap_class_names(value):
        # [FUNCTION DOC]
        # HELPER: Normalizes None to an empty list, a string to a one-element list, and
        # any other iterable to a list.
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value if isinstance(value, list) else list(value)

    default_transformer_cls_names_to_wrap = _normalize_wrap_class_names(
        getattr(model, "_no_split_modules", list())
    )
    fsdp_transformer_layer_cls_to_wrap = (
        wrap_policy.transformer_layer_cls_to_wrap if wrap_policy is not None else list()
    )
    fsdp_transformer_layer_cls_to_wrap = _normalize_wrap_class_names(
        fsdp_transformer_layer_cls_to_wrap
    )
    if not fsdp_transformer_layer_cls_to_wrap:
        fsdp_transformer_layer_cls_to_wrap = default_transformer_cls_names_to_wrap

    assert (
        len(fsdp_transformer_layer_cls_to_wrap) > 0
        and fsdp_transformer_layer_cls_to_wrap[0] is not None
    )

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    # NOTE: FSDP2 is not compatible with AutoModelForSequenceClassification, so we needs the patch
    # see: https://github.com/volcengine/verl/pull/3072
    with maybe_patch_fsdp_module(model):
        fully_shard(
            model, **fsdp_kwargs
        )  # fsdp2 will not reshard_after_forward for root module


# Adapted from verl
def fsdp2_load_full_state_dict(
    model: PreTrainedModel,
    full_state: dict,
    cpu_offload=None,
    tie_word_embeddings=False,
):
    # [FUNCTION DOC]
    # Loads a full state dict (only needed on rank 0) into an FSDP2-sharded model via
    # set_model_state_dict with broadcast_from_rank0. Moves the model to the current
    # device first (to_empty if any parameter/buffer is on the meta device, else a non-
    # blocking move). strict is disabled when tie_word_embeddings, after which
    # tie_weights is called. Then broadcasts every buffer from rank 0 (buffers like
    # rotary inv_freq are not in the state dict). If cpu_offload is not None, moves the
    # model to CPU but keeps buffers on device. Returns None; mutates model in place;
    # collectives on the default group.
    # KNOWN ISSUE: the model reassigned from to_empty/to is local only; callers must
    # rely on in-place semantics (true for nn.Module.to/to_empty).
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        model (`torch.nn.Module`): The model to load the state dict into
        full_state (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    device = current_platform.current_device()
    # Handle meta device models (from memory_efficient_load where non-rank-0
    # processes use meta tensors). to_empty() materializes meta tensors without
    # copying data; the actual weights come from set_model_state_dict broadcast.
    has_meta = any(t.is_meta for t in model.parameters()) or any(
        t.is_meta for t in model.buffers()
    )
    if has_meta:
        model = model.to_empty(device=device)
    else:
        model = model.to(device=device, non_blocking=True)
    cpu_offload = cpu_offload is not None
    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=cpu_offload,
        broadcast_from_rank0=True,
        strict=not tie_word_embeddings,
    )
    set_model_state_dict(model, full_state, options=options)

    if tie_word_embeddings:
        model.tie_weights()

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(device)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    # [FUNCTION DOC]
    # Builds a LambdaLR whose multiplier rises linearly from min_lr_ratio to 1 over
    # num_warmup_steps, then follows a cosine from 1 down to min_lr_ratio over the
    # remaining steps (num_cycles=0.5 is a half cosine), clamped below at min_lr_ratio.
    # last_epoch resumes the schedule. Returns the scheduler. Raises AssertionError if
    # min_lr_ratio is outside [0, 1].
    # KNOWN ISSUE: docstring says warmup starts at 0, but it starts at min_lr_ratio.
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        # [FUNCTION DOC]
        # HELPER: Maps a step index to the LR multiplier described above (guarding
        # divide-by-zero with max(1, ...)).
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (
                float(current_step) / float(max(1, num_warmup_steps))
            )
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

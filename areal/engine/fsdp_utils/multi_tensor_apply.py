# [MODULE DOC]
# Pure-PyTorch fallbacks (adapted from Megatron-LM) for Apex amp_C multi-tensor kernels,
# used for gradient norm computation and scaling when Apex is unavailable.
# SPDX-License-Identifier: Apache-2.0

# Adapted from Megatron-LM

import torch

from areal.infra.platforms import current_platform


def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
    # [FUNCTION DOC]
    # HELPER: Calls op with a fixed chunk size (65536), the noop flag buffer,
    # tensor_lists and extra args, returning op's result. Mirrors apex
    # multi_tensor_applier.
    return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)


# computes l2 norm for a list of contiguous tensors
# works as a drop-in replacement for amp_C.multi_tensor_l2norm
def local_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args):
    # [FUNCTION DOC]
    # Drop-in for amp_C.multi_tensor_l2norm: computes the L2 norm of every tensor in
    # every list, then the L2 norm of those norms. Returns (1-element float tensor on
    # the platform device, None). chunk_size, noop_flag and per_tensor are ignored.
    # KNOWN ISSUE: building a tensor from a list of tensors and float() force GPU-CPU
    # syncs; per_tensor norms are never returned even if requested.
    l2 = [
        [(torch.norm(tensor)) for tensor in tensor_list] for tensor_list in tensor_lists
    ]
    l2_reduced = torch.norm(torch.tensor(l2))
    l2_cuda = torch.tensor(
        [float(l2_reduced)], dtype=torch.float, device=current_platform.device_type
    )
    return l2_cuda, None


# works as a drop-in replacement for amp_C.multi_tensor_scale
def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
    # [FUNCTION DOC]
    # Drop-in for amp_C.multi_tensor_scale: for each (src, dst) pair from
    # tensor_lists[0] and [1], writes src * scale into dst in place. Returns None;
    # chunk_size and noop_flag are ignored.
    for src, dst in zip(tensor_lists[0], tensor_lists[1]):
        dst.copy_(src * scale)

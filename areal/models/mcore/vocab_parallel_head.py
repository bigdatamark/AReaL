# [MODULE DOC]
# AReaL replacement for the Megatron vocab-parallel LM head (ColumnParallelLinear output
# layer). Provides (1) TP linear autograd functions that produce fp32 logits directly
# from bf16/fp16 GEMMs (avoiding a half-precision logits tensor) and, in backward, pack
# the fp32 logit gradient into half precision in place to save memory; (2) an optional
# fp32-operand path; (3) a chunked LM head that computes per-token logprobs, entropy and
# vocab statistics without materialising full-sequence logits, recomputing logits per
# chunk in backward; and (4) an in-place class swap promoting an existing output layer
# to the AReaL head while keeping its parameters and hooks. Supports TP vocab sharding
# and sequence parallelism (inputs [S/TP, B, hidden] gathered along dim 0).
# SPDX-License-Identifier: Apache-2.0

import math
import weakref
from typing import NamedTuple

import torch
import torch.distributed as dist
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import layers as mcore_layers


def _matches_storage(
    tensor: torch.Tensor,
    tensor_ref: weakref.ReferenceType[torch.Tensor],
) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if tensor shares the same underlying storage pointer as the (still
    # alive) tensor referenced by the weakref; False if the referent was collected.
    expected_tensor = tensor_ref()
    return (
        expected_tensor is not None
        and tensor.untyped_storage().data_ptr()
        == expected_tensor.untyped_storage().data_ptr()
    )


def _pack_fp32_to_half_inplace(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    seed_elements: int,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Converts a contiguous CUDA fp32 tensor at storage offset 0 to bf16/fp16
    # values written into the first half of its own storage, returning a half-precision
    # view with the original shape (sharing the version counter). Raises ValueError for
    # other inputs or non-positive seed_elements. Copies the first seed_elements via a
    # small scratch tensor, then repeatedly doubles the processed prefix (copying [s,
    # min(2s, n))) so source and destination byte ranges never overlap. Destroys the
    # fp32 contents.
    """Pack FP32 values into the first half of their storage as BF16/FP16."""
    if (
        not tensor.is_cuda
        or tensor.dtype is not torch.float32
        or target_dtype not in (torch.bfloat16, torch.float16)
        or not tensor.is_contiguous()
        or tensor.storage_offset() != 0
    ):
        raise ValueError("in-place packing requires contiguous CUDA FP32 at offset 0")
    if seed_elements <= 0:
        raise ValueError(f"seed_elements must be positive, got {seed_elements}")

    numel = tensor.numel()
    # dtype views share the source version counter. This matters because packing
    # destructively changes the FP32 tensor's storage during backward.
    packed = tensor.view(-1).view(target_dtype)[:numel]
    if numel == 0:
        return packed.view(tensor.shape)

    source = tensor.view(-1)
    prefix_end = min(seed_elements, numel)
    scratch = source[:prefix_end].to(dtype=target_dtype)
    packed[:prefix_end].copy_(scratch)
    del scratch

    # FP32 source [s, e) occupies bytes [4s, 4e), while its packed half output
    # occupies [2s, 2e). Choosing e <= 2s makes the ranges disjoint. Doubling
    # each processed prefix therefore packs the rest in logarithmically many
    # cast-copy launches without another full-sized tensor.
    start = prefix_end
    while start < numel:
        end = min(2 * start, numel)
        packed[start:end].copy_(source[start:end])
        start = end

    return packed.view(tensor.shape)


def _prepare_linear_forward(
    ctx,
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None,
    wgrad_deferral_limit: int | None,
    tp_group: torch.distributed.ProcessGroup | None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Shared forward preamble mirroring Megatron's
    # LinearWithGradAccumulationAndAsyncCommunication: saves input and weight for
    # backward and stores flags (main_grad when gradient_accumulation_fusion and
    # available, bias use, allreduce_dgrad, sequence_parallel, wgrad deferral, grad
    # output buffer, tp_group) on ctx. Returns the input unchanged, or with sequence
    # parallelism all-gathers it along dim 0 into Megatron's global memory buffer and
    # returns that buffer.
    if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
        main_grad = weight.main_grad
    else:
        main_grad = None

    ctx.save_for_backward(input_, weight)
    ctx.main_grad = main_grad
    ctx.use_bias = bias is not None
    ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
    ctx.allreduce_dgrad = allreduce_dgrad
    ctx.sequence_parallel = sequence_parallel
    ctx.wgrad_deferral_limit = wgrad_deferral_limit
    ctx.grad_output_buffer = grad_output_buffer
    ctx.tp_group = tp_group

    if not sequence_parallel:
        return input_

    dim_size = list(input_.size())
    dim_size[0] *= tp_group.size()
    all_gather_buffer = mcore_layers.get_global_memory_buffer().get_tensor(
        dim_size, input_.dtype, "mpu"
    )
    mcore_layers.dist_all_gather_func(all_gather_buffer, input_, group=tp_group)
    return all_gather_buffer


def _lm_head_compute_weight(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    fp32_operands: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # [FUNCTION DOC]
    # HELPER: Returns (weight, bias) unchanged, or float32 copies when fp32_operands.
    if not fp32_operands:
        return weight, bias
    return weight.float(), bias.float() if bias is not None else None


class _LinearWithNativeOutput(
    mcore_layers.LinearWithGradAccumulationAndAsyncCommunication
):
    # [CLASS DOC]
    # Megatron TP linear autograd function whose forward keeps the native input/weight
    # dtype output (plain matmul plus bias) and reuses Megatron's backward (grad
    # accumulation fusion, dgrad all-reduce or reduce-scatter).
    """AReaL TP linear with the native input/weight output dtype."""

    @staticmethod
    def _forward_with_fp32_operands(
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Non-custom-autograd path computing the projection in fp32: gathers the
        # input from the sequence-parallel region (SP) or wraps it with
        # copy-to-TP-region (allreduce_dgrad) using the default TP group if none given,
        # then matmuls float input by float weight transpose and adds float bias.
        # Returns fp32 logits; gradients flow via standard autograd through the casts.
        """Project with FP32 operands while retaining PyTorch autograd semantics."""
        if sequence_parallel:
            tp_group = mcore_layers.get_tensor_model_parallel_group_if_none(tp_group)
            total_input = tensor_parallel.gather_from_sequence_parallel_region(
                input_,
                tensor_parallel_output_grad=True,
                group=tp_group,
            )
        elif allreduce_dgrad:
            tp_group = mcore_layers.get_tensor_model_parallel_group_if_none(tp_group)
            total_input = tensor_parallel.copy_to_tensor_model_parallel_region(
                input_, group=tp_group
            )
        else:
            total_input = input_

        compute_weight, compute_bias = _lm_head_compute_weight(
            weight, bias, fp32_operands=True
        )
        output = torch.matmul(total_input.float(), compute_weight.t())
        if compute_bias is not None:
            output = output + compute_bias
        return output

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: list[torch.Tensor] | None,
        wgrad_deferral_limit: int | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Runs _prepare_linear_forward and returns input @ weight^T (+ bias) in native
        # dtype.
        total_input = _prepare_linear_forward(
            ctx,
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output


class _LinearWithFrozenNativeOutput(mcore_layers.LinearWithFrozenWeight):
    # [CLASS DOC]
    # Frozen-weight TP linear (no weight grad) with native-dtype output; backward is
    # Megatron's LinearWithFrozenWeight backward (dgrad with optional all-reduce).
    """AReaL TP linear that skips wgrad for a frozen weight."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        allreduce_dgrad: bool,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Saves weight, allreduce_dgrad and tp_group on ctx and returns input @ weight^T
        # (+ bias). Input is expected to be already sequence-gathered.
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        output = torch.matmul(input_, weight.t())
        if bias is not None:
            output = output + bias
        return output


class _LinearWithFp32Output(
    mcore_layers.LinearWithGradAccumulationAndAsyncCommunication
):
    # [CLASS DOC]
    # Trainable TP linear whose forward writes the half-precision GEMM result directly
    # into an fp32 output tensor (torch.mm with out_dtype float32).
    """Megatron TP linear that writes BF16/FP16 GEMM results directly as FP32."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: list[torch.Tensor] | None,
        wgrad_deferral_limit: int | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Runs _prepare_linear_forward, then computes the GEMM into a new fp32 tensor of
        # shape [..., out_features], records the output shape, a weakref to the output
        # and its numel, adds bias in place and returns the fp32 output.
        total_input = _prepare_linear_forward(
            ctx,
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )

        input_2d = total_input.reshape(-1, total_input.size(-1))
        ctx.output_shape = (*total_input.shape[:-1], weight.size(0))
        output = torch.empty(
            ctx.output_shape,
            dtype=torch.float32,
            device=input_.device,
        )
        torch.mm(
            input_2d,
            weight.t(),
            out=output.view(-1, weight.size(0)),
            out_dtype=torch.float32,
        )
        ctx.output_tensor_ref = weakref.ref(output)
        ctx.output_numel = output.numel()
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # [FUNCTION DOC]
        # Converts the fp32 grad_output to the weight dtype: in place (packing into its
        # own storage) when it is the very tensor produced in forward (same storage and
        # size, contiguous CUDA fp32, half weight), otherwise via a regular cast. Then
        # delegates to Megatron's LinearWithGradAccumulationAndAsyncCommunication
        # backward, returning its gradient tuple.
        # The original BF16-output path receives this cast from ToCopyBackward.
        # Preserve that backward contract while removing the forward BF16 tensor.
        _, weight = ctx.saved_tensors
        can_pack_inplace = (
            grad_output.is_cuda
            and grad_output.dtype is torch.float32
            and grad_output.is_contiguous()
            and grad_output.storage_offset() == 0
            and _matches_storage(grad_output, ctx.output_tensor_ref)
            and grad_output.numel() == ctx.output_numel
            and weight.dtype in (torch.bfloat16, torch.float16)
        )
        if can_pack_inplace:
            grad_output = _pack_fp32_to_half_inplace(
                grad_output,
                target_dtype=weight.dtype,
                seed_elements=grad_output.size(-1),
            ).view(ctx.output_shape)
        else:
            grad_output = grad_output.view(ctx.output_shape).to(dtype=weight.dtype)
        return mcore_layers.LinearWithGradAccumulationAndAsyncCommunication.backward(
            ctx, grad_output
        )


class _LinearWithFrozenFp32Output(mcore_layers.LinearWithFrozenWeight):
    # [CLASS DOC]
    # Frozen-weight variant of the direct fp32-output TP linear.
    """AReaL frozen-weight TP linear with direct FP32 GEMM output."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        allreduce_dgrad: bool,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Saves weight and dgrad settings, computes input @ weight^T directly into an
        # fp32 tensor (out_dtype float32), records shape/weakref/numel, adds bias in
        # place and returns it.
        ctx.save_for_backward(weight)
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        ctx.output_shape = (*input_.shape[:-1], weight.size(0))
        output = torch.empty(
            ctx.output_shape,
            dtype=torch.float32,
            device=input_.device,
        )
        torch.mm(
            input_.reshape(-1, input_.size(-1)),
            weight.t(),
            out=output.view(-1, weight.size(0)),
            out_dtype=torch.float32,
        )
        ctx.output_tensor_ref = weakref.ref(output)
        ctx.output_numel = output.numel()
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # [FUNCTION DOC]
        # Casts grad_output to the weight dtype (in place when safe, as in the trainable
        # variant), computes grad_input = grad_output @ weight, all-reduces it over
        # tp_group if allreduce_dgrad, and returns (grad_input, None, None, None, None).
        (weight,) = ctx.saved_tensors
        can_pack_inplace = (
            grad_output.is_cuda
            and grad_output.dtype is torch.float32
            and grad_output.is_contiguous()
            and grad_output.storage_offset() == 0
            and _matches_storage(grad_output, ctx.output_tensor_ref)
            and grad_output.numel() == ctx.output_numel
            and weight.dtype in (torch.bfloat16, torch.float16)
        )
        if can_pack_inplace:
            grad_output = _pack_fp32_to_half_inplace(
                grad_output,
                target_dtype=weight.dtype,
                seed_elements=grad_output.size(-1),
            ).view(ctx.output_shape)
        else:
            grad_output = grad_output.view(ctx.output_shape).to(dtype=weight.dtype)

        grad_input = grad_output.matmul(weight)
        if ctx.allreduce_dgrad:
            torch.distributed.all_reduce(grad_input, group=ctx.tp_group)
        return grad_input, None, None, None, None


def linear_with_fp32_output(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None = None,
    wgrad_deferral_limit: int | None = 0,
    tp_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Applies _LinearWithFp32Output with the Megatron linear argument list and returns
    # fp32 output.
    """Run Megatron's TP linear with an FP32 output."""
    output = _LinearWithFp32Output.apply(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    )
    return output


def _linear_with_frozen_areal_output(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None,
    wgrad_deferral_limit: int | None,
    tp_group: torch.distributed.ProcessGroup | None,
    *,
    fp32_output: bool,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Frozen-weight dispatch. Raises AssertionError if grad_output_buffer or
    # wgrad_deferral_limit is not None (only valid for trainable weights). Resolves the
    # TP group, gathers the input from the sequence-parallel region when SP is on, and
    # applies the direct-fp32 frozen function when fp32_output is requested and
    # input/weight are the same CUDA half dtype; otherwise the native frozen function,
    # casting to float if fp32_output. gradient_accumulation_fusion is ignored.
    del gradient_accumulation_fusion
    if grad_output_buffer is not None:
        raise AssertionError(
            "grad_output_buffer is only supported for trainable LM Head weights"
        )
    if wgrad_deferral_limit is not None:
        raise AssertionError(
            "wgrad_deferral_limit is only supported for trainable LM Head weights"
        )

    tp_group = mcore_layers.get_tensor_model_parallel_group_if_none(tp_group)
    if sequence_parallel:
        input_ = mcore_layers.gather_from_sequence_parallel_region(
            input_,
            tensor_parallel_output_grad=True,
            group=tp_group,
        )

    can_use_direct_fp32 = (
        fp32_output
        and input_.is_cuda
        and input_.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == input_.dtype
    )
    function = (
        _LinearWithFrozenFp32Output
        if can_use_direct_fp32
        else _LinearWithFrozenNativeOutput
    )
    output = function.apply(input_, weight, bias, allreduce_dgrad, tp_group)
    return output.float() if fp32_output and not can_use_direct_fp32 else output


def linear_with_areal_output(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gradient_accumulation_fusion: bool,
    allreduce_dgrad: bool,
    sequence_parallel: bool,
    grad_output_buffer: list[torch.Tensor] | None = None,
    wgrad_deferral_limit: int | None = 0,
    tp_group: torch.distributed.ProcessGroup | None = None,
    *,
    fp32_output: bool,
    fp32_operands: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Replacement for Megatron's _forward_impl. With fp32_operands: raises
    # NotImplementedError if grad_output_buffer is given, else uses the fp32-operand
    # path. For frozen weights uses the frozen dispatch. Otherwise uses the direct-fp32
    # function when fp32_output and input/weight share a CUDA half dtype, else the
    # native function (cast to float if fp32_output). Returns the (local vocab shard)
    # logits.
    """Run the AReaL TP linear with native or direct FP32 output."""
    if fp32_operands:
        if grad_output_buffer is not None:
            raise NotImplementedError(
                "FP32 lm_head operands do not support deferred embedding wgrad"
            )
        return _LinearWithNativeOutput._forward_with_fp32_operands(
            input_,
            weight,
            bias,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
        )
    if not weight.requires_grad:
        return _linear_with_frozen_areal_output(
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
            fp32_output=fp32_output,
        )

    can_use_direct_fp32 = (
        fp32_output
        and input_.is_cuda
        and input_.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == input_.dtype
    )
    if can_use_direct_fp32:
        return linear_with_fp32_output(
            input_,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )

    output = _LinearWithNativeOutput.apply(
        input_,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    )
    return output.float() if fp32_output else output


class ChunkedLMHeadOutput(NamedTuple):
    # [CLASS DOC]
    # NamedTuple of per-token 1D fp32 outputs of the chunked LM head: logprobs (the only
    # differentiable output), entropy, and vocab min/max/mean/L2-norm of the logits.
    """Per-token outputs produced without materializing full-sequence logits."""

    logprobs: torch.Tensor
    entropy: torch.Tensor
    vocab_min_logits: torch.Tensor
    vocab_max_logits: torch.Tensor
    vocab_mean_logits: torch.Tensor
    vocab_norm_logits: torch.Tensor


def _tp_world_size(tp_group: dist.ProcessGroup | None) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the size of tp_group, or 1 if the group is None or
    # torch.distributed is not initialised.
    if tp_group is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(tp_group)


def _all_reduce_if_needed(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType,
    tp_group: dist.ProcessGroup | None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: In-place all-reduce of tensor with op over tp_group when its size is > 1.
    if _tp_world_size(tp_group) > 1:
        dist.all_reduce(tensor, op=op, group=tp_group)


def _gather_sequence_parallel_input(
    input_: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: All-gathers a sequence-parallel input along dim 0 across tp_group into
    # Megatron's global memory buffer (not autograd-tracked) and returns it. Raises
    # ValueError if tp_group is None.
    if tp_group is None:
        raise ValueError("sequence-parallel LM Head requires a TP process group")
    shape = list(input_.shape)
    shape[0] *= dist.get_world_size(tp_group)
    output = mcore_layers.get_global_memory_buffer().get_tensor(
        shape, input_.dtype, "mpu"
    )
    mcore_layers.dist_all_gather_func(output, input_, group=tp_group)
    return output


def _linear_chunk_fp32(
    input_: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    logit_scale: float,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Computes a 2D chunk of logits [rows, vocab_local] in fp32: direct
    # fp32-output GEMM when input and weight share a half dtype, otherwise a regular mm
    # into the fp32 buffer. Adds bias and multiplies by logit_scale in place when not 1.
    # Returns it.
    output = torch.empty(
        (input_.size(0), weight.size(0)),
        dtype=torch.float32,
        device=input_.device,
    )
    if input_.dtype in (torch.bfloat16, torch.float16) and weight.dtype == input_.dtype:
        torch.mm(input_, weight.t(), out=output, out_dtype=torch.float32)
    else:
        torch.mm(input_, weight.t(), out=output)
    if bias is not None:
        output.add_(bias)
    if logit_scale != 1.0:
        output.mul_(logit_scale)
    return output


class _ChunkedVocabParallelLMHead(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd Function computing per-token log-softmax statistics of the vocab-parallel
    # LM head chunk by chunk, saving only inputs, local targets, per-row max and sum-exp
    # so logits are recomputed in backward.
    """Recompute chunk logits in backward to bound vocab activation memory."""

    @staticmethod
    @mcore_layers.custom_fwd
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        labels: torch.Tensor,
        temperature: float,
        chunk_size: int,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        tp_group: dist.ProcessGroup | None,
        logit_scale: float,
        fp32_operands: bool,
    ) -> tuple[torch.Tensor, ...]:
        # [FUNCTION DOC]
        # Validates chunk_size > 0, finite positive temperature and logit_scale, and
        # matching input/weight dtypes (ValueError). Gathers SP input, flattens to
        # [tokens, hidden], checks token count equals labels count. Converts labels to
        # local vocab indices for this TP rank (-1 if outside the shard). For each chunk
        # of chunk_size tokens: computes fp32 logits (scaled by logit_scale; fp32
        # operands optional), records local min/max/mean/L2 norm, all-reduces the row
        # max (MAX) across TP, gathers the target logit (0 if not in shard) shifted by
        # the max and divided by temperature, computes per-row partial sums of
        # exp((logit - max)/T) and their logit-weighted sums via a fused kernel,
        # all-reduces (SUM) the [target, sum_exp, weighted_sum] workspace across TP, and
        # derives logprob = target - log(sum_exp) and entropy = log(sum_exp) -
        # weighted/sum_exp. Saves tensors and flags for backward (main_grad use when
        # grad accumulation fusion applies and the weight is not FSDP-managed). Marks
        # all outputs except logprobs non-differentiable and returns the six 1D tensors.
        # Collective calls when TP > 1.
        # KNOWN ISSUE: vocab min/max/mean/norm are computed on the local vocab shard
        # only and are not reduced across TP ranks.
        from areal.utils.functional.vocab_parallel_kernels import (
            fused_exp_sum_inplace,
            vocab_tile_count,
        )

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not math.isfinite(logit_scale) or logit_scale <= 0:
            raise ValueError(f"logit_scale must be positive, got {logit_scale}")
        if input_.dtype != weight.dtype:
            raise ValueError(
                "chunked LM Head requires matching hidden and weight dtypes, got "
                f"{input_.dtype} and {weight.dtype}"
            )

        total_input = (
            _gather_sequence_parallel_input(input_, tp_group)
            if sequence_parallel
            else input_
        )
        input_2d = total_input.reshape(-1, total_input.size(-1))
        compute_weight, compute_bias = _lm_head_compute_weight(
            weight,
            bias,
            fp32_operands=fp32_operands,
        )
        labels_1d = labels.reshape(-1)
        if input_2d.size(0) != labels_1d.numel():
            raise ValueError(
                "LM Head hidden leading dimensions must match labels: "
                f"got {tuple(total_input.shape)} and {tuple(labels.shape)}"
            )

        partition_vocab_size = weight.size(0)
        tp_rank = dist.get_rank(tp_group) if _tp_world_size(tp_group) > 1 else 0
        local_targets = labels_1d - tp_rank * partition_vocab_size
        local_targets.masked_fill_(
            (local_targets < 0) | (local_targets >= partition_vocab_size), -1
        )

        num_tokens = input_2d.size(0)
        output_tensors = [
            torch.empty(num_tokens, dtype=torch.float32, device=input_.device)
            for _ in range(6)
        ]
        logprobs, entropy, vocab_min, vocab_max, vocab_mean, vocab_norm = output_tensors
        row_maxes = torch.empty_like(logprobs)
        sum_exps = torch.empty_like(logprobs)
        workspace_rows = min(chunk_size, num_tokens)
        partial_reduction = torch.empty(
            (2, workspace_rows, vocab_tile_count(partition_vocab_size)),
            dtype=torch.float32,
            device=input_.device,
        )
        reduced_values = torch.empty(
            (3, workspace_rows), dtype=torch.float32, device=input_.device
        )
        inv_temperature = 1.0 / temperature

        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            rows = end - start
            input_chunk = input_2d[start:end]
            if fp32_operands:
                input_chunk = input_chunk.float()
            logits = _linear_chunk_fp32(
                input_chunk,
                compute_weight,
                compute_bias,
                logit_scale,
            )
            chunk_targets = local_targets[start:end]

            vocab_min[start:end] = logits.min(dim=-1).values
            vocab_max[start:end] = logits.max(dim=-1).values
            vocab_mean[start:end] = logits.mean(dim=-1)
            vocab_norm[start:end] = torch.linalg.vector_norm(logits, dim=-1)

            logits_max = logits.max(dim=-1).values
            _all_reduce_if_needed(logits_max, dist.ReduceOp.MAX, tp_group)
            row_maxes[start:end] = logits_max

            row_indices = torch.arange(rows, device=input_.device)
            predicted_logits = logits[
                row_indices, chunk_targets.clamp_min(0)
            ].masked_fill_(chunk_targets < 0, 0.0)
            predicted_logits.sub_(logits_max).mul_(inv_temperature)
            predicted_logits.masked_fill_(chunk_targets < 0, 0.0)

            partial_sums = partial_reduction[0, :rows]
            partial_weighted_sums = partial_reduction[1, :rows]
            fused_exp_sum_inplace(
                logits,
                logits_max,
                partial_sums,
                partial_weighted_sums,
                inv_temperature,
            )
            reduced = reduced_values[:, :rows]
            reduced[0].copy_(predicted_logits)
            reduced[1].copy_(partial_sums.sum(dim=-1))
            reduced[2].copy_(partial_weighted_sums.sum(dim=-1))
            if rows < workspace_rows:
                reduced_values[:, rows:].zero_()
            _all_reduce_if_needed(reduced_values, dist.ReduceOp.SUM, tp_group)

            sum_exp = reduced[1]
            sum_exps[start:end] = sum_exp
            log_sum_exp = sum_exp.log()
            logprobs[start:end] = reduced[0] - log_sum_exp
            entropy[start:end] = log_sum_exp - reduced[2] / sum_exp

        saved_bias = (
            bias
            if bias is not None
            else torch.empty(0, dtype=weight.dtype, device=weight.device)
        )
        saved_compute_bias = (
            compute_bias
            if compute_bias is not None
            else torch.empty(0, dtype=compute_weight.dtype, device=weight.device)
        )
        ctx.save_for_backward(
            input_,
            weight,
            saved_bias,
            compute_weight,
            saved_compute_bias,
            local_targets,
            row_maxes,
            sum_exps,
        )
        ctx.has_bias = bias is not None
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.use_main_grad = (
            gradient_accumulation_fusion
            and weight.requires_grad
            and hasattr(weight, "main_grad")
            and not hasattr(weight, "__fsdp_param__")
        )
        ctx.main_grad = weight.main_grad if ctx.use_main_grad else None
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group
        ctx.logit_scale = logit_scale
        ctx.fp32_operands = fp32_operands
        ctx.input_shape = input_.shape
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(
            entropy, vocab_min, vocab_max, vocab_mean, vocab_norm
        )
        return tuple(output_tensors)

    @staticmethod
    @mcore_layers.custom_bwd
    def backward(
        ctx,
        grad_logprobs: torch.Tensor | None,
        grad_entropy: torch.Tensor | None,
        grad_vocab_min: torch.Tensor | None,
        grad_vocab_max: torch.Tensor | None,
        grad_vocab_mean: torch.Tensor | None,
        grad_vocab_norm: torch.Tensor | None,
    ) -> tuple:
        # [FUNCTION DOC]
        # Only grad_logprobs is used. Re-gathers SP input and, per chunk, recomputes
        # fp32 logits and converts them in place into dlogits (softmax minus one-hot
        # target, scaled by grad_logprobs, 1/T and logit_scale) with a fused kernel.
        # Computes grad_input chunk = dlogits @ weight (fp32 path, or after packing
        # dlogits to half in place). Weight grad: fp32 operands accumulate into fp32
        # main_grad (RuntimeError if not fp32) or a local fp32 buffer; otherwise uses
        # Megatron's fused wgrad GEMM into main_grad (fp32 or half; RuntimeError for
        # other dtypes) or addmm into a local grad. Accumulates bias grad. After the
        # loop, reduce-scatters grad_input (SP) or all-reduces it (allreduce_dgrad).
        # With main_grad returns a dummy wgrad and sets grad_added_to_main_grad when the
        # attribute exists, else None; casts local grads back to parameter dtypes.
        # Returns grads for input, weight, bias and None for the remaining nine inputs.
        from areal.utils.functional.vocab_parallel_kernels import (
            fused_logits_backward_inplace,
        )

        del (
            grad_entropy,
            grad_vocab_min,
            grad_vocab_max,
            grad_vocab_mean,
            grad_vocab_norm,
        )
        (
            input_,
            weight,
            saved_bias,
            compute_weight,
            saved_compute_bias,
            local_targets,
            row_maxes,
            sum_exps,
        ) = ctx.saved_tensors
        bias = saved_bias if ctx.has_bias else None
        compute_bias = saved_compute_bias if ctx.has_bias else None
        total_input = (
            _gather_sequence_parallel_input(input_, ctx.tp_group)
            if ctx.sequence_parallel
            else input_
        )
        input_2d = total_input.reshape(-1, total_input.size(-1))
        grad_input_2d = torch.empty_like(input_2d)
        grad_logprobs_1d = (
            grad_logprobs.reshape(-1).contiguous()
            if grad_logprobs is not None
            else None
        )

        needs_weight_grad = ctx.needs_input_grad[1]
        grad_weight = (
            torch.zeros_like(
                weight,
                dtype=torch.float32 if ctx.fp32_operands else weight.dtype,
            )
            if needs_weight_grad and not ctx.use_main_grad
            else None
        )
        grad_bias = (
            torch.zeros_like(
                bias,
                dtype=torch.float32 if ctx.fp32_operands else bias.dtype,
            )
            if bias is not None and ctx.needs_input_grad[2]
            else None
        )
        if ctx.use_main_grad:
            weight.main_grad = ctx.main_grad

        num_tokens = input_2d.size(0)
        for start in range(0, num_tokens, ctx.chunk_size):
            end = min(start + ctx.chunk_size, num_tokens)
            input_chunk = input_2d[start:end]
            compute_input_chunk = (
                input_chunk.float() if ctx.fp32_operands else input_chunk
            )
            logits = _linear_chunk_fp32(
                compute_input_chunk,
                compute_weight,
                compute_bias,
                ctx.logit_scale,
            )
            fused_logits_backward_inplace(
                logits,
                row_maxes[start:end],
                sum_exps[start:end],
                local_targets[start:end],
                (grad_logprobs_1d[start:end] if grad_logprobs_1d is not None else None),
                1.0 / ctx.temperature,
                ctx.logit_scale,
            )
            if ctx.fp32_operands:
                grad_input_2d[start:end].copy_(torch.mm(logits, compute_weight))
                dlogits = logits
            else:
                dlogits = (
                    _pack_fp32_to_half_inplace(
                        logits,
                        target_dtype=weight.dtype,
                        seed_elements=weight.size(0),
                    )
                    if weight.dtype in (torch.bfloat16, torch.float16)
                    else logits
                )
                torch.mm(dlogits, weight, out=grad_input_2d[start:end])

            if needs_weight_grad:
                if ctx.fp32_operands:
                    if ctx.use_main_grad:
                        if ctx.main_grad.dtype != torch.float32:
                            raise RuntimeError(
                                "FP32 lm_head operands require an FP32 main_grad "
                                f"buffer, got {ctx.main_grad.dtype}"
                            )
                        torch.addmm(
                            weight.main_grad,
                            dlogits.t(),
                            compute_input_chunk,
                            out=weight.main_grad,
                        )
                    else:
                        torch.addmm(
                            grad_weight,
                            dlogits.t(),
                            compute_input_chunk,
                            out=grad_weight,
                        )
                    if grad_bias is not None:
                        grad_bias.add_(dlogits.sum(dim=0))
                    continue
                prepared_dlogits, prepared_input = (
                    mcore_layers.prepare_input_tensors_for_wgrad_compute(
                        dlogits, input_chunk
                    )
                )
                if ctx.use_main_grad:
                    weight.main_grad = ctx.main_grad
                    if weight.main_grad.dtype == torch.float32:
                        mcore_layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                            prepared_input, prepared_dlogits, weight.main_grad
                        )
                    elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                        mcore_layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                            prepared_input, prepared_dlogits, weight.main_grad
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported main_grad dtype {weight.main_grad.dtype}"
                        )
                else:
                    torch.addmm(
                        grad_weight,
                        prepared_dlogits.t(),
                        prepared_input,
                        out=grad_weight,
                    )
            if grad_bias is not None:
                grad_bias.add_(dlogits.sum(dim=0))

        if ctx.sequence_parallel:
            grad_input = torch.empty(
                ctx.input_shape,
                dtype=input_.dtype,
                device=input_.device,
                requires_grad=False,
            )
            mcore_layers.dist_reduce_scatter_func(
                grad_input, grad_input_2d.view_as(total_input), group=ctx.tp_group
            )
        else:
            grad_input = grad_input_2d.view_as(input_)
            if ctx.allreduce_dgrad:
                dist.all_reduce(grad_input, group=ctx.tp_group)

        if ctx.use_main_grad:
            if hasattr(weight, "grad_added_to_main_grad"):
                grad_weight = mcore_layers.get_dummy_wgrad(
                    list(weight.main_grad.shape),
                    input_.dtype,
                    zero=getattr(weight, "zero_out_wgrad", False),
                )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        elif grad_weight is not None and grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(dtype=weight.dtype)

        if grad_bias is not None and grad_bias.dtype != bias.dtype:
            grad_bias = grad_bias.to(dtype=bias.dtype)

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def chunked_lm_head_logprobs_entropy(
    output_layer: mcore_layers.ColumnParallelLinear,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    logit_scale: float = 1.0,
) -> ChunkedLMHeadOutput:
    # [FUNCTION DOC]
    # Public entry for chunked logprob/entropy computation. Requires output_layer to be
    # the AReaL LM head (TypeError), with an unmodified forward
    # (chunked_loss_supported), defer_embedding_wgrad_compute disabled and gather_output
    # False (NotImplementedError otherwise). Uses the layer's bias unless skip_bias_add,
    # and its grad-accumulation, dgrad, SP, TP group and fp32_operands settings; weight
    # is passed explicitly (supports tied embeddings). Returns ChunkedLMHeadOutput.
    """Run a Megatron-compatible LM Head and logprob backward by sequence chunk."""
    if not isinstance(output_layer, _AReaLVocabParallelLMHeadMixin):
        raise TypeError("chunked LM Head loss requires the AReaL LM Head")
    if not getattr(output_layer, "chunked_loss_supported", True):
        raise NotImplementedError(
            "chunked LM Head loss does not support this output-layer forward override"
        )
    if output_layer.config.defer_embedding_wgrad_compute:
        raise NotImplementedError(
            "chunked LM Head loss is incompatible with defer_embedding_wgrad_compute"
        )
    if output_layer.gather_output:
        raise NotImplementedError(
            "chunked LM Head loss requires vocab-parallel (ungathered) output"
        )

    bias = output_layer.bias if not output_layer.skip_bias_add else None
    values = _ChunkedVocabParallelLMHead.apply(
        hidden_states,
        weight,
        bias,
        labels,
        temperature,
        chunk_size,
        output_layer.gradient_accumulation_fusion,
        output_layer.allreduce_dgrad,
        output_layer.sequence_parallel,
        output_layer.tp_group,
        logit_scale,
        output_layer.fp32_operands,
    )
    return ChunkedLMHeadOutput(*values)


class _AReaLVocabParallelLMHeadMixin:
    # [CLASS DOC]
    # Mixin placed before ColumnParallelLinear (or a subclass) that routes _forward_impl
    # to linear_with_areal_output with the instance's fp32_output / fp32_operands flags
    # (class defaults False).
    fp32_output: bool = False
    fp32_operands: bool = False

    def _forward_impl(self, input, weight, *args, **kwargs):
        # [FUNCTION DOC]
        # Forwards all arguments to linear_with_areal_output with this layer's fp32
        # flags. Returns its output.
        return linear_with_areal_output(
            input,
            weight,
            *args,
            **kwargs,
            fp32_output=self.fp32_output,
            fp32_operands=self.fp32_operands,
        )


class AReaLVocabParallelLMHead(
    _AReaLVocabParallelLMHeadMixin,
    mcore_layers.ColumnParallelLinear,
):
    # [CLASS DOC]
    # ColumnParallelLinear with the AReaL mixin; used when the original output layer is
    # a plain ColumnParallelLinear.
    """A ColumnParallelLinear LM head with an optional direct FP32 output."""


def replace_output_layer_with_areal_lm_head(
    model: torch.nn.Module,
    *,
    fp32_output: bool,
    fp32_operands: bool = False,
) -> None:
    # [FUNCTION DOC]
    # Promotes model.output_layer in place (no-op if the model has none). Raises
    # TypeError if it is not a ColumnParallelLinear. If not already AReaL, records
    # chunked_loss_supported (true only when the original class keeps
    # ColumnParallelLinear.forward) and swaps __class__ to AReaLVocabParallelLMHead or a
    # dynamically created "AReaL<OriginalName>" subclass with the mixin first,
    # preserving parameters and hooks. Sets fp32_output/fp32_operands. If
    # Megatron-Bridge is installed, registers the new class name as a "column" parallel
    # module type for weight conversion (global registry side effect).
    """Promote an existing MCore LM head without replacing its parameters or hooks."""
    if not hasattr(model, "output_layer"):
        return

    output_layer = model.output_layer
    if not isinstance(output_layer, mcore_layers.ColumnParallelLinear):
        raise TypeError(
            "AReaL LM Head requires Megatron ColumnParallelLinear, got "
            f"{type(output_layer)}"
        )

    # Bridge model builders may add output-layer mixins such as Gemma2 logit
    # soft-capping. Add AReaL's implementation ahead of the existing class so
    # its forward behavior, parameters, and installed hooks all remain intact.
    if not isinstance(output_layer, _AReaLVocabParallelLMHeadMixin):
        original_class = output_layer.__class__
        output_layer.chunked_loss_supported = (
            original_class.forward is mcore_layers.ColumnParallelLinear.forward
        )
        if original_class is mcore_layers.ColumnParallelLinear:
            areal_class = AReaLVocabParallelLMHead
        else:
            areal_class = type(
                f"AReaL{original_class.__name__}",
                (_AReaLVocabParallelLMHeadMixin, original_class),
                {},
            )
        output_layer.__class__ = areal_class
    output_layer.fp32_output = fp32_output
    output_layer.fp32_operands = fp32_operands

    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except ImportError:
        return
    AutoMapping.register_module_type(output_layer.__class__.__name__, "column")

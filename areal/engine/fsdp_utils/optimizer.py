# [MODULE DOC]
# Optimizers and optimizer helpers for FSDPEngine: a dtype parser, AnyPrecisionAdamW
# (low-precision states with optional Kahan-compensated updates), and
# PerLayerOptimWrapper, which speeds up CPU-offloaded optimizer steps by streaming each
# FSDP layer's params/grads/states to the device on async H2D/D2H streams, with a
# pluggable OptimKernel (AdamKernel by default).
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn import Parameter
from torch.optim.adam import adam as _adam_fn

from areal.infra.platforms import current_platform


def to_precision_dtype(dtype_str: str) -> torch.dtype:
    # [FUNCTION DOC]
    # HELPER: Case-insensitively maps 'bfloat16'/'bf16' to torch.bfloat16 and
    # 'float32'/'fp32' to torch.float32. Raises ValueError for anything else.
    """
    Convert string to corresponding torch dtype, only supports bfloat16 and float32.

    Args:
        dtype_str: Data type string, supports "bfloat16" or "float32"

    Returns:
        Corresponding torch dtype

    Raises:
        ValueError: If the input dtype is not supported
    """
    dtype_str = dtype_str.lower()
    if dtype_str in ["bfloat16", "bf16"]:
        return torch.bfloat16
    elif dtype_str in ["float32", "fp32"]:
        return torch.float32
    else:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Only 'bfloat16' and 'float32' are supported."
        )


# https://github.com/meta-llama/llama-cookbook/blob/v0.0.5/src/llama_cookbook/policies/anyprecision_optimizer.py
class AnyPrecisionAdamW(torch.optim.Optimizer):
    # [CLASS DOC]
    # AdamW optimizer (from llama-cookbook) whose momentum, variance and Kahan
    # compensation buffers use configurable dtypes (bf16 or fp32 strings stored in
    # param-group defaults). Per-param state: step (float tensor), exp_avg, exp_avg_sq
    # and, if Kahan summation is on, compensation.
    def __init__(
        self,
        params: Iterator[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        use_kahan_summation: bool = True,
        momentum_dtype: str = "bfloat16",
        variance_dtype: str = "bfloat16",
        compensation_buffer_dtype: str = "bfloat16",
    ):
        # [FUNCTION DOC]
        # Builds the defaults dict (lr, betas, eps, weight_decay, use_kahan_summation
        # and the three dtype strings) and initialises the base Optimizer with params.
        # dtype strings are only validated lazily at step time.
        """
        AnyPrecisionAdamW: a flexible precision AdamW optimizer
        with optional Kahan summation for high precision weight updates.
        Allows direct control over momentum, variance and auxiliary compensation buffer dtypes.
        Optional Kahan summation is used to offset precision reduction for the weight updates.
        This allows full training in BFloat16 (equal or better than FP32 results in many cases)
        due to high precision weight updates.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay (float, optional): weight decay coefficient (default: 1e-2)

            # Any Precision specific
            use_kahan_summation = creates auxiliary buffer to ensure high precision
            model param updates (default: True)
            momentum_dtype = dtype for momentum  (default: bfloat16)
            variance_dtype = dtype for uncentered variance (default: bfloat16)
            compensation_buffer_dtype = dtype for Kahan summation buffer (default: bfloat16)

            # Usage
            This optimizer implements optimizer states, and Kahan summation
            for high precision updates, all in user controlled dtypes.
            Defaults are variance in BF16, Momentum in BF16.
            This can be run in FSDP mixed precision, amp, or full precision,
            depending on what training pipeline you wish to work with.

            Setting to use_kahan_summation = False, and changing momentum and
            variance dtypes to FP32, reverts this to a standard AdamW optimizer.

        """
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_kahan_summation": use_kahan_summation,
            "momentum_dtype": momentum_dtype,
            "variance_dtype": variance_dtype,
            "compensation_buffer_dtype": compensation_buffer_dtype,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        # [FUNCTION DOC]
        # Runs one update under no_grad. Optionally calls closure with grad enabled. For
        # each param with a grad: raises RuntimeError on sparse grads; lazily creates
        # zero states in the configured dtypes; increments step; applies decoupled
        # weight decay (param *= 1 - lr*wd); updates the EMAs; computes bias-corrected
        # step size and denominator sqrt(v)/sqrt(1-beta2^t)+eps; then either accumulates
        # the update into the compensation buffer and adds it to the param while storing
        # the rounding error back (Kahan), or applies the update directly. Mutates
        # params and state in place.
        # KNOWN ISSUE: the closure's loss is discarded and step returns None instead of
        # the loss, contrary to the Optimizer contract.
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            use_kahan_summation = group["use_kahan_summation"]

            momentum_dtype = to_precision_dtype(group["momentum_dtype"])
            variance_dtype = to_precision_dtype(group["variance_dtype"])
            compensation_buffer_dtype = to_precision_dtype(
                group["compensation_buffer_dtype"]
            )
            for p in group["params"]:
                assert isinstance(p, torch.Tensor)  # lint
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients."
                    )

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["exp_avg"] = torch.zeros_like(p, dtype=momentum_dtype)

                    # variance uncentered - EMA of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=variance_dtype)

                    # optional Kahan summation - accumulated error tracker
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p, dtype=compensation_buffer_dtype
                        )

                # Main processing
                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                grad = p.grad

                if weight_decay:  # weight decay, AdamW style
                    p.data.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # update momentum
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2
                )  # update uncentered variance

                bias_correction1 = 1 - beta1**step  # adjust using bias1
                step_size = lr / bias_correction1

                denom_correction = (
                    1 - beta2**step
                ) ** 0.5  # adjust using bias2 and avoids math import
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(
                    eps, alpha=1
                )

                if use_kahan_summation:  # lr update to compensation
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))
                else:  # usual AdamW updates
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)


# ---------------------------------------------------------------------------
# Per-layer optimizer wrapper: streaming optimizer states to device
# ---------------------------------------------------------------------------


def _get_local_tensor(t: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns the local shard of a DTensor, or the tensor itself otherwise.
    """Extract the underlying tensor from a DTensor, or return as-is."""
    return t.to_local() if isinstance(t, DTensor) else t


@dataclass
class ParamTransferState:
    # [CLASS DOC]
    # Dataclass record for one parameter during a per-layer step: the Parameter, whether
    # its local data lived off-device (p_was_cpu), the original local param tensor,
    # device copies of param and grad, and the CPU and device versions of each optimizer
    # state keyed by state name.
    """CPU and device copies of a parameter and its optimizer states for one H2D/D2H cycle."""

    param: Parameter
    p_was_cpu: bool
    cpu_p: torch.Tensor
    device_p: torch.Tensor
    device_g: torch.Tensor
    cpu_states: dict[str, torch.Tensor]
    device_states: dict[str, torch.Tensor]


class OptimKernel:
    # [CLASS DOC]
    # Abstract strategy for PerLayerOptimWrapper describing an optimizer: which state
    # keys to stream, which hyperparameters must match across param groups, how to
    # create and normalise state, and how to run the update on device tensor lists.
    # Methods raise NotImplementedError unless overridden (normalize_state defaults to a
    # no-op).
    """Base class for optimizer-specific computation in PerLayerOptimWrapper.

    Subclass to add support for new optimizers. Each kernel defines:
    - Which state keys to transfer between host and device.
    - Which hyperparameters must be consistent across param groups.
    - How to initialize and normalize states.
    - The actual optimizer update computation.
    """

    @property
    def state_keys(self) -> list[str]:
        # [FUNCTION DOC]
        # Property: list of optimizer-state keys to transfer between host and device.
        # Abstract.
        """Optimizer state keys to transfer between host and device."""
        raise NotImplementedError

    @property
    def hyperparam_keys(self) -> list[str]:
        # [FUNCTION DOC]
        # Property: list of hyperparameter keys that must be equal across param groups.
        # Abstract.
        """Hyperparameter keys that must be identical across param groups."""
        raise NotImplementedError

    def init_param_state(self, local_p: torch.Tensor) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Abstract: returns fresh CPU state tensors for a local parameter shard.
        """Create initial optimizer states on CPU for a parameter."""
        raise NotImplementedError

    def normalize_state(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Hook to fix up checkpoint-loaded state in place (e.g. type conversions).
        # Default does nothing.
        """Normalize states loaded from checkpoints (e.g., type conversions).

        Called once during initialization for pre-existing states.
        Override when checkpoint formats differ from runtime expectations.
        """

    def compute(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: dict[str, list[torch.Tensor]],
        group: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC]
        # Abstract: performs the in-place optimizer update on device lists of params,
        # grads and per-key states using hyperparameters from group.
        """Run optimizer update on device tensors. Modifies tensors in-place."""
        raise NotImplementedError


class AdamKernel(OptimKernel):
    # [CLASS DOC]
    # OptimKernel for torch Adam/AdamW, delegating the math to torch's functional adam
    # implementation.
    """Adam (AdamW) computation kernel."""

    @property
    def state_keys(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns exp_avg, exp_avg_sq and step.
        return ["exp_avg", "exp_avg_sq", "step"]

    @property
    def hyperparam_keys(self) -> list[str]:
        # [FUNCTION DOC]
        # Returns lr, betas, eps, weight_decay, amsgrad, maximize,
        # decoupled_weight_decay.
        return [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "decoupled_weight_decay",
        ]

    def init_param_state(self, local_p: torch.Tensor) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # Returns a float32 scalar step of 0 and CPU zero tensors for exp_avg and
        # exp_avg_sq with the param's shape and dtype.
        return {
            "step": torch.tensor(0.0, dtype=torch.float32),
            "exp_avg": torch.zeros(local_p.shape, dtype=local_p.dtype, device="cpu"),
            "exp_avg_sq": torch.zeros(local_p.shape, dtype=local_p.dtype, device="cpu"),
        }

    def normalize_state(self, state: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Converts a non-tensor 'step' entry (int/float from checkpoints) into a float32
        # scalar tensor, in place.
        # Checkpoints may store step as int/float instead of tensor
        if "step" in state and not isinstance(state["step"], torch.Tensor):
            state["step"] = torch.tensor(float(state["step"]), dtype=torch.float32)

    def compute(
        self,
        params: list[torch.Tensor],
        grads: list[torch.Tensor],
        states: dict[str, list[torch.Tensor]],
        group: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC]
        # Calls torch's functional adam with foreach=True on the given lists, using lr,
        # betas, weight_decay from group and eps/amsgrad/maximize/decoupled_weight_decay
        # with defaults. Mutates params and states in place.
        # KNOWN ISSUE: passes an empty max_exp_avg_sqs list, so amsgrad=True in the
        # group would fail inside torch.
        beta1, beta2 = group["betas"]
        _adam_fn(
            params,
            grads,
            states["exp_avg"],
            states["exp_avg_sq"],
            [],  # max_exp_avg_sqs (empty for non-amsgrad)
            states["step"],
            amsgrad=group.get("amsgrad", False),
            has_complex=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group.get("eps", 1e-8),
            maximize=group.get("maximize", False),
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            decoupled_weight_decay=group.get("decoupled_weight_decay", False),
        )


class PerLayerOptimWrapper:
    # [CLASS DOC]
    # Replacement for optimizer.step when optimizer states are offloaded. Groups
    # trainable params by non-root FSDP2 submodules (plus a residual group), keeps all
    # states on pinned CPU memory, and on step pipelines: prefetch layers on an H2D
    # stream (prefetch_layers ahead), compute on the current stream, copy results back
    # on a D2H stream, then frees device memory. Uses only param_groups[0]
    # hyperparameters, so all groups must agree.
    """Accelerate offloaded optimizer step by streaming states per-layer to device.

    When optimizer states are offloaded to CPU, the default CPU optimizer step is
    very slow. This wrapper streams states one layer at a time to device, runs the
    optimizer there, then streams back, using pipelined H2D/D2H transfers via
    dedicated async streams.

    Note:
        Optimizer states are automatically managed on CPU by this wrapper
        regardless of the ``offload_params`` setting.
        Requires a compatible OptimKernel (default: AdamKernel).

    Works with both offload_params=True and offload_params=False.
    When params/grads are on CPU, they are streamed to device alongside states.

    Config: ``per_layer_optim_step``, ``optim_step_prefetch_layers``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device_id: int | str | torch.device,
        prefetch_layers: int = 1,
        kernel: OptimKernel | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Stores the optimizer and kernel (default AdamKernel), resolves device (int ->
        # 'cuda:N', otherwise torch.device), then builds layer groups, validates
        # hyperparameters, initialises CPU states, pins them and creates streams/events.
        # Raises ValueError on mismatched hyperparameters. Side effects: mutates
        # optimizer.state, allocates pinned memory and device streams.
        # KNOWN ISSUE: an integer device_id is always mapped to a CUDA device, ignoring
        # non-CUDA platforms.
        self.optimizer = optimizer
        self.device = (
            torch.device(f"cuda:{device_id}")
            if isinstance(device_id, int)
            else torch.device(device_id)
        )
        self.prefetch_layers = prefetch_layers
        self.kernel = kernel or AdamKernel()
        self._layer_param_groups = self._build_layer_groups(model)
        self._validate_hyperparams()
        self._init_states()
        self._pin_states()
        self._init_streams_and_events()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _validate_hyperparams(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If there are multiple param groups, raises ValueError when any kernel
        # hyperparameter present in both a group and group 0 differs.
        """Assert all param_groups share identical hyperparameters.

        PerLayerOptimWrapper processes params by layer (not by group),
        so all groups must have the same hyperparams. Fails loudly if not.
        """
        if len(self.optimizer.param_groups) <= 1:
            return
        ref = self.optimizer.param_groups[0]
        for i, group in enumerate(self.optimizer.param_groups[1:], 1):
            for key in self.kernel.hyperparam_keys:
                if key in ref and key in group and group[key] != ref[key]:
                    raise ValueError(
                        f"PerLayerOptimWrapper requires all param_groups to have "
                        f"identical hyperparameters, but group[{i}]['{key}']={group[key]} "
                        f"differs from group[0]['{key}']={ref[key]}."
                    )

    def _build_layer_groups(self, model: nn.Module) -> list[list[Parameter]]:
        # [FUNCTION DOC]
        # HELPER: Walks named modules, collecting every non-root FSDPModule; for each,
        # gathers its requires_grad params not yet assigned (deduplicated by id) as one
        # group; remaining trainable params form a final residual group. Returns the
        # list of groups in module order.
        """Group params by FSDP2-wrapped sub-modules (excluding root).

        After apply_fsdp2, each transformer layer and embedding becomes an
        FSDPModule. We find all non-root FSDPModule instances and group their
        params. Any remaining params (final norm, lm_head) form a residual group.
        """
        fsdp_children = []
        for name, module in model.named_modules():
            if name and isinstance(module, FSDPModule):
                fsdp_children.append(module)

        assigned: set[int] = set()
        groups: list[list[Parameter]] = []
        for module in fsdp_children:
            params = [
                p
                for p in module.parameters()
                if p.requires_grad and id(p) not in assigned
            ]
            if params:
                groups.append(params)
                for p in params:
                    assigned.add(id(p))

        # Residual: final norm, lm_head, etc.
        residual = [
            p for p in model.parameters() if p.requires_grad and id(p) not in assigned
        ]
        if residual:
            groups.append(residual)
        return groups

    def _init_states(self) -> None:
        # [FUNCTION DOC]
        # HELPER: For each trainable param, creates kernel initial state on CPU if
        # empty; otherwise moves existing state tensors (local shards) to CPU and
        # normalises them.
        """Initialize optimizer states on CPU for all trainable params.

        IMPORTANT: States must be created on CPU regardless of current param device.
        When param_offload=True, params may already be on device, so zeros_like(param)
        would create states on device, causing OOM during forward/backward. We
        explicitly create on CPU and stream per-layer during optimizer step.

        Called from __init__ (before backward), so we init states for ALL
        trainable params regardless of whether they have gradients yet.
        """
        for group in self._layer_param_groups:
            for param in group:
                state = self.optimizer.state[param]
                if len(state) == 0:
                    local_p = _get_local_tensor(param.data)
                    state.update(self.kernel.init_param_state(local_p))
                else:
                    for key in self.kernel.state_keys:
                        if key in state:
                            local = _get_local_tensor(state[key])
                            if local.device.type != "cpu":
                                state[key] = local.to("cpu")
                    self.kernel.normalize_state(state)

    def _pin_states(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Replaces each CPU, unpinned state tensor with a pinned copy so non-
        # blocking transfers are truly async.
        """Pin optimizer state tensors in memory for async H2D/D2H transfers.

        Without pinning, .to(non_blocking=True) on CPU tensors is synchronous,
        killing all pipeline overlap. Pinning enables true async DMA transfers.
        """
        for group in self._layer_param_groups:
            for param in group:
                state = self.optimizer.state[param]
                for key in self.kernel.state_keys:
                    if key in state:
                        tensor = _get_local_tensor(state[key])
                        if tensor.device.type == "cpu" and not tensor.is_pinned():
                            state[key] = tensor.pin_memory()

    def refresh_states(self) -> None:
        # [FUNCTION DOC]
        # Public: after an external state load (load_state_dict/DCP), moves state
        # tensors back to CPU, normalises them and re-pins. Mutates optimizer.state.
        """Re-apply state invariants after checkpoint load.

        ``load_state_dict`` / DCP ``set_state_dict`` replace optimizer state
        tensor objects, so pinning established by ``_pin_states`` is lost and
        checkpoint formats may need normalization.  Call this after any
        external mutation of ``self.optimizer.state``.
        """
        for group in self._layer_param_groups:
            for param in group:
                state = self.optimizer.state[param]
                if len(state) == 0:
                    continue
                for key in self.kernel.state_keys:
                    if key in state:
                        local = _get_local_tensor(state[key])
                        if local.device.type != "cpu":
                            state[key] = local.to("cpu")
                self.kernel.normalize_state(state)
        self._pin_states()

    def _init_streams_and_events(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Creates dedicated H2D and D2H streams on the device and one compute-
        # end and one H2D-end event per layer group.
        """Pre-allocate streams and events for pipeline synchronization."""
        num_groups = len(self._layer_param_groups)
        self._h2d_stream = current_platform.Stream(device=self.device)
        self._d2h_stream = current_platform.Stream(device=self.device)
        self._compute_end_events = [current_platform.Event() for _ in range(num_groups)]
        self._h2d_end_events = [current_platform.Event() for _ in range(num_groups)]

    # ------------------------------------------------------------------
    # Per-layer transfer helpers
    # ------------------------------------------------------------------

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Returns t unchanged if already on the target device, else a non-
        # blocking copy to it.
        """Move tensor to device if not already there (non-blocking)."""
        return t if t.device == self.device else t.to(self.device, non_blocking=True)

    def _prefetch_layer(self, layer_idx: int) -> dict[int, ParamTransferState]:
        # [FUNCTION DOC]
        # HELPER: For each param in a layer group that has a grad, builds a
        # ParamTransferState with device copies of the local param, grad and every state
        # tensor (copies only those not already on the device). Returns a dict keyed by
        # id(param). Must be called under the H2D stream.
        """H2D: copy layer's optimizer states (and params/grads if on CPU) to device.

        Auto-detects tensor device: skips H2D for tensors already on device.
        States are already initialized and pinned by _init_states / _pin_states.
        """
        result: dict[int, ParamTransferState] = {}
        for param in self._layer_param_groups[layer_idx]:
            if param.grad is None:
                continue
            state = self.optimizer.state[param]
            local_p = _get_local_tensor(param.data)
            local_g = _get_local_tensor(param.grad.data)

            cpu_states: dict[str, torch.Tensor] = {}
            device_states: dict[str, torch.Tensor] = {}
            for key in self.kernel.state_keys:
                cpu_t = _get_local_tensor(state[key])
                cpu_states[key] = cpu_t
                device_states[key] = self._to_device(cpu_t)

            result[id(param)] = ParamTransferState(
                param=param,
                p_was_cpu=local_p.device != self.device,
                cpu_p=local_p,
                device_p=self._to_device(local_p),
                device_g=self._to_device(local_g),
                cpu_states=cpu_states,
                device_states=device_states,
            )
        return result

    def _compute_for_layer(self, layer_states: dict[int, ParamTransferState]) -> None:
        # [FUNCTION DOC]
        # HELPER: Collects device params, grads and per-key states from the layer's
        # transfer states and runs kernel.compute with param_groups[0]. No-op if the
        # layer has no grads.
        """Run optimizer computation on device tensors for one layer."""
        group = self.optimizer.param_groups[0]
        params: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        collected: dict[str, list[torch.Tensor]] = {
            key: [] for key in self.kernel.state_keys
        }

        for ts in layer_states.values():
            params.append(ts.device_p)
            grads.append(ts.device_g)
            for key in self.kernel.state_keys:
                collected[key].append(ts.device_states[key])

        if not params:
            return
        self.kernel.compute(params, grads, collected, group)

    def _offload_layer(self, layer_states: dict[int, ParamTransferState]) -> None:
        # [FUNCTION DOC]
        # HELPER: Non-blocking copies of updated params (only those originally off-
        # device) and all states back into their pinned CPU tensors.
        """D2H: copy updated param data + optimizer states back to CPU.

        Only copies back tensors that were originally on CPU.
        CPU destination tensors are pinned, enabling async D2H via non_blocking.
        """
        for ts in layer_states.values():
            if ts.p_was_cpu:
                ts.cpu_p.copy_(ts.device_p, non_blocking=True)
            for key in self.kernel.state_keys:
                ts.cpu_states[key].copy_(ts.device_states[key], non_blocking=True)

    def _record_streams_for_layer(
        self,
        layer_states: dict[int, ParamTransferState],
        *streams: torch.cuda.Stream,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls record_stream on every device tensor of the layer for each given
        # stream so the caching allocator does not reuse memory still in use by those
        # streams.
        """Mark device tensors as used by each *stream* for allocator safety.

        Device tensors are allocated on ``_h2d_stream`` but consumed on
        ``compute_stream`` and ``_d2h_stream``.  Without ``record_stream``,
        the caching allocator may reuse the memory as soon as the Python
        reference is dropped (``layer_states[i] = None``), even if another
        stream still has outstanding reads.
        """
        for ts in layer_states.values():
            for stream in streams:
                ts.device_p.record_stream(stream)
                ts.device_g.record_stream(stream)
                for st in ts.device_states.values():
                    st.record_stream(stream)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self) -> None:
        # [FUNCTION DOC]
        # Runs the pipelined optimizer step under no_grad: prefetches the first
        # prefetch_layers+1 groups, then for each group waits for its H2D event,
        # computes, records a compute event, prefetches the next group, offloads the
        # current one on the D2H stream after its compute event, records streams and
        # drops device references. Finally synchronises the D2H stream and empties the
        # device cache. Returns None; mutates params and optimizer state.
        """Per-layer optimizer step with async prefetch pipeline."""
        h2d_stream = self._h2d_stream
        d2h_stream = self._d2h_stream
        compute_stream = current_platform.current_stream(self.device)
        num_groups = len(self._layer_param_groups)
        layer_states: list[dict[int, ParamTransferState] | None] = [None] * num_groups

        compute_end_events = self._compute_end_events
        h2d_end_events = self._h2d_end_events

        # Prefetch initial layers
        for i in range(min(self.prefetch_layers + 1, num_groups)):
            with current_platform.stream(h2d_stream):
                layer_states[i] = self._prefetch_layer(i)
                h2d_stream.record_event(h2d_end_events[i])

        # Process each layer
        for i in range(num_groups):
            compute_stream.wait_event(h2d_end_events[i])

            cur_states = layer_states[i]
            assert cur_states is not None, f"Layer {i} was not prefetched"
            self._compute_for_layer(cur_states)
            compute_stream.record_event(compute_end_events[i])

            # Prefetch next layer (overlaps with D2H below)
            next_idx = i + self.prefetch_layers + 1
            if next_idx < num_groups:
                with current_platform.stream(h2d_stream):
                    layer_states[next_idx] = self._prefetch_layer(next_idx)
                    h2d_stream.record_event(h2d_end_events[next_idx])

            # Offload current layer (waits only for this layer's compute)
            d2h_stream.wait_event(compute_end_events[i])
            with current_platform.stream(d2h_stream):
                cur_states_offload = layer_states[i]
                assert cur_states_offload is not None, f"Layer {i} already freed"
                self._offload_layer(cur_states_offload)
            self._record_streams_for_layer(
                cur_states_offload, compute_stream, d2h_stream
            )
            layer_states[i] = None  # free device memory

        d2h_stream.synchronize()

        # Prevent cross-phase cache pollution: return freed optimizer state
        # blocks to driver so forward/backward can't repurpose them.
        current_platform.empty_cache()

# [MODULE DOC]
# Resource allocation and parallelism description. Defines ParallelStrategy
# (tp/pp/dp/cp/ep/etp sizes) and its FSDP/Megatron variants, ModelAllocation (backend +
# optional name + parallel strategy + scheduling strategy) parsed from strings like
# "fsdp:d4t2" or "megatron:(attn:d1p2t4|ffn:d1p2e4)", the deprecated multi-component
# _AllocationMode for SPMD launchers ("+" = separate GPUs, "|" = colocate, "|" binds
# tighter), and a Lark Earley grammar plus transformer that build these objects with
# validation. Importing the removed AllocationMode name raises an explanatory
# AttributeError. Uses the "AllocMode" logger (currently unused).
# SPDX-License-Identifier: Apache-2.0

import enum
import math
from dataclasses import dataclass, field
from typing import Literal

from lark import Lark, Transformer

from areal.api.cli_args import SchedulingStrategy, SchedulingStrategyType
from areal.utils import logging

logger = logging.getLogger("AllocMode")


class AllocationType(enum.Enum):
    # [CLASS DOC]
    # Legacy enum of allocation kinds: COLOCATE (training and inference share GPUs, also
    # training-only), DECOUPLED_TRAIN (separate GPUs), LLM_SERVER_ONLY (inference only).
    """Backward Compatible: Type of resource allocation strategy."""

    COLOCATE = 0  # Shared resources between training and inference (including SFT/training-only)
    DECOUPLED_TRAIN = 1  # Separate resources for training and inference
    LLM_SERVER_ONLY = 2  # Inference-only allocation


class AllocationValidationError(Exception):
    # [CLASS DOC]
    # Exception for semantic allocation validation failures (duplicate/missing names,
    # incompatible sizes, missing backend).
    """Raised when allocation mode validation fails."""


class InvalidAllocationModeError(Exception):
    # [CLASS DOC]
    # Legacy exception kept for compatibility; raised when hybrid MoE attention and
    # expert world sizes differ.
    """Legacy exception for backward compatibility with existing code."""


@dataclass
class ParallelStrategy:
    # [CLASS DOC]
    # Dataclass with tensor, pipeline, data, context, expert and expert-tensor parallel
    # sizes (all default 1), following Megatron MoE parallel folding: world_size =
    # dp*cp*tp*pp; expert modules re-fold the same devices as edp*ep*etp*pp. Provides
    # short alias properties (tp_size, pp_size, dp_size, cp_size, ep_size, etp_size,
    # edp_size).
    """5D parallel strategy supporting tensor, pipeline, data, context, and expert parallelism.

    This class represents a comprehensive parallelization strategy for distributed ML workloads,
    particularly designed for large language models and mixture-of-experts architectures.

    The five dimensions of parallelism are:
    - Tensor parallelism: Splits individual operations (like matrix multiplications) across devices
    - Pipeline parallelism: Splits model layers across devices in a pipeline fashion
    - Data parallelism: Replicates the model and splits data across devices
    - Context parallelism: Splits sequence length across devices (attention-specific)
    - Expert parallelism: Splits experts in MoE models across devices

    For implementation details, refer to:
    https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe#moe-parallel-folding

    Args:
        tensor_parallel_size: Number of devices for tensor model parallelism (default: 1)
        pipeline_parallel_size: Number of pipeline parallel stages (default: 1)
        data_parallel_size: Number of data parallel replicas for ZeRO optimization (default: 1)
        context_parallel_size: Number of devices for context parallelism in attention modules (default: 1)
        expert_parallel_size: Number of devices for expert parallelism in MoE models (default: 1)
        expert_tensor_parallel_size: Tensor parallelism size specifically for expert modules (default: 1)

    Note:
        - Context parallelism is only effective for attention modules
        - Expert parallelism is only effective for MoE (Mixture of Experts) modules
    """

    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Size of tensor-model parallelism"}
    )
    pipeline_parallel_size: int = field(
        default=1, metadata={"help": "Number of pipeline parallel stages"}
    )
    data_parallel_size: int = field(
        default=1, metadata={"help": "Data parallelism size for ZeRO optimization"}
    )
    context_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Context parallelism size for attention modules. "
            "Note that context parallelism is only effective for attention modules."
        },
    )
    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert parallelism size for MoE models. "
            "Note that expert parallelism is only effective for expert modules."
        },
    )
    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallelism size for expert modules. "
            "By default, it is 1 which disables expert tensor parallelism."
        },
    )

    def __post_init__(self):
        # [FUNCTION DOC]
        # When expert_parallel_size > 1, sets the dynamic attribute
        # expert_model_parallel_size = pp*etp*ep and asserts it divides world_size
        # (AssertionError otherwise).
        """Initialize computed properties and validate configuration."""
        if self.expert_parallel_size > 1:
            # Calculate expert model parallel size for validation
            self.expert_model_parallel_size = (
                self.pipeline_parallel_size
                * self.expert_tensor_parallel_size
                * self.expert_parallel_size
            )

            # Validate that world size is divisible by expert model parallel size
            assert self.world_size % self.expert_model_parallel_size == 0, (
                f"Expert model parallel size {self.expert_model_parallel_size} "
                f"cannot divide world size {self.world_size}."
            )

    @property
    def expert_data_parallel_size(self) -> int:
        # [FUNCTION DOC]
        # Property: world_size // expert_model_parallel_size if that attribute was set
        # (ep > 1), else data_parallel_size.
        """Data parallelism size for expert modules in MoE models."""
        if not hasattr(self, "expert_model_parallel_size"):
            return self.data_parallel_size
        return self.world_size // self.expert_model_parallel_size

    # Abbreviated properties for convenience
    @property
    def tp_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of tensor_parallel_size.
        """Tensor parallelism size (abbreviated)."""
        return self.tensor_parallel_size

    @property
    def pp_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of pipeline_parallel_size.
        """Pipeline parallelism size (abbreviated)."""
        return self.pipeline_parallel_size

    @property
    def dp_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of data_parallel_size.
        """Data parallelism size (abbreviated)."""
        return self.data_parallel_size

    @property
    def cp_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of context_parallel_size.
        """Context parallelism size (abbreviated)."""
        return self.context_parallel_size

    @property
    def ep_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of expert_parallel_size.
        """Expert parallelism size (abbreviated)."""
        return self.expert_parallel_size

    @property
    def etp_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of expert_tensor_parallel_size.
        """Expert tensor parallelism size (abbreviated)."""
        return self.expert_tensor_parallel_size

    @property
    def edp_size(self) -> int:
        # [FUNCTION DOC]
        # Property alias of expert_data_parallel_size.
        """Expert data parallelism size (abbreviated)."""
        return self.expert_data_parallel_size

    @property
    def world_size(self) -> int:
        # [FUNCTION DOC]
        # Property: total devices, dp * cp * tp * pp (expert dims are folded into these,
        # not multiplied).
        """Total number of devices required for this parallelization strategy."""
        return (
            self.data_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    def __str__(self):
        # [FUNCTION DOC]
        # Returns "Parallel(tp=..,pp=..,dp=..[,cp=..][,ep=..[,ep_tp=..]])", including cp
        # only if > 1, ep only if > 1 and ep_tp only alongside ep when != 1.
        # KNOWN ISSUE: The docstring says it shows only non-default dimensions, but
        # tp/pp/dp are always shown.
        """String representation showing all non-default parallelism dimensions."""
        parts = [
            f"tp={self.tensor_parallel_size}",
            f"pp={self.pipeline_parallel_size}",
            f"dp={self.data_parallel_size}",
        ]

        if self.context_parallel_size > 1:
            parts.append(f"cp={self.context_parallel_size}")
        if self.expert_parallel_size > 1:
            parts.append(f"ep={self.expert_parallel_size}")
            if self.expert_tensor_parallel_size != 1:
                parts.append(f"ep_tp={self.expert_tensor_parallel_size}")

        return f"Parallel({','.join(parts)})"

    @staticmethod
    def parallelism_eq(this, other):
        # [FUNCTION DOC]
        # Static method: True if all six parallel sizes of this and other match. Static
        # (not __eq__) to avoid OmegaConf issues.
        """Compare two parallelism configurations for equality.

        Args:
            this: First ParallelStrategy to compare
            other: Second ParallelStrategy to compare

        Returns:
            bool: True if all parallelism dimensions match

        Note:
            Implemented as static method to avoid OmegaConf compatibility issues.
        """
        return (
            (this.tensor_parallel_size == other.tensor_parallel_size)
            and (this.pipeline_parallel_size == other.pipeline_parallel_size)
            and (this.data_parallel_size == other.data_parallel_size)
            and (this.context_parallel_size == other.context_parallel_size)
            and (this.expert_parallel_size == other.expert_parallel_size)
            and (this.expert_tensor_parallel_size == other.expert_tensor_parallel_size)
        )


@dataclass
class FSDPParallelStrategy(ParallelStrategy):
    # [CLASS DOC]
    # ParallelStrategy subclass for FSDP; adds no fields.
    """FSDP parallel strategy."""

    @staticmethod
    def parallelism_eq(this, other):
        # [FUNCTION DOC]
        # Static method delegating to ParallelStrategy.parallelism_eq.
        """Compare FSDP parallelism configurations."""
        return ParallelStrategy.parallelism_eq(this, other)


@dataclass
class MegatronParallelStrategy(ParallelStrategy):
    # [CLASS DOC]
    # ParallelStrategy subclass adding virtual_pipeline_parallel_size (interleaved
    # schedule, default 1) and use_sequence_parallel (default False).
    """Megatron parallel strategy with additional sequence parallelism and virtual pipeline parallelism."""

    virtual_pipeline_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Virtual pipeline parallelism size for megatron modules "
            "for interleaved pipeline schedule. Default value is 1 (disabled)."
        },
    )
    use_sequence_parallel: bool = field(
        default=False,
        metadata={
            "help": "Enable sequence parallelism. Only used with tensor-model parallelism in Megatron",
        },
    )

    def __post_init__(self):
        # [FUNCTION DOC]
        # Runs the base validation, clamps vpp values <= 1 to 1, and raises
        # AllocationValidationError if vpp > 1 while pipeline_parallel_size <= 1.
        super().__post_init__()
        vpp = self.virtual_pipeline_parallel_size
        if vpp <= 1:
            self.virtual_pipeline_parallel_size = 1
        elif self.pipeline_parallel_size <= 1:
            raise AllocationValidationError(
                "Virtual pipeline parallelism requires pipeline_parallel_size > 1."
            )

    @staticmethod
    def parallelism_eq(this, other):
        # [FUNCTION DOC]
        # Static method: base parallelism equality plus equal virtual pipeline sizes
        # (sequence parallel flag ignored).
        """Compare Megatron parallelism configurations (excluding sequence parallelism)."""
        return ParallelStrategy.parallelism_eq(this, other) and (
            this.virtual_pipeline_parallel_size == other.virtual_pipeline_parallel_size
        )


@dataclass
class ModelAllocation:
    # [CLASS DOC]
    # Dataclass for one model component: backend (fsdp, megatron, archon, vllm, sglang),
    # optional name (role), parallel strategy and scheduling_strategy (separation or
    # colocation with a target).
    """Single model allocation with backend, name, parallel strategy, and scheduling.

    Parameters
    ----------
    backend : str
        Backend type ("sglang", "vllm", "fsdp", "megatron", "archon")
    name : str, optional
        Component name for referencing via allocation_mode[name]
    parallel : ParallelStrategy
        Parallelization strategy (tp, pp, dp, cp, ep sizes)
    scheduling_strategy : SchedulingStrategy
        Resource scheduling (separation or colocation)

    Examples
    --------
    >>> ModelAllocation("sglang", "rollout", ParallelStrategy(dp=2), SchedulingStrategy("separation"))
    """

    backend: Literal["fsdp", "megatron", "archon", "vllm", "sglang"]
    name: str | None
    parallel: ParallelStrategy
    scheduling_strategy: SchedulingStrategy

    def __post_init__(self):
        # [FUNCTION DOC]
        # Raises AllocationValidationError for the fsdp backend if pipeline or expert
        # parallel size is > 1.
        if self.backend == "fsdp":
            if (
                self.parallel.pipeline_parallel_size > 1
                or self.parallel.expert_parallel_size > 1
            ):
                raise AllocationValidationError(
                    f"FSDP backend only supports data/tensor/context parallelism. "
                    f"Got strategy: {self.parallel}"
                )

    @classmethod
    def from_str(
        cls,
        spec: str,
        name: str | None = None,
        scheduling_strategy: SchedulingStrategy | None = None,
    ) -> "ModelAllocation":
        # [FUNCTION DOC]
        # Parses a single-component spec string (backend prefix required). Raises
        # ValueError if it contains "+". Parses with _LLMParallelParser; accepts a one-
        # element list or a ModelAllocation result (ValueError otherwise, e.g. a "|"
        # spec producing several). Overrides name if given and sets scheduling_strategy
        # to the given one or a default separation strategy. Returns the
        # ModelAllocation. Parser errors propagate as
        # ValueError/AllocationValidationError.
        """Parse a single backend:parallelism string into a ModelAllocation.

        Parameters
        ----------
        spec : str
            Single component spec like ``"fsdp:d4"``, ``"sglang:d4t2"``,
            or ``"megatron:(attn:d1p12t4|ffn:d1p12e4)"``.
            An explicit backend prefix is always required.
        name : str, optional
            Role name (e.g., ``"actor"``, ``"rollout"``).
        scheduling_strategy : SchedulingStrategy, optional
            Scheduling strategy. Defaults to separation.

        Returns
        -------
        ModelAllocation

        Raises
        ------
        ValueError
            If *spec* contains ``'+'`` (multi-component strings are not allowed).
        """
        if "+" in spec:
            raise ValueError(
                "ModelAllocation.from_str() accepts a single component spec. "
                "Multi-component strings containing '+' are not allowed. "
                "Use separate per-engine 'backend' fields instead "
                "(e.g., actor.backend='fsdp:d4', rollout.backend='sglang:d4')."
            )

        parser = _LLMParallelParser()
        result = parser.parse(spec)

        # Extract the single ModelAllocation from the parse result
        if isinstance(result, list):
            if len(result) != 1:
                raise ValueError(
                    f"Expected a single allocation from spec '{spec}', "
                    f"got {len(result)} allocations."
                )
            alloc = result[0]
        elif isinstance(result, ModelAllocation):
            alloc = result
        else:
            raise ValueError(f"Unexpected parse result type: {type(result)}")

        # Override name if provided
        if name is not None:
            alloc.name = name

        # Override scheduling_strategy if provided, otherwise default to separation
        if scheduling_strategy is not None:
            alloc.scheduling_strategy = scheduling_strategy
        else:
            alloc.scheduling_strategy = SchedulingStrategy(
                type=SchedulingStrategyType.separation, target=None
            )

        return alloc

    @property
    def world_size(self):
        # [FUNCTION DOC]
        # Property: 0 for colocated allocations (they reuse the target's GPUs), else
        # parallel.world_size.
        if self.scheduling_strategy.type == SchedulingStrategyType.colocation.value:
            return 0
        return self.parallel.world_size

    def __str__(self):
        # [FUNCTION DOC]
        # Returns "backend[(name)]:dims" where dims concatenates d,p,t,c,e sizes that
        # differ from 1 in that order, or "d1" if all are 1.
        dims = []
        if self.parallel.data_parallel_size != 1:
            dims.append(f"d{self.parallel.data_parallel_size}")
        if self.parallel.pipeline_parallel_size != 1:
            dims.append(f"p{self.parallel.pipeline_parallel_size}")
        if self.parallel.tensor_parallel_size != 1:
            dims.append(f"t{self.parallel.tensor_parallel_size}")
        if self.parallel.context_parallel_size != 1:
            dims.append(f"c{self.parallel.context_parallel_size}")
        if self.parallel.expert_parallel_size != 1:
            dims.append(f"e{self.parallel.expert_parallel_size}")

        if not dims:  # Show at least data parallel if all dimensions are 1
            dims.append(f"d{self.parallel.data_parallel_size}")

        result = "".join(dims)
        if self.name:
            result = f"{self.backend}({self.name}):{result}"
        else:
            result = f"{self.backend}:{result}"
        return result


@dataclass
class _AllocationMode:
    # [CLASS DOC]
    # Deprecated container of ModelAllocations for SPMD launchers (local/ray/slurm).
    # Supports lookup by name and legacy properties (type_, gen, train, gen_backend,
    # train_backend, gen_instance_size) that assume at most one inference and one
    # training component.
    """DEPRECATED — Legacy resource allocation configuration for SPMD launchers only.

    Use :class:`ModelAllocation` with per-engine ``backend`` fields instead.

    This class is retained only for backward compatibility with SPMD launchers
    (local, ray, slurm) and will be removed in a future version.

    Parameters
    ----------
    allocations : list[ModelAllocation]
        List of ModelAllocation objects, each representing a component

    Notes
    -----
    Access patterns:
        - allocation_mode[name]: Get allocation by name
        - allocation_mode.allocations: Get all allocations
        - allocation_mode.gen: Backward-compatible (single inference only)
        - allocation_mode.train: Backward-compatible (single training only)

    Examples
    --------
    Two named components:

    >>> mode = _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4")
    >>> rollout = mode["rollout"]

    Three components (names required):

    >>> mode = _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4+fsdp[c]:d4")

    Colocation (actor and critic share 4 GPUs):

    >>> mode = _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4|fsdp[c]:d4")
    """

    allocations: list[ModelAllocation] = field(default_factory=list)

    @classmethod
    def from_str(cls, allocation_mode: str):
        # [FUNCTION DOC]
        # Classmethod: parses a full allocation-mode string (with "+" and "|") via
        # _LLMParallelParser and wraps the result in _AllocationMode. Raises
        # AllocationValidationError / InvalidAllocationModeError for semantic errors,
        # ValueError (with syntax hints) for parse errors.
        """Parse allocation mode string into _AllocationMode object.

        Parameters
        ----------
        allocation_mode : str
            String representation of allocation mode

        Returns
        -------
        _AllocationMode
            Parsed allocation configuration

        Raises
        ------
        AllocationValidationError
            When validation fails (duplicate names, missing names, etc.)
        ValueError
            When parsing fails

        Notes
        -----
        Syntax:
            - backend(name):dims - Named component
            - component+component - Disaggregation (separate GPUs)
            - component|component - Colocation (shared GPUs, names required)
            - Operator precedence: | binds tighter than +

        Examples
        --------
        Two components, no names:

        >>> _AllocationMode.from_str("sglang:d4t2+fsdp:d8")

        Two named components:

        >>> _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4")

        Three+ components (names required):

        >>> _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4+fsdp[c]:d4")

        Colocation (r separated, a|c share GPUs):

        >>> _AllocationMode.from_str("sglang[r]:d2+fsdp[a]:d4|fsdp[c]:d4")
        """
        parser = _LLMParallelParser()
        result = parser.parse(allocation_mode)
        return parser._convert_to_allocation_mode(result)

    def __getitem__(self, name: str) -> ModelAllocation:
        # [FUNCTION DOC]
        # Returns the allocation whose name equals name; raises KeyError if none.
        """Get allocation by name."""
        for alloc in self.allocations:
            if alloc.name == name:
                return alloc
        raise KeyError(f"No allocation found with name: {name}")

    @property
    def world_size(self):
        # [FUNCTION DOC]
        # Property: sum of component world sizes (colocated components contribute 0).
        return sum(alloc.world_size for alloc in self.allocations)

    def _get_inference_allocations(self) -> list[ModelAllocation]:
        # [FUNCTION DOC]
        # HELPER: Returns allocations with sglang or vllm backend.
        """Get all inference allocations (sglang, vllm backends)."""
        return [a for a in self.allocations if a.backend in ("sglang", "vllm")]

    def _get_training_allocations(self) -> list[ModelAllocation]:
        # [FUNCTION DOC]
        # HELPER: Returns allocations with fsdp, megatron or archon backend.
        """Get all training allocations (fsdp, megatron, archon backends)."""
        return [
            a for a in self.allocations if a.backend in ("fsdp", "megatron", "archon")
        ]

    ########### Legacy Attributes for Backward Compatibility ###########
    # DEPRECATED: AllocationMode and these properties are legacy. Use ModelAllocation directly.
    # These properties are only retained for SPMD launcher compatibility and will be removed.
    @property
    def type_(self) -> AllocationType:
        # [FUNCTION DOC]
        # Deprecated property. Requires 1 or 2 allocations (AttributeError otherwise).
        # One allocation: LLM_SERVER_ONLY if inference backend else COLOCATE. Two:
        # requires exactly one inference and one training allocation (AttributeError
        # otherwise); DECOUPLED_TRAIN if both use separation scheduling, else COLOCATE.
        """DEPRECATED: Infer allocation type from allocations. Use ModelAllocation directly."""
        if len(self.allocations) not in [1, 2]:
            raise AttributeError(
                "Can only infer allocation type from 1 or 2 allocations."
            )

        if len(self.allocations) == 1:
            if self.allocations[0].backend in ("sglang", "vllm"):
                return AllocationType.LLM_SERVER_ONLY
            return AllocationType.COLOCATE

        inf_alloc = self._get_inference_allocations()
        train_alloc = self._get_training_allocations()
        if not (len(inf_alloc) == 1 and len(train_alloc) == 1):
            raise AttributeError(
                "Ambiguous allocation type: expected one inference and one training allocation."
            )
        if (
            inf_alloc[0].scheduling_strategy.type
            == SchedulingStrategyType.separation.value
            and train_alloc[0].scheduling_strategy.type
            == SchedulingStrategyType.separation.value
        ):
            return AllocationType.DECOUPLED_TRAIN
        return AllocationType.COLOCATE

    @property
    def gen(self) -> ParallelStrategy:
        # [FUNCTION DOC]
        # Deprecated property: parallel strategy of the single inference allocation,
        # None if none, AttributeError if several.
        # KNOWN ISSUE: Annotated as returning ParallelStrategy but may return None.
        """Backward compatible: returns parallel strategy for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            return None
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name] or allocation_mode.allocations instead."
            )
        return inf_allocs[0].parallel

    @property
    def train(self) -> ParallelStrategy | None:
        # [FUNCTION DOC]
        # Deprecated property: parallel strategy of the single training allocation, None
        # if none, AttributeError if several.
        """Backward compatible: returns parallel strategy for single training allocation."""
        train_allocs = self._get_training_allocations()
        if len(train_allocs) == 0:
            return None
        if len(train_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'train' property: found {len(train_allocs)} training allocations. "
                f"Use allocation_mode[name] or allocation_mode.allocations instead."
            )
        return train_allocs[0].parallel

    @property
    def gen_backend(self) -> str | None:
        # [FUNCTION DOC]
        # Deprecated property: backend of the single inference allocation, None if none,
        # AttributeError if several.
        """Backward compatible: returns backend for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            return None
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen_backend' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name].backend or allocation_mode.allocations instead."
            )
        return inf_allocs[0].backend

    @property
    def train_backend(self) -> str | None:
        # [FUNCTION DOC]
        # Deprecated property: backend of the single training allocation, None if none,
        # AttributeError if several.
        """Backward compatible: returns backend for single training allocation."""
        train_allocs = self._get_training_allocations()
        if len(train_allocs) == 0:
            return None
        if len(train_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'train_backend' property: found {len(train_allocs)} training allocations. "
                f"Use allocation_mode[name].backend or allocation_mode.allocations instead."
            )
        return train_allocs[0].backend

    @property
    def gen_instance_size(self) -> int:
        # [FUNCTION DOC]
        # Deprecated property: tp_size * pp_size (GPUs per server instance) of the
        # single inference allocation; AttributeError if none or several.
        """Backward compatible: returns instance size for single inference allocation."""
        inf_allocs = self._get_inference_allocations()
        if len(inf_allocs) == 0:
            raise AttributeError("No inference allocations found")
        if len(inf_allocs) > 1:
            raise AttributeError(
                f"Ambiguous 'gen_instance_size' property: found {len(inf_allocs)} inference allocations. "
                f"Use allocation_mode[name].parallel.tp_size * pp_size instead."
            )
        return inf_allocs[0].parallel.tp_size * inf_allocs[0].parallel.pp_size


# Grammar-based parser using Lark
# Operator precedence: | (colocation) binds tighter than + (disaggregation)
# Example: "a+b|c" parses as "a+(b|c)", not "(a+b)|c"
ALLOCATION_GRAMMAR = """
    start: expression

    expression: disaggregate_chain | component

    disaggregate_chain: component ("+" component)+
    component: colocate_expr | single_allocation
    single_allocation: inf_para | train_para
    colocate_expr: single_allocation ("|" single_allocation)+

    inf_para: modern_inf_para
    modern_inf_para: INFER_BACKEND ("[" NAME "]")? ":" inf_dim+
    train_para: train_backend_name_hybrid | train_backend_with_name | train_backend_hybrid | train_backend_only | train_name_only | train_dims_only | hybrid_moe_syntax
    train_backend_name_hybrid: TRAIN_BACKEND "[" NAME "]" ":" hybrid_moe_syntax
    train_backend_with_name: TRAIN_BACKEND "[" NAME "]" ":" common_dim+
    train_backend_hybrid: TRAIN_BACKEND ":" hybrid_moe_syntax
    train_backend_only: TRAIN_BACKEND ":" common_dim+
    train_name_only: "[" NAME "]" ":" common_dim+
    train_dims_only: common_dim+

    hybrid_moe_syntax: "("? attn_section "|" ffn_section ")"?
    attn_section: "attn" ":" attn_dim+
    ffn_section: "ffn" ":" ffn_dim+

    // Training parallelism strategy
    common_dim: DIM_TYPE NUMBER
    attn_dim: ATTN_DIM_TYPE NUMBER
    ffn_dim: FFN_DIM_TYPE NUMBER

    // Inference parallelism strategy
    inf_dim: INF_DIM_TYPE NUMBER

    DIM_TYPE: "p" | "d" | "t" | "c" | "e"
    ATTN_DIM_TYPE: "c" | "d" | "t" | "p"
    FFN_DIM_TYPE: "d" | "e" | "t" | "p"
    INF_DIM_TYPE: "d" | "t" | "p"

    INFER_BACKEND: "sglang" | "vllm"
    TRAIN_BACKEND: "fsdp" | "megatron" | "archon"

    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[1-9][0-9]*/

    %import common.WS
    %ignore WS
"""


@dataclass
class ParallelDimension:
    # [CLASS DOC]
    # HELPER: Parser intermediate holding one dimension letter (d/t/p/c/e) and its size;
    # str gives e.g. "d4".
    """Single parallelism dimension with type and size.

    Used internally by the grammar parser to represent individual
    parallelism specifications before combining them into strategies.
    """

    type_: str  # Dimension type ("d", "t", "p", "c", "e")
    size: int  # Parallelism degree

    def __str__(self):
        # [FUNCTION DOC]
        # HELPER: Returns type letter followed by size.
        return f"{self.type_}{self.size}"


@dataclass
class InferenceParallelism:
    # [CLASS DOC]
    # Legacy dataclass pairing an inference backend name with a ParallelStrategy.
    """Backward Compatible: Inference parallelism configuration with backend and validation.

    Represents the parallelization strategy for inference workloads,
    including the specific backend (SGLang, vLLM) and associated
    validation rules.
    """

    backend: str
    strategy: ParallelStrategy

    def __str__(self):
        # [FUNCTION DOC]
        # Returns "backend:dims" listing d, t, p sizes that differ from 1 (or "d1" if
        # none).
        dims = []
        if self.strategy.data_parallel_size != 1:
            dims.append(f"d{self.strategy.data_parallel_size}")
        if self.strategy.tensor_parallel_size != 1:
            dims.append(f"t{self.strategy.tensor_parallel_size}")
        if self.strategy.pipeline_parallel_size != 1:
            dims.append(f"p{self.strategy.pipeline_parallel_size}")
        if not dims:  # Show at least data parallel if all dimensions are 1
            dims.append(f"d{self.strategy.data_parallel_size}")
        return f"{self.backend}:{''.join(dims)}"


class _ParallelStrategyTransformer(Transformer):
    # [CLASS DOC]
    # HELPER: Lark Transformer turning the allocation parse tree into ModelAllocation
    # objects or lists of them. Methods are named after grammar rules/terminals and
    # receive child results as items. Tracks seen_names to enforce unique component
    # names within one parse.
    """Lark transformer to convert parse tree to lists of ModelAllocation objects."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the base Transformer and an empty seen_names set.
        super().__init__()
        self.seen_names = set()

    def _validate_name(self, name: str | None):
        # [FUNCTION DOC]
        # HELPER: Raises AllocationValidationError if a non-None name was already seen,
        # else records it.
        """Validate and track component names for uniqueness."""
        if name is not None:
            if name in self.seen_names:
                raise AllocationValidationError(f"Duplicate component name: {name}")
            self.seen_names.add(name)

    def _build_model_allocation(
        self,
        backend: str,
        name: str | None,
        strategy: ParallelStrategy,
        scheduling: SchedulingStrategy,
    ) -> ModelAllocation:
        # [FUNCTION DOC]
        # HELPER: Validates name uniqueness and constructs a ModelAllocation (which may
        # raise for invalid fsdp strategies).
        """Build ModelAllocation with validation."""
        self._validate_name(name)
        return ModelAllocation(
            backend=backend,
            name=name,
            parallel=strategy,
            scheduling_strategy=scheduling,
        )

    def start(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the single child result.
        return items[0]

    def expression(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the single child result.
        return items[0]

    def disaggregate_chain(self, items):
        # [FUNCTION DOC]
        # HELPER: Flattens child allocations/lists into one list; with 3 or more
        # allocations every one must be named (AllocationValidationError otherwise).
        # Returns the list.
        """Handle multi-component disaggregation: comp1 + comp2 + comp3..."""
        all_allocations = []
        for item in items:
            if isinstance(item, list):
                all_allocations.extend(item)
            else:
                all_allocations.append(item)

        # Validate: 3+ components must all have names
        if len(all_allocations) >= 3:
            unnamed = [a for a in all_allocations if a.name is None]
            if unnamed:
                raise AllocationValidationError(
                    f"When using 3+ components, all must have names. "
                    f"Found {len(unnamed)} unnamed components."
                )

        return all_allocations

    def component(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the single child result.
        return items[0]

    def single_allocation(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the single child result.
        return items[0]

    def colocate_expr(self, items):
        # [FUNCTION DOC]
        # HELPER: Builds a colocation group: the first allocation is the anchor
        # (separation scheduling); each later one must be named
        # (AllocationValidationError), gets colocation scheduling targeting the anchor
        # name, and must have the same parallel world_size as the anchor. Returns the
        # list.
        # KNOWN ISSUE: The anchor itself may be unnamed, producing colocation targets of
        # None.
        """Handle colocation: comp1 | comp2 | comp3..."""
        allocations = []
        anchor_name = None

        for i, item in enumerate(items):
            if isinstance(item, list):
                # This shouldn't happen in practice
                allocations.extend(item)
            else:
                alloc = item
                if i == 0:
                    # First component is the anchor
                    anchor_name = alloc.name
                    alloc.scheduling_strategy = SchedulingStrategy(
                        type=SchedulingStrategyType.separation, target=None
                    )
                else:
                    # Rest colocate with anchor
                    if alloc.name is None:
                        raise AllocationValidationError(
                            "Components in colocation group must have names"
                        )
                    alloc.scheduling_strategy = SchedulingStrategy(
                        type=SchedulingStrategyType.colocation, target=anchor_name
                    )

                    # Validate world sizes match
                    if alloc.parallel.world_size != allocations[0].parallel.world_size:
                        raise AllocationValidationError(
                            f"Colocated components must have matching world sizes. "
                            f"'{anchor_name}' has {allocations[0].parallel.world_size}, "
                            f"'{alloc.name}' has {alloc.parallel.world_size}."
                        )
                allocations.append(alloc)

        return allocations

    def inf_para(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the single child result.
        return items[0]

    def modern_inf_para(self, items):
        # [FUNCTION DOC]
        # HELPER: Builds an inference ModelAllocation from backend, optional [name] and
        # d/t/p dims with separation scheduling.
        backend = str(items[0])
        name = None
        dim_start_idx = 1

        # Check if name is provided
        if len(items) > 1 and isinstance(items[1], str):
            name = str(items[1])
            dim_start_idx = 2

        dimensions = items[dim_start_idx:]

        # Build ParallelStrategy
        strategy_kwargs = {}
        for dim in dimensions:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return self._build_model_allocation(
            backend,
            name,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_para(self, items):
        # [FUNCTION DOC]
        # HELPER: Passes through the train alternative's result; a bare ParallelStrategy
        # (hybrid MoE syntax without backend) raises AllocationValidationError.
        """Pass through result from one of the train_* alternatives."""
        result = items[0]

        # If result is a ParallelStrategy (from hybrid_moe_syntax), it means
        # no backend prefix was given — reject it.
        if isinstance(result, ParallelStrategy):
            raise AllocationValidationError(
                "Backend must be explicitly specified for hybrid MoE parallelism. "
                "Use e.g. 'megatron:(attn:d1p12t4|ffn:d1p12e4)'. "
                "Auto-backend selection is no longer supported."
            )

        return result

    def train_backend_with_name(self, items):
        # [FUNCTION DOC]
        # HELPER: Builds a named training ModelAllocation from backend and d/t/p/c/e
        # dims with separation scheduling.
        """Handle: TRAIN_BACKEND ( NAME ) : common_dim+"""
        backend = str(items[0])
        name = str(items[1])
        dims = items[2:]

        strategy_kwargs = {}
        for dim in dims:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                strategy_kwargs["context_parallel_size"] = dim.size
            elif dim.type_ == "e":
                strategy_kwargs["expert_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return self._build_model_allocation(
            backend,
            name,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_backend_name_hybrid(self, items):
        # [FUNCTION DOC]
        # HELPER: Builds a named training ModelAllocation from backend and a hybrid MoE
        # ParallelStrategy.
        """Handle: TRAIN_BACKEND [ NAME ] : hybrid_moe_syntax"""
        backend = str(items[0])
        name = str(items[1])
        strategy = items[2]  # ParallelStrategy from hybrid_moe_syntax

        return self._build_model_allocation(
            backend,
            name,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_backend_hybrid(self, items):
        # [FUNCTION DOC]
        # HELPER: Builds an unnamed training ModelAllocation from backend and a hybrid
        # MoE ParallelStrategy.
        """Handle: TRAIN_BACKEND : hybrid_moe_syntax"""
        backend = str(items[0])
        strategy = items[1]  # ParallelStrategy from hybrid_moe_syntax

        return self._build_model_allocation(
            backend,
            None,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_backend_only(self, items):
        # [FUNCTION DOC]
        # HELPER: Builds an unnamed training ModelAllocation from backend and d/t/p/c/e
        # dims.
        """Handle: TRAIN_BACKEND : common_dim+"""
        backend = str(items[0])
        dims = items[1:]

        strategy_kwargs = {}
        for dim in dims:
            if dim.type_ == "d":
                strategy_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                strategy_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                strategy_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                strategy_kwargs["context_parallel_size"] = dim.size
            elif dim.type_ == "e":
                strategy_kwargs["expert_parallel_size"] = dim.size

        strategy = ParallelStrategy(**strategy_kwargs)
        return self._build_model_allocation(
            backend,
            None,
            strategy,
            SchedulingStrategy(type=SchedulingStrategyType.separation, target=None),
        )

    def train_name_only(self, items):
        # [FUNCTION DOC]
        # HELPER: Always raises AllocationValidationError (backend prefix required).
        """Handle: ( NAME ) : common_dim+"""
        raise AllocationValidationError(
            "Backend must be explicitly specified. "
            "Got a named component without a backend prefix. "
            "Use e.g. 'fsdp[actor]:d4', 'megatron[actor]:d4t2p2'. "
            "Auto-backend selection is no longer supported."
        )

    def train_dims_only(self, items):
        # [FUNCTION DOC]
        # HELPER: Always raises AllocationValidationError (backend prefix required).
        """Handle: common_dim+"""
        raise AllocationValidationError(
            "Backend must be explicitly specified. "
            "Got bare parallelism dimensions without a backend prefix. "
            "Use e.g. 'fsdp:d4', 'megatron:d4t2p2', 'sglang:d4'. "
            "Auto-backend selection is no longer supported."
        )

    def common_dim(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns a ParallelDimension from the dim letter and size.
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def attn_dim(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns a ParallelDimension from the dim letter and size.
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def ffn_dim(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns a ParallelDimension from the dim letter and size.
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def inf_dim(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns a ParallelDimension from the dim letter and size.
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def expert_dim(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns a ParallelDimension; no grammar rule named expert_dim exists,
        # so it is unused.
        dim_type = str(items[0])
        size = int(items[1])
        return ParallelDimension(type_=dim_type, size=size)

    def attn_para(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the dims list; unused by the current grammar.
        return items  # Return list of dimensions

    def expert_para(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the dims list; unused by the current grammar.
        return items  # Return list of dimensions

    def hybrid_train_para(self, items):
        # [FUNCTION DOC]
        # HELPER: Combines attention dims (d/t/p/c, default 1) and FFN dims into one
        # ParallelStrategy. Expert pp inherits attention pp if omitted and must equal it
        # otherwise (AllocationValidationError). Expert tp/ep default 1; expert dp, if
        # omitted, is attention world size divided by ep*etp*pp
        # (AllocationValidationError if not divisible). Raises
        # InvalidAllocationModeError if attention and expert world sizes differ. Returns
        # a ParallelStrategy with attention sizes plus expert_parallel_size and
        # expert_tensor_parallel_size (expert dp is implied, not stored).
        attn_dims = items[0]  # List of dimensions for attention modules
        expert_dims = items[1]  # List of dimensions for expert modules

        # Build attention strategy
        attn_kwargs = {
            "data_parallel_size": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "context_parallel_size": 1,
        }
        for dim in attn_dims:
            if dim.type_ == "d":
                attn_kwargs["data_parallel_size"] = dim.size
            elif dim.type_ == "t":
                attn_kwargs["tensor_parallel_size"] = dim.size
            elif dim.type_ == "p":
                attn_kwargs["pipeline_parallel_size"] = dim.size
            elif dim.type_ == "c":
                attn_kwargs["context_parallel_size"] = dim.size

        # Build expert strategy parameters
        # - pp: inherits from attn if omitted, must match if specified
        # - dp: derived from world_size if omitted
        # - tp/ep: default to 1 if omitted
        expert_data_parallel_size = None
        expert_pipeline_parallel_size = None
        expert_tensor_parallel_size = 1
        expert_parallel_size = 1

        for dim in expert_dims:
            if dim.type_ == "d":
                expert_data_parallel_size = dim.size
            elif dim.type_ == "p":
                expert_pipeline_parallel_size = dim.size
            elif dim.type_ == "t":
                expert_tensor_parallel_size = dim.size
            elif dim.type_ == "e":
                expert_parallel_size = dim.size

        # expert PP: inherit from attn if omitted, validate match if specified
        if expert_pipeline_parallel_size is None:
            expert_pipeline_parallel_size = attn_kwargs["pipeline_parallel_size"]
        elif expert_pipeline_parallel_size != attn_kwargs["pipeline_parallel_size"]:
            raise AllocationValidationError(
                f"Pipeline parallel size for attention and FFN modules must be identical. "
                f"Got attention: {attn_kwargs['pipeline_parallel_size']}, FFN: {expert_pipeline_parallel_size}."
            )

        # Calculate attn world size
        attn_world_size = math.prod(
            [
                attn_kwargs["data_parallel_size"],
                attn_kwargs["tensor_parallel_size"],
                attn_kwargs["pipeline_parallel_size"],
                attn_kwargs["context_parallel_size"],
            ]
        )

        # expert DP: derive from world_size if omitted
        if expert_data_parallel_size is None:
            ffn_non_dp_size = (
                expert_parallel_size
                * expert_tensor_parallel_size
                * expert_pipeline_parallel_size
            )
            if attn_world_size % ffn_non_dp_size != 0:
                raise AllocationValidationError(
                    f"Cannot derive expert dp: attn world_size ({attn_world_size}) "
                    f"is not divisible by ffn ep*tp*pp ({ffn_non_dp_size})."
                )
            expert_data_parallel_size = attn_world_size // ffn_non_dp_size

        # Validate world sizes match
        expert_world_size = math.prod(
            [
                expert_data_parallel_size,
                expert_pipeline_parallel_size,
                expert_tensor_parallel_size,
                expert_parallel_size,
            ]
        )

        if attn_world_size != expert_world_size:
            raise InvalidAllocationModeError(
                f"World size for expert modules and attention modules must be identical. "
                f"Got attention: {attn_world_size}, expert: {expert_world_size}."
            )

        # Create final strategy combining both
        final_strategy_kwargs = attn_kwargs.copy()
        final_strategy_kwargs["expert_parallel_size"] = expert_parallel_size
        final_strategy_kwargs["expert_tensor_parallel_size"] = (
            expert_tensor_parallel_size
        )

        strategy = ParallelStrategy(**final_strategy_kwargs)
        return strategy  # Return ParallelStrategy, will be wrapped by train_para

    def hybrid_moe_syntax(self, items):
        # [FUNCTION DOC]
        # HELPER: Forwards the attention and FFN dim lists to hybrid_train_para.
        # items should be [attn_section_result, ffn_section_result]
        attn_dims = items[0]
        ffn_dims = items[1]
        return self.hybrid_train_para([attn_dims, ffn_dims])

    def attn_section(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the list of attention dims.
        # items will be the attn_dim+ results (ignoring "attn" and ":" literals)
        return items

    def ffn_section(self, items):
        # [FUNCTION DOC]
        # HELPER: Returns the list of FFN dims.
        # items will be the ffn_dim+ results (ignoring "ffn" and ":" literals)
        return items

    def DIM_TYPE(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)

    def ATTN_DIM_TYPE(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)

    def FFN_DIM_TYPE(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)

    def EXPERT_DIM_TYPE(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str; no such terminal exists
        # in the grammar.
        return str(token)

    def INF_DIM_TYPE(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)

    def INFER_BACKEND(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)

    def TRAIN_BACKEND(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)

    def NUMBER(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to int.
        return int(token)

    def NAME(self, token):
        # [FUNCTION DOC]
        # HELPER: Terminal callback converting the token to str.
        return str(token)


class _LLMParallelParser:
    # [CLASS DOC]
    # HELPER: Wraps a Lark Earley parser (explicit ambiguity) over ALLOCATION_GRAMMAR
    # and the transformer, normalizing errors.
    """Internal LLM parallel strategy parser using Lark grammar.

    This parser handles the modern allocation mode syntax with explicit
    backend specifications, comprehensive validation, and support for
    complex allocation patterns including disaggregated and colocated
    configurations.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Builds the Lark parser from ALLOCATION_GRAMMAR.
        self.parser = Lark(ALLOCATION_GRAMMAR, parser="earley", ambiguity="explicit")

    def parse(self, expression: str):
        # [FUNCTION DOC]
        # HELPER: Parses expression and transforms it with a fresh
        # _ParallelStrategyTransformer, returning a ModelAllocation or list. Validation
        # errors are re-raised directly; when wrapped by Lark (VisitError) they are
        # recovered by scanning the formatted traceback text for the exception name and
        # re-raised with the extracted message. Any other failure becomes ValueError
        # with syntax hints.
        try:
            tree = self.parser.parse(expression)
            transformer = _ParallelStrategyTransformer()
            result = transformer.transform(tree)
            return result
        except (AllocationValidationError, InvalidAllocationModeError):
            # Re-raise validation errors without modification
            raise
        except Exception as e:
            # Check for wrapped validation errors in lark VisitError
            import traceback

            tb = traceback.format_exception(type(e), e, e.__traceback__)
            tb_str = "".join(tb)

            if "AllocationValidationError" in tb_str:
                # Extract the validation error message
                lines = tb_str.split("\n")
                for line in lines:
                    if "AllocationValidationError:" in line:
                        msg = line.split("AllocationValidationError:")[-1].strip()
                        raise AllocationValidationError(msg)
                raise AllocationValidationError(str(e))
            elif "InvalidAllocationModeError" in tb_str:
                # Extract the invalid allocation error message
                lines = tb_str.split("\n")
                for line in lines:
                    if "InvalidAllocationModeError:" in line:
                        msg = line.split("InvalidAllocationModeError:")[-1].strip()
                        raise InvalidAllocationModeError(msg)
                raise InvalidAllocationModeError(str(e))

            err_hint = """
Hints:
1. The parsing logic requires colons instead of dots to separate backends from dimensions, e.g., use "sglang:d4+fsdp:d4" instead of "sglang.d4+fsdp.d4".
2. Check https://areal-project.github.io/AReaL/en/tutorial/megatron.html for allowed syntax and examples with complex MoE models.
"""
            raise ValueError(f"Parsing error: {e}\n{err_hint}")

    def _convert_to_allocation_mode(self, result):
        # [FUNCTION DOC]
        # HELPER: Wraps a list result or single ModelAllocation into _AllocationMode;
        # ValueError for other types.
        """Convert parsed result to _AllocationMode object.

        Args:
            result: Parsed result (list of ModelAllocation)

        Returns:
            _AllocationMode: Converted allocation mode configuration

        Raises:
            ValueError: When expression type is not recognized
        """
        if isinstance(result, list):
            # Main case: list of ModelAllocation objects
            return _AllocationMode(allocations=result)
        elif isinstance(result, ModelAllocation):
            # Single allocation
            return _AllocationMode(allocations=[result])
        else:
            raise ValueError(f"Unknown result type: {type(result)}")


# ---------------------------------------------------------------------------
# ``AllocationMode`` is removed.  Use :class:`ModelAllocation` with per-engine
# ``backend`` fields instead.  The internal ``_AllocationMode`` class is
# retained only for backward-compatible SPMD launcher parsing.
# ---------------------------------------------------------------------------

# Provide a clear error if someone still tries to import the old name.
# This avoids silent breakage.


def __getattr__(name):
    # [FUNCTION DOC]
    # HELPER: Module-level hook raising an AttributeError with migration guidance for
    # the removed name AllocationMode, and a standard AttributeError for anything else.
    if name == "AllocationMode":
        raise AttributeError(
            "AllocationMode has been removed. Use ModelAllocation with per-engine "
            "'backend' fields instead (e.g., actor.backend='fsdp:d4', "
            "rollout.backend='sglang:d4'). "
            "See docs/en/reference/alloc_mode.md for migration details."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

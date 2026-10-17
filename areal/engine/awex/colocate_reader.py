# [MODULE DOC]
# Inference-side AWEX colocated weight reader that runs inside the SGLang scheduler
# process. At import time it patches torch_memory_saver's hook_mode setter and rebuilds
# awex's model registry (both side effects), then defines AwexColocateReader: publishes
# inference metadata to the awex MetaServer, lazily builds awex's native
# NCCLWorkerWeightsReader, performs per-version weight updates, and implements the
# writer handshake (wait-for-offload, wait-for-ready, signal-finished) plus SGLang
# memory release/resume.
# SPDX-License-Identifier: Apache-2.0

"""AWEX colocate weight reader (native awex worker-reader adapter).

Runs inside the SGLang scheduler process. This is a thin shell around awex's
native ``NCCLWorkerWeightsReader`` that:

1. Eager-registers the inference-side metadata the train writer waits for
   (``infer_conf`` + ``num_infer_engines``), computed via awex's own
   ``InferParamMetaResolver._get_model_param_info`` + ``_build_params_meta``
   (no hand-rolled name normalization or shard merging).
2. Lazily constructs the awex ``NCCLWorkerWeightsReader`` on the first weight
   update (it needs ``training_params_meta``, which only appears after the
   first training step) and delegates the whole IPC-collect + StreamBatch
   transport + writer handshake to it.

Why the awex-native reader instead of a hand-rolled receiver: the community
SGLang scheduler has no ``execute_task_in_model_worker`` driver layer, so we
build the awex *worker* reader directly in-process. The native worker reader
uses ``NcclColocateStreamBatchTransport`` (recursive partition), the transport
AWEX ships -- a hand-rolled ring-shift transport deadlocks on mismatched
train/infer pipeline layouts (e.g. train PP=4 vs infer PP=1).

The plugin shell still owns the steps awex's *driver* would normally do
(``_pre_update_weights`` wait-for-offload + resume weights, ``_resume_kvcache``
signal-finished); see ``awex_sglang_plugin.process_awex_queue``.
"""

from __future__ import annotations

from typing import Any

import torch


def _patch_tms_hook_mode() -> None:
    # [FUNCTION DOC]
    # HELPER: If torch_memory_saver is importable and its singleton exists, replaces the
    # class's hook_mode property with one whose setter silently ignores assignments once
    # the singleton is initialised (detected by the missing _impl_ctor_kwargs attribute)
    # and otherwise delegates to the original setter. Idempotent via an _awex_safe
    # marker. Side effect: monkey-patches a third-party class globally. Returns None.
    """Make ``torch_memory_saver.hook_mode`` setter a no-op once initialized.

    ``megatron.core.inference.contexts.dynamic_context`` (pulled in transitively
    by ``awex.converter.mcore_converter`` -> ``megatron.core``) runs a
    module-level ``torch_memory_saver.hook_mode = "torch"``. In the SGLang
    scheduler process the memory_saver singleton is already initialized (sglang
    ran ``_ensure_initialized``, which ``del``s ``_impl_ctor_kwargs``), so that
    late assignment raises ``AttributeError``. awex's model registry swallows the
    import error, the BailingMoe converter never registers, and weight transfer
    later dies with ``Unsupported attention parameter name: attention.g_proj``.
    The singleton's own assert already declares post-init configuration
    unsupported, so dropping the late set is the intended behavior.
    """
    try:
        import torch_memory_saver as _tms
    except Exception:
        return
    inst = getattr(_tms, "torch_memory_saver", None)
    if inst is None:
        return
    cls = type(inst)
    prop = cls.hook_mode
    if getattr(prop.fset, "_awex_safe", False):
        return

    def _safe_setter(self, value):
        # [FUNCTION DOC]
        # HELPER: Replacement setter; no-op after init, else calls the original setter.
        if not hasattr(self, "_impl_ctor_kwargs"):
            return  # singleton already initialized; late set is a design no-op
        prop.fset(self, value)

    _safe_setter._awex_safe = True
    cls.hook_mode = property(prop.fget, _safe_setter)


# Must run before any awex import: awex.models.registry auto-imports model
# modules at module load, and the BailingMoe module's transitive megatron import
# trips the hook_mode race above.
_patch_tms_hook_mode()

from awex.meta.infer_meta_resolver import InferParamMetaResolver  # noqa: E402
from awex.meta.meta_resolver import ParamMetaResolver  # noqa: E402
from awex.reader.nccl_reader import NCCLWorkerWeightsReader  # noqa: E402
from awex.sharding import get_sharding_strategy_builder  # noqa: E402
from awex.util.common import simple_hf_config  # noqa: E402

from areal.utils.logging import getLogger  # noqa: E402

logger = getLogger("AwexColocateReader")


def _get_router_dtype(config):
    # [FUNCTION DOC]
    # HELPER: Returns config.router_dtype, else text_config.router_dtype, else the
    # default 'bf16'.
    """Read router dtype from a flat or multimodal Hugging Face config."""
    router_dtype = getattr(config, "router_dtype", None)
    if router_dtype is not None:
        return router_dtype
    text_config = getattr(config, "text_config", config)
    return getattr(text_config, "router_dtype", "bf16")


def _get_awex_infer_hf_config(model, model_runner=None):
    # [FUNCTION DOC]
    # HELPER: Picks the original HF config from model_runner.model_config.hf_config
    # (keeps vision_config for composite VLMs) or falls back to model.config, serialises
    # it with awex's simple_hf_config, and fills architectures with the model class name
    # if missing. Returns the serialised config object.
    """Serialize the complete runtime config for AWEX metadata exchange."""
    # SGLang keeps only ``text_config`` on Qwen3-VL-MoE's runtime model while
    # retaining the original composite config on ``ModelRunner.model_config``.
    # Prefer that original config so AWEX also receives ``vision_config``.
    model_config = getattr(model_runner, "model_config", None)
    config = getattr(model_config, "hf_config", None)
    if config is None:
        config = model.config
    serialized_config = simple_hf_config(config)
    if not getattr(serialized_config, "architectures", None):
        serialized_config.architectures = [type(model).__name__]
    return serialized_config


def _ensure_awex_models_registered() -> None:
    # [FUNCTION DOC]
    # HELPER: Clears awex's cached model-config import and rebuilds
    # ModelRegistry.models, then logs a warning listing any expected
    # Bailing/Qwen3-VL/Qwen3.5 architectures still missing. Any exception is caught and
    # logged as a warning. Runs at import time.
    """Rebuild awex's model registry in case it cached a failed auto-import.

    ``import_model_configs`` is ``lru_cache``-d and ``ModelRegistry`` is built
    once at module load. If anything imported the registry before our hook_mode
    patch took effect, a native converter could be silently missing. Clear the
    cache and rebuild now that the patch is in place.

    The explicit architecture list is diagnostic only: AWEX remains the source
    of truth for model registration, transfer plans, conversion, and sharding.
    In particular, Qwen3.5 Dense and MoE reuse AWEX's native implementations
    for both causal-language-model and multimodal conditional-generation
    runtimes. Keeping those architecture names here makes an unavailable or
    failed AWEX model import visible before the first colocated weight update,
    without duplicating any model-specific transfer logic in AReaL.
    """
    try:
        from awex.models import registry as _reg

        _reg.import_model_configs.cache_clear()
        _reg.ModelRegistry.models = _reg.import_model_configs()
        missing = [
            m
            for m in (
                "BailingMoeV2_5ForCausalLM",
                "BailingMoeV2ForCausalLM",
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
                "Qwen3_5ForCausalLM",
                "Qwen3_5MoeForCausalLM",
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            )
            if m not in _reg.ModelRegistry.models
        ]
        if missing:
            logger.warning(f"awex model registry still missing converters: {missing}")
    except Exception as e:  # pragma: no cover - diagnostics only
        logger.warning(f"Failed to rebuild awex model registry: {e}")


_ensure_awex_models_registered()


class _SingleInstanceMetaResolver(ParamMetaResolver):
    # [CLASS DOC]
    # awex ParamMetaResolver subclass that aggregates per-rank raw parameter metadata
    # for one inference instance (exchanged via MetaServer instead of awex's driver fan-
    # out) into awex ParameterMeta, using awex's sharding strategy builder for the
    # engine.
    """Aggregate per-rank raw meta of ONE inference instance into ParameterMeta.

    awex's ``InferParamMetaResolver`` normally drives this via
    ``execute_task_in_model_worker`` (a driver fan-out we do not have). We
    instead exchange the per-rank raw meta dicts through the MetaServer
    (see ``_build_instance_params_meta``) and reuse awex's ``_build_params_meta``
    for the aggregation, plus awex's own sharding strategy builder for
    ``_get_sharding_info``. This yields the exact same ``parameters_meta`` the
    native reader expects, with awex converter parameter names (no hand-rolled
    normalization).
    """

    def __init__(self, hf_config, engine_name, infer_engine_config, raw_meta_list):
        # [FUNCTION DOC]
        # Stores the raw meta list, picks the rank-0 entry for the model architecture
        # name and builds the sharding strategy from engine_name, the inference engine
        # config and rank 0's rank_info.
        super().__init__(hf_config)
        self._raw_meta_list = raw_meta_list
        rank0 = self._select_rank0(raw_meta_list)
        self._model_arch_name = rank0["model_arch_name"]
        self._sharding_strategy = get_sharding_strategy_builder(engine_name)(
            self._model_arch_name,
            infer_engine_config,
            rank0["rank_info"],
        )

    @staticmethod
    def _select_rank0(raw_meta_list):
        # [FUNCTION DOC]
        # HELPER: Returns the entry whose rank_info.global_rank is 0, else the first
        # entry.
        for info in raw_meta_list:
            if info["rank_info"].global_rank == 0:
                return info
        return raw_meta_list[0]

    def get_model_arch_name(self) -> str:
        # [FUNCTION DOC]
        # Returns the cached model architecture name.
        return self._model_arch_name

    def get_parameters_meta(self):
        # [FUNCTION DOC]
        # Returns the aggregated parameter metadata via the base class's
        # _build_params_meta.
        return self._build_params_meta()

    def _get_params_raw_meta(self):
        # [FUNCTION DOC]
        # HELPER: Override returning the provided raw meta list.
        return self._raw_meta_list

    def _get_sharding_info(self, name, rank_info, param_meta):
        # [FUNCTION DOC]
        # HELPER: Override delegating to the sharding strategy for the given name,
        # rank_info and param_meta.
        return self._sharding_strategy.get_sharding_strategy(
            name, rank_info=rank_info, param_meta=param_meta
        )


class AwexColocateReader:
    # [CLASS DOC]
    # Adapter owned by the SGLang plugin. State: scheduler reference, MetaServer client,
    # lazily created awex reader, released memory tags, transfer/world sizes, instance
    # decomposition (instance world = tp*pp, engine_rank = transfer_rank // instance
    # world, instance-local rank), cached inference params meta and infer_conf, and an
    # initialised flag.
    """Thin adapter binding awex's native worker reader into a SGLang scheduler."""

    def __init__(self, scheduler: Any):
        # [FUNCTION DOC]
        # Stores the scheduler and initialises all state fields to None/empty. No I/O.
        self._scheduler = scheduler
        self._meta_server_client = None
        self._reader: NCCLWorkerWeightsReader | None = None
        self._released_tags: set[str] = set()

        self._transfer_rank: int | None = None
        self._local_gpu_id: int | None = None
        self._infer_world_size: int | None = None
        self._train_world_size: int | None = None
        self._meta_server_addr: str | None = None

        # External-instance decomposition (computed in initialize()).
        self._infer_instance_world_size: int | None = None
        self._num_infer_engines: int | None = None
        self._engine_rank: int | None = None
        self._instance_local_rank: int | None = None

        # Inference-side parameters_meta for ONE engine instance, computed via
        # awex resolver + MetaServer raw-meta exchange. Reused as the native
        # reader's ``parameters_meta`` constructor arg.
        self._infer_params_meta = None
        self._infer_conf: dict | None = None
        self._initialized = False

    # ── model / context helpers ───────────────────────────────────────

    def _get_model(self) -> torch.nn.Module:
        # [FUNCTION DOC]
        # HELPER: Returns the SGLang model module from scheduler.tp_worker.model_runner.
        return self._scheduler.tp_worker.model_runner.model

    def _build_model_context(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Builds the awex model_context dict for one inference instance from
        # server_args and scheduler attributes: scheduler, infer_engine_config, tp/pp
        # ranks and sizes, dp_size, attention TP/DP ranks/sizes, local_rank (tp_rank),
        # and world_size/global_rank which are the instance world and instance-local
        # rank after initialize, otherwise tp*pp and tp_rank.
        """awex model_context describing ONE inference engine instance.

        ``world_size`` is the single-server tp*pp; ``global_rank`` is the
        instance-local rank (= tp_rank for pp=1). The cross-server NCCL identity
        (engine_rank / global transfer_rank) is tracked separately by the awex
        reader. ``infer_engine_config`` (== server_args) is required by
        ``WorkerWeightsReader.__init__`` and the backport's model_context omits
        it, so we add it here.
        """
        scheduler = self._scheduler
        server_args = scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        dp_size = int(getattr(server_args, "dp_size", 1))
        tp_rank = int(getattr(scheduler, "tp_rank", 0))

        if self._infer_instance_world_size is not None:
            world_size = self._infer_instance_world_size
            global_rank = self._instance_local_rank
        else:
            world_size = tp_size * pp_size
            global_rank = tp_rank

        return {
            "scheduler": scheduler,
            "infer_engine_config": server_args,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "pp_rank": int(getattr(scheduler, "pp_rank", 0)),
            "pp_size": pp_size,
            "dp_size": dp_size,
            "world_size": world_size,
            "global_rank": global_rank,
            "local_rank": tp_rank,
            "attn_tp_rank": int(getattr(scheduler, "attn_tp_rank", tp_rank)),
            "attn_tp_size": int(getattr(scheduler, "attn_tp_size", tp_size)),
            "attn_dp_rank": int(getattr(scheduler, "attn_dp_rank", 0)),
        }

    def get_parallelism(self) -> dict:
        # [FUNCTION DOC]
        # Returns a dict with world_size (from the model context), tp/pp/dp/ep sizes
        # from server_args, and num_engines (defaults to 1).
        ctx = self._build_model_context()
        server_args = self._scheduler.server_args
        return {
            "world_size": ctx["world_size"],
            "tp_size": int(getattr(server_args, "tp_size", ctx["tp_size"])),
            "pp_size": int(getattr(server_args, "pp_size", ctx["pp_size"])),
            "dp_size": int(getattr(server_args, "dp_size", ctx["dp_size"])),
            "ep_size": int(getattr(server_args, "ep_size", 1)),
            "num_engines": self._num_infer_engines or 1,
        }

    # ── metadata (awex-native, no hand-rolled normalization) ──────────

    def _compute_local_raw_meta(self) -> dict:
        # [FUNCTION DOC]
        # HELPER: Calls awex InferParamMetaResolver._get_model_param_info for 'sglang'
        # with converted (HF) param names, this engine rank, the model and model
        # context. Returns this rank's raw meta dict.
        """Per-rank raw meta via awex's own staticmethod (HF-converted names)."""
        server_args = self._scheduler.server_args
        model_context = self._build_model_context()
        return InferParamMetaResolver._get_model_param_info(
            "sglang",
            server_args,
            convert_params=True,
            engine_rank=self._engine_rank or 0,
            model=self._get_model(),
            model_context=model_context,
        )

    def _build_instance_params_meta(self):
        # [FUNCTION DOC]
        # HELPER: Computes local raw meta; if the instance has >1 rank, puts it in the
        # MetaServer under a key namespaced by engine rank and instance-local rank and
        # fetches all ranks' entries (300 s timeout each); otherwise uses only the local
        # one. Rebuilds rank_info dicts into awex RankInfo objects and aggregates with
        # _SingleInstanceMetaResolver. Returns the instance's parameters_meta list.
        # Avoids process-group collectives to prevent deadlocks with the scheduler main
        # thread.
        """Gather single-instance raw meta via the MetaServer, then aggregate.

        Returns the awex ``parameters_meta`` (list[ParameterMeta]) for ONE
        inference engine instance (the ``instance_world`` instance-local ranks).

        We exchange per-rank raw meta through the MetaServer instead of an
        ``all_gather`` over ``tp_cpu_group``: that group is sglang's TP
        request-broadcast group, driven by the scheduler MainThread's
        ``recv_requests`` -> ``broadcast_pyobj``. This method runs on the
        plugin's background thread, so a collective on the shared group races
        the MainThread broadcast and deadlocks (two ops in flight on one
        non-thread-safe group). The MetaServer exchange needs no process-group
        collective, is isolated per engine instance by ``engine_rank``, and also
        sidesteps the ``dist.new_group`` collective-ordering trap (train + infer
        share the default world in colocate mode).
        """
        local_raw = self._compute_local_raw_meta()

        instance_world = self._infer_instance_world_size or 1
        if instance_world > 1:
            client = self._meta_server_client
            prefix = f"infer_instance_raw_meta_{self._engine_rank}"
            client.put_object(f"{prefix}_{self._instance_local_rank}", local_raw)
            raw_meta_list = [
                client.get_object(f"{prefix}_{r}", timeout=300.0)
                for r in range(instance_world)
            ]
        else:
            raw_meta_list = [local_raw]

        # MetaServer serializes RankInfo to a dict on the wire (as did the
        # legacy all_gather); rebuild the object before awex's resolver reads it.
        from awex.sharding.rank_info import RankInfo

        for info in raw_meta_list:
            ri = info.get("rank_info")
            if isinstance(ri, dict):
                info["rank_info"] = RankInfo(**ri)

        resolver = _SingleInstanceMetaResolver(
            self._get_model().config,
            "sglang",
            self._scheduler.server_args,
            raw_meta_list,
        )
        return resolver.get_parameters_meta()

    def get_weight_metadata(self):
        # [FUNCTION DOC]
        # Returns the cached instance parameters_meta, computing it on first call.
        # Raises RuntimeError if called before initialize set engine_rank.
        """Inference-side parameters_meta for ONE engine instance."""
        if self._engine_rank is None:
            raise RuntimeError(
                "AwexColocateReader must be initialized before getting weight metadata"
            )
        if self._infer_params_meta is None:
            self._infer_params_meta = self._build_instance_params_meta()
        return self._infer_params_meta

    # ── eager init: register infer_conf + num_infer_engines ───────────

    def initialize(
        self,
        meta_server_addr: str,
        transfer_rank: int,
        infer_world_size: int,
        train_world_size: int,
        local_gpu_id: int,
        timeout_s: float = 300.0,
    ) -> None:
        # [FUNCTION DOC]
        # Eager, non-blocking setup. Raises ValueError if infer and train world sizes
        # differ or infer_world_size is not divisible by tp*pp. Records sizes and the
        # instance decomposition, creates a MetaServerClient from 'host:port', computes
        # weight metadata, builds infer_conf (engine name, attention TP size, infer
        # world size, serialised HF config, router dtype) and, on transfer rank 0 only,
        # publishes infer_conf and num_infer_engines to the MetaServer. Marks
        # initialised and logs. Side effects: network I/O to MetaServer, logging.
        # KNOWN ISSUE: timeout_s parameter is accepted but never used.
        """Eager init: publish the metadata the train writer waits for.

        Must NOT block on the training side (runs before the first training step
        finishes). The native ``NCCLWorkerWeightsReader`` is built lazily in
        ``update_weights`` once ``training_params_meta`` is available. Device
        entry registration (``inference_device_rank_entries``) is left to the
        native reader's ``_init_reader_in_colocate_mode``.
        """
        from awex.meta.meta_server import MetaServerClient

        if infer_world_size != train_world_size:
            raise ValueError(
                f"Colocate mode requires equal total rank counts "
                f"(same physical GPUs), got infer={infer_world_size} "
                f"vs train={train_world_size}"
            )

        self._transfer_rank = transfer_rank
        self._local_gpu_id = local_gpu_id
        self._infer_world_size = infer_world_size
        self._train_world_size = train_world_size
        self._meta_server_addr = meta_server_addr

        server_args = self._scheduler.server_args
        tp_size = int(getattr(server_args, "tp_size", 1))
        pp_size = int(getattr(server_args, "pp_size", 1))
        instance_world = max(1, tp_size * pp_size)
        if infer_world_size % instance_world != 0:
            raise ValueError(
                f"infer_world_size ({infer_world_size}) must be divisible by the "
                f"per-instance world tp*pp ({instance_world})"
            )
        self._infer_instance_world_size = instance_world
        self._num_infer_engines = infer_world_size // instance_world
        self._engine_rank = transfer_rank // instance_world
        self._instance_local_rank = transfer_rank % instance_world
        logger.info(
            "AWEX instance decomposition: transfer_rank=%d -> engine_rank=%d, "
            "instance_local_rank=%d (instance_world=%d, num_engines=%d)",
            transfer_rank,
            self._engine_rank,
            self._instance_local_rank,
            instance_world,
            self._num_infer_engines,
        )

        host, port = meta_server_addr.rsplit(":", 1)
        self._meta_server_client = MetaServerClient(host, int(port))

        # Compute single-instance parameters_meta (also reused as the native
        # reader's constructor arg later).
        self.get_weight_metadata()

        par = self.get_parallelism()
        model_runner = self._scheduler.tp_worker.model_runner
        awex_hf_config = _get_awex_infer_hf_config(self._get_model(), model_runner)
        infer_conf = {
            "engine_name": "sglang",
            "infer_atten_tp_size": par["tp_size"],
            "infer_world_size": infer_world_size,
            "hf_config": awex_hf_config,
            # AWEX's native reader publishes router_dtype so the train-side
            # converter casts mlp.gate.weight to the dtype the inference
            # engine actually holds (fp32 for BailingMoe). Omitting it makes
            # the converter fall back to its bf16 default: gate shards go out
            # as 2N bytes against a 4N irecv and the transfer wedges
            # deterministically. The wire-level dtype reconciliation below
            # papers over any such mismatch generically, but keep the
            # semantic path whole so new models behave identically to native
            # awex.
            "router_dtype": _get_router_dtype(self._get_model().config),
        }
        self._infer_conf = infer_conf

        # Only one rank publishes the engine-instance-wide info the writer waits
        # for. transfer_rank 0 is engine_rank 0, instance_local_rank 0.
        if transfer_rank == 0:
            self._meta_server_client.put_object("infer_conf", infer_conf)
            self._meta_server_client.put_object(
                "num_infer_engines", self._num_infer_engines
            )
            logger.info(
                "Registered infer_conf + num_infer_engines=%d with MetaServer",
                self._num_infer_engines,
            )

        self._initialized = True
        logger.info(
            "Eager init done: transfer_rank=%d, local_gpu_id=%d, infer_world_size=%d "
            "(native worker reader construction deferred to first update_weights)",
            transfer_rank,
            local_gpu_id,
            infer_world_size,
        )

    # ── lazy native-reader construction + weight update ───────────────

    def _ensure_reader(self) -> NCCLWorkerWeightsReader:
        # [FUNCTION DOC]
        # HELPER: Returns the cached native reader or creates it: blocks on MetaServer
        # for training_params_meta (timeout 10000 s), constructs NCCLWorkerWeightsReader
        # in colocate mode with CUDA IPC, using the cached infer metadata, conf, engine
        # rank and count, then initialises and caches it.
        if self._reader is not None:
            return self._reader

        client = self._meta_server_client
        training_params_meta = client.get_object(
            "training_params_meta", timeout=10000.0
        )
        logger.info("Got training_params_meta from MetaServer")

        model_context = self._build_model_context()
        reader = NCCLWorkerWeightsReader(
            engine_name="sglang",
            model=self._get_model(),
            model_context=model_context,
            infer_conf=self._infer_conf,
            engine_rank=self._engine_rank,
            num_engines=self._num_infer_engines,
            meta_server_addr=self._meta_server_addr,
            parameters_meta=self._infer_params_meta,
            training_params_meta=training_params_meta,
            enable_colocate_mode=True,
            ipc_backend="cuda",
            enable_debug_mode=False,
        )
        reader.initialize()
        self._reader = reader
        logger.info(
            "Constructed native NCCLWorkerWeightsReader (transfer_rank=%d, "
            "engine_rank=%d, num_engines=%d)",
            reader.transfer_rank,
            self._engine_rank,
            self._num_infer_engines,
        )
        return reader

    def update_weights(self, version: int) -> None:
        # [FUNCTION DOC]
        # Raises RuntimeError if not initialised; otherwise ensures the native reader,
        # runs its update_weights for step_id=version (IPC collect, NCCL transport,
        # handshake, cache flush) and then rebuilds derived weights. Side effects: GPU
        # communication, in-place model weight mutation, logging.
        """Run one colocate weight update via the native awex worker reader.

        The native reader internally does: IPC collect -> StreamBatch transport
        -> put ``weights_update_finished`` -> barrier -> get_then_delete
        ``write_finished`` -> flush_cache. The plugin only needs to wrap this
        with the driver-equivalent wait-for-offload + resume + signal steps.
        """
        if not self._initialized:
            raise RuntimeError("AwexColocateReader not initialized")
        reader = self._ensure_reader()
        reader.update_weights(step_id=version)
        self._rebuild_derived_weights()
        logger.info("Colocate weight update completed: version=%d", version)

    def _rebuild_derived_weights(self) -> None:
        # [FUNCTION DOC]
        # HELPER: If the model defines post_load_weights, calls it to re-derive non-
        # parameter tensors (e.g. absorbed MLA w_kc/w_vc) after the in-place weight
        # write, then synchronises CUDA and logs. No-op otherwise.
        # KNOWN ISSUE: uses torch.cuda.synchronize directly rather than the platform
        # abstraction.
        """Re-derive non-parameter tensors after an in-place AWEX weight write.

        Root cause: sglang's ``load_model`` ends with
        ``post_load_weights()``, which splits each MLA layer's
        ``kv_b_proj.weight`` into the absorbed-path tensors ``w_kc``/``w_vc``
        — ``.contiguous()`` copies stored as plain attributes, in neither
        ``named_parameters`` nor ``named_buffers``. The memory-saver
        release/resume cycle remaps their pages to zeros, and the AWEX reader
        rewrites only named parameters via in-place ``copy_`` (bypassing
        ``model.load_weights``), so nothing ever rebuilds them: decode's
        forward_absorb then consumes zeros and the 4 MLA layers degenerate
        while the 28 Lightning layers stay healthy (reward 0.77 -> ~0 within
        5 steps). Rebuild after EVERY transfer — train weights move each
        version, so a one-time fix would go stale. ``bind_or_assign`` copies
        into the existing tensors in place, which keeps captured CUDA-graph
        addresses valid.
        """
        model = self._get_model()
        fn = getattr(model, "post_load_weights", None)
        if fn is None:
            return
        fn()
        torch.cuda.synchronize()
        logger.info("post_load_weights() re-derived absorbed MLA weights")

    # ── memory release/resume (delegate to SGLang native) ─────────────

    def release_memory(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Releases SGLang memory occupation for the given tags (default ['kv_cache'])
        # that are not already released, via the scheduler's release_memory_occupation,
        # and records them as released. Logs the requested tags.
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        native_tags = [t for t in tags if t not in self._released_tags]
        if native_tags:
            req = ReleaseMemoryOccupationReqInput(tags=native_tags)
            self._scheduler.release_memory_occupation(req)
            self._released_tags.update(native_tags)
        logger.info("release_memory: tags=%s", tags)

    def resume_memory(self, tags: list[str] | None = None) -> None:
        # [FUNCTION DOC]
        # Resumes SGLang memory occupation for requested tags (default ['kv_cache'])
        # that are currently released, then removes them from the released set. Logs the
        # requested tags.
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput

        tags = tags or ["kv_cache"]
        resume_tags = [t for t in tags if t in self._released_tags]
        if resume_tags:
            req = ResumeMemoryOccupationReqInput(tags=resume_tags)
            self._scheduler.resume_memory_occupation(req)
            self._released_tags.difference_update(resume_tags)
        logger.info("resume_memory: tags=%s", tags)

    # ── writer-coordination handshake (driver-equivalent shell steps) ──

    def wait_for_training_offloaded(self, version: int) -> None:
        # [FUNCTION DOC]
        # Blocks until the MetaServer set all_training_offloaded_weights reaches
        # train_world_size entries, with the AWEX colocate timeout.
        # KNOWN ISSUE: the version argument is ignored, so a stale set from a previous
        # version could satisfy the wait.
        """Wait for the writer to offload its model weights (avoid 2x weights).

        Equivalent to awex driver ``_pre_update_weights``'s wait on
        ``all_training_offloaded_weights``.
        """
        from areal.engine.awex.colocate_writer import awex_colocate_timeout_s

        self._meta_server_client.wait_set_until_size(
            "all_training_offloaded_weights",
            self._train_world_size,
            timeout=awex_colocate_timeout_s(),
        )

    def wait_for_weights_ready(
        self, version: int, timeout_s: float | None = None
    ) -> None:
        # [FUNCTION DOC]
        # Blocks (existence check only, no deserialisation) until the MetaServer key
        # training_serialized_weights_<ip>_<local_gpu>_<version> exists, using timeout_s
        # or the AWEX colocate timeout. Provides per-version gating for the plugin's
        # background thread.
        """Block until the writer has published THIS version's IPC handles.

        Used by the plugin's background thread as the per-version trigger to
        enqueue a weight-update marker. We probe the per-version
        ``training_serialized_weights_{ip}_{gpu}_{version}`` key with MetaServer
        ``wait_key`` (existence-only, NO deserialization), for two reasons:

        1. Per-version gating. The unversioned ``all_training_offloaded_weights``
           set is only deleted by the writer's rank0 in ``finish_colocate_weight_update``
           (a later phase than the engine's signal_finished), so gating on it
           lets the background thread fire v+1 off a *stale* satisfied set while
           the writer is still in v's finish phase. The collected v+1 IPC then
           blocks waiting for a not-yet-published key, hogging the scheduler main
           loop so it cannot serve rollout -> train waits on rollout -> deadlock.
           The writer only puts the v+1 serialized key in the NEXT training cycle,
           so gating on it cannot fire early.
        2. No double-attach. ``get_object`` would deserialize the CUDA IPC handle
           in the background thread, racing the worker reader's own collect inside
           update_weights. ``wait_key`` only checks presence (``_has_key``).
        """
        from awex.util.common import get_ip_address

        from areal.engine.awex.colocate_writer import awex_colocate_timeout_s

        ip = get_ip_address()
        key = f"training_serialized_weights_{ip}_{self._local_gpu_id}_{version}"
        self._meta_server_client.wait_key(
            key,
            timeout=awex_colocate_timeout_s() if timeout_s is None else timeout_s,
        )

    def signal_finished_weights_update(self) -> None:
        # [FUNCTION DOC]
        # On instance-local rank 0 only, adds this engine_rank to the MetaServer set
        # finished_weights_update_engines; other ranks do nothing.
        """Signal this engine finished, so the writer can resume kv_cache.

        Equivalent to awex driver ``_resume_kvcache``'s add to
        ``finished_weights_update_engines``. Only one rank per engine instance
        (instance_local_rank == 0) signals, with its real engine_rank, so the
        set collects exactly num_infer_engines unique entries.
        """
        if self._instance_local_rank != 0:
            return
        self._meta_server_client.add_object_to_set(
            "finished_weights_update_engines", self._engine_rank
        )

    def teardown(self) -> None:
        # [FUNCTION DOC]
        # Drops the reference to the native reader (no explicit cleanup of NCCL or
        # MetaServer resources).
        self._reader = None


__all__ = ["AwexColocateReader"]

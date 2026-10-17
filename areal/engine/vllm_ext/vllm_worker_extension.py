# [MODULE DOC]
# vLLM worker extension (passed to vLLM as worker_extension_cls) implementing the
# worker-side methods invoked by areal_vllm_server through collective_rpc: reload full
# weights or a LoRA adapter from disk, create custom NCCL/HCCL weight-update process
# groups with the trainer, store broadcast metadata, and receive full-model or LoRA
# weights via broadcast. Each method runs on every vLLM worker (TP/PP rank) and returns
# a (success, message) tuple. Provides a no-op fallback for set_current_vllm_config on
# vLLM < 0.19. Logs via vLLM's init_logger.
# SPDX-License-Identifier: Apache-2.0

import traceback
from contextlib import contextmanager

import torch
import torch.distributed as dist
from vllm.logger import init_logger
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader import get_model_loader

try:
    from vllm.config import set_current_vllm_config
except ImportError:
    # vLLM < 0.19 does not require an explicit config context for weight loading.
    @contextmanager
    def set_current_vllm_config(*_args, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Fallback no-op context manager used when vLLM lacks
        # set_current_vllm_config; ignores all args and yields.
        yield


from areal.engine.core.distributed import init_custom_process_group
from areal.infra.platforms import current_platform
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT

logger = init_logger("vllm_worker_extension")


class VLLMWorkerExtension:
    # [CLASS DOC]
    # Mixin merged into vLLM's Worker class; relies on the worker's model_runner, rank
    # and device. Adds state attributes: weight_update_groups (group name ->
    # ProcessGroup), areal_weight_meta_* (names/dtypes/shapes/group for full updates),
    # areal_lora_* (LoRA XCCL metadata) and _lora_partial_shards ((lora_name, id) ->
    # group name -> received tensors) for buffering per-PP-stage LoRA shards.
    # KNOWN ISSUE: docstring says "Iherited from vllm codebase" (typo; it is an
    # extension mixin, not inherited).
    """
    Iherited from vllm codebase
    """

    def sync(self):
        # [FUNCTION DOC]
        # HELPER: Device synchronize via current_platform followed by a default-group
        # torch.distributed barrier across vLLM workers.
        current_platform.synchronize()
        torch.distributed.barrier()

    def areal_update_weights(self, model_path):
        # [FUNCTION DOC]
        # Reload all model weights from model_path in place: sets model_config.model to
        # the path, gets the model loader for the load config and calls load_weights on
        # the live model within the vLLM config context, then syncs. Returns (True,
        # "Success") or (False, error message) on exception (logged).
        # KNOWN ISSUE: the first logger.info call passes flush=True, which standard
        # logging.Logger methods reject with TypeError (outside the try block), unless
        # vLLM's logger accepts it; same pattern in the LoRA and XCCL update methods.
        logger.info(f"start update weights, {model_path}", flush=True)
        try:
            # load weight
            self.model_runner.model_config.model = model_path
            model_loader = get_model_loader(self.model_runner.vllm_config.load_config)
            logger.info("Reloading weights inplace...")
            with set_current_vllm_config(self.model_runner.vllm_config):
                model_loader.load_weights(
                    self.model_runner.model, model_config=self.model_runner.model_config
                )
            self.sync()

            return True, "Success"
        except Exception as e:
            error_msg = f"failed to upload weights! {e}"
            logger.error(error_msg)
            return False, error_msg

    def areal_update_weights_lora(
        self,
        lora_model_path: str,
        lora_name: str,
        lora_int_id: int,
        base_model_name: str,
    ):
        # [FUNCTION DOC]
        # Replace a LoRA adapter from disk: removes adapter lora_int_id from the LoRA
        # manager, builds a LoRARequest (name, id, path, base model) and adds it through
        # model_runner.add_lora inside the config context, then syncs. Returns (success,
        # message); errors are logged and returned.
        logger.info(
            f"start lora update weights, lora_model_path-{lora_model_path}, lora_name-{lora_name}, lora_int_id-{lora_int_id}, base_model_name-{base_model_name}",
            flush=True,
        )
        try:
            # load lora weight
            self.model_runner.lora_manager.remove_adapter(lora_int_id)
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_model_path,
                base_model_name=base_model_name,
            )
            logger.info(f"Reloading lora weights with request {lora_request}")
            with set_current_vllm_config(self.model_runner.vllm_config):
                self.model_runner.add_lora(lora_request)

            self.sync()
            return True, "Success"
        except Exception as e:
            error_msg = f"failed to upload lora weights! {e}"
            logger.error(error_msg)
            return False, error_msg

    def areal_set_weight_meta(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
    ):
        # [FUNCTION DOC]
        # Store full-model broadcast metadata (names, dtype strings, shapes, group name)
        # on the worker for the next areal_update_weight_xccl. Returns (True,
        # "Success").
        logger.info("start set weights meta")
        self.areal_weight_meta_names = names
        self.areal_weight_meta_dtypes = dtypes
        self.areal_weight_meta_shapes = shapes
        self.areal_weight_meta_group_name = group_name
        return True, "Success"

    def areal_set_weight_meta_lora(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        lora_name: str,
        lora_int_id: int,
        lora_target_modules: list[str] | str,
        lora_rank: int,
        lora_alpha: int,
        lora_bias: str,
        base_model_name: str,
    ):
        # [FUNCTION DOC]
        # Store LoRA broadcast metadata (names/dtypes/shapes, group name, LoRA name, id,
        # target modules, rank, alpha, bias, base model name) for the next
        # areal_update_weight_lora_xccl. Returns (True, "Success").
        logger.info(
            f"start set lora weights meta for lora_name={lora_name}, lora_int_id={lora_int_id}"
        )
        self.areal_lora_weight_meta_names = names
        self.areal_lora_weight_meta_dtypes = dtypes
        self.areal_lora_weight_meta_shapes = shapes
        self.areal_weight_meta_group_name = group_name
        self.areal_lora_name = lora_name
        self.areal_lora_int_id = lora_int_id
        self.areal_lora_target_modules = lora_target_modules
        self.areal_lora_rank = lora_rank
        self.areal_lora_alpha = lora_alpha
        self.areal_lora_bias = lora_bias
        self.areal_lora_base_model_name = base_model_name
        return True, "Success"

    def areal_update_weight_xccl(self):
        # [FUNCTION DOC]
        # Receive full-model weights from the trainer: looks up the process group by the
        # stored group name (raises KeyError if absent, not caught), then for each
        # stored (name, dtype, shape) allocates an empty tensor on the worker device,
        # receives it by broadcast from group rank 0 (blocking) and immediately loads it
        # into the model with load_weights inside the config context. Syncs at the end.
        # Returns (success, message); other exceptions are logged and returned as
        # failure.
        logger.info("start update weights by nccl or hccl", flush=True)
        names = self.areal_weight_meta_names
        dtypes = self.areal_weight_meta_dtypes
        shapes = self.areal_weight_meta_shapes
        try:
            group = self.weight_update_groups[self.areal_weight_meta_group_name]
        except KeyError:
            raise KeyError(
                f"Weight update group named `{self.areal_weight_meta_group_name}` not found"
            )
        try:
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                tensor = torch.empty(
                    shape, dtype=target_dtype, device=self.model_runner.device
                )
                torch.distributed.broadcast(
                    tensor,
                    src=0,
                    group=group,
                    async_op=False,
                )
                with set_current_vllm_config(self.model_runner.vllm_config):
                    self.model_runner.model.load_weights(weights=[(name, tensor)])
            self.sync()
            return True, "Success"
        except Exception as e:
            error_msg = f"Failed to update parameter! {e}."
            logger.error(error_msg)
            return False, error_msg

    def areal_update_weight_lora_xccl(self):
        # [FUNCTION DOC]
        # Receive LoRA weights via broadcast and hot-swap the adapter (uses private vLLM
        # adapter-manager APIs). Raises KeyError if the group is unknown. Requires an
        # initialized LoRA manager with adapter lora_int_id already registered (else
        # failure). Receives each tensor from group rank 0, moves it to CPU, strips
        # "default." from names, and buffers the shard under (lora_name, id) and the
        # group name. Until shards from as many groups as are registered have arrived
        # (one per training PP stage), syncs and returns success. Once complete, merges
        # shards, builds a PEFTHelper (r, lora_alpha, target_modules, bias), creates a
        # CPU LoRAModel from the tensors with the manager's LoRA dtype, vocab size
        # (+extra vocab) and the model's hf_to_vllm mapper, removes the old adapter,
        # adds and activates the new one, logs (warning if the module count differs from
        # the old adapter), and syncs. Returns (success, message) with traceback on
        # failure.
        # NOTE: This code relies on vLLM private APIs: _adapter_manager, _registered_adapters,
        # and _add_adapter/activate_adapter, which may change/ breakdown due to newer vllm versions.

        logger.info(
            f"start update lora weights by xccl, lora_name={self.areal_lora_name}, lora_int_id={self.areal_lora_int_id}",
            flush=True,
        )
        names = self.areal_lora_weight_meta_names
        dtypes = self.areal_lora_weight_meta_dtypes
        shapes = self.areal_lora_weight_meta_shapes
        try:
            group = self.weight_update_groups[self.areal_weight_meta_group_name]
        except KeyError:
            raise KeyError(
                f"Weight update group named `{self.areal_weight_meta_group_name}` not found"
            )
        lora_int_id = self.areal_lora_int_id

        try:
            # Check if LoRA manager and adapter exist
            if self.model_runner.lora_manager is None:
                raise RuntimeError("LoRA manager is not initialized")

            # Check if the LoRA adapter exists
            adapter_ids = self.model_runner.lora_manager.list_adapters()
            if lora_int_id not in adapter_ids:
                raise RuntimeError(
                    f"LoRA adapter {lora_int_id} not found. Available: {adapter_ids}"
                )

            # Get the currently registered LoRA model (used for diagnostics).
            lora_model = (
                self.model_runner.lora_manager._adapter_manager._registered_adapters[
                    lora_int_id
                ]
            )
            logger.info(f"Found LoRA model with {len(lora_model.loras)} LoRA modules")

            # Receive all weights via XCCL broadcast
            logger.info(f"Receiving {len(names)} LoRA parameters via XCCL")
            received_weights = {}
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )

                tensor = torch.empty(
                    shape, dtype=target_dtype, device=self.model_runner.device
                )

                torch.distributed.broadcast(
                    tensor,
                    src=0,
                    group=group,
                    async_op=False,
                )

                received_weights[name] = tensor.cpu()

            logger.info(f"Received {len(received_weights)} LoRA parameters via XCCL")

            normalized_weights = {
                k.replace("default.", ""): v for k, v in received_weights.items()
            }

            lora_partial_shard_key = (self.areal_lora_name, lora_int_id)

            group_shards = self._lora_partial_shards.setdefault(
                lora_partial_shard_key, {}
            )
            group_shards[self.areal_weight_meta_group_name] = normalized_weights
            buffered_count = len(group_shards)

            # Assumes that every registered weight update group contributes to the update cycle
            if buffered_count < len(self.weight_update_groups):
                logger.info(
                    "Buffered LoRA shard for "
                    f"{self.areal_lora_name}: group={self.areal_weight_meta_group_name}, "
                    f"buffered={buffered_count}/{len(self.weight_update_groups)} PP stages."
                )
                self.sync()
                return True, "Success"

            merged_weights: dict[str, torch.Tensor] = {}
            for shard in group_shards.values():
                merged_weights.update(shard)
            self._lora_partial_shards.pop(lora_partial_shard_key, None)

            peft_config = {
                "r": self.areal_lora_rank,
                "lora_alpha": self.areal_lora_alpha,
                "target_modules": self.areal_lora_target_modules,
                "bias": self.areal_lora_bias,
            }
            peft_helper = PEFTHelper.from_dict(peft_config)

            extra_vocab = getattr(
                self.model_runner.lora_manager.lora_config,
                "lora_extra_vocab_size",
                0,
            )
            model_vocab_size = self.model_runner.lora_manager.vocab_size + extra_vocab

            new_lora_model = LoRAModel.from_lora_tensors(
                lora_model_id=self.areal_lora_int_id,
                tensors=merged_weights,
                peft_helper=peft_helper,
                device="cpu",
                dtype=self.model_runner.lora_manager.lora_config.lora_dtype,
                model_vocab_size=model_vocab_size,
                weights_mapper=getattr(
                    self.model_runner.model, "hf_to_vllm_mapper", None
                ),
            )

            self.model_runner.lora_manager.remove_adapter(lora_int_id)

            with set_current_vllm_config(self.model_runner.vllm_config):
                self.model_runner.lora_manager._adapter_manager._add_adapter(
                    new_lora_model
                )
                self.model_runner.lora_manager._adapter_manager.activate_adapter(
                    new_lora_model.id
                )
            logger.info(
                f"Updated New LoRA model with {len(new_lora_model.loras)} LoRA modules "
                f"from {len(merged_weights)} tensors across {len(self.weight_update_groups)} groups"
            )
            if len(new_lora_model.loras) != len(lora_model.loras):
                logger.warning(
                    f"Number of modules in the new LoRA model ({len(new_lora_model.loras)}) "
                    f"does not match the old LoRA model ({len(lora_model.loras)})."
                )

            self.sync()
            return True, "Success"

        except Exception as e:
            error_msg = f"Failed to update LoRA parameter via XCCL!   {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg

    def areal_init_update_weight_group(
        self,
        master_address: str,
        master_port: str,
        rank_offset: int,
        world_size: int,
        backend: str,
        group_name: str,
    ):
        # [FUNCTION DOC]
        # Join a custom process group with the trainer: lazily creates
        # weight_update_groups and, when LoRA is enabled, _lora_partial_shards; calls
        # init_custom_process_group with the backend, world_size,
        # tcp://master_address:master_port init method, rank = worker rank +
        # rank_offset, the group name and the default timeout (blocks until all members
        # join), and stores the group. Returns (success, message).
        if not hasattr(self, "weight_update_groups"):
            self.weight_update_groups: dict[str, dist.ProcessGroup] = {}

        # This is required for buffering weights during lora weight update, as vLLM
        # expects the partial PP shards to be buffered until all groups have sent their shards.
        _is_vllm_lora_enabled = (
            getattr(self.model_runner, "lora_manager", None) is not None
        )
        if _is_vllm_lora_enabled and not hasattr(self, "_lora_partial_shards"):
            # (lora_name, lora_int_id) -> group_name -> normalized weight dict
            self._lora_partial_shards: dict[
                tuple[str, int], dict[str, dict[str, torch.Tensor]]
            ] = {}

        try:
            group = init_custom_process_group(
                backend=backend,
                world_size=world_size,
                init_method=f"tcp://{master_address}:{master_port}",
                rank=self.rank + rank_offset,
                group_name=group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.weight_update_groups[group_name] = group
            return True, "Success"
        except Exception as e:
            error_msg = f"Failed to init group! {e}."
            logger.error(error_msg)
            return False, error_msg

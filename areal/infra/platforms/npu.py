# [MODULE DOC]
# Huawei Ascend NPU platform implementation using torch.npu and the hccl backend.
# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("NPUPlatform")


class NPUPlatform(Platform):
    # [CLASS DOC]
    # Platform for Ascend NPUs. Class attributes: device_name/dispatch_key/ray key
    # "NPU", device_type "npu", visibility env var ASCEND_RT_VISIBLE_DEVICES, Ray noset
    # flag RAY_EXPERIMENTAL_NOSET_NPU_VISIBLE_DEVICES, backend "hccl". Does not override
    # clear_cublas_workspaces, get_vllm_worker_class, set_allocator_settings or
    # get_custom_env_vars, so those raise NotImplementedError from the base class.
    device_name: str = "NPU"
    device_type: str = "npu"
    dispatch_key: str = "NPU"
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_NPU_VISIBLE_DEVICES"
    communication_backend: str = "hccl"

    @classmethod
    def synchronize(cls) -> None:
        # [FUNCTION DOC]
        # Blocks until queued NPU work completes (torch.npu.synchronize).
        torch.npu.synchronize()

    def clear_memory(self) -> None:
        # [FUNCTION DOC]
        # Synchronizes the NPU, runs garbage collection, then empties the NPU cache.
        self.synchronize()
        gc.collect()
        torch.npu.empty_cache()

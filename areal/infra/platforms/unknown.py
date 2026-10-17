# [MODULE DOC]
# Fallback platform for CUDA-compatible devices whose name does not contain "NVIDIA";
# behaves like the CUDA platform without NUMA affinity or allocator env tuning.
# SPDX-License-Identifier: Apache-2.0

import gc

import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("UnknownPlatform")


class UnknownPlatform(Platform):
    # [CLASS DOC]
    # Platform with device_name "UNKNOWN" but CUDA settings: device_type "cuda",
    # dispatch key "CUDA", Ray key "GPU", CUDA_VISIBLE_DEVICES visibility and nccl
    # backend.
    device_name: str = "UNKNOWN"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    def clear_memory(self) -> None:
        # [FUNCTION DOC]
        # Synchronizes CUDA, runs garbage collection, then empties the CUDA cache.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        # [FUNCTION DOC]
        # Calls the private torch C hook that frees cuBLAS workspaces.
        torch._C._cuda_clearCublasWorkspaces()

    @classmethod
    def get_vllm_worker_class(clas):
        # [FUNCTION DOC]
        # Returns vLLM's Worker class, trying the V1 gpu_worker module then the V0
        # module; logs the result. Raises RuntimeError (chained) if neither import
        # works.
        # KNOWN ISSUE: First parameter is named "clas" instead of "cls" (harmless typo).
        try:
            from vllm.v1.worker.gpu_worker import Worker

            logger.info("Successfully imported vLLM V1 Worker.")
            return Worker
        except ImportError:
            pass

        try:
            from vllm.worker.worker import Worker

            logger.info("Successfully imported vLLM V0 Worker.")
            return Worker
        except ImportError as e:
            logger.error(
                "Failed to import vLLM Worker. "
                "Make sure vLLM is installed correctly: %s",
                e,
            )
            raise RuntimeError(
                "vLLM is not installed or not properly configured."
            ) from e

    @classmethod
    def set_allocator_settings(cls) -> None:
        # [FUNCTION DOC]
        # Sets the CUDA caching allocator to expandable_segments:False.
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        # [FUNCTION DOC]
        # Returns {"TORCHINDUCTOR_COMPILE_THREADS": "2"}.
        env_vars = {
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
        }
        return env_vars

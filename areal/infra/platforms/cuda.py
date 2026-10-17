# [MODULE DOC]
# NVIDIA CUDA platform implementation: nccl backend, CUDA_VISIBLE_DEVICES visibility,
# vLLM worker class lookup, allocator/NUMA configuration.
# SPDX-License-Identifier: Apache-2.0

import gc
import os

import torch

import areal.utils.logging as logging

from .platform import Platform

logger = logging.getLogger("CUDAPlatform")


class CudaPlatform(Platform):
    # [CLASS DOC]
    # Platform for NVIDIA GPUs. Class attributes: device_name "NVIDIA", device_type
    # "cuda", dispatch_key "CUDA", ray_device_key "GPU", device_control_env_var
    # CUDA_VISIBLE_DEVICES, Ray noset flag RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES,
    # backend "nccl". Unknown attributes fall through to torch.cuda via the base class.
    device_name: str = "NVIDIA"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_experimental_noset: str = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    communication_backend: str = "nccl"

    def clear_memory(self) -> None:
        # [FUNCTION DOC]
        # Synchronizes CUDA, runs Python garbage collection, then empties the CUDA
        # cache.
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
        # Returns vLLM's Worker class: tries the V1 gpu_worker module first, then the V0
        # worker module, logging which succeeded. If both imports fail, logs an error
        # and raises RuntimeError chained from the ImportError.
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
        # Sets the CUDA caching allocator to expandable_segments:False at runtime.
        # KNOWN ISSUE: Conflicts with get_custom_env_vars, which sets
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        torch.cuda.memory._set_allocator_settings("expandable_segments:False")

    @classmethod
    def set_numa_affinity(cls, local_rank: int) -> None:
        # [FUNCTION DOC]
        # Binds the current process's CPU affinity to cores local to GPU local_rank
        # using pynvml (init, get handle by index, set CPU affinity), then logs the
        # number of cores bound. Missing pynvml or any other failure is logged as a
        # warning, never raised. Always shuts down NVML if it was initialized.
        """Bind the current process to CPU cores local to the assigned GPU."""

        nvml_initialized = False
        try:
            import pynvml

            pynvml.nvmlInit()
            nvml_initialized = True
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)
            cpu_set = os.sched_getaffinity(0)
            logger.info(
                "Set NUMA affinity for GPU %s: bound to %s CPU cores.",
                local_rank,
                len(cpu_set),
            )
        except ImportError:
            logger.warning(
                "pynvml (nvidia-ml-py) not available, skipping NUMA affinity setup."
            )
        except Exception as e:
            logger.warning("Failed to set NUMA affinity for GPU %s: %s", local_rank, e)
        finally:
            if nvml_initialized:
                pynvml.nvmlShutdown()

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        # [FUNCTION DOC]
        # Returns env vars for CUDA workers: TORCHINDUCTOR_COMPILE_THREADS=2 and
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
        env_vars = {
            # "RAY_DEBUG": "legacy"
            "TORCHINDUCTOR_COMPILE_THREADS": "2",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        return env_vars

    @classmethod
    def synchronize(cls) -> None:
        # [FUNCTION DOC]
        # Blocks until all CUDA kernels on the current device complete.
        torch.cuda.synchronize()

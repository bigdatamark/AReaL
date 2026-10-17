# [MODULE DOC]
# CPU fallback platform: no accelerator, gloo communication backend, and no-op memory
# and synchronization operations.
# SPDX-License-Identifier: Apache-2.0

from .platform import Platform


class CpuPlatform(Platform):
    # [CLASS DOC]
    # Platform implementation for CPU-only hosts. Class attributes: device_name "CPU",
    # device_type "cpu", dispatch_key "CPU", ray_device_key "CPU", empty visibility env
    # var and Ray noset flag, communication_backend "gloo". All device memory queries
    # return zeros.
    device_name: str = "CPU"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    ray_device_key: str = "CPU"
    device_control_env_var: str = ""
    ray_experimental_noset: str = ""
    communication_backend: str = "gloo"

    def clear_memory(self) -> None:
        # [FUNCTION DOC]
        # No-op.
        pass

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        # [FUNCTION DOC]
        # No-op class method.
        pass

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        # [FUNCTION DOC]
        # Returns an empty dict (no extra env vars for CPU).
        return {}

    @classmethod
    def synchronize(cls) -> None:
        # [FUNCTION DOC]
        # No-op class method.
        pass

    @classmethod
    def update_env_vars_for_visible_devices(
        cls, env_vars: dict, gpu_ranks: list
    ) -> None:
        # [FUNCTION DOC]
        # No-op; env_vars is left unchanged regardless of gpu_ranks.
        # No-op for CPU platform
        pass

    @classmethod
    def get_visible_devices(cls) -> list:
        # [FUNCTION DOC]
        # Returns an empty list (no devices).
        # No-devices for CPU platform
        return []

    def memory_allocated(self) -> int:
        # [FUNCTION DOC]
        # Returns 0.
        """No device memory; return 0."""
        return 0

    def memory_reserved(self) -> int:
        # [FUNCTION DOC]
        # Returns 0.
        """No device memory; return 0."""
        return 0

    def mem_get_info(self) -> tuple[int, int]:
        # [FUNCTION DOC]
        # Returns (0, 0) as (free, total).
        """No device memory; return (0, 0)."""
        return (0, 0)

    def empty_cache(self) -> None:
        # [FUNCTION DOC]
        # No-op.
        """No-op."""
        pass

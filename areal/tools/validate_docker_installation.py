#!/usr/bin/env python3
# [MODULE DOC]
# CLI script validating the AReaL Docker image. Extends BaseInstallationValidator
# (imported as top-level module validation_base, so run with areal/tools on sys.path)
# with packages built in the Dockerfile (flash-attn, grouped_gemm, apex,
# transformer_engine, flash_attn_3, Megatron stack, DeepSeek-V3 kernels, fla,
# causal_conv1d), detection of the sglang-vs-vllm image variant, and extra CUDA smoke
# tests. Exits 0 on success, 1 on failure.
# SPDX-License-Identifier: Apache-2.0

"""
Docker Installation Validation Script for AReaL

This script validates dependencies in the Docker environment, which includes
additional packages (grouped_gemm, apex, transformer_engine, flash_attn_3)
and flash-attn version 2.8.3 (installed separately in Dockerfile).

Also validates DeepSeek-V3 related packages:
- FlashMLA: Multi-head Latent Attention (requires SM90+ GPU)
- DeepGEMM: FP8 GEMM library (requires SM90+ GPU)
- DeepEP: Expert Parallelism communication (requires SM80+ GPU)
- flash-linear-attention (fla): Linear attention with Triton kernels
"""

import sys
from pathlib import Path

from validation_base import BaseInstallationValidator


class DockerInstallationValidator(BaseInstallationValidator):
    # [CLASS DOC]
    # Docker-specific validator. Extends CUDA_SUBMODULES with Docker packages' extension
    # modules and CRITICAL_PACKAGES with Docker build packages. sglang/vllm criticality
    # is decided at runtime in parse_pyproject.
    # KNOWN ISSUE: CRITICAL_PACKAGES entries use underscores (e.g. grouped_gemm) while
    # the base class compares against dash-normalized names; this only matters for
    # pyproject deps, since additional packages use their own required flag.
    """Validates installation in Docker environment with additional packages."""

    # Extend CUDA sub-modules with Docker-specific packages
    CUDA_SUBMODULES = {
        **BaseInstallationValidator.CUDA_SUBMODULES,
        "grouped_gemm": ["grouped_gemm"],
        "apex": ["apex.optimizers", "apex.normalization"],
        "transformer_engine": ["transformer_engine.pytorch"],
        "flash_attn_3": ["flash_attn_3"],
        "vllm": ["vllm._C"],
        "sglang": ["sgl_kernel", "sgl_kernel.flash_attn"],
        "megatron-core": [
            "megatron.core.parallel_state",
            "megatron.core.tensor_parallel",
        ],
        # DeepSeek-V3 related packages
        "flash_mla": ["flash_mla"],
        "deep_gemm": ["deep_gemm"],
        "deep_ep": ["deep_ep"],
        "fla": ["fla.ops", "fla.layers", "fla.modules"],
        "causal_conv1d": ["causal_conv1d_cuda"],
    }

    # Add Docker-specific packages to critical list
    # Note: sglang/vllm are NOT listed here — they are mutually exclusive
    # and are added dynamically in parse_pyproject() based on what's installed
    CRITICAL_PACKAGES = {
        *BaseInstallationValidator.CRITICAL_PACKAGES,
        "grouped_gemm",
        "apex",
        "transformer_engine",
        "flash_attn_3",
        "megatron-core",
        "mbridge",
        "megatron-bridge",
        "causal_conv1d",
    }

    def __init__(self, pyproject_path: Path | None = None):
        # [FUNCTION DOC]
        # Delegates to the base constructor with the pyproject path.
        super().__init__(pyproject_path)

    @staticmethod
    def _is_package_importable(name: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: Static; returns True if importing name succeeds, False on ImportError.
        # Side effect: actually imports the package. Non-import exceptions propagate.
        """Check if a package can be actually imported (not just spec-found)."""
        try:
            __import__(name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def parse_pyproject(self):
        # [FUNCTION DOC]
        # Runs the base parse, then registers additional packages with version pins
        # taken from optional-dependencies where available: flash-attn (default ==2.8.3,
        # required), grouped_gemm/apex/transformer_engine (required), flash_attn_3
        # (optional). Detects the inference backend by importing sglang and vllm: both
        # present or neither present appends a critical failure; the present one is
        # added as required and added to an instance-level CRITICAL_PACKAGES copy
        # (sglang also adds optional nvidia-cudnn-cu12); the absent one is added as
        # optional. Then adds megatron-core, mbridge, megatron-bridge, kernels
        # (required), torch_memory_saver, flash_mla, deep_gemm, deep_ep (optional), fla
        # (required) and causal_conv1d ==1.6.0 (required, hardcoded pin). Prints notes.
        # KNOWN ISSUE: flash-attn may also appear in pyproject dependencies, in which
        # case it is validated twice (once as dependency, once as additional package).
        """Parse pyproject.toml and add Docker-specific packages."""
        super().parse_pyproject()

        # Version pins are read from pyproject.toml optional-dependencies
        # so they stay in sync automatically.
        opt_versions = self._get_optional_dep_versions()

        # flash-attn (pre-built wheel installed in Dockerfile)
        fa_spec = opt_versions.get("flash-attn", "==2.8.3")
        self.add_additional_package("flash-attn", fa_spec, required=True)
        print(f"  Note: Expecting flash-attn {fa_spec} for Docker environment")

        # C++ packages built from source in Dockerfile (not in pyproject.toml)
        self.add_additional_package("grouped_gemm", required=True)
        self.add_additional_package("apex", required=True)
        self.add_additional_package("transformer_engine", required=True)
        self.add_additional_package("flash_attn_3", required=False)

        # Auto-detect which inference backend variant is installed.
        # Each Docker image has exactly one of sglang or vllm.
        has_sglang = self._is_package_importable("sglang")
        has_vllm = self._is_package_importable("vllm")

        if has_sglang and has_vllm:
            print(
                "  ⚠ ERROR: Both sglang and vllm detected"
                " — Docker image should have exactly one"
            )
            self.critical_failures.append(
                "Both sglang and vllm installed (should be mutually exclusive)"
            )

        if has_sglang:
            self.add_additional_package(
                "sglang", opt_versions.get("sglang", ""), required=True
            )
            self.CRITICAL_PACKAGES = {*self.CRITICAL_PACKAGES, "sglang"}
            print("  Detected variant: sglang")
            self.add_additional_package(
                "nvidia-cudnn-cu12",
                opt_versions.get("nvidia-cudnn-cu12", ""),
                required=False,
            )
        else:
            self.add_additional_package("sglang", required=False)

        if has_vllm:
            self.add_additional_package(
                "vllm", opt_versions.get("vllm", ""), required=True
            )
            self.CRITICAL_PACKAGES = {*self.CRITICAL_PACKAGES, "vllm"}
            print("  Detected variant: vllm")
        else:
            self.add_additional_package("vllm", required=False)

        if not has_sglang and not has_vllm:
            print(
                "  ⚠ ERROR: Neither sglang nor vllm detected"
                " — Docker image must have exactly one"
            )
            self.critical_failures.append(
                "No inference backend installed (need either sglang or vllm)"
            )

        # Megatron packages (from cuda-train > megatron extra)
        self.add_additional_package(
            "megatron-core",
            opt_versions.get("megatron-core", ""),
            required=True,
        )
        self.add_additional_package(
            "mbridge", opt_versions.get("mbridge", ""), required=True
        )
        self.add_additional_package(
            "megatron-bridge",
            opt_versions.get("megatron-bridge", ""),
            required=True,
        )

        # Training packages (from cuda-train extra)
        self.add_additional_package(
            "torch_memory_saver",
            opt_versions.get("torch-memory-saver", ""),
            required=False,
        )
        self.add_additional_package(
            "kernels", opt_versions.get("kernels", ""), required=True
        )

        # DeepSeek-V3 related packages (built from source in Dockerfile)
        self.add_additional_package("flash_mla", required=False)  # SM90+ only
        self.add_additional_package("deep_gemm", required=False)  # SM90+ only
        self.add_additional_package("deep_ep", required=False)  # SM80+ only
        self.add_additional_package(
            "fla", required=True
        )  # Pure Triton, works everywhere

        # Mamba-related packages (built from source in Dockerfile)
        self.add_additional_package("causal_conv1d", "==1.6.0", required=True)

    def test_cuda_functionality(self):
        # [FUNCTION DOC]
        # Runs the base CUDA tests, then (returning early if CUDA unavailable) a
        # Transformer Engine FP8 Linear forward/backward under DelayedScaling E4M3, an
        # Apex FusedAdam step on a small Linear, import checks for flash_attn_3,
        # grouped_gemm, causal_conv1d_cuda, flash_mla, deep_gemm (prints SM count) and
        # deep_ep, and an fla MultiScaleRetention bf16 forward on cuda:0 checking output
        # shape (import-only if no CUDA). All failures are printed only and not recorded
        # as failures or warnings. Uses GPU memory.
        """Run CUDA functionality tests including Docker-specific packages."""
        super().test_cuda_functionality()

        print("\n=== Docker-Specific CUDA Tests ===")

        # Test transformer engine FP8 if available
        try:
            import torch

            if not torch.cuda.is_available():
                print("⚠ CUDA not available - skipping transformer engine tests")
                return

            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            # Set dimensions for a small test
            in_features = 128
            out_features = 256
            hidden_size = 64

            # Initialize model and inputs
            model = te.Linear(in_features, out_features, bias=True)
            inp = torch.randn(hidden_size, in_features, device="cuda")

            # Create an FP8 recipe
            fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

            # Enable autocasting for the forward pass
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(inp)

            loss = out.sum()
            loss.backward()
            print("✓ Transformer Engine FP8 operations")

        except ImportError:
            print("⚠ Transformer Engine not available - skipping FP8 tests")
        except Exception as e:
            print(f"⚠ Transformer Engine FP8 test failed: {e}")

        # Test Apex fused optimizers if available
        try:
            import torch
            from apex.optimizers import FusedAdam

            # Create a simple model and optimizer
            model = torch.nn.Linear(10, 10).cuda()
            optimizer = FusedAdam(model.parameters(), lr=0.001)

            # Test a forward-backward pass
            x = torch.randn(5, 10, device="cuda")
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            print("✓ Apex FusedAdam optimizer")

        except ImportError:
            print("⚠ Apex not available - skipping Apex tests")
        except Exception as e:
            print(f"⚠ Apex optimizer test failed: {e}")

        # Test flash_attn_3 if available
        try:
            import flash_attn_3  # noqa: F401

            print("✓ Flash Attention 3 (Hopper) imported successfully")
        except ImportError:
            print("⚠ Flash Attention 3 not available (optional for Hopper GPUs)")

        # Test grouped_gemm if available
        try:
            import grouped_gemm  # noqa: F401

            print("✓ Grouped GEMM imported successfully")
        except ImportError:
            print("⚠ Grouped GEMM not available")
        except Exception as e:
            print(f"⚠ Grouped GEMM test failed: {e}")

        # Test causal_conv1d CUDA extension
        try:
            import causal_conv1d_cuda  # noqa: F401

            print("✓ causal-conv1d CUDA extension imported successfully")
        except ImportError:
            print("⚠ causal-conv1d CUDA extension not available")
        except Exception as e:
            print(f"⚠ causal-conv1d CUDA extension test failed: {e}")

        print("\n=== DeepSeek-V3 Package Tests ===")

        # Test FlashMLA (requires SM90+ GPU)
        try:
            from flash_mla import flash_mla_with_kvcache, get_mla_metadata  # noqa: F401

            print(
                "✓ FlashMLA imported successfully "
                "(functions: get_mla_metadata, flash_mla_with_kvcache)"
            )
        except ImportError:
            print("⚠ FlashMLA not available (requires SM90+ GPU)")
        except Exception as e:
            print(f"⚠ FlashMLA import failed: {e}")

        # Test DeepGEMM (requires SM90+ GPU)
        try:
            import deep_gemm

            num_sms = deep_gemm.get_num_sms()
            print(f"✓ DeepGEMM imported successfully (detected {num_sms} SMs)")
        except ImportError:
            print("⚠ DeepGEMM not available (requires SM90+ GPU)")
        except Exception as e:
            print(f"⚠ DeepGEMM test failed: {e}")

        # Test DeepEP (requires SM80+ GPU and NVSHMEM)
        try:
            from deep_ep import Buffer, EventOverlap  # noqa: F401

            print("✓ DeepEP imported successfully (classes: Buffer, EventOverlap)")
        except ImportError:
            print("⚠ DeepEP not available (requires SM80+ GPU and NVSHMEM)")
        except Exception as e:
            print(f"⚠ DeepEP import failed: {e}")

        # Test flash-linear-attention (fla) with actual layer instantiation
        try:
            import fla  # noqa: F401
            import torch
            from fla.layers import MultiScaleRetention

            if torch.cuda.is_available():
                device, dtype = "cuda:0", torch.bfloat16
                retnet = MultiScaleRetention(hidden_size=1024, num_heads=4).to(
                    device=device, dtype=dtype
                )
                x = torch.randn(1, 64, 1024, device=device, dtype=dtype)
                y, *_ = retnet(x)
                assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
                print(
                    "✓ flash-linear-attention (fla) - MultiScaleRetention forward pass OK"
                )
            else:
                print(
                    "✓ flash-linear-attention (fla) imported successfully "
                    "(CUDA not available for layer test)"
                )
        except ImportError:
            print("⚠ flash-linear-attention (fla) not available")
        except Exception as e:
            print(f"⚠ flash-linear-attention (fla) test failed: {e}")

    def get_validation_title(self) -> str:
        # [FUNCTION DOC]
        # Returns "AReaL Docker Installation Validation".
        """Get the title for validation output."""
        return "AReaL Docker Installation Validation"


def _detect_pyproject(project_root: Path) -> Path:
    # [FUNCTION DOC]
    # HELPER: Returns project_root/pyproject.vllm.toml if vllm is importable and that
    # file exists, otherwise project_root/pyproject.toml (existence not checked). Side
    # effect: imports vllm.
    """Auto-detect the correct pyproject file based on installed variant."""
    vllm_path = project_root / "pyproject.vllm.toml"
    default_path = project_root / "pyproject.toml"

    try:
        __import__("vllm")
        if vllm_path.exists():
            return vllm_path
    except (ImportError, ModuleNotFoundError):
        pass

    return default_path


def main():
    # [FUNCTION DOC]
    # Entry point: resolves the repo root two levels above this script, picks the
    # pyproject file via _detect_pyproject, exits 1 if it is missing, prints its name,
    # runs the Docker validator and exits 0/1 by result.
    """Main entry point."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    pyproject_path = _detect_pyproject(project_root)

    if not pyproject_path.exists():
        print(f"Error: pyproject not found at {pyproject_path}")
        sys.exit(1)

    print(f"Using: {pyproject_path.name}")
    validator = DockerInstallationValidator(pyproject_path)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

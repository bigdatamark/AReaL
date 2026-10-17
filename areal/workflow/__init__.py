# [PACKAGE DOC]
# Rollout workflow implementations (RLVR, multi-turn, vision RLVR) plus agent-framework
# example workflows in subpackages. Top-level workflow classes are lazily imported on
# first attribute access to avoid heavy imports.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "RLVRWorkflow",
    "MultiTurnWorkflow",
    "VisionRLVRWorkflow",
]

_LAZY_IMPORTS = {
    "RLVRWorkflow": "areal.workflow.rlvr",
    "MultiTurnWorkflow": "areal.workflow.multi_turn",
    "VisionRLVRWorkflow": "areal.workflow.vision_rlvr",
}


def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module __getattr__: for names in the lazy-import table imports the target module,
    # caches the attribute in module globals and returns it; otherwise raises
    # AttributeError.
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # [FUNCTION DOC]
    # Returns the public names (__all__) for dir().
    return list(__all__)

# [PACKAGE DOC]
# Engine package. Exposes FSDP, Megatron and remote inference engine classes (listed in
# __all__) via lazy module-level attribute access, so importing areal.engine does not
# pull in heavy backends until a class is actually requested.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "FSDPEngine",
    "FSDPPPOActor",
    "FSDPPPOCritic",
    "FSDPLMEngine",
    "FSDPRWEngine",
    "FSDPDPOEngine",
    "MegatronEngine",
    "MegatronPPOActor",
    "MegatronPPOCritic",
    "MegatronLMEngine",
    "MegatronRWEngine",
    "MegatronDPOEngine",
    "RemoteSGLangEngine",
    "RemotevLLMEngine",
]

_LAZY_IMPORTS = {
    "FSDPEngine": "areal.engine.fsdp_engine",
    "FSDPPPOActor": "areal.engine.fsdp_engine",
    "FSDPPPOCritic": "areal.engine.fsdp_engine",
    "FSDPLMEngine": "areal.engine.fsdp_engine",
    "FSDPRWEngine": "areal.engine.fsdp_engine",
    "FSDPDPOEngine": "areal.engine.fsdp_engine",
    "MegatronEngine": "areal.engine.megatron_engine",
    "MegatronPPOActor": "areal.engine.megatron_engine",
    "MegatronPPOCritic": "areal.engine.megatron_engine",
    "MegatronLMEngine": "areal.engine.megatron_engine",
    "MegatronRWEngine": "areal.engine.megatron_engine",
    "MegatronDPOEngine": "areal.engine.megatron_engine",
    "RemoteSGLangEngine": "areal.engine.sglang_remote",
    "RemotevLLMEngine": "areal.engine.vllm_remote",
}


def __getattr__(name: str):
    # [FUNCTION DOC]
    # Module-level PEP 562 hook. If name is in the lazy-import table, imports the
    # backing module, fetches the attribute, caches it in the package globals (so later
    # lookups bypass this hook) and returns it. Otherwise raises AttributeError. Side
    # effect: imports heavy backend modules on first access.
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # [FUNCTION DOC]
    # HELPER: Returns the list of public names from __all__ for dir() / autocompletion.
    return list(__all__)

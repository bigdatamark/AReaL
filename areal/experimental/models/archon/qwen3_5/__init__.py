# [PACKAGE DOC]
# Archon implementation of Qwen3.5 (dense and MoE, hybrid linear-attention /
# full-attention layers). Re-exports the model class, args dataclass, HF state-dict
# adapter and the parallelization function. Importing has no side effects beyond
# importing those submodules (spec registration lives in qwen3_5/spec.py).
# SPDX-License-Identifier: Apache-2.0

from areal.experimental.models.archon.qwen3_5.infra.parallelize import (
    parallelize_qwen3_5,
)
from areal.experimental.models.archon.qwen3_5.model.args import Qwen3_5ModelArgs
from areal.experimental.models.archon.qwen3_5.model.model import Qwen3_5Model
from areal.experimental.models.archon.qwen3_5.model.state_dict_adapter import (
    Qwen3_5StateDictAdapter,
)

__all__ = [
    "Qwen3_5Model",
    "Qwen3_5ModelArgs",
    "Qwen3_5StateDictAdapter",
    "parallelize_qwen3_5",
]

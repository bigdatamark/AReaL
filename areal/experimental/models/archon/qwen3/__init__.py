# [PACKAGE DOC]
# Archon implementation of the Qwen3 family (dense and MoE). Re-exports the model, args,
# HF state-dict adapter and parallelize function; spec registration lives in qwen3/spec.py.
# SPDX-License-Identifier: Apache-2.0

from areal.experimental.models.archon.qwen3.infra.parallelize import (
    parallelize_qwen3,
)
from areal.experimental.models.archon.qwen3.model.args import Qwen3ModelArgs
from areal.experimental.models.archon.qwen3.model.model import Qwen3Model
from areal.experimental.models.archon.qwen3.model.state_dict_adapter import (
    Qwen3StateDictAdapter,
)

__all__ = [
    "Qwen3Model",
    "Qwen3ModelArgs",
    "Qwen3StateDictAdapter",
    "parallelize_qwen3",
]

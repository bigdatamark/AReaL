# [PACKAGE DOC]
# Archon implementation of the Qwen2 dense decoder family. Re-exports the model, its args,
# the HF state-dict adapter and the parallelize function. (Spec registration lives in
# qwen2/spec.py, imported by the archon package.)
# SPDX-License-Identifier: Apache-2.0

from areal.experimental.models.archon.qwen2.infra.parallelize import (
    parallelize_qwen2,
)
from areal.experimental.models.archon.qwen2.model.args import Qwen2ModelArgs
from areal.experimental.models.archon.qwen2.model.model import Qwen2Model
from areal.experimental.models.archon.qwen2.model.state_dict_adapter import (
    Qwen2StateDictAdapter,
)

__all__ = [
    "Qwen2Model",
    "Qwen2ModelArgs",
    "Qwen2StateDictAdapter",
    "parallelize_qwen2",
]

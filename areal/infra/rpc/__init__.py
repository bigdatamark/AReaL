# [PACKAGE DOC]
# RPC package for remote communication between controllers and workers. Re-exports
# RTensor/TensorShardInfo (remote tensor handles) and serialize_value/deserialize_value
# (JSON-safe encoding of tensors and other objects).
# SPDX-License-Identifier: Apache-2.0

"""RPC modules for remote communication."""

from .rtensor import RTensor, TensorShardInfo
from .serialization import deserialize_value, serialize_value

__all__ = [
    "RTensor",
    "TensorShardInfo",
    "deserialize_value",
    "serialize_value",
]

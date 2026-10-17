# [MODULE DOC]
# Process-wide random seeding utilities. Module globals hold the current seed
# (_SEED), the base seed (_BASE_SEED) and a lazily created Shuffler (_SHUFFLER)
# that yields a deterministic sequence of dataset shuffle seeds.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import random

import numpy as np
import torch
import transformers

from areal.infra.platforms import current_platform

_SEED = None
_BASE_SEED = None
_SHUFFLER = None


def _seed_from_key(key: str) -> int:
    # [FUNCTION DOC]
    # HELPER: Hashes key with SHA-256 and returns the low 32 bits as a non-negative int.
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF


def derive_deterministic_seed(identity: str, request_index: int) -> int:
    # [FUNCTION DOC]
    # Returns a stable seed in [0, 2^31) from the first 4 bytes (big-endian) of
    # SHA-256("<identity>:<request_index>"). Raises ValueError for negative
    # request_index. Pure.
    """Derive a stable non-negative sampling seed from a logical request identity."""
    if request_index < 0:
        raise ValueError(f"request_index must be non-negative, got {request_index}")
    digest = hashlib.sha256(f"{identity}:{request_index}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) & 0x7FFFFFFF


def set_random_seed(base_seed: int, key: str) -> None:
    # [FUNCTION DOC]
    # Sets the process seed to base_seed + hash(key) (key distinguishes ranks/roles),
    # stores base and derived seeds in module globals, sets PYTHONHASHSEED env var and
    # seeds transformers, random, numpy, torch and the current accelerator platform
    # (AttributeError from the platform is ignored). May initialize CUDA.
    global _SEED, _BASE_SEED
    _BASE_SEED = base_seed
    seed = base_seed + _seed_from_key(key)
    _SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # FIXME: seeding initializes CUDA
    try:
        current_platform.manual_seed(seed)
    except AttributeError:
        pass


def get_seed() -> int:
    # [FUNCTION DOC]
    # Returns the seed set by set_random_seed; raises ValueError if unset.
    global _SEED
    if _SEED is None:
        raise ValueError("Random seed is not set. Please call set_random_seed first.")
    return _SEED


class Shuffler:
    # [CLASS DOC]
    # Deterministic seed generator: each call hashes "<base_key>_<counter>" and
    # increments the counter.
    def __init__(self, key="default"):
        # [FUNCTION DOC]
        # Initializes counter to 0 and stores the base key (default "default").
        self.cnt = 0
        self.base_key = key

    def next_shuffle(self) -> int:
        # [FUNCTION DOC]
        # Returns the 32-bit hash seed for the current counter value and increments it.
        shuffle_key = f"{self.base_key}_{self.cnt}"
        self.cnt += 1
        return _seed_from_key(shuffle_key)


def get_shuffle_seed() -> int:
    # [FUNCTION DOC]
    # Returns the next seed from a global Shuffler keyed "AReaL-seed<base seed>",
    # creating it on first use.
    # KNOWN ISSUE: if called before set_random_seed, the shuffler is permanently keyed
    # with "None" as the base seed.
    global _BASE_SEED, _SHUFFLER
    if _SHUFFLER is None:
        _SHUFFLER = Shuffler(f"AReaL-seed{_BASE_SEED}")
    return _SHUFFLER.next_shuffle()

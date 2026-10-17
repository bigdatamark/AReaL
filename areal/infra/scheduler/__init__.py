# [PACKAGE DOC]
# Scheduler package exposing LocalScheduler, RayScheduler and SlurmScheduler, which
# launch and manage worker processes across local, Ray and Slurm backends. Importing it
# imports all three backend modules.
# SPDX-License-Identifier: Apache-2.0

from .local import LocalScheduler
from .ray import RayScheduler
from .slurm import SlurmScheduler

__all__ = [
    "LocalScheduler",
    "RayScheduler",
    "SlurmScheduler",
]

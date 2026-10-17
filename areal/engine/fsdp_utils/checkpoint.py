# [MODULE DOC]
# Distributed Checkpoint (DCP) adapter for FSDP models: a Stateful wrapper so
# dcp.save/dcp.load can (de)serialize sharded model and optimizer state.
# SPDX-License-Identifier: Apache-2.0

"""FSDP checkpointing utilities for DCP (Distributed Checkpoint) integration."""

from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful


class DCPState(Stateful):
    # [CLASS DOC]
    # Stateful holder of a model and optional optimizer; DCP calls its
    # state_dict/load_state_dict during save/load. Attributes: model (nn.Module),
    # optimizer (Optimizer or None).
    """Wrapper for checkpointing the State using DCP.

    This class implements the Stateful protocol, so DCP will automatically call
    state_dict/load_state_dict as needed in the dcp.save/load APIs.

    It handles calling distributed state dict methods on the model and optimizer.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
    ):
        # [FUNCTION DOC]
        # Stores model and optional optimizer. No side effects.
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns {'model': ..., 'optim': ...} from torch's get_state_dict when an
        # optimizer is set, else {'model': ...} from get_model_state_dict. Uses torch's
        # default (sharded) state-dict options.
        """
        Get state dict for model and optimizer using DCP utilities.
        This automatically manages FSDP FQN's and
        sets default state dict type to FSDP.SHARDED_STATE_DICT
        """
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
            state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}
        else:
            state_dict = {"model": get_model_state_dict(self.model)}
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # [FUNCTION DOC]
        # Loads the 'model' (and 'optim' if an optimizer is set) entries back via
        # set_state_dict / set_model_state_dict, mutating model and optimizer in place.
        # Raises KeyError if required keys are missing.
        """
        Load state dicts onto model and optimizer.
        """
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
            )
        else:
            set_model_state_dict(
                self.model,
                model_state_dict=state_dict["model"],
            )

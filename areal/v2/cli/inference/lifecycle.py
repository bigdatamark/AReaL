# [MODULE DOC]
# Lifecycle object for inference services (inf_lifecycle), using RuntimeState which
# spans a service state file plus a model registry file.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.v2.cli.inference.state import (
    INF_NAMESPACE,
    ModelState,
    RuntimeState,
    ServiceState,
    store,
)
from areal.v2.cli.lifecycle import ServiceLifecycle
from areal.v2.cli.process import kill_pids


class InferenceLifecycle(ServiceLifecycle):
    # [CLASS DOC]
    # ServiceLifecycle subclass whose slot replacement understands the two state files.
    """Adds the inf-specific behavior on top of scaffold's lifecycle:

    - State spans two files (service + model registry). ``force_replace_slot``
      walks the inf raw-JSON helper which knows about both files, and
      removes both on cleanup.
    """

    def force_replace_slot(self, service: str, *, grace_s: float = 5.0) -> None:
        # [FUNCTION DOC]
        # If a state file exists for service: collects pids from the loaded state, or
        # (if loading fails) recovers pids from the raw JSON via the store (empty on
        # failure), kills them with grace_s, then removes both service and model state
        # files. No-op if no state file.
        path = self.state_path(service)
        if not path.exists():
            return
        pids: list[int] = []
        try:
            state = self.load_state(service)
            pids = self._collect_pids(state)
        except Exception:
            try:
                pids = store.recover_pids_from_raw_state(service)
            except Exception:
                pids = []
        if pids:
            kill_pids(pids, grace_s=grace_s)
        ServiceState.remove(service)
        ModelState.remove(service)


inf_lifecycle = InferenceLifecycle(
    namespace=INF_NAMESPACE,
    state_class=RuntimeState,
    stop_command="areal inf stop",
)

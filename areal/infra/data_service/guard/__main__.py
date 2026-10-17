# [MODULE DOC]
# Module entrypoint for the data-service guard: simply delegates to the generic RPC
# guard main function when run as a script.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.infra.rpc.guard.__main__ import main

if __name__ == "__main__":
    main()

# [MODULE DOC]
# Script entry point: runs the root "areal" click CLI when executed as __main__.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.v2.cli.cli import cli

if __name__ == "__main__":
    cli()

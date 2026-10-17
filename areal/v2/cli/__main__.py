# [MODULE DOC]
# Entry point for running the v2 CLI as a module. Imports the root click group
# from the v2 cli module and invokes it when executed as the main program.
# SPDX-License-Identifier: Apache-2.0

from areal.v2.cli.cli import cli

if __name__ == "__main__":
    cli()

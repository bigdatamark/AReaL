# [PACKAGE DOC]
# Click group "areal train" for running training experiments; registers the run command.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from areal.v2.cli.training.commands.run import run_cmd


@click.group(help="Run AReaL training experiments.")
def train() -> None:
    # [FUNCTION DOC]
    # Group callback; does nothing.
    pass


train.add_command(run_cmd)

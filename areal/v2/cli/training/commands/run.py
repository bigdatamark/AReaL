# [MODULE DOC]
# Click command "areal train run": imports a user-specified training driver function and
# invokes it in-process with a --config argv plus pass-through hydra overrides.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path

import click


@click.command(
    name="run",
    help="Invoke a training driver with the given config and hydra overrides.",
    context_settings={"ignore_unknown_options": True},
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the experiment yaml.",
)
@click.option(
    "--driver",
    "driver_spec",
    required=True,
    help="Driver entry point as 'module.path:func', e.g. examples.math.gsm8k_rl:main.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def run_cmd(config_path: Path, driver_spec: str, overrides: tuple[str, ...]) -> None:
    # [FUNCTION DOC]
    # Click entry point. Options: --config (existing YAML file, required), --driver
    # ("module.path:func", required), and any number of unprocessed override arguments
    # (unknown options allowed). Exits via SystemExit with do_run's return code.
    raise SystemExit(do_run(config_path, driver_spec, list(overrides)) or 0)


def do_run(config_path: Path, driver_spec: str, overrides: list[str]) -> int:
    # [FUNCTION DOC]
    # Validates the driver spec contains ":" (UsageError otherwise), imports the module
    # (ClickException on ImportError) and gets the function (ClickException if missing).
    # Calls it with argv ["--config", <absolute config path>, *overrides]. Returns the
    # function's result if it is an int, else 0. Runs the training in the current
    # process.
    if ":" not in driver_spec:
        raise click.UsageError(
            f"--driver must be in 'module.path:func' form, got: {driver_spec!r}"
        )
    mod_path, func_name = driver_spec.split(":", 1)
    try:
        module = importlib.import_module(mod_path)
    except ImportError as e:
        raise click.ClickException(
            f"failed to import driver module {mod_path!r}: {e}"
        ) from e
    try:
        fn = getattr(module, func_name)
    except AttributeError as e:
        raise click.ClickException(
            f"module {mod_path!r} has no attribute {func_name!r}"
        ) from e

    argv = ["--config", str(config_path.resolve()), *overrides]
    result = fn(argv)
    if isinstance(result, int):
        return result
    return 0

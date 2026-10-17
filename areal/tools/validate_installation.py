#!/usr/bin/env python3
# [MODULE DOC]
# CLI script that validates an AReaL installation against the dependencies declared in
# the repo's pyproject.toml (presence, versions, CUDA extensions) using
# BaseInstallationValidator from validation_base. Imports validation_base as a top-level
# module, so it must be run with areal/tools on sys.path (e.g. executed directly as a
# script). Exits 0 on success, 1 on failure.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamic Installation Validation Script for AReaL

This script validates that all dependencies listed in pyproject.toml are properly
installed with correct versions and that CUDA extensions are functional.
"""

import sys
from pathlib import Path

from validation_base import BaseInstallationValidator


class DynamicInstallationValidator(BaseInstallationValidator):
    # [CLASS DOC]
    # Concrete validator for the standard (non-docker) installation; inherits all
    # checking logic from BaseInstallationValidator and only supplies the output title.
    """Validates installation based on pyproject.toml dependencies."""

    def get_validation_title(self) -> str:
        # [FUNCTION DOC]
        # Returns the fixed header string "AReaL Installation Validation" used in
        # printed output.
        """Get the title for validation output."""
        return "AReaL Installation Validation"


def main():
    # [FUNCTION DOC]
    # Entry point: resolves pyproject.toml two directories above this script's folder
    # (repo root), prints an error and exits 1 if missing; otherwise constructs the
    # validator, calls run(), and exits the process with 0 if it returned truthy else 1.
    # Never returns.
    """Main entry point."""
    # Find pyproject.toml
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)

    validator = DynamicInstallationValidator(pyproject_path)
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

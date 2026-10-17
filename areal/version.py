# [MODULE DOC]
# Version information for AReaL: package version from installed metadata plus git
# branch/commit/dirty state probed at import time. Exposes a module-level version_info
# singleton and backward-compatible __version__/__branch__/__commit__/__is_dirty__
# globals. Side effect on import: runs git subprocesses in the package directory.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _get_package_version() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the installed 'areal' distribution version, or '0.0.0.dev' if not
    # installed.
    """Get version from installed package metadata (single source of truth)."""
    try:
        return version("areal")
    except PackageNotFoundError:
        # Package not installed, fallback for development
        return "0.0.0.dev"


class VersionInfo:
    # [CLASS DOC]
    # Holds version, git branch, short commit hash and dirty flag; provides formatted
    # full version strings.
    def __init__(self):
        # [FUNCTION DOC]
        # Initializes version from package metadata, empty branch/commit, not dirty,
        # then populates git info.
        self.__version__ = _get_package_version()
        self.__branch__ = ""
        self.__commit__ = ""
        self.__is_dirty__ = False
        self._populate_version_info()

    @property
    def version(self) -> str:
        # [FUNCTION DOC]
        # Returns the package version string.
        return self.__version__

    @property
    def branch(self) -> str:
        # [FUNCTION DOC]
        # Returns the git branch ('' if unknown).
        return self.__branch__

    @property
    def commit(self) -> str:
        # [FUNCTION DOC]
        # Returns the short git commit ('' if unknown).
        return self.__commit__

    @property
    def is_dirty(self) -> bool:
        # [FUNCTION DOC]
        # Returns True if the git work tree has uncommitted changes.
        return self.__is_dirty__

    @property
    def full_version(self) -> str:
        # [FUNCTION DOC]
        # Returns '<version>[-<commit>][-dirty]'.
        version = self.version
        if self.commit != "":
            version = f"{self.version}-{self.commit}"
        if self.is_dirty:
            version = f"{version}-dirty"
        return version

    @property
    def full_version_with_dirty_description(self) -> str:
        # [FUNCTION DOC]
        # Returns full_version plus an explanatory note when dirty.
        version = self.full_version
        if self.is_dirty:
            version = (
                f"{version} ('-dirty' means there are uncommitted code changes in git)"
            )
        return version

    @lru_cache(maxsize=1)
    def _populate_version_info(self):
        # [FUNCTION DOC]
        # HELPER: Runs 'git branch --show-current', 'git rev-parse --short HEAD' and
        # 'git diff-index --quiet HEAD --' in the package directory to fill branch,
        # commit and dirty flag; silently ignores git errors or missing git.
        # KNOWN ISSUE: lru_cache on an instance method keys on self and keeps the
        # instance alive; it only prevents repeat calls for the same instance.
        try:
            self.__branch__ = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
                .decode("utf-8")
                .strip()
            )
            self.__commit__ = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    cwd=Path(__file__).parent,
                )
                .decode("utf-8")
                .strip()
            )
            git_diff_result = subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).parent,
            )
            self.__is_dirty__ = git_diff_result.returncode != 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass


version_info = VersionInfo()

# for backward compatibility
__version__ = version_info.version
__branch__ = version_info.branch
__commit__ = version_info.commit
__is_dirty__ = version_info.is_dirty

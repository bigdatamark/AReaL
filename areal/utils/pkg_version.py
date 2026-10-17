# [MODULE DOC]
# Helpers for querying installed package versions (importlib.metadata) and comparing
# them with PEP 440 semantics (packaging.version.Version).
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version

from packaging.version import Version


def is_available(pkg_name):
    # [FUNCTION DOC]
    # Returns True if package metadata for pkg_name is installed (non-empty version),
    # False if PackageNotFoundError.
    try:
        return bool(get_version(pkg_name))
    except PackageNotFoundError:
        return False


def compare_versions(version1: str, version2: str) -> int:
    # [FUNCTION DOC]
    # Parses both strings as PEP 440 versions and returns -1, 0 or 1 for <, ==, >.
    # Raises InvalidVersion for unparseable strings.
    """
    Compare two version strings.

    :param version1: First version string.
    :param version2: Second version string.
    :return: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    v1 = Version(version1)
    v2 = Version(version2)
    if v1 < v2:
        return -1
    elif v1 == v2:
        return 0
    else:
        return 1


def is_version_greater_or_equal(package_name: str, target_version: str) -> bool:
    # [FUNCTION DOC]
    # True if the installed version of package_name >= target_version. Raises
    # PackageNotFoundError if the package is not installed.
    """
    Check if the installed version of a package is greater than or equal to the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is greater than or equal to the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) >= 0


def is_version_less(package_name: str, target_version: str) -> bool:
    # [FUNCTION DOC]
    # True if the installed version of package_name < target_version. Raises
    # PackageNotFoundError if not installed.
    """
    Check if the installed version of a package is less than the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is less than the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) < 0


def is_version_equal(package_name: str, target_version: str) -> bool:
    # [FUNCTION DOC]
    # True if the installed version equals target_version under PEP 440 comparison.
    # Raises PackageNotFoundError if not installed.
    """
    Check if the installed version of a package is equal to the target version.

    :param package_name: Name of the package.
    :param target_version: Target version to compare against.
    :return: True if the installed version is equal to the target version, False otherwise.
    """
    installed_version = get_version(package_name)
    return compare_versions(installed_version, target_version) == 0

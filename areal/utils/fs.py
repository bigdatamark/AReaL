# [MODULE DOC]
# Filesystem helpers: a per-user cache directory and validation that shared paths exist
# and live on a network filesystem (detected via mount table keyword matching).
# SPDX-License-Identifier: Apache-2.0

import getpass
import os

import psutil

from areal.utils.logging import getLogger

logger = getLogger("FileSystemUtils")

# Keywords to match in fstype or device path (case-insensitive substring match)
NETWORK_FS_KEYWORDS = {
    # Standard network filesystems
    "nfs",
    "lustre",
    "gpfs",
    "mmfs",
    "beegfs",
    "pvfs",
    "orangefs",
    "ceph",
    "glusterfs",
    "afs",
    "cifs",
    "smb",
    "rbd",
    "9p",
    # Cloud providers
    "alinas",  # Alibaba Cloud NAS/CPFS
    "cpfs",  # Alibaba Cloud CPFS
    "vepfs",  # Volcano Engine (ByteDance) PFS
    "goosefs",  # Tencent Cloud GooseFS
    "chdfs",  # Tencent Cloud HDFS
    "obsfs",  # Huawei Cloud OBS
    "gcsfuse",  # Google Cloud Storage FUSE
    "efs",  # AWS Elastic File System
    "blobfuse",  # Azure Blob FUSE
    "s3fs",  # S3-compatible storage FUSE
    "fsx",  # AWS FSx
    # Distributed filesystems
    "juicefs",
    "alluxio",
    "seaweedfs",
    "hdfs",
    "moosefs",
    "lizardfs",
    "xtreemfs",
    "sshfs",
}


def get_user_tmp():
    # [FUNCTION DOC]
    # Returns /home/<user>/.cache/areal, creating it if needed (side effect: mkdir).
    # KNOWN ISSUE: hardcodes /home/<user> instead of the real home directory, which
    # fails for users whose home is elsewhere (e.g. root).
    user = getpass.getuser()
    user_tmp = os.path.join("/home", user, ".cache", "areal")
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


def validate_shared_path(path: str, name: str = "path", warn_nfs: bool = True) -> None:
    # [FUNCTION DOC]
    # Validates a shared path. Empty path is a no-op. Raises FileNotFoundError if it
    # does not exist. If warn_nfs, resolves the real path, finds the longest-prefix
    # mount from psutil.disk_partitions(all=True), and treats it as network if its
    # fstype+device contains any NETWORK_FS_KEYWORDS, the device has ":" and is not
    # under /dev, or starts with "//". Logs a warning (does not raise) if not on network
    # storage. Returns None.
    # KNOWN ISSUE: the mountpoint prefix match is a plain string prefix, so "/data" also
    # matches "/data2/..."; and substring keywords like "afs" or "efs" can false-
    # positive.
    """
    Validate that a path exists for shared/distributed access.

    Args:
        path: Path to validate.
        name: Descriptive name for the path (used in messages).
        warn_nfs: If True, warn when not on network filesystem.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} '{path}' does not exist.")

    if not warn_nfs:
        return

    # Check if path is on network filesystem
    real_path = os.path.realpath(path)
    partitions = psutil.disk_partitions(all=True)

    # Find the best matching mountpoint (longest prefix match)
    best_match = None
    best_match_len = 0

    for partition in partitions:
        mountpoint = partition.mountpoint
        if real_path.startswith(mountpoint) and len(mountpoint) > best_match_len:
            best_match = partition
            best_match_len = len(mountpoint)

    is_network = False
    if best_match:
        fstype = best_match.fstype.lower()
        device = best_match.device.lower()
        combined = fstype + device

        is_network = (
            any(kw in combined for kw in NETWORK_FS_KEYWORDS)
            or (":" in device and not device.startswith("/dev"))
            or device.startswith("//")
        )

    if not is_network:
        logger.warning(
            f"{name} '{path}' is not on a network filesystem. "
            "This may cause issues in distributed training where all nodes "
            "need access to the same files. Consider using NFS, Lustre, or "
            "other shared storage."
        )

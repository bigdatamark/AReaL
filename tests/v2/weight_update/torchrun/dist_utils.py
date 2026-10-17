# [MODULE DOC]
# HELPER module: small utilities shared by torchrun-launched weight-transfer test
# scripts (result-file writing and rank-0 printing).
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist


def write_result(out: str, succ: bool, error: str = "") -> None:
    # [FUNCTION DOC]
    # HELPER: Writes 'Passed' or 'Failed' (per succ) to file path out, overwriting it,
    # followed by a newline and error text if error is non-empty. The parent pytest
    # reads this file to judge the subprocess outcome.
    with open(out, "w") as f:
        f.write("Passed" if succ else "Failed")
        if error:
            f.write(f"\n{error}")


def print_rank0(msg: str) -> None:
    # [FUNCTION DOC]
    # HELPER: Prints msg only on global rank 0. Requires an initialized
    # torch.distributed process group (get_rank raises otherwise).
    if dist.get_rank() == 0:
        print(msg)

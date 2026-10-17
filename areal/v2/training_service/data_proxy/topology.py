# [MODULE DOC]
# Discovers the parallel topology of a worker group by querying each worker's /topology
# endpoint and building DP grouping information.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import aiohttp

from areal.utils import logging

logger = logging.getLogger("TrainDataProxy")


@dataclass
class WorkerInfo:
    # [CLASS DOC]
    # Dataclass describing one worker rank: addr, global rank, world_size, dp_rank,
    # dp_size, whether it is the head (data-receiving) rank of its DP group, local_rank.
    addr: str
    rank: int = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    is_dp_head: bool = True
    local_rank: int = 0


@dataclass
class WorkerTopology:
    # [CLASS DOC]
    # Dataclass describing the whole group: workers (in worker_addrs order), indices of
    # DP heads, dp_size, dp_groups (worker indices per dp rank) and pp/tp/cp/ep sizes.
    workers: list[WorkerInfo] = field(default_factory=list)
    dp_heads: list[int] = field(default_factory=list)
    dp_size: int = 1
    dp_groups: list[list[int]] = field(default_factory=list)
    pp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1


async def discover_topology(
    worker_addrs: list[str],
    timeout: float = 10.0,
) -> WorkerTopology:
    # [FUNCTION DOC]
    # Concurrently GETs /topology from every address using one aiohttp session with a
    # total timeout. Any failure (exception or HTTP >= 400) raises RuntimeError naming
    # the address. Parallel sizes (pp/tp/cp/ep) are taken from the first response; per-
    # worker fields default to a single-rank layout when absent. dp_size comes from the
    # first worker; dp_groups lists worker indices by dp_rank (extended if a rank
    # exceeds dp_size). Returns WorkerTopology.
    workers: list[WorkerInfo] = []
    meta: dict[str, int] = {}

    async def _fetch(session: aiohttp.ClientSession, addr: str) -> dict:
        # [FUNCTION DOC]
        # HELPER: GETs addr/topology and returns the decoded JSON (ignoring content
        # type); raises RuntimeError with status and body on HTTP errors.
        async with session.get(f"{addr}/topology") as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(
                    f"Failed to discover topology from {addr}: "
                    f"HTTP {resp.status}: {text}"
                )
            return await resp.json(content_type=None)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        tasks = [_fetch(session, addr) for addr in worker_addrs]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, resp in enumerate(responses):
        addr = worker_addrs[i]
        if isinstance(resp, BaseException):
            raise RuntimeError(f"Failed to discover topology from {addr}: {resp}")
        data = resp
        if not meta:
            meta = {
                "pp_size": int(data.get("pp_size", 1)),
                "tp_size": int(data.get("tp_size", 1)),
                "cp_size": int(data.get("cp_size", 1)),
                "ep_size": int(data.get("ep_size", 1)),
            }
        workers.append(
            WorkerInfo(
                addr=addr,
                rank=data.get("rank", 0),
                world_size=data.get("world_size", 1),
                dp_rank=data.get("dp_rank", 0),
                dp_size=data.get("dp_size", 1),
                is_dp_head=data.get("is_dp_head", True),
                local_rank=data.get("local_rank", 0),
            )
        )

    dp_heads = [i for i, w in enumerate(workers) if w.is_dp_head]
    dp_size = workers[0].dp_size if workers else 1
    dp_groups: list[list[int]] = [[] for _ in range(max(dp_size, 1))]
    for i, w in enumerate(workers):
        if w.dp_rank >= len(dp_groups):
            dp_groups.extend([] for _ in range(w.dp_rank - len(dp_groups) + 1))
        dp_groups[w.dp_rank].append(i)

    return WorkerTopology(
        workers=workers,
        dp_heads=dp_heads,
        dp_size=dp_size,
        dp_groups=dp_groups,
        pp_size=meta.get("pp_size", 1),
        tp_size=meta.get("tp_size", 1),
        cp_size=meta.get("cp_size", 1),
        ep_size=meta.get("ep_size", 1),
    )

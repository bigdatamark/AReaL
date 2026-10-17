# [MODULE DOC]
# CPU unit tests for grouped colocation in the Ray scheduler: group_colocated_gpus
# chunking each node's GPUs into per-worker groups, create_workers routing colocation
# jobs whose total GPU count matches the target role to the grouped path, and
# _create_grouped_colocated_workers launching one node-pinned launcher per node with
# role-specific env and physical device ids. Ray calls and launchers are stubbed; scheduler
# helpers are reused from tests/test_ray_scheduler.py.
# SPDX-License-Identifier: Apache-2.0
"""Grouped colocation on the HTTP-based Ray scheduler.

Grouped colocation places multi-GPU workers (e.g. 16 x 4-GPU SGLang rollout
workers) on the same nodes and physical GPUs as an existing target role made
of more, smaller workers (e.g. 64 x 1-GPU Megatron actors on 8 nodes).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_ray_scheduler import _scheduler, _worker_info

import areal.infra.scheduler.ray as ray_scheduler
from areal.api import Job
from areal.api.cli_args import (
    SchedulingSpec,
    SchedulingStrategy,
    SchedulingStrategyType,
)
from areal.infra.scheduler.ray import group_colocated_gpus


@pytest.fixture(autouse=True)
def _disable_scheduler_destructor(monkeypatch):
    # [FUNCTION DOC]
    # Autouse fixture: replaces RayScheduler.__del__ with a no-op so test schedulers do not
    # try to tear down Ray resources on garbage collection.
    monkeypatch.setattr(ray_scheduler.RayScheduler, "__del__", lambda self: None)


def _node_info(node_idx: int, n_gpus: int = 8) -> dict:
    # [FUNCTION DOC]
    # HELPER: Returns a fake node info dict (host 10.0.0.<idx>, node_id "node<idx>",
    # visible_devices "0".."n_gpus-1").
    return {
        "host": f"10.0.0.{node_idx}",
        "node_id": f"node{node_idx}",
        "visible_devices": [str(gpu) for gpu in range(n_gpus)],
    }


def test_group_colocated_gpus_chunks_each_node_without_crossing_nodes():
    # [FUNCTION DOC]
    # Four 8-GPU nodes with 4 GPUs per worker give 8 groups, each of 4 contiguous devices
    # within a single node, covering every (node, gpu) exactly once, in node order.
    node_infos = [_node_info(node) for node in range(4)]

    groups = group_colocated_gpus(node_infos, gpus_per_worker=4)

    assert len(groups) == 8
    assert all(len(group["gpu_devices"]) == 4 for group in groups)
    seen = {(group["node_id"], gpu) for group in groups for gpu in group["gpu_devices"]}
    assert seen == {(f"node{node}", str(gpu)) for node in range(4) for gpu in range(8)}
    assert groups[0]["gpu_devices"] == ["0", "1", "2", "3"]
    assert groups[1]["gpu_devices"] == ["4", "5", "6", "7"]
    assert groups[0]["host"] == groups[1]["host"] == "10.0.0.0"
    assert [group["node_id"] for group in groups[:3]] == ["node0", "node0", "node1"]


def test_group_colocated_gpus_rejects_non_divisible_node():
    # [FUNCTION DOC]
    # A node with 6 GPUs and group size 4 raises ValueError ("not a multiple").
    with pytest.raises(ValueError, match="not a multiple"):
        group_colocated_gpus([_node_info(0, n_gpus=6)], gpus_per_worker=4)


def test_group_colocated_gpus_rejects_invalid_group_size():
    # [FUNCTION DOC]
    # gpus_per_worker of 0 raises ValueError.
    with pytest.raises(ValueError, match="gpus_per_worker"):
        group_colocated_gpus([_node_info(0)], gpus_per_worker=0)


def test_create_workers_routes_gpu_conserving_colocation_to_grouped_path(tmp_path):
    # [FUNCTION DOC]
    # A non-fork colocation job of 2 x 4-GPU workers targeting 8 x 1-GPU actors is routed to
    # _create_grouped_colocated_workers (mocked) and is not registered as a plain colocated
    # role.
    scheduler = _scheduler(tmp_path)
    scheduler._workers["actor"] = [
        _worker_info(f"actor/{index}", role="actor") for index in range(8)
    ]
    for info in scheduler._workers["actor"]:
        info.spec = SchedulingSpec(cpu=1, gpu=1, mem=1)
    grouped = Mock(return_value=["rollout/0", "rollout/1"])
    scheduler._create_grouped_colocated_workers = grouped
    job = Job(
        role="rollout",
        replicas=2,
        tasks=[SchedulingSpec(cpu=1, gpu=4, mem=1)] * 2,
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor", fork=False
        ),
    )

    worker_ids = scheduler.create_workers(job)

    assert worker_ids == ["rollout/0", "rollout/1"]
    grouped.assert_called_once()
    assert "rollout" not in scheduler._colocated_roles


def test_create_workers_still_rejects_non_conserving_replica_mismatch(tmp_path):
    # [FUNCTION DOC]
    # A colocation job whose replica count and GPU total do not match the target still raises
    # WorkerCreationError ("Replica count").
    scheduler = _scheduler(tmp_path)
    scheduler._workers["actor"] = [_worker_info("actor/0", role="actor")]
    job = Job(
        role="ref",
        replicas=2,
        tasks=[SchedulingSpec(cpu=1, gpu=1, mem=1)] * 2,
        scheduling_strategy=SchedulingStrategy(
            type=SchedulingStrategyType.colocation, target="actor"
        ),
    )

    with pytest.raises(ray_scheduler.WorkerCreationError, match="Replica count"):
        scheduler.create_workers(job)


class _LauncherStub:
    # [CLASS DOC]
    # HELPER: Fake RayWorkerProcessLauncher actor class that records options, constructor
    # args, and specs passed to start_workers into a shared captured list.
    def __init__(self, captured: list):
        # [FUNCTION DOC]
        # HELPER: Stores the shared captured list.
        self._captured = captured

    def options(self, **options):
        # [FUNCTION DOC]
        # HELPER: Mimics Ray's ActorClass.options; snapshots the options and returns a factory
        # whose remote(*args) records options/init args, returns a fake launcher whose
        # start_workers.remote stores the specs in the record, and saves the launcher in it.
        captured = self._captured
        options_snapshot = dict(options)

        class _Factory:
            # [CLASS DOC]
            # HELPER: Fake remote-actor factory returned by options().
            @staticmethod
            def remote(*args):
                # [FUNCTION DOC]
                # HELPER: Creates and records a fake launcher as described in options().
                record = {"options": options_snapshot, "init_args": args}
                captured.append(record)
                launcher = SimpleNamespace()
                launcher.start_workers = SimpleNamespace(
                    remote=lambda specs, _r=record: _r.setdefault("specs", specs)
                )
                record["launcher"] = launcher
                return launcher

        return _Factory


def test_grouped_launcher_env_is_role_specific_and_devices_are_physical(
    tmp_path, monkeypatch
):
    # [FUNCTION DOC]
    # Grouped creation of 2 x 2-GPU rollout workers over 4 actor workers on one 4-GPU node
    # creates one launcher pinned (hard NodeAffinity) to the node's hex id with no GPU
    # reservation, passes only the rollout env vars (target env unchanged), gives each worker
    # all physical devices "0".."3", sets SLURM_LOCALID/NODEID/NNODES per worker, records the
    # role's workers and launcher, and creates no placement group or colocated-role entry.
    scheduler = _scheduler(tmp_path)
    target_env = {"AWEX_ACTOR_ALLOC_CONF": "expandable_segments:True"}
    target_spec = SchedulingSpec(cpu=1, gpu=1, mem=1, env_vars=target_env)
    scheduler._workers["actor"] = [
        _worker_info(f"actor/{index}", role="actor") for index in range(4)
    ]
    for info in scheduler._workers["actor"]:
        info.spec = target_spec
    node_launcher = SimpleNamespace(
        get_node_info=SimpleNamespace(remote=lambda: "node-info-ref")
    )
    scheduler._launchers["actor"] = [node_launcher]
    hex_node_id = "ab" * 28

    captured: list = []
    monkeypatch.setattr(
        ray_scheduler, "RayWorkerProcessLauncher", _LauncherStub(captured)
    )
    node_infos = [dict(_node_info(0, n_gpus=4), node_id=hex_node_id)]
    monkeypatch.setattr(
        ray_scheduler.ray,
        "get",
        lambda refs, timeout=None: node_infos if refs == ["node-info-ref"] else refs,
    )

    rollout_env = {"LD_PRELOAD": "/x/tms.so", "NCCL_NVLS_ENABLE": "0"}
    rollout_spec = SchedulingSpec(cpu=1, gpu=2, mem=1, env_vars=rollout_env)

    worker_ids = scheduler._create_grouped_colocated_workers(
        "rollout", "actor", [rollout_spec, rollout_spec]
    )

    assert worker_ids == ["rollout/0", "rollout/1"]
    assert len(captured) == 1
    assert captured[0]["init_args"][-1] == rollout_env
    assert captured[0]["options"].get("num_gpus") in (0, None)
    strategy = captured[0]["options"]["scheduling_strategy"]
    assert strategy.node_id == hex_node_id
    assert strategy.soft is False
    specs = captured[0]["specs"]
    assert [spec["gpu_devices"] for spec in specs] == [
        ["0", "1", "2", "3"],
        ["0", "1", "2", "3"],
    ]
    assert [spec["extra_env"]["SLURM_LOCALID"] for spec in specs] == ["0", "1"]
    assert all(spec["extra_env"]["SLURM_NODEID"] == "0" for spec in specs)
    assert all(spec["extra_env"]["SLURM_NNODES"] == "1" for spec in specs)
    assert target_env == {"AWEX_ACTOR_ALLOC_CONF": "expandable_segments:True"}
    assert "rollout" not in scheduler._colocated_roles
    assert "rollout" in scheduler._workers
    assert scheduler._launchers["rollout"] == [captured[0]["launcher"]]
    assert "rollout" not in scheduler._placement_groups

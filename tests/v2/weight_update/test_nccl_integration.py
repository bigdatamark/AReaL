# [MODULE DOC]
# GPU integration tests for the v2 weight-update (awex) path. Part 1 launches torchrun
# subprocesses running torchrun/run_nccl_weight_transfer.py to test raw NCCL group
# creation and P2P transfer. Part 2 runs end-to-end round trips: a single LocalScheduler
# owns all GPUs; RolloutControllerV2 starts SGLang inference workers (whose weights are
# randomized via /awex/debug/randomize_parameters so the update is not a no-op),
# GatewayTrainController starts FSDP or Megatron training workers, and
# WeightUpdateController launches the weight-update gateway and drives connect ->
# update_weights -> disconnect. Afterwards inference /generate is smoke-tested and
# selected parameters are fetched from both sides via /awex/debug/get_parameters (saved
# to .pt files) and compared bitwise. Covers FSDP DP, Megatron DP/TP/PP/CP/EP
# combinations, and colocated (shared GPU, CUDA IPC) mode. All tests skip without CUDA
# or with too few GPUs; most are marked multi_gpu/slow/sglang.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import httpx
import pytest
import torch

from areal.infra.platforms import current_platform
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Project root so that torchrun workers can resolve `from tests.*` imports.
# pytest adds "." via pyproject.toml `pythonpath`, but subprocesses don't inherit that.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _run_weight_update_test(n_gpus: int, test_type: str, output: str):
    # [FUNCTION DOC]
    # HELPER: Runs run_nccl_weight_transfer.py under torchrun on one node with n_gpus
    # processes, a free master port, the given test_type and output result path, with
    # the project root prepended to PYTHONPATH and output streamed to the test's stdout.
    # Waits for completion (killing the process tree on any interruption and re-
    # raising), fails the test on non-zero exit, then asserts the result file content
    # equals 'Passed'.
    # KNOWN ISSUE: the script path is relative, so it only works when pytest runs from
    # the repo root.
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master_port={port}",
            "tests/v2/weight_update/torchrun/run_nccl_weight_transfer.py",
            f"--test_type={test_type}",
            f"--output={output}",
        ],
        text=True,
        stderr=sys.stdout,
        stdout=sys.stdout,
        env=env,
    )
    try:
        proc.wait()
    except BaseException:
        kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        pytest.fail(f"torchrun exited with code {proc.returncode}")

    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_nccl_group_init_2gpu(tmp_path_factory):
    # [FUNCTION DOC]
    # 2-GPU test (skipped if fewer GPUs): runs the 'nccl_group_init' torchrun test and
    # checks its result file.
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "nccl_group_init.out"
    _run_weight_update_test(2, "nccl_group_init", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_batch_isend_irecv_2gpu(tmp_path_factory):
    # [FUNCTION DOC]
    # 2-GPU test: runs the 'batch_isend_irecv' torchrun test and checks its result file.
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "batch_isend_irecv.out"
    _run_weight_update_test(2, "batch_isend_irecv", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_weight_transfer_lifecycle_2gpu(tmp_path_factory):
    # [FUNCTION DOC]
    # 2-GPU test: runs the 'weight_transfer_lifecycle' torchrun test (1 inference + 1
    # training rank) and checks its result file.
    # KNOWN ISSUE: output file is named weight_transfer_4gpu.out although only 2 GPUs
    # are used (cosmetic).
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "weight_transfer_4gpu.out"
    _run_weight_update_test(2, "weight_transfer_lifecycle", str(output))


# ---------------------------------------------------------------------------
# E2E awex weight update: real FSDPEngine + SGLang server + gateway
# ---------------------------------------------------------------------------


def _get_test_model_path() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the local Qwen3-0.6B path under /storage/openpsi/models if
    # present, else the HF id 'Qwen/Qwen3-0.6B'.
    local = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-0.6B"


def _get_test_moe_model_path() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the local Qwen3-30B-A3B MoE path if present, else the HF id
    # 'Qwen/Qwen3-30B-A3B'.
    local = "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-30B-A3B"


def _make_truncated_moe_model(tmp_path, num_layers: int = 4) -> str:
    # [FUNCTION DOC]
    # HELPER: Builds a truncated MoE model directory at tmp_path/truncated_moe: copies
    # config.json with num_hidden_layers overridden to num_layers (default 4), copies
    # existing tokenizer/generation config files, symlinks every *.safetensors / *.bin
    # weight file and copies their index JSONs from the source. Returns the new
    # directory path. Filesystem side effects only. Intended for use with
    # init_from_scratch so full weights need not be loaded.
    # KNOWN ISSUE: if the local model is absent the source is an HF hub id, which is not
    # a directory, so opening config.json raises FileNotFoundError. Re-calling with the
    # same tmp_path would fail on os.symlink because the links already exist.
    import glob
    import json
    import shutil

    src = _get_test_moe_model_path()
    dst = str(tmp_path / "truncated_moe")
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    config["num_hidden_layers"] = num_layers
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f)

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        src_file = os.path.join(src, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst)

    for weight_file in glob.glob(os.path.join(src, "*.safetensors")) + glob.glob(
        os.path.join(src, "*.bin")
    ):
        os.symlink(weight_file, os.path.join(dst, os.path.basename(weight_file)))

    for index_file in glob.glob(
        os.path.join(src, "*.safetensors.index.json")
    ) + glob.glob(os.path.join(src, "*.bin.index.json")):
        shutil.copy2(index_file, dst)

    return dst


def _make_local_scheduler(tmp_path, name: str, gpu_devices: list[int]):
    # [FUNCTION DOC]
    # HELPER: Creates <name>_fileroot and <name>_name_resolve dirs under tmp_path and
    # returns a LocalScheduler over gpu_devices with logs at <name>_logs, experiment
    # 'test-awex-<name>', trial 't0'.
    from areal.infra.scheduler.local import LocalScheduler

    fileroot = tmp_path / f"{name}_fileroot"
    fileroot.mkdir(exist_ok=True)
    nr_root = tmp_path / f"{name}_name_resolve"
    nr_root.mkdir(exist_ok=True)

    return LocalScheduler(
        gpu_devices=gpu_devices,
        log_dir=str(tmp_path / f"{name}_logs"),
        experiment_name=f"test-awex-{name}",
        trial_name="t0",
        fileroot=str(fileroot),
        nfs_record_root=str(nr_root),
    )


# Representative parameters spanning different fusion/sharding cases.
# Covers QKV unfusing (q/k/v_proj), gate_up unfusing (gate/up_proj),
# a deeper layer, and a small non-attention weight (layer norm).
_VALIDATE_PARAM_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.27.self_attn.q_proj.weight",
    "model.norm.weight",
]

# Qwen3-30B-A3B (truncated to 4 layers) MoE validation params.
# This model has no shared experts — pure MoE with 128 routed experts.
_VALIDATE_PARAM_NAMES_MOE = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.1.mlp.experts.0.gate_proj.weight",
    "model.layers.1.mlp.experts.0.up_proj.weight",
    "model.layers.1.mlp.experts.0.down_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.norm.weight",
]


def _validate_weight_update_correctness(
    train_worker_urls: list[str],
    inf_worker_url: str,
    param_dir,
) -> None:
    # [FUNCTION DOC]
    # HELPER (FSDP validation): For each training worker URL, POSTs
    # /awex/debug/get_parameters asking it to save _VALIDATE_PARAM_NAMES to
    # train_params_rank<i>.pt in param_dir; does the same for the single inference
    # worker (infer_params.pt). Asserts every request returns 200. Loads all files on
    # CPU, asserts every name exists on every side, reconstructs each full training
    # param by concatenating rank shards along dim 0 (FSDP Shard(0)) when there are
    # several ranks, and asserts exact equality (rtol=atol=0) with the inference tensor.
    # Prints progress. Returns None; raises AssertionError on mismatch.
    # KNOWN ISSUE: assumes inference TP=1 and that shard order equals URL order; params
    # not sharded on dim 0 would be reconstructed incorrectly.
    """Fetch params from both sides via HTTP and compare bitwise.

    Training workers return FSDP local shards (Shard(0)), so we concatenate
    along dim 0 to reconstruct the full parameter.  Inference (TP=1) returns
    full parameters directly.
    """
    n_train = len(train_worker_urls)
    print(
        f"\n[weight-validation] Fetching parameters from {n_train} training "
        f"worker(s) and 1 inference worker …"
    )

    train_shard_paths = []
    for i, url in enumerate(train_worker_urls):
        p = str(param_dir / f"train_params_rank{i}.pt")
        train_shard_paths.append(p)
        resp = httpx.post(
            f"{url}/awex/debug/get_parameters",
            json={"save_path": p, "names": _VALIDATE_PARAM_NAMES},
            timeout=120.0,
        )
        assert resp.status_code == 200, (
            f"get_parameters failed on training worker {i}: {resp.text}"
        )

    inf_path = str(param_dir / "infer_params.pt")
    resp = httpx.post(
        f"{inf_worker_url}/awex/debug/get_parameters",
        json={"save_path": inf_path, "names": _VALIDATE_PARAM_NAMES},
        timeout=120.0,
    )
    assert resp.status_code == 200, (
        f"get_parameters failed on inference worker: {resp.text}"
    )

    infer_params = torch.load(inf_path, map_location="cpu", weights_only=True)
    train_shards = [
        torch.load(p, map_location="cpu", weights_only=True) for p in train_shard_paths
    ]

    print(f"[weight-validation] Comparing {len(_VALIDATE_PARAM_NAMES)} parameters …")
    for name in _VALIDATE_PARAM_NAMES:
        assert name in infer_params, f"Inference missing param: {name}"
        for i, shard in enumerate(train_shards):
            assert name in shard, f"Training rank {i} missing param: {name}"

        # Reconstruct full training param from FSDP Shard(0) chunks
        if len(train_shards) > 1:
            full_train = torch.cat([s[name] for s in train_shards], dim=0)
        else:
            full_train = train_shards[0][name]

        torch.testing.assert_close(
            full_train,
            infer_params[name],
            rtol=0,
            atol=0,
            msg=f"Parameter mismatch after weight update: {name}",
        )
        print(
            f"[weight-validation]   {name}: OK "
            f"(shape={list(full_train.shape)}, dtype={full_train.dtype})"
        )

    print(
        f"[weight-validation] All {len(_VALIDATE_PARAM_NAMES)} parameters "
        f"match between training and inference ✓"
    )


def _validate_weight_update_correctness_megatron(
    train_worker_urls: list[str],
    inf_worker_url: str,
    param_dir,
    tag: str = "megatron",
    param_names: list[str] | None = None,
) -> None:
    # [FUNCTION DOC]
    # HELPER (Megatron validation): Like the FSDP variant but fetches training params
    # concurrently (thread pool, one thread per worker) into
    # <tag>_train_params_rank<i>.pt, uses param_names (default _VALIDATE_PARAM_NAMES),
    # and instead of concatenating shards it unions the per-rank dicts (later ranks
    # overwrite earlier) since with PP/EP each rank owns a disjoint subset and with
    # DP/TP/CP ranks return full gathered params. Asserts every name is present on both
    # sides and exactly equal. Returns None.
    import concurrent.futures

    names = param_names or _VALIDATE_PARAM_NAMES
    n_train = len(train_worker_urls)
    print(
        f"\n[weight-validation] Fetching parameters from {n_train} Megatron "
        f"worker(s) and 1 inference worker …"
    )

    train_paths = [
        str(param_dir / f"{tag}_train_params_rank{i}.pt") for i in range(n_train)
    ]

    def _fetch_train(args):
        # [FUNCTION DOC]
        # HELPER: Takes a tuple (rank index, worker url, save path), POSTs
        # get_parameters for the enclosing names list, and asserts HTTP 200.
        i, url, p = args
        resp = httpx.post(
            f"{url}/awex/debug/get_parameters",
            json={"save_path": p, "names": names},
            timeout=120.0,
        )
        assert resp.status_code == 200, (
            f"get_parameters failed on training worker {i}: {resp.text}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_train) as pool:
        list(
            pool.map(
                _fetch_train,
                [
                    (i, url, p)
                    for i, (url, p) in enumerate(zip(train_worker_urls, train_paths))
                ],
            )
        )

    inf_path = str(param_dir / f"{tag}_infer_params.pt")
    resp = httpx.post(
        f"{inf_worker_url}/awex/debug/get_parameters",
        json={"save_path": inf_path, "names": names},
        timeout=120.0,
    )
    assert resp.status_code == 200, (
        f"get_parameters failed on inference worker: {resp.text}"
    )

    infer_params = torch.load(inf_path, map_location="cpu", weights_only=True)

    # Union params across all training ranks: with PP each rank owns a disjoint
    # subset of layers, so we need all ranks to cover the validate param names.
    train_params: dict[str, torch.Tensor] = {}
    for p in train_paths:
        train_params.update(torch.load(p, map_location="cpu", weights_only=True))

    print(f"[weight-validation] Comparing {len(names)} parameters …")
    for name in names:
        assert name in infer_params, f"Inference missing param: {name}"
        assert name in train_params, (
            f"No training rank owns param: {name}. "
            f"Available: {sorted(train_params.keys())[:5]}…"
        )

        torch.testing.assert_close(
            train_params[name],
            infer_params[name],
            rtol=0,
            atol=0,
            msg=f"Parameter mismatch after weight update: {name}",
        )
        print(
            f"[weight-validation]   {name}: OK "
            f"(shape={list(train_params[name].shape)}, dtype={train_params[name].dtype})"
        )

    print(
        f"[weight-validation] All {len(names)} parameters "
        f"match between Megatron training and inference ✓"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_fsdp_e2e_weight_update(n_gpus, tmp_path_factory):
    # [FUNCTION DOC]
    # E2E for FSDPEngine pure DP, parametrized over 2/4/8 GPUs: first half for SGLang
    # (sglang:d<half>, mem fraction 0.7), second half for FSDP (fsdp:d<half>,
    # NCCL_CUMEM/NVLS disabled). Initializes inference, randomizes its weights,
    # initializes training (FinetuneSpec 1 epoch, 100 samples, batch 2), launches the
    # weight-update gateway on 127.0.0.1, checks health, connects with training worker
    # URLs, updates to version 1 asserting status ok, disconnects, checks /generate
    # returns 200, then runs _validate_weight_update_correctness. Always destroys
    # gateway, training, inference controllers and scheduler workers in finally.
    """Full round trip: FSDPEngine (pure DP) → weight-update gateway → SGLang.

    A single :class:`LocalScheduler` owns all *n_gpus* devices.  Inference
    is initialized first so the round-robin allocator assigns the first half
    of the GPUs to SGLang and the second half to FSDP training — mirroring
    :class:`PPOTrainer` where one scheduler serves every engine.

    Orchestrates connect → update_weights → disconnect, passing the
    training *worker* URLs (not the training gateway) to the weight-update
    service since the awex blueprint lives on the workers.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_half = n_gpus // 2
    tmp = tmp_path_factory.mktemp("awex_e2e")
    model_path = _get_test_model_path()

    scheduler = _make_local_scheduler(tmp, "e2e", gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_half}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"fsdp:d{n_half}",
        experiment_name="test-awex-e2e",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.fsdp_engine.FSDPEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None

    try:
        # -- 1. SGLang via inference controller ----------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        # Randomize inference weights so the transfer is NOT a no-op.
        # Without this, both sides load from the same checkpoint and the
        # comparison would trivially pass even if the transfer never happened.
        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. FSDP engine via training controller -------------------------
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_ctrl.initialize(role="actor", ft_spec=ft_spec, wait=True)
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway -------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(
                host="127.0.0.1",
                request_timeout=300.0,
            )
        )
        wu_ctrl.initialize()

        # -- 4. Weight update lifecycle -----------------------------------
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name="test_e2e",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
        )

        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1

        wu_ctrl.disconnect()

        # -- 5. Verify inference server still works post-update -----------
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight update: {gen_resp.text}"
        )

        # -- 6. Validate training ↔ inference parameter equality ----------
        _validate_weight_update_correctness(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
        )

    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_megatron_dp_e2e_weight_update(n_gpus, tmp_path_factory):
    # [FUNCTION DOC]
    # E2E for Megatron pure DP over 2/4/8 GPUs via _run_megatron_awex_e2e with backend
    # megatron:d<half>.
    """Full round trip: MegatronEngine (pure DP) → weight-update gateway → SGLang.

    Each training rank holds a full copy of every parameter. Validation unions
    params across all training ranks and compares bitwise against SGLang.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    n_half = n_gpus // 2
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{n_half}",
        pair_name="test_megatron_dp_e2e",
        tag="megatron_dp_e2e",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,tp_size",
    [(4, 2), (8, 2), (8, 4)],
    ids=["4gpu-dp1tp2", "8gpu-dp2tp2", "8gpu-dp1tp4"],
)
def test_awex_megatron_dp_tp_e2e_weight_update(n_gpus, tp_size, tmp_path_factory):
    # [FUNCTION DOC]
    # E2E for Megatron DP+TP with (n_gpus, tp_size) in (4,2),(8,2),(8,4): dp_size =
    # training GPUs // tp_size, skips if < 1, runs _run_megatron_awex_e2e with backend
    # megatron:d<dp>t<tp>.
    """Full round trip: MegatronEngine (DP+TP) → weight-update gateway → SGLang.

    TP ranks within a DP group each hold the same full parameter after
    all_gather_param. dp_replicated=True tells awex only one rank per group
    needs to send, avoiding redundant transfers.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    n_infer = n_gpus // 2
    n_train = n_gpus - n_infer
    dp_size = n_train // tp_size
    if dp_size < 1:
        pytest.skip(f"Not enough GPUs for dp={dp_size} tp={tp_size}")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}t{tp_size}",
        pair_name=f"test_megatron_dp{dp_size}tp{tp_size}",
        tag=f"megatron_dp{dp_size}tp{tp_size}",
        tmp_path_factory=tmp_path_factory,
    )


def _run_megatron_awex_e2e(
    *,
    n_gpus: int,
    backend: str,
    pair_name: str,
    tag: str,
    tmp_path_factory,
    model_path: str | None = None,
    validate_param_names: list[str] | None = None,
    init_from_scratch: bool = False,
):
    # [FUNCTION DOC]
    # HELPER: Shared Megatron E2E driver (keyword-only args). n_gpus split half
    # inference (sglang:d<n/2>) and remainder training with the given Megatron backend
    # string; pair_name for the gateway pair; tag names the tmp dir, scheduler,
    # experiment and saved param files; model_path defaults to Qwen3-0.6B;
    # validate_param_names defaults to the dense list; init_from_scratch is forwarded to
    # TrainEngineConfig. Flow identical to the FSDP E2E test (init + randomize
    # inference, init MegatronLMEngine training, gateway connect/update v1/disconnect,
    # /generate smoke test) ending with _validate_weight_update_correctness_megatron.
    # Cleans up all controllers and scheduler workers in finally.
    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_infer = n_gpus // 2
    tmp = tmp_path_factory.mktemp(tag)
    model_path = model_path or _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_infer}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=backend,
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        init_from_scratch=init_from_scratch,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None
    try:
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
            wait=True,
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
        )
        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1
        wu_ctrl.disconnect()

        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight update: {gen_resp.text}"
        )

        _validate_weight_update_correctness_megatron(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            tag=tag,
            param_names=validate_param_names,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,pp_size",
    [(4, 2), (8, 4)],
    ids=["4gpu-dp1pp2", "8gpu-dp1pp4"],
)
def test_awex_megatron_pp_e2e_weight_update(n_gpus, pp_size, tmp_path_factory):
    # [FUNCTION DOC]
    # E2E for Megatron pure PP ((4,2),(8,4)) with backend megatron:d1p<pp>.
    """Full round trip: MegatronEngine (pure PP) → weight-update gateway → SGLang.

    Each PP stage owns a disjoint subset of layers. Validation unions params
    across all PP ranks to reconstruct the full parameter set for comparison.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d1p{pp_size}",
        pair_name=f"test_megatron_pp{pp_size}",
        tag=f"megatron_pp{pp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,dp_size,pp_size",
    [(8, 2, 2)],
    ids=["8gpu-dp2pp2"],
)
def test_awex_megatron_dp_pp_e2e_weight_update(
    n_gpus, dp_size, pp_size, tmp_path_factory
):
    # [FUNCTION DOC]
    # E2E for Megatron DP+PP (8 GPUs, dp2 pp2) with backend megatron:d<dp>p<pp>.
    """Full round trip: MegatronEngine (DP+PP) → weight-update gateway → SGLang.

    Combines data parallelism (multiple replicas) with pipeline parallelism
    (each replica split across PP stages). Each PP stage of each DP replica
    sends its own layer subset independently.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}p{pp_size}",
        pair_name=f"test_megatron_dp{dp_size}pp{pp_size}",
        tag=f"megatron_dp{dp_size}pp{pp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,pp_size,tp_size",
    [(8, 2, 2)],
    ids=["8gpu-dp1pp2tp2"],
)
def test_awex_megatron_pp_tp_e2e_weight_update(
    n_gpus, pp_size, tp_size, tmp_path_factory
):
    # [FUNCTION DOC]
    # E2E for Megatron PP+TP (8 GPUs, pp2 tp2) with backend megatron:d1p<pp>t<tp>.
    """Full round trip: MegatronEngine (PP+TP) → weight-update gateway → SGLang.

    PP splits layers across stages; TP shards each stage's weights across ranks.
    Both dp_replicated=True (TP) and disjoint layer ownership (PP) apply.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d1p{pp_size}t{tp_size}",
        pair_name=f"test_megatron_pp{pp_size}tp{tp_size}",
        tag=f"megatron_pp{pp_size}tp{tp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,cp_size",
    [(4, 2), (8, 4)],
    ids=["4gpu-dp1cp2", "8gpu-dp1cp4"],
)
def test_awex_megatron_cp_e2e_weight_update(n_gpus, cp_size, tmp_path_factory):
    # [FUNCTION DOC]
    # E2E for Megatron pure CP ((4,2),(8,4)) with backend megatron:d1c<cp>.
    """Full round trip: MegatronEngine (pure CP) → weight-update gateway → SGLang.

    CP splits the sequence across ranks for attention but all CP ranks hold
    identical parameters. dp_replicated=True tells awex only one CP rank per
    group needs to send.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d1c{cp_size}",
        pair_name=f"test_megatron_cp{cp_size}",
        tag=f"megatron_cp{cp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,dp_size,cp_size",
    [(8, 2, 2)],
    ids=["8gpu-dp2cp2"],
)
def test_awex_megatron_dp_cp_e2e_weight_update(
    n_gpus, dp_size, cp_size, tmp_path_factory
):
    # [FUNCTION DOC]
    # E2E for Megatron DP+CP (8 GPUs, dp2 cp2) with backend megatron:d<dp>c<cp>.
    """Full round trip: MegatronEngine (DP+CP hybrid) → weight-update gateway → SGLang.

    Combines data parallelism with context parallelism. Each DP replica has
    cp_size CP ranks all holding identical parameters. dp_replicated=True
    ensures only one CP rank per DP group sends, avoiding redundant transfers.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}c{cp_size}",
        pair_name=f"test_megatron_dp{dp_size}cp{cp_size}",
        tag=f"megatron_dp{dp_size}cp{cp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,ep_size",
    [(4, 2), (8, 4)],
    ids=["4gpu-dp2ep2", "8gpu-dp4ep4"],
)
def test_awex_megatron_ep_e2e_weight_update(n_gpus, ep_size, tmp_path_factory):
    # [FUNCTION DOC]
    # E2E for Megatron EP ((4,2),(8,4)) using a 4-layer truncated Qwen3-30B-A3B model
    # initialized from scratch, backend megatron:d<n_train>e<ep>, validated with
    # _VALIDATE_PARAM_NAMES_MOE.
    # KNOWN ISSUE: requires the local MoE model directory (see
    # _make_truncated_moe_model).
    """Full round trip: MegatronEngine (EP) → weight-update gateway → SGLang.

    Each EP rank owns a different subset of expert parameters while attention
    and norm weights are replicated. Uses a truncated Qwen3-30B-A3B MoE model
    (4 layers) with init_from_scratch=True to avoid loading full weights.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    n_train = n_gpus - n_gpus // 2
    tmp = tmp_path_factory.mktemp(f"megatron_ep{ep_size}_model")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{n_train}e{ep_size}",
        pair_name=f"test_megatron_ep{ep_size}",
        tag=f"megatron_ep{ep_size}",
        tmp_path_factory=tmp_path_factory,
        model_path=_make_truncated_moe_model(tmp, num_layers=4),
        validate_param_names=_VALIDATE_PARAM_NAMES_MOE,
        init_from_scratch=True,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,dp_size,ep_size",
    [(8, 4, 2)],
    ids=["8gpu-dp4ep2"],
)
def test_awex_megatron_dp_ep_e2e_weight_update(
    n_gpus, dp_size, ep_size, tmp_path_factory
):
    # [FUNCTION DOC]
    # E2E for Megatron DP+EP (8 GPUs, dp4 ep2) with the truncated MoE model,
    # init_from_scratch, backend megatron:d<dp>e<ep> and MoE validation names.
    """Full round trip: MegatronEngine (DP+EP hybrid) → weight-update gateway → SGLang.

    Combines data parallelism with expert parallelism. Each DP replica has
    ep_size EP ranks owning different expert subsets. Non-expert params
    (attention, norms) are replicated across all ranks.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    tmp = tmp_path_factory.mktemp(f"megatron_dp{dp_size}ep{ep_size}_model")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}e{ep_size}",
        pair_name=f"test_megatron_dp{dp_size}ep{ep_size}",
        tag=f"megatron_dp{dp_size}ep{ep_size}",
        tmp_path_factory=tmp_path_factory,
        model_path=_make_truncated_moe_model(tmp, num_layers=4),
        validate_param_names=_VALIDATE_PARAM_NAMES_MOE,
        init_from_scratch=True,
    )


# ---------------------------------------------------------------------------
# Colocated weight update: Megatron + SGLang on the SAME GPUs (pure DP)
# ---------------------------------------------------------------------------


def _run_megatron_colocate_e2e(
    *,
    n_gpus: int,
    pair_name: str,
    tag: str,
    tmp_path_factory,
    model_path: str | None = None,
):
    # [FUNCTION DOC]
    # HELPER: Colocated Megatron E2E driver (keyword-only). Both SGLang
    # (sglang:d<n_gpus>) and Megatron (megatron:d<n_gpus>, pure DP) span all n_gpus
    # GPUs; the scheduler's round-robin wraps so each GPU hosts one inference and one
    # training worker. Same flow as _run_megatron_awex_e2e except controllers are
    # initialized without wait=True and connect passes colocate=True (transfer via CUDA
    # IPC). Validates with the default dense param list and cleans up in finally.
    """Colocated weight transfer: MegatronEngine + SGLang share the same GPUs.

    Unlike the separated tests where inference and training each own a
    disjoint half of the GPUs, colocated mode puts both on every GPU.
    The LocalScheduler round-robin counter naturally wraps, giving
    inference GPUs 0..N-1 and training GPUs 0..N-1 (same devices).

    Only pure DP is supported for colocated mode (TP=1, PP=1, EP=1).
    """
    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    tmp = tmp_path_factory.mktemp(tag)
    model_path = model_path or _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    # Both inference and training use ALL n_gpus GPUs (colocated).
    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_gpus}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"megatron:d{n_gpus}",
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None
    try:
        # -- 1. SGLang inference (uses GPUs 0..N-1) -------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        # Randomize inference weights so the transfer is NOT a no-op.
        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. Megatron training (wraps to same GPUs 0..N-1) ---------------
        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway ---------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        # -- 4. Connect with colocate=True ----------------------------------
        wu_ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            colocate=True,
        )

        # -- 5. Colocated weight update -------------------------------------
        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1
        wu_ctrl.disconnect()

        # -- 6. Verify inference server still works post-update -------------
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight update: {gen_resp.text}"
        )

        # -- 7. Validate training ↔ inference parameter equality ------------
        _validate_weight_update_correctness_megatron(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            tag=tag,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_megatron_colocate_dp_e2e_weight_update(n_gpus, tmp_path_factory):
    # [FUNCTION DOC]
    # Colocated pure-DP E2E over 2/4/8 GPUs via _run_megatron_colocate_e2e.
    """Full round trip: colocated MegatronEngine (pure DP) + SGLang on same GPUs.

    Unlike separated tests that split GPUs between training and inference,
    colocated mode shares all GPUs.  Weight transfer uses CUDA IPC
    (zero-copy on same device) instead of NCCL P2P across devices.

    Only pure DP (TP=1, PP=1, EP=1) is supported for colocated mode.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_colocate_e2e(
        n_gpus=n_gpus,
        pair_name=f"test_megatron_colocate_dp{n_gpus}",
        tag=f"megatron_colocate_dp{n_gpus}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
def test_awex_megatron_colocate_dp_multi_version_e2e(tmp_path_factory):
    # [FUNCTION DOC]
    # Colocated 2-GPU Megatron+SGLang test performing two sequential updates (version 1
    # then 2) on the same connection, asserting both succeed, then the /generate smoke
    # test and Megatron param validation. Setup and cleanup duplicate
    # _run_megatron_colocate_e2e inline.
    """Colocated weight update with multiple sequential versions.

    Verifies that the colocated IPC path correctly handles version
    sequencing: version 1 → version 2.  The KV store keys include
    the version number, so each round must produce fresh IPC handles.
    """
    n_gpus = 2
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    tag = "megatron_colocate_multi_ver"
    tmp = tmp_path_factory.mktemp(tag)
    model_path = _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_gpus}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"megatron:d{n_gpus}",
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None
    try:
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check()

        wu_ctrl.connect(
            pair_name="test_colocate_multi_ver",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            colocate=True,
        )

        # Version 1
        result1 = wu_ctrl.update_weights(version=1)
        assert result1.status == "ok"
        assert result1.version == 1

        # Version 2
        result2 = wu_ctrl.update_weights(version=2)
        assert result2.status == "ok"
        assert result2.version == 2

        wu_ctrl.disconnect()

        # Verify inference still works after two sequential updates
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight updates: {gen_resp.text}"
        )

        # Final parameter equality check
        _validate_weight_update_correctness_megatron(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            tag=tag,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)

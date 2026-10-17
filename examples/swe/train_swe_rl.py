# [MODULE DOC]
# Training entry script for SWE-bench agent RL in AReaL proxy mode: loads SWE-bench
# JSONL datasets, optionally installs AReaL-SWEAgent dependencies on Ray GPU nodes, and
# runs PPOTrainer with examples.swe.agent.SWEAgentWorkflow.
"""Training script for SWE-bench agent RL with AReaL proxy mode."""

import json
import sys
import warnings
from pathlib import Path
from typing import Any

from datasets import Dataset

from examples.swe.utils import SWEPPOConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import logging

logger = logging.getLogger("SWETrain")


def get_swe_dataset(
    dataset_path: str,
    split: str = "train",
    min_items: int = 64,
) -> Dataset:
    # [FUNCTION DOC]
    # Reads a JSONL file of SWE-bench instances into a HuggingFace Dataset. Raises
    # FileNotFoundError if missing. Skips blank lines; skips (with a warning) records
    # lacking
    # instance_id or problem_statement; raises ValueError if nothing valid remains;
    # malformed
    # JSON raises from json. If fewer than min_items (64) records, repeatedly appends
    # copies
    # of the original list until at least min_items. split is only used in the log
    # message.
    # Returns the Dataset.
    """Create a HuggingFace Dataset from a SWE-bench JSONL file.

    Each line in the JSONL file should be a SWE-bench instance with at minimum:
    - instance_id: The SWE-bench instance ID (e.g., "django__django-10097")
    - problem_statement: The GitHub issue description
    - eval_script: Shell script to evaluate the agent's fix

    Args:
        dataset_path: Path to the SWE-bench JSONL file.
        split: Informational split label (not used for filtering).
        min_items: Minimum dataset size; items are duplicated if fewer exist.

    Returns:
        HuggingFace Dataset of SWE-bench instances.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench dataset not found: {dataset_path}")

    dataset_items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "instance_id" not in item:
                logger.warning(f"Skipping item missing 'instance_id': {line[:100]}")
                continue
            if "problem_statement" not in item:
                logger.warning(
                    "Skipping item missing 'problem_statement': "
                    f"{item.get('instance_id')}"
                )
                continue
            dataset_items.append(item)

    if not dataset_items:
        raise ValueError(f"No valid items found in dataset: {dataset_path}")

    # Duplicate dataset if fewer than min_items for efficient batching
    if len(dataset_items) < min_items:
        original_items = dataset_items.copy()
        while len(dataset_items) < min_items:
            dataset_items.extend(original_items)

    dataset = Dataset.from_list(dataset_items)
    logger.info(
        f"Created SWE dataset with {len(dataset)} items "
        f"from {dataset_path} (split={split})"
    )
    return dataset


def group_filter(x: dict[str, Any]):
    # [FUNCTION DOC]
    # Dynamic filter: keeps a group iff the mean of its "rewards" tensor is <= 0.95.
    # Not used by main (which uses config.should_accept_fn).
    """Filter out groups where all rollouts already solved the task."""
    return x["rewards"].mean() <= 0.95


def _install_aweagent_deps_on_ray_nodes(aweagent_root: str):
    # [FUNCTION DOC]
    # HELPER: If Ray is initialized, runs a zero-GPU remote task pinned to every alive
    # GPU
    # node (via node:<ip> custom resource) that runs uv pip install -r on the agent
    # root's
    # requirements.txt (120 s timeout), waits up to 180 s, and logs per-node success or
    # the
    # stderr tail on failure. aweagent_root defaults to ../AReaL-SWEAgent next to the
    # repo
    # when empty. All exceptions are caught and logged as warnings. Spawns subprocesses
    # on
    # remote nodes.
    """Install AReaL-SWEAgent dependencies on all Ray GPU nodes.

    Each node runs in a separate container with its own venv,
    so we must ensure packages like ``aenv`` are installed everywhere.
    """
    if not aweagent_root:
        aweagent_root = str(
            Path(__file__).resolve().parents[2].parent / "AReaL-SWEAgent"
        )
    try:
        import ray

        if not ray.is_initialized():
            return

        @ray.remote(num_gpus=0)
        def _install():
            # [FUNCTION DOC]
            # HELPER: Ray remote task: installs the requirements on the current node and
            # returns
            # (node IP, return code, last 200 characters of stderr).
            import os
            import socket
            import subprocess

            ip = socket.gethostbyname(socket.gethostname())
            req_path = os.path.join(aweagent_root, "requirements.txt")
            result = subprocess.run(
                ["uv", "pip", "install", "-r", req_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return (
                ip,
                result.returncode,
                result.stderr[-200:] if result.stderr else "",
            )

        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0
        ]
        refs = []
        for node in nodes:
            node_ip = node["NodeManagerAddress"]
            refs.append(_install.options(resources={f"node:{node_ip}": 0.01}).remote())

        results = ray.get(refs, timeout=180)
        for ip, rc, err in results:
            if rc != 0:
                logger.warning(f"Failed to install AReaL-SWEAgent deps on {ip}: {err}")
            else:
                logger.info(f"AReaL-SWEAgent deps installed on {ip}")
    except Exception as e:
        logger.warning(f"Could not install AReaL-SWEAgent deps on Ray nodes: {e}")


def _resolve_aweagent_root(econfig) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the first non-empty of econfig.agent_root, aweagent_root and
    # swe_agent_root attributes (missing attributes count as empty), else "".
    return (
        getattr(econfig, "agent_root", "")
        or getattr(econfig, "aweagent_root", "")
        or getattr(econfig, "swe_agent_root", "")
    )


def main(args):
    # [FUNCTION DOC]
    # Suppresses pydantic UserWarnings, parses SWEPPOConfig, and for the Ray scheduler
    # connects to the existing cluster and installs agent deps on all nodes. Resolves
    # train
    # and valid dataset paths (absolute or existing paths kept; otherwise tried relative
    # to
    # econfig.dataset_path), loads both datasets, and runs PPOTrainer.train with
    # SWEAgentWorkflow kwargs (econfig as dict, gen_args with gconfig temperature and
    # max_new_tokens, timeout from econfig) and dynamic_filter_fn =
    # config.should_accept_fn.
    # KNOWN ISSUE: eval_workflow is None, so the temperature-0 eval kwargs (and the
    # loaded
    # validation dataset) are effectively unused for evaluation.
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    config, _ = load_expr_config(args, SWEPPOConfig)
    econfig = config.econfig

    # When using Ray scheduler, ensure SWEAgent deps are on all nodes
    if config.scheduler.type == "ray":
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        _install_aweagent_deps_on_ray_nodes(_resolve_aweagent_root(econfig))

    # Resolve dataset paths from config
    train_path = config.train_dataset.path
    valid_path = config.valid_dataset.path

    def resolve_path(p: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Returns p if absolute or existing; otherwise econfig.dataset_path
        # joined with p
        # if that exists; otherwise p unchanged.
        if Path(p).is_absolute() or Path(p).exists():
            return p
        if econfig.dataset_path:
            candidate = Path(econfig.dataset_path) / p
            if candidate.exists():
                return str(candidate)
        return p

    train_dataset = get_swe_dataset(
        dataset_path=resolve_path(train_path),
        split="train",
    )
    valid_dataset = get_swe_dataset(
        dataset_path=resolve_path(valid_path),
        split="test",
    )

    # Build workflow kwargs
    from dataclasses import asdict

    econfig_dict = asdict(econfig)
    workflow_kwargs = dict(
        econfig=econfig_dict,
        gen_args=dict(
            temperature=config.gconfig.temperature,
            max_completion_tokens=config.gconfig.max_new_tokens,
        ),
        timeout=econfig.timeout,
    )

    # Eval workflow with lower temperature for deterministic evaluation
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gen_args"] = dict(
        temperature=0.0,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.swe.agent.SWEAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dynamic_filter_fn=getattr(config, "should_accept_fn", None),
        )


if __name__ == "__main__":
    main(sys.argv[1:])

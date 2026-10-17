# [MODULE DOC]
# Utility to pre-build Terminal-Bench task docker images before training. DATASET_ROOT
# is
# the "dataset" directory three levels above this file's directory (repo-relative).
from pathlib import Path

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager

DATASET_ROOT = Path(__file__).resolve().parents[3] / "dataset"


def build_docker_image(task: dict, timeout=1200.0):
    # [FUNCTION DOC]
    # Builds the docker-compose images for one Terminal-Bench task. Inputs: task (dict
    # with
    # "task_path" relative to DATASET_ROOT), timeout (seconds for the build, default
    # 1200).
    # Creates a TrialHandler (trial name "build_run", outputs to ./build_outputs),
    # prints the
    # task path to stdout, constructs a DockerComposeManager from the handler's
    # container/image
    # names and paths with no_rebuild=True and cleanup=True, then calls build. Returns
    # None;
    # side effects are docker builds and filesystem output. Raises whatever
    # terminal_bench
    # raises (e.g. missing task_path gives a TypeError when joining None).
    # KNOWN ISSUE: uses print instead of the AReaL logger.
    task_path = DATASET_ROOT / task.get("task_path")
    trial_handler = TrialHandler(
        trial_name="build_run",
        input_path=task_path,
        output_path=Path("build_outputs"),
    )
    print(f"Task path: {task_path}")

    compose_manager = DockerComposeManager(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        no_rebuild=True,
        cleanup=True,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
    )
    compose_manager.build(timeout=timeout)

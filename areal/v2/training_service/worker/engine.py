# [MODULE DOC]
# Builds the training worker's engine Flask blueprint. Dependencies (flask module,
# engine accessors, engine-thread submitter, parsing and execution helpers) are injected
# by app.py. Compute routes localize incoming RTensor references, run the engine method
# with DP-head broadcast and return results as RTensor references owned by this node;
# other routes call an engine method directly on the engine thread.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from typing import Any

from areal.api import TrainEngine
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("TrainWorker")


def create_engine_module(
    *,
    flask_module: Any,
    config: Any,
    get_engine: Callable[[], TrainEngine | None],
    set_engine: Callable[[TrainEngine], None],
    submit_to_engine_thread: Callable[..., Any],
    parse_args_kwargs: Callable[[dict[str, Any] | None], tuple[Any, Any]],
    require_engine: Callable[[], TrainEngine],
    run_endpoint: Callable[[str, Callable[[], Any]], Any],
    execute_compute: Callable[..., Any],
    get_node_addr: Callable[[], str],
) -> Any:
    # [FUNCTION DOC]
    # Creates blueprint "worker_engine" with: GET /health, POST /create_engine, POST
    # /configure, GET /topology, GET /get_param_info, POST /destroy_engine; plain engine
    # routes /train and /eval (no result), /create_process_group, /initialize,
    # /set_version, /save, /load, /offload, /onload, /optimizer_zero_grad,
    # /optimizer_step, /step_lr_scheduler, /get_device_stats, /config_perf_tracer,
    # /save_perf_tracer, /clear_batches (POST) and /get_version, /export_stats (GET);
    # and compute routes /train_batch, /forward_batch, /eval_batch, /sft/train
    # (train_lm), /sft/evaluate (evaluate_lm), /ppo/actor/compute_logp,
    # /ppo/actor/compute_advantages, /ppo/actor/update (ppo_update),
    # /ppo/critic/compute_values, /ppo/critic/update (ppo_update), /rw/train (train_rw),
    # /rw/evaluate (evaluate_rw). Returns the blueprint.
    Blueprint = flask_module.Blueprint
    jsonify = flask_module.jsonify
    request = flask_module.request

    bp = Blueprint("worker_engine", __name__)

    # -- core routes -------------------------------------------------------

    @bp.route("/health", methods=["GET"])
    def health_check():
        # [FUNCTION DOC]
        # GET /health: returns status healthy, RANK and ROLE env values (defaults 0 and
        # "train_worker") and ready = whether an engine exists.
        rank = int(os.environ.get("RANK", 0))
        role = os.environ.get("ROLE", "train_worker")
        ready = get_engine() is not None
        return jsonify(
            {
                "status": "healthy",
                "rank": rank,
                "role": role,
                "ready": ready,
            }
        )

    @bp.route("/create_engine", methods=["POST"])
    def create_engine():
        # [FUNCTION DOC]
        # POST /create_engine. Body: engine_class (or legacy "engine") import path,
        # serialized init_args and init_kwargs. 400 for invalid JSON, missing class,
        # existing engine, import failure or a class that is not a TrainEngine subclass.
        # Instantiates the engine on the engine thread, stores it globally and returns
        # success; unexpected errors give 500.
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON in request body"}), 400

            engine_class_path = data.get("engine_class")
            if engine_class_path is None:
                engine_class_path = data.get("engine")
            init_args = deserialize_value(data.get("init_args", []))
            init_kwargs = deserialize_value(data.get("init_kwargs", {}))

            if not engine_class_path:
                return jsonify(
                    {"error": "Missing 'engine_class' field in request"}
                ), 400
            if get_engine() is not None:
                return jsonify({"error": "Engine already exists on this worker"}), 400

            try:
                engine_class = import_from_string(engine_class_path)
                if not issubclass(engine_class, TrainEngine):
                    raise TypeError(
                        "Engine class must be a subclass of TrainEngine, "
                        f"got {engine_class}."
                    )
            except (ValueError, ImportError, AttributeError) as e:
                return (
                    jsonify(
                        {
                            "error": (
                                f"Failed to import engine class '{engine_class_path}': {str(e)}"
                            )
                        }
                    ),
                    400,
                )
            except TypeError as e:
                return jsonify({"error": str(e)}), 400

            def create_in_engine_thread():
                # [FUNCTION DOC]
                # HELPER: Constructs the engine instance with the deserialized init
                # args.
                return engine_class(*init_args, **init_kwargs)

            engine = submit_to_engine_thread("create_engine", create_in_engine_thread)
            set_engine(engine)
            return jsonify(
                {
                    "status": "success",
                    "message": "Engine created and initialized",
                    "result": None,
                }
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in create_engine: {e}\n{traceback.format_exc()}"
            )
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @bp.route("/configure", methods=["POST"])
    def configure():
        # [FUNCTION DOC]
        # POST /configure: calls engine.configure(*args, **kwargs) on the engine thread
        # if the engine has it (else returns None) and returns the serialized result.
        # KNOWN ISSUE: args are parsed outside the error wrapper, so an invalid JSON
        # body raises ValueError as an unhandled Flask 500 instead of 400.
        data = request.get_json(silent=True)
        raw_args, raw_kwargs = parse_args_kwargs(data)

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call of the optional configure method.
            engine = require_engine()
            configure_fn = getattr(engine, "configure", None)
            if callable(configure_fn):
                return configure_fn(*raw_args, **raw_kwargs)
            return None

        return run_endpoint(
            "configure",
            lambda: submit_to_engine_thread("configure", action),
        )

    @bp.route("/topology", methods=["GET"])
    def topology():
        # [FUNCTION DOC]
        # GET /topology: returns rank/world_size/local_rank from env and dp_rank,
        # dp_size, is_dp_head from the engine; 400 if no engine, 500 on other errors.
        # Runs on the request thread, not the engine thread.
        try:
            engine = require_engine()
            return jsonify(
                {
                    "rank": int(os.environ.get("RANK", 0)),
                    "world_size": int(os.environ.get("WORLD_SIZE", 1)),
                    "dp_rank": engine.data_parallel_rank,
                    "dp_size": engine.data_parallel_world_size,
                    "is_dp_head": engine.is_data_parallel_head(),
                    "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
                }
            )
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error in topology: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @bp.route("/get_param_info", methods=["GET"])
    def get_param_info():
        # [FUNCTION DOC]
        # GET /get_param_info: returns engine.get_param_info() or, as fallback,
        # get_parameter_info(), or None, executed on the engine thread.
        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread lookup of the parameter-info method.
            engine = require_engine()
            get_param_info_fn = getattr(engine, "get_param_info", None)
            if callable(get_param_info_fn):
                return get_param_info_fn()
            get_parameter_info_fn = getattr(engine, "get_parameter_info", None)
            if callable(get_parameter_info_fn):
                return get_parameter_info_fn()
            return None

        return run_endpoint(
            "get_param_info",
            lambda: submit_to_engine_thread("get_param_info", action),
        )

    # -- dispatch helpers --------------------------------------------------

    def _register_compute_route(
        path: str, method_name: str, *, endpoint_prefix: str = ""
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Registers POST path whose handler parses args/kwargs, localizes
        # RTensors, executes method_name with require_broadcast and converts the result
        # to RTensor references for this node, wrapped by run_endpoint. Endpoint name
        # uses endpoint_prefix when given to keep names unique.
        def handler():
            # [FUNCTION DOC]
            # HELPER: Request handler implementing the compute flow described above.
            data = request.get_json(silent=True)
            raw_args, raw_kwargs = parse_args_kwargs(data)
            args = RTensor.localize(raw_args)
            kwargs = RTensor.localize(raw_kwargs)
            result = execute_compute(
                method_name,
                args,
                kwargs,
                require_broadcast=True,
            )
            return RTensor.remotize(result, node_addr=get_node_addr())

        ep_name = (
            f"{endpoint_prefix}{method_name}_endpoint"
            if endpoint_prefix
            else f"{method_name}_compute_endpoint"
        )
        bp.add_url_rule(
            path,
            ep_name,
            lambda: run_endpoint(method_name, handler),
            methods=["POST"],
        )

    def _register_engine_route(
        path: str,
        method_name: str,
        *,
        methods: list[str] | None = None,
        return_result: bool = True,
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Registers path (default POST) calling engine.method_name on the engine
        # thread; GET requests pass no args, POST bodies are parsed (missing body means
        # no args). return_result False suppresses the result in the response.
        http_methods = methods or ["POST"]

        def handler():
            # [FUNCTION DOC]
            # HELPER: Request handler implementing the direct-call flow described above.
            if request.method == "GET":
                args, kwargs = [], {}
            else:
                data = request.get_json(silent=True) or {}
                args, kwargs = parse_args_kwargs(data)

            return run_endpoint(
                method_name,
                lambda: submit_to_engine_thread(
                    method_name,
                    lambda: getattr(require_engine(), method_name)(*args, **kwargs),
                ),
                return_result=return_result,
            )

        bp.add_url_rule(path, f"{method_name}_endpoint", handler, methods=http_methods)

    # -- lifecycle routes --------------------------------------------------

    @bp.route("/destroy_engine", methods=["POST"])
    def destroy_engine():
        # [FUNCTION DOC]
        # POST /destroy_engine: if an engine exists, calls engine.destroy() on the
        # engine thread (tears down the process group; must be called on all ranks
        # before exit); returns success, or a "No engine to destroy" message. The global
        # engine reference is not cleared.
        """Gracefully destroy the engine and release distributed resources.

        Calls ``engine.destroy()`` which runs
        ``dist.destroy_process_group()`` (a local ``ncclCommAbort`` +
        HeartbeatMonitor join).  Must be called on **all** workers before
        any process exits so that rank-0's TCPStore server outlives every
        other rank's HeartbeatMonitor thread.
        """
        engine = get_engine()
        if engine is None:
            return jsonify({"status": "success", "message": "No engine to destroy"})

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call of engine.destroy.
            engine.destroy()

        return run_endpoint(
            "destroy_engine",
            lambda: submit_to_engine_thread("destroy_engine", action),
            return_result=False,
        )

    # -- engine routes -----------------------------------------------------

    _register_engine_route("/train", "train", return_result=False)
    _register_engine_route("/eval", "eval", return_result=False)
    _register_compute_route("/train_batch", "train_batch")
    _register_compute_route("/forward_batch", "forward_batch")
    _register_compute_route("/eval_batch", "eval_batch")
    _register_engine_route("/create_process_group", "create_process_group")
    _register_engine_route("/initialize", "initialize")
    _register_engine_route("/set_version", "set_version")
    _register_engine_route("/get_version", "get_version", methods=["GET"])
    _register_engine_route("/save", "save")
    _register_engine_route("/load", "load")
    _register_engine_route("/offload", "offload")
    _register_engine_route("/onload", "onload")
    _register_engine_route("/optimizer_zero_grad", "optimizer_zero_grad")
    _register_engine_route("/optimizer_step", "optimizer_step")
    _register_engine_route("/step_lr_scheduler", "step_lr_scheduler")
    _register_engine_route("/get_device_stats", "get_device_stats")
    _register_engine_route("/config_perf_tracer", "config_perf_tracer")
    _register_engine_route("/save_perf_tracer", "save_perf_tracer")
    _register_engine_route("/clear_batches", "clear_batches")
    _register_engine_route("/export_stats", "export_stats", methods=["GET"])

    # -- SFT routes --------------------------------------------------------

    _register_compute_route("/sft/train", "train_lm")
    _register_compute_route("/sft/evaluate", "evaluate_lm")

    # -- PPO actor routes --------------------------------------------------

    _register_compute_route(
        "/ppo/actor/compute_logp", "compute_logp", endpoint_prefix="ppo_actor_"
    )
    _register_compute_route(
        "/ppo/actor/compute_advantages",
        "compute_advantages",
        endpoint_prefix="ppo_actor_",
    )
    _register_compute_route(
        "/ppo/actor/update", "ppo_update", endpoint_prefix="ppo_actor_"
    )

    # -- PPO critic routes -------------------------------------------------

    _register_compute_route(
        "/ppo/critic/compute_values",
        "compute_values",
        endpoint_prefix="ppo_critic_",
    )
    _register_compute_route(
        "/ppo/critic/update", "ppo_update", endpoint_prefix="ppo_critic_"
    )

    # -- RW routes ---------------------------------------------------------

    _register_compute_route("/rw/train", "train_rw", endpoint_prefix="rw_")
    _register_compute_route("/rw/evaluate", "evaluate_rw", endpoint_prefix="rw_")

    return bp

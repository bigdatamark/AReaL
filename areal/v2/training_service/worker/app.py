# [MODULE DOC]
# Flask app factory and process-global engine state for a training worker. All engine
# operations run serially on a single daemon "EngineWorker" thread (so CUDA/NCCL work
# happens on one thread), fed through a queue with futures; HTTP handler threads block
# until their work completes. Holds the global engine instance and this node's address
# used for remote-tensor (RTensor) references.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from queue import Queue
from threading import Lock, Thread
from typing import Any

from areal.api import TrainEngine
from areal.infra.platforms import current_platform
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging
from areal.utils.data import broadcast_tensor_container, tensor_container_to
from areal.v2.training_service.worker.awex import create_awex_blueprint
from areal.v2.training_service.worker.config import TrainWorkerConfig
from areal.v2.training_service.worker.engine import create_engine_module

logger = logging.getLogger("TrainWorker")

_engine: TrainEngine | None = None
_node_addr: str = ""

_engine_thread: Thread | None = None
_engine_work_queue: Queue | None = None
_engine_thread_lock = Lock()


def _init_engine_thread() -> None:
    # [FUNCTION DOC]
    # HELPER: Idempotently starts the engine thread under a lock: returns if it is
    # alive, raises RuntimeError if it exists but died, else creates the work queue and
    # starts the daemon thread.
    global _engine_thread, _engine_work_queue

    with _engine_thread_lock:
        if _engine_thread is not None:
            if _engine_thread.is_alive():
                return
            else:
                raise RuntimeError("Engine thread is dead.")

        _engine_work_queue = Queue()

        def engine_worker():
            # [FUNCTION DOC]
            # HELPER: Engine thread loop: takes (func, args, kwargs, future, name) items
            # from the queue; a None item stops the loop. Runs func, setting the result
            # or exception on the future (exceptions are logged with traceback), and
            # marks the item done.
            logger.info("Engine thread started")
            work_queue = _engine_work_queue
            if work_queue is None:
                raise RuntimeError("Engine work queue not initialized")
            while True:
                work_item = None
                func_name = "<unknown>"
                try:
                    work_item = work_queue.get()
                    if work_item is None:
                        logger.info("Engine thread shutting down")
                        break

                    func, args, kwargs, future, func_name = work_item
                    try:
                        result = func(*args, **kwargs)
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
                        logger.error(
                            f"Error in engine thread when "
                            f"running {func_name}: {e}\n{traceback.format_exc()}"
                        )
                    finally:
                        work_queue.task_done()
                except Exception as e:
                    logger.error(
                        f"Error in engine thread when "
                        f"running {func_name}: {e}\n{traceback.format_exc()}"
                    )
                    if work_item and len(work_item) > 3:
                        work_item[3].set_exception(e)

        _engine_thread = Thread(target=engine_worker, daemon=True, name="EngineWorker")
        _engine_thread.start()
        logger.info("Engine thread initialized")


def _submit_to_engine_thread(
    func_name: str, func: Callable, *args: Any, **kwargs: Any
) -> Any:
    # [FUNCTION DOC]
    # HELPER: Ensures the engine thread is running, enqueues func with args/kwargs and a
    # new Future, and blocks until it completes, returning the result or re-raising the
    # exception from the engine thread.
    global _engine_work_queue

    _init_engine_thread()
    if _engine_work_queue is None:
        raise RuntimeError("Engine work queue not initialized")

    future: Future = Future()
    _engine_work_queue.put((func, args, kwargs, future, func_name))
    return future.result()


def _require_engine() -> TrainEngine:
    # [FUNCTION DOC]
    # HELPER: Returns the global engine or raises RuntimeError if none was created.
    if _engine is None:
        raise RuntimeError("Engine not created. Call /create_engine first.")
    return _engine


def _execute_compute(
    method_name: str,
    args: Any,
    kwargs: Any,
    *,
    require_broadcast: bool = False,
) -> Any:
    # [FUNCTION DOC]
    # HELPER: Runs engine.method_name(*args, **kwargs) on the engine thread;
    # RuntimeError if the method does not exist. With require_broadcast, first moves
    # args/kwargs to the current device and broadcasts them from the DP head over the
    # engine's context-and-model-parallel group so non-head ranks (which received empty
    # payloads) get the real inputs; if that group is None it raises when DP world size
    # > 1 and otherwise uses local inputs. Returns the method result.
    engine = _require_engine()
    method = getattr(engine, method_name, None)
    if not callable(method):
        raise RuntimeError(f"Engine does not implement method '{method_name}'")

    def execute():
        # [FUNCTION DOC]
        # HELPER: Engine-thread closure performing the optional broadcast and the call.
        nonlocal args, kwargs
        if require_broadcast:
            group = engine.context_and_model_parallel_group
            if group is None:
                if engine.data_parallel_world_size > 1:
                    raise RuntimeError(
                        "Broadcast required for endpoint, but "
                        "engine.context_and_model_parallel_group is None"
                    )
            else:
                args = broadcast_tensor_container(
                    tensor_container_to(args, current_platform.current_device()),
                    src_rank=engine.current_data_parallel_head(),
                    group=group,
                )
                kwargs = broadcast_tensor_container(
                    tensor_container_to(kwargs, current_platform.current_device()),
                    src_rank=engine.current_data_parallel_head(),
                    group=group,
                )
        return method(*args, **kwargs)

    return _submit_to_engine_thread(method_name, execute)


def _parse_args_kwargs(data: dict[str, Any] | None) -> tuple[Any, Any]:
    # [FUNCTION DOC]
    # HELPER: Deserializes "args" (default empty list) and "kwargs" (default empty dict)
    # from a request JSON dict; raises ValueError if data is None.
    if data is None:
        raise ValueError("Invalid JSON in request body")
    raw_args = deserialize_value(data.get("args", []))
    raw_kwargs = deserialize_value(data.get("kwargs", {}))
    return raw_args, raw_kwargs


def create_app(config: TrainWorkerConfig):
    # [FUNCTION DOC]
    # Records this node's address as host:port, builds a Flask app (flask imported
    # dynamically) and registers the engine blueprint, the awex blueprint (both wired
    # through the callbacks below) and the shared RPC guard data blueprint. Returns the
    # app.
    global _node_addr
    _node_addr = f"{config.host}:{config.port}"

    flask = importlib.import_module("flask")
    jsonify = flask.jsonify

    app = flask.Flask(__name__)

    def _run_endpoint(
        endpoint_name: str,
        action: Callable[[], Any],
        return_result: bool = True,
    ):
        # [FUNCTION DOC]
        # HELPER: Runs action and returns JSON {"status": "success", "result":
        # serialized result or None}; RuntimeError/ValueError -> 400 with error; other
        # exceptions are logged and return 500.
        try:
            result = action()
            if return_result:
                return jsonify({"status": "success", "result": serialize_value(result)})
            return jsonify({"status": "success", "result": None})
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error in {endpoint_name}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    def _get_engine() -> TrainEngine | None:
        # [FUNCTION DOC]
        # HELPER: Returns the global engine or None.
        return _engine

    def _set_engine(engine: TrainEngine) -> None:
        # [FUNCTION DOC]
        # HELPER: Sets the global engine instance.
        global _engine
        _engine = engine

    def _get_node_addr() -> str:
        # [FUNCTION DOC]
        # HELPER: Returns this node's host:port address.
        return _node_addr

    app.register_blueprint(
        create_engine_module(
            flask_module=flask,
            config=config,
            get_engine=_get_engine,
            set_engine=_set_engine,
            submit_to_engine_thread=_submit_to_engine_thread,
            parse_args_kwargs=_parse_args_kwargs,
            require_engine=_require_engine,
            run_endpoint=_run_endpoint,
            execute_compute=_execute_compute,
            get_node_addr=_get_node_addr,
        )
    )

    app.register_blueprint(
        create_awex_blueprint(
            flask_module=flask,
            get_engine=_get_engine,
            submit_to_engine_thread=_submit_to_engine_thread,
            run_endpoint=_run_endpoint,
        )
    )

    from areal.infra.rpc.guard.data_blueprint import data_bp

    app.register_blueprint(data_bp)

    return app

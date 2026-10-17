# [MODULE DOC]
# Flask blueprint exposing awex weight-update endpoints (/awex/*) on a training worker.
# An Awex adapter (FSDP or Megatron) is lazily created from the engine and held in
# blueprint-local state; heavy operations run on the engine thread through the injected
# submitter and are wrapped by run_endpoint.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.utils import logging

if TYPE_CHECKING:
    from flask import Blueprint

logger = logging.getLogger("AwexBlueprint")


def create_awex_blueprint(
    *,
    flask_module: Any,
    get_engine: Any,
    submit_to_engine_thread: Any,
    run_endpoint: Any,
) -> Blueprint:
    # [FUNCTION DOC]
    # Creates the "awex" blueprint with: GET report_parallelism, POST
    # report_weight_meta, init_weights_update_group, update_weights, batch_isend_irecv,
    # teardown, init_colocate_weight_update, execute_colocate_weight_update,
    # release_memory, resume_memory and debug/get_parameters. Bodies are parsed with
    # force=True JSON. Returns the blueprint.
    """Create Flask blueprint for awex weight update endpoints.

    Registered alongside the engine blueprint in the training service worker.
    The adapter is lazily created when first needed.

    Follows the same callback injection pattern as create_engine_module().
    """
    bp = flask_module.Blueprint("awex", __name__, url_prefix="/awex")

    _state: dict[str, Any] = {"adapter": None}

    def _require_adapter():
        # [FUNCTION DOC]
        # HELPER: Returns the cached adapter, creating it from the engine on first use;
        # RuntimeError if no engine exists.
        if _state["adapter"] is None:
            engine = get_engine()
            if engine is None:
                raise RuntimeError("Engine not initialized")
            _state["adapter"] = _create_training_adapter(engine)
        return _state["adapter"]

    @bp.route("/report_parallelism", methods=["GET"])
    def report_parallelism():
        # [FUNCTION DOC]
        # GET: returns the adapter's parallelism strategy as JSON (400 on RuntimeError).
        # Runs on the request thread.
        try:
            adapter = _require_adapter()
            return flask_module.jsonify(adapter.parallelism_strategy)
        except RuntimeError as e:
            return flask_module.jsonify({"error": str(e)}), 400

    @bp.route("/report_weight_meta", methods=["POST"])
    def report_weight_meta():
        # [FUNCTION DOC]
        # POST: returns serialized weight metadata computed on the engine thread.
        def action():
            # [FUNCTION DOC]
            # HELPER: Gets weight metadata from the adapter.
            adapter = _require_adapter()
            return adapter.get_weight_metadata()

        return run_endpoint(
            "report_weight_meta",
            lambda: submit_to_engine_thread("report_weight_meta", action),
        )

    @bp.route("/init_weights_update_group", methods=["POST"])
    def init_weights_update_group():
        # [FUNCTION DOC]
        # POST: passes the JSON body as kwargs to adapter.init_weight_update_group.
        data = flask_module.request.get_json(force=True)

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.init_weight_update_group(**data)

        return run_endpoint(
            "init_weights_update_group",
            lambda: submit_to_engine_thread("init_weights_update_group", action),
        )

    @bp.route("/update_weights", methods=["POST"])
    def update_weights():
        # [FUNCTION DOC]
        # POST: runs adapter.execute_weight_update(version) (version from body, default
        # 0).
        data = flask_module.request.get_json(force=True)
        version = data.get("version", 0)

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.execute_weight_update(version)

        return run_endpoint(
            "update_weights",
            lambda: submit_to_engine_thread("update_weights", action),
        )

    @bp.route("/batch_isend_irecv", methods=["POST"])
    def batch_isend_irecv():
        # [FUNCTION DOC]
        # POST: passes the JSON body as kwargs to adapter.batch_isend_irecv (p2p
        # transfers).
        data = flask_module.request.get_json(force=True)

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.batch_isend_irecv(**data)

        return run_endpoint(
            "batch_isend_irecv",
            lambda: submit_to_engine_thread("batch_isend_irecv", action),
        )

    @bp.route("/teardown", methods=["POST"])
    def teardown():
        # [FUNCTION DOC]
        # POST: if an adapter exists, tears down its weight-update group on the engine
        # thread and drops the cached adapter; otherwise returns success immediately.
        adapter = _state.get("adapter")
        if adapter is None:
            return flask_module.jsonify({"status": "success"})

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread teardown and cache reset.
            adapter.teardown_weight_update_group()
            _state["adapter"] = None

        return run_endpoint(
            "awex_teardown",
            lambda: submit_to_engine_thread("awex_teardown", action),
            return_result=False,
        )

    @bp.route("/init_colocate_weight_update", methods=["POST"])
    def init_colocate_weight_update():
        # [FUNCTION DOC]
        # POST: passes the JSON body as kwargs to adapter.init_colocate_weight_update.
        data = flask_module.request.get_json(force=True)

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.init_colocate_weight_update(**data)

        return run_endpoint(
            "init_colocate_weight_update",
            lambda: submit_to_engine_thread("init_colocate_weight_update", action),
            return_result=False,
        )

    @bp.route("/execute_colocate_weight_update", methods=["POST"])
    def execute_colocate_weight_update():
        # [FUNCTION DOC]
        # POST: runs adapter.execute_colocate_weight_update(version) (default 0).
        data = flask_module.request.get_json(force=True)
        version = data.get("version", 0)

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.execute_colocate_weight_update(version)

        return run_endpoint(
            "execute_colocate_weight_update",
            lambda: submit_to_engine_thread("execute_colocate_weight_update", action),
            return_result=False,
        )

    @bp.route("/release_memory", methods=["POST"])
    def release_memory():
        # [FUNCTION DOC]
        # POST: runs adapter.release_memory(tags) with optional body tags.
        data = flask_module.request.get_json(force=True)
        tags = data.get("tags")

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.release_memory(tags)

        return run_endpoint(
            "release_memory",
            lambda: submit_to_engine_thread("release_memory", action),
            return_result=False,
        )

    @bp.route("/resume_memory", methods=["POST"])
    def resume_memory():
        # [FUNCTION DOC]
        # POST: runs adapter.resume_memory(tags) with optional body tags.
        data = flask_module.request.get_json(force=True)
        tags = data.get("tags")

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call described above.
            adapter = _require_adapter()
            adapter.resume_memory(tags)

        return run_endpoint(
            "resume_memory",
            lambda: submit_to_engine_thread("resume_memory", action),
            return_result=False,
        )

    @bp.route("/debug/get_parameters", methods=["POST"])
    def get_parameters():
        # [FUNCTION DOC]
        # POST /debug/get_parameters: saves local shard parameters (required save_path,
        # optional names) to a file for tests. A missing save_path raises KeyError
        # outside the error wrapper (Flask 500).
        """Save local shard parameters to a file for test validation."""
        data = flask_module.request.get_json(force=True)
        save_path = data["save_path"]
        names = data.get("names")

        def action():
            # [FUNCTION DOC]
            # HELPER: Engine-thread call of adapter.save_parameters.
            adapter = _require_adapter()
            adapter.save_parameters(save_path, names)

        return run_endpoint(
            "get_parameters",
            lambda: submit_to_engine_thread("get_parameters", action),
            return_result=False,
        )

    return bp


def _create_training_adapter(engine):
    # [FUNCTION DOC]
    # HELPER: Returns AwexFSDPAdapter for FSDPEngine or AwexMegatronAdapter for
    # MegatronEngine (imports are lazy); raises TypeError for other engine types (e.g.
    # Archon).
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.engine.megatron_engine import MegatronEngine
    from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter
    from areal.v2.weight_update.awex.megatron_adapter import (
        AwexMegatronAdapter,
    )

    if isinstance(engine, FSDPEngine):
        return AwexFSDPAdapter(engine)

    if isinstance(engine, MegatronEngine):
        return AwexMegatronAdapter(engine)

    raise TypeError(
        f"Unsupported engine type for weight update: {type(engine).__name__}"
    )

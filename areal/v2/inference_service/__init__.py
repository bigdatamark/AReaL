# [PACKAGE DOC]
# AReaL v2 inference service package: gateway/router/data-proxy services plus a rollout
# controller and workflow that drive RL rollouts through them. Re-exports
# InferenceServiceWorkflow.
# SPDX-License-Identifier: Apache-2.0

from areal.v2.inference_service.controller.workflow import (
    InferenceServiceWorkflow,
)

__all__ = ["InferenceServiceWorkflow"]

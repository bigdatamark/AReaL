# [PACKAGE DOC]
# Router subpackage: session-affine routing of sessions to DataProxy instances.
# Re-exports create_router_app and the async RouterClient.
# SPDX-License-Identifier: Apache-2.0

from .app import create_router_app
from .client import RouterClient

__all__ = ["RouterClient", "create_router_app"]

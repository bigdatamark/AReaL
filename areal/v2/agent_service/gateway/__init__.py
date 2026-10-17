# [PACKAGE DOC]
# Gateway subpackage: the public HTTP/WebSocket entry point. Re-exports the app
# factory create_gateway_app plus the OpenResponsesBridge and mount_bridge helpers
# that add the OpenAI-Responses-compatible API.
# SPDX-License-Identifier: Apache-2.0

from .app import create_gateway_app
from .bridge import OpenResponsesBridge, mount_bridge

__all__ = ["OpenResponsesBridge", "create_gateway_app", "mount_bridge"]

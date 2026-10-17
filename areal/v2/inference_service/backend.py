# [MODULE DOC]
# Defines the InfBridgeBackend protocol: per-inference-server (SGLang, vLLM) translation
# between AReaL request/response objects and backend HTTP requests, plus control
# requests (pause, resume, offload, onload).
# SPDX-License-Identifier: Apache-2.0

"""Inference bridge backend protocol.

Defines the :class:`InfBridgeBackend` protocol that concrete backends
(SGLang, vLLM, …) must satisfy.  See ``sglang/bridge.py`` and
``vllm/bridge.py`` for implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from areal.api.io_struct import HttpGenerationResult, HttpRequest

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class InfBridgeBackend(Protocol):
    # [CLASS DOC]
    # Runtime-checkable structural protocol; implementations are stateless adapters
    # returning HttpRequest objects (endpoint + payload) and parsing backend JSON.
    """Protocol for inference-server backends used by :class:`InfBridge`.

    Each method converts between AReaL domain objects and HTTP payloads
    specific to a particular inference server (SGLang, vLLM, …).
    """

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: builds the backend generation HttpRequest for req, including LoRA
        # adapter info when with_lora (version identifies the weight version).
        """Translate a :class:`ModelRequest` into a backend-specific HTTP request.

        Parameters
        ----------
        req:
            The model-level generation request.
        with_lora:
            Whether to include LoRA adapter info in the payload.
        version:
            Current weight version (used for LoRA versioning).

        Returns
        -------
        HttpRequest
            An endpoint + JSON payload ready for :pymethod:`InfBridge._send_request`.
        """
        ...

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        # [FUNCTION DOC]
        # Protocol: converts the backend's raw JSON response into an
        # HttpGenerationResult.
        """Parse a raw JSON response from the backend into an
        :class:`HttpGenerationResult`.
        """
        ...

    def get_pause_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: returns the request that pauses generation on the backend.
        """Return the HTTP request that pauses generation on the backend."""
        ...

    def get_resume_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: returns the request that resumes generation.
        """Return the HTTP request that resumes generation on the backend."""
        ...

    def get_offload_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: returns the request that offloads model memory.
        """Return the HTTP request that offloads model memory on the backend."""
        ...

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Protocol: returns the request that reloads model memory, optionally limited to
        # tags.
        """Return the HTTP request that reloads model memory on the backend."""
        ...

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        # [FUNCTION DOC]
        # Protocol: returns the max-new-tokens budget encoded in http_req.
        """Return the current generation budget encoded in ``http_req``."""
        ...

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        # [FUNCTION DOC]
        # Protocol: mutates http_req in place for resubmission after an abort, appending
        # accumulated_tokens to the prompt and setting the budget to remaining_tokens.
        """Mutate ``http_req`` for an abort/resubmit iteration."""
        ...

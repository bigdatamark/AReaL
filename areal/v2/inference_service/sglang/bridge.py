# [MODULE DOC]
# SGLang implementation of the InfBridgeBackend protocol: translates ModelRequest into
# SGLang /generate payloads, parses responses, and supplies control endpoints.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-specific inference bridge backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    get_versioned_lora_name,
)

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


class SGLangBridgeBackend:
    # [CLASS DOC]
    # Stateless backend object mirroring the relevant subset of the remote SGLang engine
    # backend.
    """SGLang-specific backend for :class:`InfBridge`.

    Mirrors the relevant subset of
    :class:`areal.engine.sglang_remote.SGLangBackend`.
    """

    # -- generation ---------------------------------------------------------

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Builds an HttpRequest for /generate. Raises NotImplementedError for beam
        # search. max_new_tokens = min(max_tokens - input length, max_new_tokens).
        # Sampling params: top_p, top_k, max_new_tokens, temperature (0 when greedy),
        # stop_token_ids, ignore_eos, skip_special_tokens, frequency_penalty, plus stop
        # strings and sampling_seed when set. Payload: input_ids, image_data,
        # sampling_params, return_logprob true, stream false, return_routed_experts when
        # requested in metadata. With with_lora, requires gconfig.lora_name (else
        # ValueError) and sets lora_path to the versioned LoRA name for version.
        """Build a ``/generate`` request for SGLang."""
        gconfig = req.gconfig

        if gconfig.use_beam_search:
            raise NotImplementedError(
                "Beam search is not supported in the SGLang bridge backend."
            )

        # Compute effective max_new_tokens
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids),
            gconfig.max_new_tokens,
        )

        sampling_params: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "frequency_penalty": gconfig.frequency_penalty,
        }
        if gconfig.stop:
            sampling_params["stop"] = gconfig.stop
        if gconfig.seed is not None:
            sampling_params["sampling_seed"] = gconfig.seed

        payload: dict[str, Any] = {
            "input_ids": list(req.input_ids),
            "image_data": req.image_data,
            "sampling_params": sampling_params,
            "return_logprob": True,
            "stream": False,
        }

        if req.metadata.get("return_routed_experts", False):
            payload["return_routed_experts"] = True

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora "
                    "is enabled."
                )
            payload["lora_path"] = get_versioned_lora_name(lora_name, version)

        return HttpRequest(endpoint="/generate", payload=payload)

    # -- response parsing ---------------------------------------------------

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        # [FUNCTION DOC]
        # Parses SGLang JSON into HttpGenerationResult: stop_reason is
        # meta_info.finish_reason.type. If routed_experts is present, base64-decodes it
        # into an int32 array reshaped to (prompt_tokens + completion_tokens - 1, -1).
        # An abort whose message starts with "Abort before prefill" returns no tokens.
        # Otherwise requires output_token_logprobs (ValueError if missing) and splits
        # each (logprob, token_id, ...) entry into tokens and logprobs.
        """Parse SGLang ``/generate`` JSON into :class:`HttpGenerationResult`."""
        import pybase64

        meta_info = response["meta_info"]
        finish_reason = meta_info["finish_reason"]
        stop_reason: str = finish_reason["type"]
        stop_message: str = finish_reason.get("message", "")

        # Routed experts (MoE)
        routed_experts: np.ndarray | None = None
        raw_experts = meta_info.get("routed_experts", None)
        if raw_experts is not None:
            num_sgl_token = (
                meta_info["prompt_tokens"] + meta_info["completion_tokens"] - 1
            )
            routed_experts = np.frombuffer(
                pybase64.b64decode(raw_experts.encode("utf-8")),
                dtype=np.int32,
            ).reshape(num_sgl_token, -1)

        if stop_reason == "abort" and stop_message.startswith("Abort before prefill"):
            return HttpGenerationResult(
                output_tokens=[],
                output_logprobs=[],
                stop_reason=stop_reason,
                routed_experts=routed_experts,
            )

        if "output_token_logprobs" not in meta_info:
            raise ValueError(
                "Malformed SGLang response: output_token_logprobs is missing "
                "from meta_info."
            )
        output_token_logprobs = meta_info["output_token_logprobs"]
        output_tokens = [x[1] for x in output_token_logprobs]
        output_logprobs = [x[0] for x in output_token_logprobs]

        return HttpGenerationResult(
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            stop_reason=stop_reason,
            routed_experts=routed_experts,
        )

    # -- pause / resume -----------------------------------------------------

    def get_pause_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /pause_generation with empty payload.
        return HttpRequest(endpoint="/pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /continue_generation with empty payload.
        return HttpRequest(endpoint="/continue_generation", payload={})

    def get_offload_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /release_memory_occupation with empty payload.
        return HttpRequest(endpoint="/release_memory_occupation", payload={})

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /resume_memory_occupation with {"tags": tags} when tags given,
        # else {}.
        payload = {"tags": tags} if tags is not None else {}
        return HttpRequest(endpoint="/resume_memory_occupation", payload=payload)

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        # [FUNCTION DOC]
        # Reads sampling_params.max_new_tokens from a built request as int.
        return int(http_req.payload["sampling_params"]["max_new_tokens"])

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        # [FUNCTION DOC]
        # Mutates the request in place for resubmission: input_ids becomes original
        # input plus accumulated output tokens; max_new_tokens becomes remaining_tokens.
        http_req.payload["input_ids"] = list(req.input_ids) + accumulated_tokens
        http_req.payload["sampling_params"]["max_new_tokens"] = remaining_tokens

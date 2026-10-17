# [MODULE DOC]
# vLLM implementation of the InfBridgeBackend protocol: builds OpenAI-compatible
# completion / chat-completion payloads with token-id outputs, parses responses and
# supplies AReaL pause/continue and sleep/wake_up control endpoints.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific inference bridge backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from areal.api.io_struct import (
    HttpGenerationResult,
    HttpRequest,
    detect_image_mime,
    get_versioned_lora_name,
)
from areal.utils.vllm_response import parse_vllm_generation_response

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


class VLLMBridgeBackend:
    # [CLASS DOC]
    # Stateless backend object mirroring the relevant subset of the remote vLLM engine
    # backend.
    """vLLM-specific backend for :class:`InfBridge`.

    Mirrors the relevant subset of
    :class:`areal.engine.vllm_remote.VLLMBackend`.
    """

    def build_generation_request(
        self,
        req: ModelRequest,
        with_lora: bool,
        version: int = -1,
    ) -> HttpRequest:
        # [FUNCTION DOC]
        # Builds an HttpRequest. max_tokens = min(max_tokens - input length,
        # max_new_tokens). Payload: top_p, top_k, max_tokens, temperature (0 when
        # greedy), stop_token_ids, ignore_eos, skip_special_tokens,
        # return_tokens_as_token_ids, logprobs 0, use_beam_search, stream false,
        # optional seed. With with_lora, requires lora_name (ValueError otherwise) and
        # sets "model" to the versioned LoRA name. If the request carries vision
        # messages, fills each image_url content item in order with a data URL built
        # from req.image_data (MIME sniffed; ValueError if too few images), sets
        # messages and logprobs true, and targets /v1/chat/completions; otherwise sets
        # prompt to the input ids and targets /v1/completions.
        # KNOWN ISSUE: image_url entries of req.vision_msg_vllm are mutated in place
        # (only a shallow copy of the list is stored in the payload).
        """Build a ``/v1/completions`` or ``/v1/chat/completions`` request."""
        gconfig = req.gconfig

        # Compute effective max_new_tokens (cap by remaining context window)
        max_new_tokens = min(
            gconfig.max_tokens - len(req.input_ids),
            gconfig.max_new_tokens,
        )

        payload: dict[str, Any] = {
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "stop_token_ids": gconfig.stop_token_ids,
            "ignore_eos": gconfig.ignore_eos,
            "skip_special_tokens": gconfig.skip_special_tokens,
            "return_tokens_as_token_ids": True,
            "logprobs": 0,
            "use_beam_search": gconfig.use_beam_search,
            "stream": False,
        }
        if gconfig.seed is not None:
            payload["seed"] = gconfig.seed

        if with_lora:
            lora_name = gconfig.lora_name
            if not lora_name:
                raise ValueError(
                    "LoRA name (gconfig.lora_name) is required when use_lora is enabled."
                )
            payload["model"] = get_versioned_lora_name(lora_name, version)

        if req.vision_msg_vllm:
            images = iter(req.image_data or [])
            parsed_input = req.vision_msg_vllm[0]
            for msg in parsed_input:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if content.get("type") == "image_url":
                            try:
                                base64_img = next(images)
                            except StopIteration as exc:
                                raise ValueError(
                                    "Not enough images in req.image_data to match image_url entries."
                                ) from exc
                            mime = detect_image_mime(base64_img)
                            content["image_url"] = {
                                "url": f"data:{mime};base64,{base64_img}"
                            }
            payload["messages"] = parsed_input.copy()
            payload["logprobs"] = True
            return HttpRequest(endpoint="/v1/chat/completions", payload=payload)

        payload["prompt"] = list(req.input_ids)
        return HttpRequest(endpoint="/v1/completions", payload=payload)

    def parse_generation_response(
        self,
        response: dict[str, Any],
    ) -> HttpGenerationResult:
        # [FUNCTION DOC]
        # Delegates to areal.utils.vllm_response.parse_vllm_generation_response.
        """Parse vLLM JSON into :class:`HttpGenerationResult`."""
        return parse_vllm_generation_response(response)

    def get_pause_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /areal_pause_generation with empty payload.
        return HttpRequest(endpoint="/areal_pause_generation", payload={})

    def get_resume_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /areal_continue_generation with empty payload.
        return HttpRequest(endpoint="/areal_continue_generation", payload={})

    def get_offload_request(self) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /sleep with empty payload.
        return HttpRequest(endpoint="/sleep", payload={}, method="POST")

    def get_onload_request(self, tags: list[str] | None = None) -> HttpRequest:
        # [FUNCTION DOC]
        # Returns POST /wake_up, with tags url-encoded as repeated query params when
        # given.
        if tags is not None:
            from urllib.parse import urlencode

            tags_query = urlencode({"tags": tags}, doseq=True)
            endpoint = f"/wake_up?{tags_query}"
        else:
            endpoint = "/wake_up"
        return HttpRequest(endpoint=endpoint, payload={}, method="POST")

    def get_generation_max_new_tokens(self, http_req: HttpRequest) -> int:
        # [FUNCTION DOC]
        # Reads payload max_tokens as int.
        return int(http_req.payload["max_tokens"])

    def patch_generation_request(
        self,
        http_req: HttpRequest,
        req: ModelRequest,
        accumulated_tokens: list[int],
        remaining_tokens: int,
    ) -> None:
        # [FUNCTION DOC]
        # Sets max_tokens to remaining_tokens and, for completion requests only, sets
        # prompt to original input ids plus accumulated tokens.
        # KNOWN ISSUE: chat (vision) requests are not extended with accumulated tokens,
        # so a resubmit after abort regenerates from scratch while InfBridge keeps the
        # earlier partial output, producing inconsistent token sequences.
        http_req.payload["max_tokens"] = remaining_tokens
        if "prompt" in http_req.payload:
            http_req.payload["prompt"] = list(req.input_ids) + accumulated_tokens

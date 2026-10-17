# [MODULE DOC]
# Thin async wrapper around a HuggingFace tokenizer used by the inference data proxy.
# Blocking tokenizer calls are offloaded to the default thread-pool executor so the
# asyncio event loop is not stalled.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from areal.utils.hf_utils import apply_chat_template as _apply_chat_template


class TokenizerProxy:
    # [CLASS DOC]
    # Holds a loaded HF tokenizer (private attribute) and exposes async encode / chat
    # template helpers plus sync decode helpers and eos/pad token id properties.
    """Wraps HuggingFace tokenizer with async-safe methods for the data proxy."""

    def __init__(self, tokenizer_path: str):
        # [FUNCTION DOC]
        # Loads the tokenizer from tokenizer_path (local dir or hub id) via areal's
        # load_hf_tokenizer (imported lazily) and stores it. Performs disk/network I/O.
        from areal.utils.hf_utils import load_hf_tokenizer

        self._tok = load_hf_tokenizer(tokenizer_path)

    async def tokenize(self, text: str) -> list[int]:
        # [FUNCTION DOC]
        # Encodes text into a list of token ids using the tokenizer's encode (default
        # special token handling) on the running loop's default executor; awaits and
        # returns the list.
        """Tokenize string -> token IDs. Runs in executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._tok.encode, text)

    async def apply_chat_template(self, messages: list[dict], **kw) -> list[int]:
        # [FUNCTION DOC]
        # Renders an OpenAI-style message list with the chat template, tokenized, with
        # add_generation_prompt enabled; extra keyword args (e.g. tools) are forwarded
        # to areal.utils.hf_utils.apply_chat_template. Runs in the default executor and
        # returns the token id list.
        """Apply chat template -> token IDs. Runs in executor."""
        loop = asyncio.get_running_loop()

        def _apply():
            # [FUNCTION DOC]
            # HELPER: Closure that performs the blocking chat-template call with the
            # captured messages and kwargs.
            return _apply_chat_template(
                self._tok, messages, tokenize=True, add_generation_prompt=True, **kw
            )

        return await loop.run_in_executor(None, _apply)

    def decode_token(self, token_id: int) -> str:
        # [FUNCTION DOC]
        # Decodes a single token id to its string piece, keeping special tokens.
        # Synchronous.
        """Decode single token ID -> string piece. Sync (fast dict lookup)."""
        return self._tok.decode([token_id], skip_special_tokens=False)

    def decode_tokens(self, token_ids: list[int]) -> str:
        # [FUNCTION DOC]
        # Decodes a list of token ids into text, skipping special tokens. Synchronous.
        """Decode a list of token IDs -> full string. Used by ChatCompletionHandler."""
        return self._tok.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_id(self) -> int:
        # [FUNCTION DOC]
        # Property: the tokenizer's eos token id.
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int:
        # [FUNCTION DOC]
        # Property: the pad token id, falling back to eos id when pad id is None or
        # falsy.
        # KNOWN ISSUE: a legitimate pad id of 0 is treated as missing and replaced by
        # eos id.
        return self._tok.pad_token_id or self._tok.eos_token_id

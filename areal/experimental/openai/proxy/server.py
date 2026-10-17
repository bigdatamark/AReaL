# [MODULE DOC]
# Shared protocol definitions for the OpenAI proxy servers: pydantic request/response
# models, per-session state (SessionData wrapping an InteractionCache), interaction
# (de)serialization for HTTP transport, URL path constants used by both servers and
# clients, the default admin API key, and SESSION_TIMEOUT_SECONDS (3600) for stale
# session cleanup.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from areal.experimental.openai.cache import InteractionCache

if TYPE_CHECKING:
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    # [CLASS DOC]
    # Body of rl/start_session: task_id and optional api_key to reuse a previously
    # issued
    # session key (refresh).
    """Request to start a new RL session."""

    task_id: str
    api_key: str | None = None  # Reuse a previously-issued key (refresh)


class StartSessionResponse(BaseModel):
    # [CLASS DOC]
    # Reply of rl/start_session: session_id and the session API key.
    """Response from start_session endpoint."""

    session_id: str
    api_key: str


class SetRewardRequest(BaseModel):
    # [CLASS DOC]
    # Body of rl/set_reward: optional interaction_id (None = latest) and reward.
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float


class ExportTrajectoriesRequest(BaseModel):
    # [CLASS DOC]
    # Body of export_trajectories: session_id, discount (default 1.0), style (default
    # "individual") and drop_retry_orphans flag.
    """Request to export trajectories for a session."""

    session_id: str
    discount: float = 1.0
    style: str = "individual"
    drop_retry_orphans: bool = False


class ExportTrajectoriesResponse(BaseModel):
    # [CLASS DOC]
    # Reply of export_trajectories: serialized interactions dict.
    """Response containing serialized interactions."""

    interactions: dict[str, Any]


# =============================================================================
# Session Data
# =============================================================================


class SessionData:
    # [CLASS DOC]
    # Server-side state for one RL session: id, sampling seed identity, completion flag
    # and threading.Event, the InteractionCache of recorded interactions, start/last
    # access/end timestamps, a lock, and a counter for per-request sampling indices.
    """Data associated with a single RL session."""

    def __init__(
        self,
        session_id: str,
        prefix_matcher=None,
        sampling_seed_identity: str | None = None,
    ):
        # [FUNCTION DOC]
        # Initializes state; sampling_seed_identity defaults to session_id; creates the
        # InteractionCache with the session id and optional prefix matcher.
        self.session_id = session_id
        self.sampling_seed_identity = sampling_seed_identity or session_id

        self._completed = False
        self._completions = InteractionCache(
            session_id=session_id,
            prefix_matcher=prefix_matcher,
        )
        self._completed_event = threading.Event()
        self._start_time = time.time()
        self._last_access_time = time.time()
        self._end_time = None
        self._lock = threading.Lock()
        self._next_sampling_request_index = 0

    def next_sampling_request_index(self) -> int:
        # [FUNCTION DOC]
        # Atomically returns and increments the per-session request counter (used to
        # derive deterministic sampling seeds).
        """Reserve a unique request index without serializing request execution."""
        with self._lock:
            request_index = self._next_sampling_request_index
            self._next_sampling_request_index += 1
        return request_index

    def update_last_access(self):
        # [FUNCTION DOC]
        # Sets the last access time to now under the lock.
        """Update the last access time for this session."""
        with self._lock:
            self._last_access_time = time.time()

    def is_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> bool:
        # [FUNCTION DOC]
        # True if the last access is older than timeout_seconds (default one hour).
        """Check if this session has been inactive for too long."""
        with self._lock:
            return time.time() - self._last_access_time > timeout_seconds

    def finish(self):
        # [FUNCTION DOC]
        # Marks the session completed, records end time and sets the completion event.
        self._completed = True
        self._end_time = time.time()
        self._completed_event.set()

    @property
    def is_completed(self) -> bool:
        # [FUNCTION DOC]
        # Property: whether finish() has been called.
        """Whether this session has been completed via ``finish()``."""
        return self._completed

    @property
    def completions(self):
        # [FUNCTION DOC]
        # Property: the session's InteractionCache.
        return self._completions

    async def wait_for_finish(self, timeout: float | None = None) -> bool:
        # [FUNCTION DOC]
        # Async-waits for finish() by polling the threading.Event in the default
        # executor
        # in slices of at most 1s (so task cancellation is responsive). Returns True
        # when
        # finished, False when timeout seconds elapse.
        # KNOWN ISSUE: timeout=0 is treated like None (falsy) and waits forever.
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout if timeout else None
        while not self._completed_event.is_set():
            remaining = (deadline - time.monotonic()) if deadline else 1.0
            if deadline and remaining <= 0:
                return False
            poll = min(remaining, 1.0)  # Poll every 1s so cancellation works
            await loop.run_in_executor(None, self._completed_event.wait, poll)
        return True

    def export_interactions(
        self, discount: float, style: str, drop_retry_orphans: bool = False
    ) -> dict[str, InteractionWithTokenLogpReward]:
        # [FUNCTION DOC]
        # Returns {} if there are no interactions; otherwise optionally drops retry
        # orphans,
        # applies reward discount and exports with the given style.
        # KNOWN ISSUE: a second export of the same session raises RuntimeError because
        # apply_reward_discount is one-shot.
        if len(self.completions) == 0:
            return {}
        if drop_retry_orphans:
            self.completions.drop_retry_orphans()
        self.completions.apply_reward_discount(turn_discount=discount)
        return self.completions.export_interactions(style=style)


# =============================================================================
# Serialization Helpers
# =============================================================================


def serialize_interactions(
    interactions: dict[str, InteractionWithTokenLogpReward],
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Converts interactions to a JSON-compatible dict via the RPC serialize_value: for
    # interactions with tensor data, {tensor_dict, reward, interaction_id}; otherwise
    # {messages, output_message_list, reward, interaction_id}.
    """Serialize interactions into a json-compatible format for HTTP transport."""
    from areal.infra.rpc.serialization import serialize_value

    result = {}
    for key, interaction in interactions.items():
        if interaction.has_tensor_data:
            result[key] = {
                "tensor_dict": interaction.to_tensor_dict(),
                "reward": interaction.reward,
                "interaction_id": interaction.interaction_id,
            }
        else:
            result[key] = {
                "messages": interaction.messages,
                "output_message_list": interaction.output_message_list,
                "reward": interaction.reward,
                "interaction_id": interaction.interaction_id,
            }
    return serialize_value(result)


def deserialize_interactions(
    data: dict[str, Any],
) -> dict[str, InteractionWithTokenLogpReward]:
    # [FUNCTION DOC]
    # Inverse of serialize_interactions: deserializes values and builds bare
    # InteractionWithTokenLogpReward objects, setting the private tensor cache or the
    # message fields, plus reward and interaction_id. Parent links and model responses
    # are
    # not restored.
    """Deserialize interactions from HTTP response."""
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.infra.rpc.serialization import deserialize_value

    data = deserialize_value(data)
    result = {}
    for key, item in data.items():
        interaction = InteractionWithTokenLogpReward()
        if "tensor_dict" in item:
            interaction._cache = item["tensor_dict"]
        else:
            interaction.messages = item["messages"]
            interaction.output_message_list = item["output_message_list"]
        interaction.reward = item["reward"]
        interaction.interaction_id = item["interaction_id"]
        result[key] = interaction
    return result


# =============================================================================
# Path Constants (must match client_session.py expectations)
# =============================================================================

RL_START_SESSION_PATHNAME = "rl/start_session"
RL_END_SESSION_PATHNAME = "rl/end_session"
RL_SET_REWARD_PATHNAME = "rl/set_reward"
CHAT_COMPLETIONS_PATHNAME = "chat/completions"
RESPONSES_PATHNAME = "responses"
ANTHROPIC_MESSAGES_PATHNAME = "v1/messages"
GRANT_CAPACITY_PATHNAME = "grant_capacity"
EXPORT_TRAJECTORIES_PATHNAME = "export_trajectories"
INTERNAL_WAIT_FOR_SESSION_PATHNAME = "internal/wait_for_session"

# Shared default for admin API key — used by cli_args.py and workflow.py
# to avoid independent duplication.
DEFAULT_ADMIN_API_KEY = "areal-admin-key"


class WaitForSessionRequest(BaseModel):
    # [CLASS DOC]
    # Body of internal/wait_for_session sent by _OnlineAgent: worker_addr.
    """Request from _OnlineAgent to register a worker and wait for a session."""

    worker_addr: str


class WaitForSessionResponse(BaseModel):
    # [CLASS DOC]
    # Reply of internal/wait_for_session: session_api_key, session_id and worker_addr of
    # a completed session.
    """Response with completed session credentials."""

    session_api_key: str
    session_id: str
    worker_addr: str

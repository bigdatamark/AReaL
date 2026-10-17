# [MODULE DOC]
# Session bookkeeping for the data proxy: pydantic request/response models for the RL
# session endpoints, per-session interaction caches that are cut into reward-bounded
# "ready" trajectories, and a thread-safe store mapping session ids and API keys.
# Sessions idle longer than SESSION_TIMEOUT_SECONDS (1h) are considered stale.
# SPDX-License-Identifier: Apache-2.0

"""Session lifecycle management for the data proxy."""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from areal.experimental.openai.cache import InteractionCache, PrefixMatcher
from areal.experimental.openai.types import InteractionWithTokenLogpReward

# Session timeout for cleanup (1 hour)
SESSION_TIMEOUT_SECONDS = 3600


# =============================================================================
# Request/Response Models
# =============================================================================


class StartSessionRequest(BaseModel):
    # [CLASS DOC]
    # Body of /rl/start_session: task_id, optional api_key to reuse, group_size (default
    # 1).
    # KNOWN ISSUE: the docstring says group sessions are created atomically, but the
    # endpoint does not roll back already-created sessions on failure.
    """Request to start one or more offline RL sessions.

    When ``group_size`` is greater than 1, the data proxy creates multiple
    sessions atomically. The response always contains a flat ``sessions``
    list — single-session is just ``group_size=1``.
    """

    task_id: str
    api_key: str | None = None  # Reuse a previously-issued key (refresh)
    group_size: int = 1


class SessionCredentials(BaseModel):
    # [CLASS DOC]
    # One session's id and API key.
    """One session's identity and authentication key."""

    session_id: str
    session_api_key: str


class StartSessionResponse(BaseModel):
    # [CLASS DOC]
    # group_id plus the list of session credentials.
    """Response from start_session — always a list of session credentials."""

    group_id: str
    sessions: list[SessionCredentials]


class SetRewardRequest(BaseModel):
    # [CLASS DOC]
    # Body of /rl/set_reward: optional interaction_id (default: last), reward, optional
    # model (unused by the proxy).
    """Request to set reward for an interaction."""

    interaction_id: str | None = None
    reward: float
    model: str | None = None


class ExportTrajectoriesRequest(BaseModel):
    # [CLASS DOC]
    # Body of /export_trajectories: session ids, optional group_id (unused here),
    # trajectory_id (default latest), discount, style ("individual"/"concat"),
    # remove_session, drop_retry_orphans, reward_normalization, discard_trajectory.
    """Request to export trajectories for one or more sessions.

    All sessions are exported and merged into a single interactions dict.
    ``group_id`` is optional metadata used by the gateway for router cleanup;
    the data proxy itself does not use it.
    """

    session_ids: list[str]
    group_id: str | None = None
    trajectory_id: int | None = None
    discount: float = 1.0
    style: str = "individual"
    remove_session: bool = True
    drop_retry_orphans: bool = False
    reward_normalization: bool = False
    discard_trajectory: bool = False


class ExportTrajectoriesResponse(BaseModel):
    # [CLASS DOC]
    # Holds the serialized merged trajectory dict.
    """Response containing merged serialized interactions."""

    traj: dict[str, Any]


@dataclass(frozen=True)
class RewardResult:
    # [CLASS DOC]
    # Immutable result of set_reward/finalization: session id, trajectory id (None while
    # still active), interaction count, and whether a ready transition just happened.
    """Internal result returned when an online session closes a trajectory."""

    session_id: str
    trajectory_id: int | None
    interaction_count: int
    ready_transition: bool


@dataclass(frozen=True)
class ReadyNotification:
    # [CLASS DOC]
    # Immutable (session_id, trajectory_id) pair for an online-ready callback.
    session_id: str
    trajectory_id: int


@dataclass
class ReadyTrajectory:
    # [CLASS DOC]
    # A finalized, not yet exported trajectory: id, id of the rewarded interaction, its
    # interaction cache, creation time and online-callback flags.
    """One ready-but-not-yet-exported online trajectory."""

    trajectory_id: int
    interaction_id: str
    completions: InteractionCache
    created_at: float
    needs_online_callback: bool = False
    callback_delivered: bool = False


# =============================================================================
# Session Data
# =============================================================================


class SessionData:
    # [CLASS DOC]
    # Per-session state guarded by a threading lock: the active InteractionCache that
    # chat completions append to, an ordered map of ready trajectories, trajectory id
    # counter, deterministic-sampling request counter and seed identity (defaults to
    # session id), last access time, and the pending reward (interaction id and time)
    # awaiting finalization after set_reward_finish_timeout.
    """Unified session data for both offline and online modes.

    Maintains ``active_completions`` (the current in-progress interaction
    cache) and ``ready_trajectories`` (reward-bounded, exportable
    trajectories).

    - **Offline**: one session → one trajectory via ``set_reward`` →
      ``export_trajectory``.
    - **Online**: one persistent session → many reward-bounded trajectories
      via repeated ``set_reward`` → ``export_trajectory`` calls.
    """

    def __init__(
        self,
        session_id: str,
        set_reward_finish_timeout: float = 0.0,
        sampling_seed_identity: str | None = None,
        prefix_matcher: PrefixMatcher | None = None,
    ):
        # [FUNCTION DOC]
        # Initializes the state described above with a fresh interaction cache.
        self.session_id = session_id
        self.sampling_seed_identity = sampling_seed_identity or session_id
        self._set_reward_finish_timeout = set_reward_finish_timeout
        self._prefix_matcher = prefix_matcher
        self._last_access_time = time.time()
        self._lock = threading.Lock()
        self._active_completions = self._new_interaction_cache()
        self._ready_trajectories: OrderedDict[int, ReadyTrajectory] = OrderedDict()
        self._next_trajectory_id = 0
        self._next_sampling_request_index = 0
        self._last_set_reward_time: float | None = None
        self._last_reward_interaction_id: str | None = None

    def next_sampling_request_index(self) -> int:
        # [FUNCTION DOC]
        # Atomically returns and increments the sampling request index.
        """Reserve a request index before inference without serializing generation."""
        with self._lock:
            request_index = self._next_sampling_request_index
            self._next_sampling_request_index += 1
        return request_index

    def _new_interaction_cache(self) -> InteractionCache:
        # [FUNCTION DOC]
        # HELPER: Returns a new InteractionCache for this session using the prefix
        # matcher.
        return InteractionCache(
            session_id=self.session_id,
            prefix_matcher=self._prefix_matcher,
        )

    def update_last_access(self) -> None:
        # [FUNCTION DOC]
        # Sets the last access time to now under the lock.
        with self._lock:
            self._last_access_time = time.time()

    def is_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> bool:
        # [FUNCTION DOC]
        # Returns True if idle longer than timeout_seconds (default 1h).
        with self._lock:
            return time.time() - self._last_access_time > timeout_seconds

    @property
    def active_completions(self) -> InteractionCache:
        # [FUNCTION DOC]
        # Property: the current in-progress interaction cache (not lock protected).
        return self._active_completions

    @property
    def has_ready_trajectories(self) -> bool:
        # [FUNCTION DOC]
        # Property: True if any ready trajectory awaits export.
        with self._lock:
            return bool(self._ready_trajectories)

    def _latest_ready_trajectory_locked(self) -> ReadyTrajectory | None:
        # [FUNCTION DOC]
        # HELPER: Returns the most recently created ready trajectory or None; lock must
        # be held.
        """Return the latest ready trajectory.

        Caller must already hold ``self._lock``.
        """
        if not self._ready_trajectories:
            return None
        return next(reversed(self._ready_trajectories.values()))

    def _resolve_duplicate_ready_locked(
        self, interaction_id: str | None
    ) -> ReadyTrajectory | None:
        # [FUNCTION DOC]
        # HELPER: Detects a repeated set_reward after finalization: returns the latest
        # ready trajectory if the active cache is empty and interaction_id is None or
        # equals that trajectory's rewarded interaction; else None. Lock must be held.
        latest_ready = self._latest_ready_trajectory_locked()
        if latest_ready is None:
            return None
        if len(self._active_completions) != 0:
            return None
        resolved_interaction_id = interaction_id or latest_ready.interaction_id
        if resolved_interaction_id == latest_ready.interaction_id:
            return latest_ready
        return None

    def _mark_active_trajectory_ready_locked(
        self,
        now: float,
    ) -> RewardResult:
        # [FUNCTION DOC]
        # HELPER: Moves the active cache into a new ReadyTrajectory (next id, flagged
        # for online callback), starts a fresh cache and clears the pending reward.
        # Raises ValueError if the cache is empty or no reward was set. Returns a
        # RewardResult with ready_transition True. Lock must be held.
        completions = self._active_completions
        if len(completions) == 0:
            raise ValueError("No interactions in session")

        resolved_interaction_id = self._last_reward_interaction_id
        if resolved_interaction_id is None:
            raise ValueError("No reward has been set for the active trajectory")

        trajectory_id = self._next_trajectory_id
        self._next_trajectory_id += 1
        ready = ReadyTrajectory(
            trajectory_id=trajectory_id,
            interaction_id=resolved_interaction_id,
            completions=completions,
            created_at=now,
            needs_online_callback=True,
        )
        self._ready_trajectories[trajectory_id] = ready
        self._active_completions = self._new_interaction_cache()
        self._last_set_reward_time = None
        self._last_reward_interaction_id = None

        return RewardResult(
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            interaction_count=len(completions),
            ready_transition=True,
        )

    def _finalize_if_reward_timeout_elapsed_locked(
        self,
        now: float,
    ) -> RewardResult | None:
        # [FUNCTION DOC]
        # HELPER: If a reward is pending and at least set_reward_finish_timeout seconds
        # have passed since it, finalizes the active trajectory and returns the result;
        # else None.
        if self._last_set_reward_time is None:
            return None
        if now - self._last_set_reward_time < self._set_reward_finish_timeout:
            return None
        return self._mark_active_trajectory_ready_locked(now)

    def set_reward(
        self,
        interaction_id: str | None,
        reward: float,
    ) -> RewardResult:
        # [FUNCTION DOC]
        # Records reward on interaction_id (or the last interaction) of the active
        # cache. A duplicate call after finalization returns the existing ready
        # trajectory without change. Raises ValueError if the cache is empty or the
        # interaction is unknown. Remembers the pending reward and finalizes immediately
        # when the finish timeout has elapsed (always for timeout 0). Returns a
        # RewardResult (trajectory_id None while still pending).
        """Record reward for the active trajectory."""
        with self._lock:
            now = time.time()
            self._last_access_time = now

            duplicate_ready = self._resolve_duplicate_ready_locked(interaction_id)
            if duplicate_ready is not None:
                return RewardResult(
                    session_id=self.session_id,
                    trajectory_id=duplicate_ready.trajectory_id,
                    interaction_count=len(duplicate_ready.completions),
                    ready_transition=False,
                )

            completions = self._active_completions
            if len(completions) == 0:
                raise ValueError("No interactions in session")

            resolved_interaction_id = interaction_id or completions.last_interaction_id
            if resolved_interaction_id not in completions:
                raise ValueError(f"Interaction {resolved_interaction_id} not found")

            completions.set_reward(resolved_interaction_id, reward)
            self._last_reward_interaction_id = resolved_interaction_id
            self._last_set_reward_time = now

            ready_result = self._finalize_if_reward_timeout_elapsed_locked(now)
            if ready_result is not None:
                return ready_result

            return RewardResult(
                session_id=self.session_id,
                trajectory_id=None,
                interaction_count=len(completions),
                ready_transition=False,
            )

    def finalize_if_reward_timeout_elapsed(
        self,
        now: float | None = None,
    ) -> RewardResult | None:
        # [FUNCTION DOC]
        # Thread-safe wrapper finalizing a pending trajectory whose timeout elapsed (now
        # defaults to current time). Returns the RewardResult or None.
        with self._lock:
            return self._finalize_if_reward_timeout_elapsed_locked(now or time.time())

    def pending_online_callbacks(self) -> list[ReadyNotification]:
        # [FUNCTION DOC]
        # Returns notifications for ready trajectories that need an online callback not
        # yet delivered.
        # KNOWN ISSUE: every finalized trajectory is flagged for an online callback,
        # including offline (agent-driven) sessions, so the controller may receive
        # callbacks for trajectories that offline workflows export themselves.
        with self._lock:
            return [
                ReadyNotification(
                    session_id=self.session_id,
                    trajectory_id=ready.trajectory_id,
                )
                for ready in self._ready_trajectories.values()
                if ready.needs_online_callback and not ready.callback_delivered
            ]

    def mark_online_callback_delivered(self, trajectory_id: int) -> bool:
        # [FUNCTION DOC]
        # Marks a ready trajectory's callback as delivered; returns False if it no
        # longer exists or needs none, True otherwise.
        with self._lock:
            ready = self._ready_trajectories.get(trajectory_id)
            if ready is None or not ready.needs_online_callback:
                return False
            if ready.callback_delivered:
                return True
            ready.callback_delivered = True
            return True

    def add_string_interaction(self, messages: list[dict], response: str) -> str:
        # [FUNCTION DOC]
        # Adds a text-only interaction (messages plus one assistant message with
        # response) under a new UUID to the active cache, touches access time, and
        # returns the id.
        # KNOWN ISSUE: mutates the active cache without holding the session lock.
        interaction_id = str(uuid.uuid4())
        interaction = InteractionWithTokenLogpReward(
            messages=messages,
            output_message_list=[{"role": "assistant", "content": response}],
        )
        interaction._interaction_id = interaction_id
        self._active_completions[interaction_id] = interaction
        self.update_last_access()
        return interaction_id

    def export_trajectory(
        self,
        discount: float,
        style: str,
        trajectory_id: int | None = None,
        drop_retry_orphans: bool = False,
    ) -> tuple[int, dict[str, InteractionWithTokenLogpReward]]:
        # [FUNCTION DOC]
        # Pops the requested (or latest) ready trajectory under the lock, raising
        # KeyError if none exist or the id is unknown, then exports its interactions
        # with the given style, reward discount and retry-orphan dropping. Returns
        # (trajectory_id, interactions dict).
        """Export a ready trajectory.

        Parameters
        ----------
        discount : float
            Reward discount factor passed to
            :pymethod:`InteractionCache.export_interactions`.
        style : str
            Export style (``"individual"`` or ``"concat"``).
        trajectory_id : int | None
            Specific trajectory to export.  When ``None``, the latest
            ready trajectory is exported.
        drop_retry_orphans : bool
            Remove duplicate responses left by upstream request retries before
            reward discounting and concat export.

        Returns
        -------
        tuple[int, dict[str, InteractionWithTokenLogpReward]]
            ``(trajectory_id, interactions)``

        Raises
        ------
        KeyError
            If no ready trajectories exist, or the requested
            ``trajectory_id`` is not found.
        """
        with self._lock:
            if not self._ready_trajectories:
                raise KeyError(f"No ready trajectories for session {self.session_id}")

            target_trajectory_id = trajectory_id
            if target_trajectory_id is None:
                target_trajectory_id = next(reversed(self._ready_trajectories))

            ready = self._ready_trajectories.pop(target_trajectory_id, None)
            if ready is None:
                raise KeyError(
                    f"Trajectory {target_trajectory_id} not found for session {self.session_id}"
                )

        interactions = ready.completions.export_interactions(
            style=style,
            reward_discount=discount,
            drop_retry_orphans=drop_retry_orphans,
        )
        return ready.trajectory_id, interactions


# =============================================================================
# Session Store
# =============================================================================


class SessionStore:
    # [CLASS DOC]
    # Thread-safe registry of sessions keyed by id, with bidirectional session <-> API
    # key maps, the admin key (default "areal-admin-key"), an unused capacity value, and
    # settings passed to new sessions (finish timeout, prefix matcher).
    """Thread-safe store for session lifecycle management."""

    def __init__(
        self,
        set_reward_finish_timeout: float = 0.0,
        prefix_matcher: PrefixMatcher | None = None,
    ):
        # [FUNCTION DOC]
        # Initializes empty maps and stores the session settings.
        self._sessions: dict[str, SessionData] = {}
        self._api_key_to_session: dict[str, str] = {}
        self._session_to_api_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._capacity: int = 0
        self._admin_api_key: str = "areal-admin-key"
        self._set_reward_finish_timeout = set_reward_finish_timeout
        self._prefix_matcher = prefix_matcher

    def set_capacity(self, n: int) -> None:
        # [FUNCTION DOC]
        # Stores a capacity value (not used elsewhere in this module).
        with self._lock:
            self._capacity = n

    def set_admin_key(self, key: str) -> None:
        # [FUNCTION DOC]
        # Sets the admin API key.
        with self._lock:
            self._admin_api_key = key

    @property
    def admin_api_key(self) -> str:
        # [FUNCTION DOC]
        # Property: the admin API key.
        return self._admin_api_key

    def start_session(
        self,
        task_id: str,
        api_key: str | None = None,
        sampling_seed_identity: str | None = None,
    ) -> tuple[str, str]:
        # [FUNCTION DOC]
        # Creates a session with id "<task_id>-<n>" (first unused n). With api_key,
        # reuses it: raises ValueError if it is bound to another session that has no
        # ready trajectories, otherwise unbinds it from that session. Without api_key,
        # generates a random URL-safe key distinct from existing keys and the admin key.
        # Registers the session and key mapping and returns (session_id, api_key).
        """Start a new session, returning (session_id, session_api_key).

        If *api_key* is provided the key is reused (refreshed); otherwise a
        fresh opaque key is generated.
        """
        with self._lock:
            idx = 0
            while f"{task_id}-{idx}" in self._sessions:
                idx += 1
            session_id = f"{task_id}-{idx}"

            if api_key:
                session_api_key = api_key
                existing_sid = self._api_key_to_session.get(session_api_key)
                if existing_sid is not None:
                    existing_session = self._sessions.get(existing_sid)
                    if (
                        existing_session is not None
                        and not existing_session.has_ready_trajectories
                    ):
                        raise ValueError(
                            f"API key is already bound to active session {existing_sid}."
                        )
                    self._remove_api_keys_for_session(existing_sid)
            else:
                session_api_key = secrets.token_urlsafe(32)
                while (
                    session_api_key in self._api_key_to_session
                    or session_api_key == self._admin_api_key
                ):
                    session_api_key = secrets.token_urlsafe(32)

            self._sessions[session_id] = SessionData(
                session_id=session_id,
                set_reward_finish_timeout=self._set_reward_finish_timeout,
                sampling_seed_identity=sampling_seed_identity,
                prefix_matcher=self._prefix_matcher,
            )
            self._api_key_to_session[session_api_key] = session_id
            self._session_to_api_key[session_id] = session_api_key

        return (session_id, session_api_key)

    def get_session_by_api_key(self, api_key: str) -> SessionData | None:
        # [FUNCTION DOC]
        # Returns the session bound to api_key, or None.
        with self._lock:
            session_id = self._api_key_to_session.get(api_key)
            if session_id is None:
                return None
            return self._sessions.get(session_id)

    def get_or_create_hitl_session(self) -> SessionData:
        # [FUNCTION DOC]
        # Returns the persistent "__hitl__" session used for admin-key traffic, creating
        # it if missing.
        """Return the persistent HITL session, creating it if needed."""
        with self._lock:
            session = self._sessions.get("__hitl__")
            if session is None:
                session = SessionData(
                    session_id="__hitl__",
                    set_reward_finish_timeout=self._set_reward_finish_timeout,
                    prefix_matcher=self._prefix_matcher,
                )
                self._sessions["__hitl__"] = session
            return session

    def get_session(self, session_id: str) -> SessionData | None:
        # [FUNCTION DOC]
        # Returns the session by id or None.
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        # [FUNCTION DOC]
        # Deletes the session and its API key mapping (no-op if absent).
        with self._lock:
            self._sessions.pop(session_id, None)
            self._remove_api_keys_for_session(session_id)

    def _remove_api_keys_for_session(self, session_id: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Removes the session's key from both maps; caller must hold the lock.
        api_key = self._session_to_api_key.pop(session_id, None)
        if api_key:
            self._api_key_to_session.pop(api_key, None)

    def cleanup_stale(self, timeout_seconds: float = SESSION_TIMEOUT_SECONDS) -> None:
        # [FUNCTION DOC]
        # Removes sessions idle longer than timeout_seconds that have no ready
        # trajectories.
        with self._lock:
            stale_sessions: list[str] = []
            for sid, session in self._sessions.items():
                if not session.is_stale(timeout_seconds):
                    continue
                if session.has_ready_trajectories:
                    continue
                stale_sessions.append(sid)

            for sid in stale_sessions:
                self._sessions.pop(sid, None)
                self._remove_api_keys_for_session(sid)

    def finalize_rewarded_trajectories(
        self,
        now: float | None = None,
    ) -> list[RewardResult]:
        # [FUNCTION DOC]
        # Snapshots sessions and finalizes each whose reward finish timeout has elapsed;
        # returns the RewardResults of newly ready trajectories.
        with self._lock:
            sessions = list(self._sessions.values())

        finalized: list[RewardResult] = []
        resolved_now = time.time() if now is None else now
        for session in sessions:
            ready_result = session.finalize_if_reward_timeout_elapsed(resolved_now)
            if ready_result is not None:
                finalized.append(ready_result)
        return finalized

    def pending_online_callbacks(self) -> list[ReadyNotification]:
        # [FUNCTION DOC]
        # Collects undelivered online-callback notifications from all sessions.
        with self._lock:
            sessions = list(self._sessions.values())

        notifications: list[ReadyNotification] = []
        for session in sessions:
            notifications.extend(session.pending_online_callbacks())
        return notifications

    def mark_online_callback_delivered(
        self, session_id: str, trajectory_id: int
    ) -> bool:
        # [FUNCTION DOC]
        # Marks the given session's trajectory callback delivered; False if the session
        # is gone.
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return False
        return session.mark_online_callback_delivered(trajectory_id)

    @property
    def session_count(self) -> int:
        # [FUNCTION DOC]
        # Property: number of sessions (including the HITL session).
        with self._lock:
            return len(self._sessions)

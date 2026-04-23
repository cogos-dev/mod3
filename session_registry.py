"""Session-aware communication bus registry (ADR-082 Phase 1).

Evolves Mod3 from a stateless TTS engine toward a session-aware communication
bus. Each registered session owns a SessionChannel with an assigned voice, a
preferred output device (live-queried by default), and its own output queue.
A global round-robin serializer picks the next job across sessions with
pending work so multi-agent sessions can share one physical speaker without
collisions.

Scope for Phase 1 (per the ADR's "Migration Path" section):

  - register_session / deregister_session + list_sessions
  - Per-session output queues with a global serializer (round-robin default,
    priority / fifo-global policies pluggable)
  - Voice assignment from the ranked Kokoro pool
  - preferred_output_device field on SessionChannel with live OS-default
    re-query per playback (2026-04-22 amendment)

Out of scope for Phase 1: input routing, barge-in state machine, native input
provider. See later phases of the ADR.

Backward compatibility: an implicit "default" session is created on first
use so existing callers that do not supply session_id keep working exactly
as before.
"""

from __future__ import annotations

import atexit
import heapq
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

logger = logging.getLogger("mod3.session_registry")


# ---------------------------------------------------------------------------
# Voice pool — ranked allocation per ADR-082 § Voice Assignment
# ---------------------------------------------------------------------------

# The canonical Kokoro voice pool, ranked. Greedy allocation walks this list.
# bm_lewis is first because it is the legacy default; remaining Kokoro voices
# follow in the order the ADR specifies (heart / emma / adam / bella / isabella),
# then the unlisted Kokoro voices round out the pool so we do not run dry.
VOICE_POOL: tuple[str, ...] = (
    "bm_lewis",
    "af_heart",
    "bf_emma",
    "am_adam",
    "af_bella",
    "bf_isabella",
    "bm_george",
    "am_michael",
    "af_sarah",
    "af_nicole",
    "af_sky",
)

DEFAULT_SESSION_ID = "default"
DEFAULT_PARTICIPANT_ID = "legacy"
DEFAULT_PARTICIPANT_TYPE = "agent"

SERIALIZATION_POLICIES = ("round-robin", "priority", "fifo-global")


# ---------------------------------------------------------------------------
# Output-device resolution (ADR-082 amendment — 2026-04-22)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedOutputDevice:
    """Resolution result for an output-device lookup.

    ``preferred`` is the string the session asked for ("system-default" or a
    named device). ``index`` is the sounddevice index that was chosen, or
    ``None`` to let PortAudio pick — callers should treat ``None`` as
    "PortAudio fallback" rather than relying on it implicitly resolving to
    the current system default (PortAudio behavior is inconsistent across
    versions, per the ADR amendment).

    ``name`` mirrors the resolved device's name when known. ``fallback`` is
    True when the requested name was unavailable and we backed off to the
    system default.
    """

    preferred: str
    index: int | None
    name: str
    fallback: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred": self.preferred,
            "index": self.index,
            "name": self.name,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def resolve_output_device(
    preferred: str | None = "system-default",
    *,
    query_devices: Callable[[], list[dict[str, Any]]] | None = None,
    default_output_index: Callable[[], int | None] | None = None,
) -> ResolvedOutputDevice:
    """Resolve ``preferred`` to a concrete sounddevice index, live.

    For ``"system-default"`` (or None/empty) the OS default is re-queried
    every call — no caching. When pinned to a named device we enumerate the
    current device list and pick by substring match; if nothing matches we
    fall back to system-default and flag ``fallback=True`` so callers can
    emit a device_fallback warning event.

    The ``query_devices`` and ``default_output_index`` callables are test
    seams; they default to sounddevice when omitted. This keeps the live
    requery contract explicit — every call goes through the callable, no
    module-level state caches the previous result.
    """

    if query_devices is None or default_output_index is None:

        def _query() -> list[dict[str, Any]]:
            import sounddevice as sd

            return list(sd.query_devices())

        def _default() -> int | None:
            import sounddevice as sd

            # sd.default.device is (input_index, output_index). We only care
            # about the output side. Re-read every call so we pick up whatever
            # the OS chose just now.
            value = sd.default.device
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                idx = value[1]
                return int(idx) if isinstance(idx, int) and idx >= 0 else None
            return None

        query_devices = query_devices or _query
        default_output_index = default_output_index or _default

    pref = (preferred or "system-default").strip()
    if not pref:
        pref = "system-default"

    try:
        devices = query_devices()
    except Exception as exc:  # noqa: BLE001 — device enumeration can fail noisily
        logger.warning("resolve_output_device: query_devices failed: %s", exc)
        return ResolvedOutputDevice(
            preferred=pref,
            index=None,
            name="(unresolved)",
            fallback=True,
            reason=f"query_devices failed: {exc}",
        )

    def _default_resolution() -> ResolvedOutputDevice:
        try:
            idx = default_output_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("resolve_output_device: default_output_index failed: %s", exc)
            return ResolvedOutputDevice(
                preferred=pref,
                index=None,
                name="(system default, unresolved)",
                fallback=True,
                reason=f"default_output_index failed: {exc}",
            )

        if idx is None or idx < 0 or idx >= len(devices):
            # Falling back to PortAudio implicit — we would rather be explicit
            # but the platform did not give us a usable default.
            return ResolvedOutputDevice(
                preferred=pref,
                index=None,
                name="(system default)",
                fallback=False,
                reason="OS default unknown — PortAudio implicit",
            )

        name = devices[idx].get("name", "(unknown)")
        return ResolvedOutputDevice(
            preferred=pref,
            index=idx,
            name=name,
            fallback=False,
            reason="OS default",
        )

    if pref.lower() in ("system-default", "default"):
        return _default_resolution()

    # Named device — try numeric index first, then substring on name.
    if pref.isdigit():
        idx = int(pref)
        if 0 <= idx < len(devices) and devices[idx].get("max_output_channels", 0) > 0:
            return ResolvedOutputDevice(
                preferred=pref,
                index=idx,
                name=devices[idx].get("name", "(unknown)"),
                fallback=False,
                reason="index match",
            )

    pref_lower = pref.lower()
    for i, d in enumerate(devices):
        if d.get("max_output_channels", 0) <= 0:
            continue
        if pref_lower in str(d.get("name", "")).lower():
            return ResolvedOutputDevice(
                preferred=pref,
                index=i,
                name=d.get("name", "(unknown)"),
                fallback=False,
                reason="name match",
            )

    # No match — fall back to system default.
    resolved = _default_resolution()
    resolved.fallback = True
    resolved.reason = f"named device '{pref}' unavailable — fell back to system default"
    logger.warning(
        "resolve_output_device: %s unavailable, falling back to system default (idx=%s name=%s)",
        pref,
        resolved.index,
        resolved.name,
    )
    return resolved


# ---------------------------------------------------------------------------
# Session channel
# ---------------------------------------------------------------------------


SessionState = str  # "idle" | "speaking" | "blocked" | "waiting_for_input"


@dataclass
class SessionChannel:
    """A registered session's bus endpoint.

    Holds identity, voice, device preference, a per-session output queue,
    and lifecycle state. The instance is mutated in place under
    ``SessionRegistry``'s lock — callers should not hold references across
    deregistration.
    """

    session_id: str
    participant_id: str
    participant_type: str
    assigned_voice: str
    voice_conflict: bool = False
    preferred_voice: str | None = None
    preferred_output_device: str = "system-default"
    state: SessionState = "idle"
    priority: int = 0
    registered_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    # Internal: pending jobs waiting for the global serializer to pick them up.
    # Elements are opaque to the registry; the serializer pops them in FIFO
    # order within a session.
    _queue: deque[Any] = field(default_factory=deque, repr=False)
    # Per-session monotonic submit counter — used as a tie-breaker for the
    # global round-robin policy and for fifo-global arrival ordering.
    _submit_seq: int = field(default=0, repr=False)

    def to_dict(self, *, device_resolver: Callable[[str], ResolvedOutputDevice] | None = None) -> dict[str, Any]:
        """Serialize for API responses. ``device_resolver`` lets the caller
        re-query the current device (per the ADR amendment — no caching).
        """
        d = {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "participant_type": self.participant_type,
            "assigned_voice": self.assigned_voice,
            "voice_conflict": self.voice_conflict,
            "preferred_voice": self.preferred_voice,
            "preferred_output_device": self.preferred_output_device,
            "state": self.state,
            "priority": self.priority,
            "registered_at": self.registered_at,
            "last_active": self.last_active,
            "queue_depth": len(self._queue),
        }
        if device_resolver is not None:
            try:
                d["output_device"] = device_resolver(self.preferred_output_device).to_dict()
            except Exception as exc:  # noqa: BLE001 — never fail serialization on device enumeration
                d["output_device"] = {
                    "preferred": self.preferred_output_device,
                    "index": None,
                    "name": "(unresolved)",
                    "fallback": True,
                    "reason": f"resolver error: {exc}",
                }
        return d


# ---------------------------------------------------------------------------
# Global serializer
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _SerializedJob:
    """Internal envelope for the global serializer.

    Priority-tuple ordering: lower sort-key first. We sort by
    (negative_priority, arrival_seq) so higher-priority jobs win and FIFO
    breaks ties. ``payload`` is opaque — the serializer only forwards it to
    the dispatch callback.
    """

    sort_key: tuple[int, int]
    session_id: str = field(compare=False)
    submit_seq: int = field(compare=False)
    payload: Any = field(compare=False, default=None)


class GlobalSerializer:
    """Layer-2 serializer over per-session output queues.

    Policies (ADR-082 § Output Serialization):

      * ``round-robin`` — alternate across sessions with pending work. This
        is the default and the reason Phase 1 exists at all: two concurrent
        sessions should not be able to starve each other.
      * ``priority`` — highest-priority session drains first; ties fall back
        to round-robin ordering.
      * ``fifo-global`` — strict arrival order across all sessions (matches
        today's single-global-queue behavior for migration parity).

    The serializer is push-driven: registrations submit jobs, and a
    dedicated dispatcher thread picks the next one per policy. Callers
    receive a ``QueuedJob``-shaped handle so existing call sites (speech
    queue, bus.act) can swap in without type changes.
    """

    def __init__(
        self,
        *,
        policy: str = "round-robin",
        dispatcher: Callable[[str, Any], Any] | None = None,
        now: Callable[[], float] = time.time,
    ):
        if policy not in SERIALIZATION_POLICIES:
            raise ValueError(f"unknown serialization policy: {policy}")
        self._policy = policy
        self._dispatcher = dispatcher
        self._now = now

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        # round-robin cursor — list of session_ids in rotation order; we pop
        # from the front and append to the back when a session has more work.
        self._rr_cursor: deque[str] = deque()
        self._rr_seen: set[str] = set()
        # priority heap for "priority" policy: (neg_prio, submit_seq, session, job_id)
        self._priority_heap: list[tuple[int, int, str, str]] = []
        # fifo-global heap: (submit_seq, session, job_id)
        self._fifo_heap: list[tuple[int, str, str]] = []
        # monotonic arrival counter shared across all sessions
        self._global_seq = 0

        self._sessions: dict[str, SessionChannel] = {}
        self._thread: threading.Thread | None = None
        self._stopping = False
        # Diagnostics: order in which jobs are dispatched. Bounded; newest
        # first. Primarily for tests and the /v1/sessions dashboard.
        self._dispatch_log: deque[tuple[float, str, str]] = deque(maxlen=256)

    # -- Policy plumbing ----------------------------------------------------

    @property
    def policy(self) -> str:
        return self._policy

    def set_policy(self, policy: str) -> None:
        if policy not in SERIALIZATION_POLICIES:
            raise ValueError(f"unknown serialization policy: {policy}")
        with self._lock:
            self._policy = policy

    def attach_dispatcher(self, dispatcher: Callable[[str, Any], Any]) -> None:
        """Set or replace the per-job dispatcher.

        The dispatcher receives ``(session_id, payload)`` on the dispatcher
        thread and runs synchronously — its completion is what advances the
        queue. The intent is: dispatcher blocks until the playback for this
        job finishes, preserving the single-speaker contract.
        """
        with self._lock:
            self._dispatcher = dispatcher

    # -- Registration management -------------------------------------------

    def attach_session(self, session: SessionChannel) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def detach_session(self, session_id: str) -> int:
        """Drop a session and any queued jobs. Returns count of jobs dropped."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            dropped = 0
            if session is not None:
                dropped = len(session._queue)
                session._queue.clear()
            # Best-effort: prune cursor entries for this session. The heaps
            # may still reference stale jobs; _next_job() will skip them.
            self._rr_seen.discard(session_id)
            self._rr_cursor = deque(s for s in self._rr_cursor if s != session_id)
            return dropped

    # -- Submission ---------------------------------------------------------

    def submit(
        self,
        session_id: str,
        payload: Any,
        *,
        priority: int | None = None,
    ) -> str:
        """Submit a job for ``session_id``.

        Returns an opaque job_id string. The payload is handed verbatim to
        the dispatcher when its turn comes up. Raises KeyError if the
        session is not registered — callers that want auto-registration
        should go through SessionRegistry.submit() instead.
        """
        with self._cond:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"session '{session_id}' is not registered")
            job_id = uuid.uuid4().hex[:8]
            self._global_seq += 1
            session._submit_seq += 1
            prio = priority if priority is not None else session.priority
            session._queue.append((self._global_seq, job_id, payload))
            session.last_active = self._now()

            # Update scheduling structures
            if session_id not in self._rr_seen:
                self._rr_seen.add(session_id)
                self._rr_cursor.append(session_id)
            heapq.heappush(self._priority_heap, (-prio, self._global_seq, session_id, job_id))
            heapq.heappush(self._fifo_heap, (self._global_seq, session_id, job_id))

            self._cond.notify_all()
            return job_id

    # -- Dispatch thread ----------------------------------------------------

    def start(self) -> None:
        """Start the dispatcher thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name="mod3-global-serializer",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._stopping and not self._has_pending_unlocked():
                    self._cond.wait(timeout=0.5)
                if self._stopping:
                    return
                picked = self._pop_next_unlocked()
                dispatcher = self._dispatcher
            if picked is None:
                continue
            session_id, job_id, payload = picked
            self._dispatch_log.appendleft((self._now(), session_id, job_id))
            if dispatcher is None:
                logger.debug(
                    "GlobalSerializer: no dispatcher attached — dropping job %s for %s",
                    job_id,
                    session_id,
                )
                continue
            try:
                dispatcher(session_id, payload)
            except Exception as exc:  # noqa: BLE001 — keep dispatcher robust
                logger.exception("GlobalSerializer dispatcher raised: %s", exc)

    def _has_pending_unlocked(self) -> bool:
        for s in self._sessions.values():
            if s._queue:
                return True
        return False

    def _pop_next_unlocked(self) -> tuple[str, str, Any] | None:
        """Pick and remove the next job according to policy."""
        if self._policy == "round-robin":
            return self._pop_round_robin_unlocked()
        if self._policy == "priority":
            return self._pop_priority_unlocked()
        # fifo-global
        return self._pop_fifo_unlocked()

    def _pop_round_robin_unlocked(self) -> tuple[str, str, Any] | None:
        # Walk the cursor until we find a session with pending work. Skip
        # sessions whose queues are empty — we rebuild the cursor lazily.
        checked = 0
        total = len(self._rr_cursor)
        while checked < total and self._rr_cursor:
            sid = self._rr_cursor.popleft()
            session = self._sessions.get(sid)
            if session is None or not session._queue:
                self._rr_seen.discard(sid)
                checked += 1
                continue
            seq, job_id, payload = session._queue.popleft()
            # If the session still has more, put it at the back of the
            # rotation so other sessions go first.
            if session._queue:
                self._rr_cursor.append(sid)
            else:
                self._rr_seen.discard(sid)
            return sid, job_id, payload
        return None

    def _pop_priority_unlocked(self) -> tuple[str, str, Any] | None:
        while self._priority_heap:
            neg_prio, seq, sid, job_id = heapq.heappop(self._priority_heap)
            session = self._sessions.get(sid)
            if session is None or not session._queue:
                continue
            # Pop the matching (seq, job_id) from the session queue. We match
            # by seq because heap order and queue order can diverge when two
            # sessions submit at once.
            found_idx = None
            for i, (qseq, qjid, _payload) in enumerate(session._queue):
                if qseq == seq and qjid == job_id:
                    found_idx = i
                    break
            if found_idx is None:
                continue
            qseq, qjid, payload = session._queue[found_idx]
            del session._queue[found_idx]
            return sid, qjid, payload
        return None

    def _pop_fifo_unlocked(self) -> tuple[str, str, Any] | None:
        while self._fifo_heap:
            seq, sid, job_id = heapq.heappop(self._fifo_heap)
            session = self._sessions.get(sid)
            if session is None or not session._queue:
                continue
            found_idx = None
            for i, (qseq, qjid, _payload) in enumerate(session._queue):
                if qseq == seq and qjid == job_id:
                    found_idx = i
                    break
            if found_idx is None:
                continue
            qseq, qjid, payload = session._queue[found_idx]
            del session._queue[found_idx]
            return sid, qjid, payload
        return None

    # -- Introspection ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy": self._policy,
                "sessions": {
                    sid: {
                        "queue_depth": len(s._queue),
                        "last_active": s.last_active,
                        "state": s.state,
                    }
                    for sid, s in self._sessions.items()
                },
                "rr_cursor": list(self._rr_cursor),
                "recent_dispatches": list(self._dispatch_log),
            }


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------


@dataclass
class RegistrationResult:
    session: SessionChannel
    created: bool  # False when re-registering an existing session_id
    voice_conflict: bool


class SessionRegistry:
    """Thread-safe registry of SessionChannels.

    Owns voice-pool allocation, device preference, and the global serializer.
    The registry is deliberately independent of ModalityBus — tests can spin
    it up without the full bus stack, and the bus can adopt it incrementally.
    """

    def __init__(
        self,
        *,
        voice_pool: Iterable[str] | None = None,
        serializer: GlobalSerializer | None = None,
        device_resolver: Callable[[str], ResolvedOutputDevice] | None = None,
    ):
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionChannel] = {}
        self._voice_pool: list[str] = list(voice_pool if voice_pool is not None else VOICE_POOL)
        # Track which voice is currently held by which session, first-come
        # first-served. A second request for the same voice is honored (voice
        # is identity — collisions should be rare) but flagged.
        self._voice_holder: dict[str, str] = {}
        self._serializer = serializer or GlobalSerializer()
        self._device_resolver = device_resolver or resolve_output_device

    # -- Lifecycle ----------------------------------------------------------

    @property
    def serializer(self) -> GlobalSerializer:
        return self._serializer

    def start(self) -> None:
        self._serializer.start()

    def stop(self) -> None:
        self._serializer.stop()

    # -- Session management -------------------------------------------------

    def register(
        self,
        *,
        session_id: str,
        participant_id: str,
        participant_type: str = DEFAULT_PARTICIPANT_TYPE,
        preferred_voice: str | None = None,
        preferred_output_device: str = "system-default",
        priority: int = 0,
    ) -> RegistrationResult:
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                existing.participant_id = participant_id
                existing.participant_type = participant_type
                existing.preferred_output_device = preferred_output_device or "system-default"
                existing.last_active = time.time()
                # Don't reshuffle voice on re-register. If the caller wants a
                # different voice they should deregister first.
                return RegistrationResult(existing, created=False, voice_conflict=existing.voice_conflict)

            voice, conflict = self._allocate_voice(session_id, preferred_voice)
            session = SessionChannel(
                session_id=session_id,
                participant_id=participant_id,
                participant_type=participant_type,
                assigned_voice=voice,
                voice_conflict=conflict,
                preferred_voice=preferred_voice,
                preferred_output_device=preferred_output_device or "system-default",
                priority=priority,
            )
            self._sessions[session_id] = session
            self._serializer.attach_session(session)
            logger.info(
                "registered session: id=%s participant=%s voice=%s conflict=%s device=%s",
                session_id,
                participant_id,
                voice,
                conflict,
                preferred_output_device,
            )
            return RegistrationResult(session, created=True, voice_conflict=conflict)

    def deregister(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return {"status": "not_found", "session_id": session_id}
            dropped = self._serializer.detach_session(session_id)
            # Return the voice to the pool if we still hold it for this session.
            if self._voice_holder.get(session.assigned_voice) == session_id:
                del self._voice_holder[session.assigned_voice]
            logger.info(
                "deregistered session: id=%s voice=%s dropped_jobs=%d",
                session_id,
                session.assigned_voice,
                dropped,
            )
            return {
                "status": "ok",
                "session_id": session_id,
                "released_voice": session.assigned_voice,
                "dropped_jobs": dropped,
            }

    def get(self, session_id: str) -> SessionChannel | None:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create_default(self) -> SessionChannel:
        """Backward-compat path — route legacy callers to an implicit session."""
        with self._lock:
            session = self._sessions.get(DEFAULT_SESSION_ID)
            if session is not None:
                return session
        result = self.register(
            session_id=DEFAULT_SESSION_ID,
            participant_id=DEFAULT_PARTICIPANT_ID,
            participant_type=DEFAULT_PARTICIPANT_TYPE,
            preferred_voice=None,
            preferred_output_device="system-default",
        )
        return result.session

    def list(self) -> list[SessionChannel]:
        with self._lock:
            return list(self._sessions.values())

    def list_serialized(self) -> list[dict[str, Any]]:
        with self._lock:
            resolver = self._device_resolver
            return [s.to_dict(device_resolver=resolver) for s in self._sessions.values()]

    def voice_pool(self) -> list[str]:
        return list(self._voice_pool)

    def voice_holder_snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._voice_holder)

    # -- Submission ---------------------------------------------------------

    def submit(
        self,
        session_id: str | None,
        payload: Any,
        *,
        priority: int | None = None,
        auto_create_default: bool = True,
    ) -> tuple[str, str]:
        """Enqueue ``payload`` on ``session_id``.

        When session_id is None and auto_create_default is True, falls back
        to the "default" session so legacy call sites keep working.

        Returns ``(resolved_session_id, job_id)``.
        """
        if session_id is None or session_id == "":
            if not auto_create_default:
                raise ValueError("session_id is required when auto_create_default=False")
            session = self.get_or_create_default()
            session_id = session.session_id
        else:
            with self._lock:
                if session_id not in self._sessions:
                    raise KeyError(f"session '{session_id}' is not registered")

        job_id = self._serializer.submit(session_id, payload, priority=priority)
        return session_id, job_id

    # -- Device routing -----------------------------------------------------

    def resolve_device(self, session_id: str) -> ResolvedOutputDevice:
        """Live-resolve the output device for ``session_id``.

        Per the 2026-04-22 amendment, this is a live property: the OS default
        is re-queried every call when the session's preference is
        ``system-default``. Never cache the return value on the session.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            preferred = session.preferred_output_device if session else "system-default"
        return self._device_resolver(preferred)

    def set_preferred_device(self, session_id: str, preferred: str) -> ResolvedOutputDevice:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            session.preferred_output_device = preferred or "system-default"
        return self.resolve_device(session_id)

    # -- Internals ----------------------------------------------------------

    def _allocate_voice(self, session_id: str, preferred: str | None) -> tuple[str, bool]:
        # Explicit preference — honor it, flag if someone else already holds
        # it. Voice is identity, not exclusive; collisions are operator bugs
        # at worst.
        if preferred:
            if preferred not in self._voice_pool:
                # Extend the pool lazily so out-of-band voices still work.
                self._voice_pool.append(preferred)
            holder = self._voice_holder.get(preferred)
            conflict = holder is not None and holder != session_id
            self._voice_holder.setdefault(preferred, session_id)
            return preferred, conflict

        # Greedy allocation — first voice in the pool whose holder is absent
        # or dead.
        for voice in self._voice_pool:
            if voice not in self._voice_holder:
                self._voice_holder[voice] = session_id
                return voice, False

        # Pool exhausted — fall back to the first voice and flag a collision.
        fallback = self._voice_pool[0] if self._voice_pool else "bm_lewis"
        return fallback, True


# ---------------------------------------------------------------------------
# Process-global default registry — shared across server.py, http_api.py,
# and tests. Tests can instantiate their own SessionRegistry if they need
# isolation; the module-level singleton is just a convenience for runtime.
# ---------------------------------------------------------------------------


_default_registry: SessionRegistry | None = None
_default_registry_lock = threading.Lock()


def get_default_registry() -> SessionRegistry:
    global _default_registry
    with _default_registry_lock:
        if _default_registry is None:
            _default_registry = SessionRegistry()
            _default_registry.start()
        return _default_registry


def reset_default_registry() -> None:
    """For tests — tear down the module-level registry."""
    global _default_registry
    with _default_registry_lock:
        if _default_registry is not None:
            _default_registry.stop()
            _default_registry = None


@atexit.register
def _shutdown_default_registry() -> None:
    """Stop the dispatcher thread cleanly at interpreter shutdown.

    Without this the daemon thread gets killed mid-callback during
    finalization and Python emits a Fatal Python error about a NULL
    thread state. The dispatcher is idempotent for stop() so this is
    safe to call even if the registry was never created.
    """
    try:
        reset_default_registry()
    except Exception:
        pass


__all__ = [
    "DEFAULT_SESSION_ID",
    "DEFAULT_PARTICIPANT_ID",
    "DEFAULT_PARTICIPANT_TYPE",
    "GlobalSerializer",
    "RegistrationResult",
    "ResolvedOutputDevice",
    "SessionChannel",
    "SessionRegistry",
    "SERIALIZATION_POLICIES",
    "VOICE_POOL",
    "get_default_registry",
    "reset_default_registry",
    "resolve_output_device",
]

"""Mod³ session-seat registry — in-memory seat management for channel clients.

Architecture
------------
A "seat" is a logical slot held by a channel_client.py subprocess within a
mod³ session.  Multiple channel clients (one per Claude Code session) can
attach to the same mod³ session and receive events fanned out by the daemon.

The seat registry is process-local (single FastAPI process, single daemon).
Seats are keyed by (session_id, seat_id).  Each seat owns an asyncio.Queue
that drives its SSE stream; when a dashboard message arrives it is pushed
into every seat queue attached to that session.

HTTP endpoints (wired in http_api.py):
  POST   /v1/sessions/{session_id}/seats
  DELETE /v1/sessions/{session_id}/seats/{seat_id}
  GET    /v1/sessions/{session_id}/seats/{seat_id}/events  (SSE)
  POST   /v1/sessions/{session_id}/messages               (dashboard fan-out)

Fan-out policy (v1): broadcast to all seats in the session, optionally
skipping the originating seat to prevent echo loops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("mod3.seats")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

VALID_CLIENT_TYPES = frozenset({"claude-code-channel", "generic", "rtvi-client"})

_SEAT_TTL_SECONDS = 3600  # seats auto-expire after 1 hour of inactivity


@dataclass
class Seat:
    seat_id: str
    session_id: str
    client_type: str
    device_uuid: str
    created_at: float = field(default_factory=time.time)
    # Wave 6b: user identity OIDC claims — who the human operator is.
    # None means unattributed (pre-Wave-6b callers; backward compatible).
    user_iss: str | None = None
    user_sub: str | None = None
    # Wave 6c / Primitive 2: agent identity claims — the LLM entity co-present
    # with the user. Only set for agentic harnesses (Claude Code, Cursor, etc.).
    # Non-agentic seats leave these None.
    assistant_iss: str | None = None
    assistant_sub: str | None = None
    # Primitive 4: channel pipeline mode for this seat.
    # "intentional" (default) = session-scoped, explicit participation.
    # "ambient" = always-on, VAD-gated, continuous diarization.
    channel_mode: str = "intentional"
    # SSE event queue — one entry per pending event
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # asyncio loop that owns this seat's queue
    loop: asyncio.AbstractEventLoop | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seat_id": self.seat_id,
            "session_id": self.session_id,
            "client_type": self.client_type,
            "device_uuid": self.device_uuid,
            "created_at": self.created_at,
            # All four identity fields are always present in the dict so
            # callers can pattern-match on None rather than key presence.
            "user_iss": self.user_iss,
            "user_sub": self.user_sub,
            "assistant_iss": self.assistant_iss,
            "assistant_sub": self.assistant_sub,
            "channel_mode": self.channel_mode,
        }
        return d


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SeatRegistry:
    """Thread-safe in-memory registry of channel-client seats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {session_id: {seat_id: Seat}}
        self._seats: dict[str, dict[str, Seat]] = {}
        # Live SSE stream counts keyed by session_id. A seat's liveness is
        # defined by an open SSE stream (GET .../events), not by the registry
        # record — a seat with no live connection is not a seat (per the
        # seat-as-coordination-surface model). ``sse_stream`` increments on
        # entry and decrements on disconnect via mark_stream_open/closed.
        self._live_streams: dict[str, int] = {}
        # Callback fired when a session's last live SSE stream closes. Wired by
        # http_api to deregister the orphaned session from the SessionRegistry.
        # Kept as a plain callable to avoid a circular import on session_registry.
        self._on_session_idle: Callable[[str], None] | None = None

    # ------------------------------------------------------------------
    # Liveness tracking
    # ------------------------------------------------------------------

    def set_on_session_idle(self, callback: Callable[[str], None] | None) -> None:
        """Register a callback fired when a session's last SSE stream closes."""
        with self._lock:
            self._on_session_idle = callback

    def mark_stream_open(self, session_id: str) -> None:
        with self._lock:
            self._live_streams[session_id] = self._live_streams.get(session_id, 0) + 1

    def mark_stream_closed(self, session_id: str) -> None:
        """Decrement the live-stream count; fire the idle callback at zero."""
        with self._lock:
            count = self._live_streams.get(session_id, 0) - 1
            if count <= 0:
                self._live_streams.pop(session_id, None)
                idle = True
                callback = self._on_session_idle
            else:
                self._live_streams[session_id] = count
                idle = False
                callback = None
        if idle and callback is not None:
            try:
                callback(session_id)
            except Exception as exc:  # noqa: BLE001 — never let a stream teardown crash on the callback
                logger.warning("on_session_idle callback failed for %s: %s", session_id, exc)

    def has_live_stream(self, session_id: str) -> bool:
        with self._lock:
            return self._live_streams.get(session_id, 0) > 0

    def live_session_ids(self) -> set[str]:
        with self._lock:
            return {sid for sid, n in self._live_streams.items() if n > 0}

    # ------------------------------------------------------------------
    # Seat lifecycle
    # ------------------------------------------------------------------

    def register(
        self,
        session_id: str,
        client_type: str,
        device_uuid: str,
        user_iss: str | None = None,
        user_sub: str | None = None,
        assistant_iss: str | None = None,
        assistant_sub: str | None = None,
        # Wave 6b backward-compat aliases: callers passing iss/sub before the
        # Primitive-2 rename are mapped to user_iss/user_sub transparently.
        iss: str | None = None,
        sub: str | None = None,
        channel_mode: str = "intentional",
    ) -> Seat:
        """Create a new seat in *session_id*.  Auto-creates the session bucket.

        Args:
            session_id: Target session (auto-created if absent).
            client_type: One of VALID_CLIENT_TYPES; falls back to "generic".
            device_uuid: Persistent client-side UUID.
            user_iss: OIDC issuer for the human operator identity.
            user_sub: OIDC subject slug for the user (e.g. "chaz").
            assistant_iss: OIDC issuer for the agent identity (agentic harnesses only).
            assistant_sub: OIDC subject slug for the agent (e.g. "cog").
            iss: Deprecated alias for user_iss (Wave 6b callers).
            sub: Deprecated alias for user_sub (Wave 6b callers).
            channel_mode: Pipeline mode for this seat. "intentional" (default)
                or "ambient". Stored on the seat for downstream routing.
        """
        # Merge Wave-6b aliases into the canonical user_* fields.
        resolved_user_iss = user_iss or iss or None
        resolved_user_sub = user_sub or sub or None

        if client_type not in VALID_CLIENT_TYPES:
            client_type = "generic"
        seat_id = str(uuid.uuid4())
        seat = Seat(
            seat_id=seat_id,
            session_id=session_id,
            client_type=client_type,
            device_uuid=device_uuid,
            user_iss=resolved_user_iss,
            user_sub=resolved_user_sub,
            assistant_iss=assistant_iss,
            assistant_sub=assistant_sub,
            channel_mode=channel_mode,
        )
        with self._lock:
            if session_id not in self._seats:
                self._seats[session_id] = {}
            self._seats[session_id][seat_id] = seat
        if resolved_user_sub or assistant_sub:
            identity_info = f" user={resolved_user_sub!r} agent={assistant_sub!r}"
        else:
            identity_info = ""
        logger.info(
            "Seat %s registered in session %s (client=%s mode=%s%s)",
            seat_id,
            session_id,
            client_type,
            channel_mode,
            identity_info,
        )
        return seat

    def get(self, session_id: str, seat_id: str) -> Seat | None:
        with self._lock:
            return self._seats.get(session_id, {}).get(seat_id)

    def revoke(self, session_id: str, seat_id: str) -> bool:
        """Remove a seat.  Returns True if the seat existed."""
        with self._lock:
            session_seats = self._seats.get(session_id)
            if not session_seats:
                return False
            seat = session_seats.pop(seat_id, None)
            if seat is None:
                return False
            # Signal SSE stream to close
            _enqueue_nowait(seat, {"type": "_close"})
            logger.info("Seat %s revoked from session %s", seat_id, session_id)
            return True

    def list_session_seats(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [s.to_dict() for s in self._seats.get(session_id, {}).values()]

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._seats.keys())

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    def fan_out(
        self,
        session_id: str,
        event: dict[str, Any],
        exclude_seat: str | None = None,
    ) -> int:
        """Broadcast *event* to all seats attached to *session_id*.

        Args:
            session_id: Target session.
            event: Event dict to enqueue on each seat's SSE queue.
            exclude_seat: Optional seat_id to skip.  Pass the originating
                seat so it does not receive its own outbound message back
                (prevents dashboard-chat echo loops).

        Returns the number of seats that received the event.
        """
        with self._lock:
            seats = list(self._seats.get(session_id, {}).values())
        count = 0
        for seat in seats:
            if exclude_seat and seat.seat_id == exclude_seat:
                logger.debug("Fan-out skipping originating seat %s (echo suppression)", exclude_seat)
                continue
            _enqueue_nowait(seat, event)
            count += 1
        if count:
            logger.debug("Fan-out to %d seats in session %s: type=%s", count, session_id, event.get("type"))
        return count

    def fan_out_all(self, event: dict[str, Any], exclude_seat: str | None = None) -> int:
        """Broadcast *event* to ALL seats across all sessions.

        Args:
            event: Event dict to enqueue.
            exclude_seat: Optional seat_id to skip across all sessions.
        """
        with self._lock:
            all_seats = [seat for session_seats in self._seats.values() for seat in session_seats.values()]
        count = 0
        for seat in all_seats:
            if exclude_seat and seat.seat_id == exclude_seat:
                logger.debug("Fan-out-all skipping originating seat %s (echo suppression)", exclude_seat)
                continue
            _enqueue_nowait(seat, event)
            count += 1
        return count


def _enqueue_nowait(seat: Seat, event: dict[str, Any]) -> None:
    """Thread-safe enqueue — uses the seat's loop if available."""
    if seat.loop is not None and seat.loop.is_running():
        seat.loop.call_soon_threadsafe(seat.queue.put_nowait, event)
    else:
        try:
            seat.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.debug("Seat %s queue full — dropping event %s", seat.seat_id, event.get("type"))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_registry: SeatRegistry | None = None
_registry_lock = threading.Lock()


def get_seat_registry() -> SeatRegistry:
    """Return the process-singleton SeatRegistry."""
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = SeatRegistry()
    return _default_registry


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


async def sse_stream(seat: Seat):
    """Async generator yielding raw SSE text lines for a seat's event queue.

    Yields formatted SSE strings.  Yields a keep-alive comment every 15 s.
    Closes cleanly when a ``{"type": "_close"}`` sentinel is dequeued or
    when the caller cancels the task.
    """
    seat.loop = asyncio.get_running_loop()
    KEEPALIVE_INTERVAL = 15.0
    # Liveness: an open SSE stream is what makes this seat's session "live".
    # Mark it open here and closed in the finally so a killed/crashed client
    # (which never runs the DELETE seat hook) still drops liveness when its
    # stream tears down — letting the reaper prune the orphaned session.
    registry = get_seat_registry()
    registry.mark_stream_open(seat.session_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(seat.queue.get(), timeout=KEEPALIVE_INTERVAL)
            except asyncio.TimeoutError:
                # Keep-alive ping
                yield ": keepalive\n\n"
                continue

            if event.get("type") == "_close":
                break

            etype = event.get("type", "event")
            data = json.dumps(event, separators=(",", ":"))
            yield f"event: {etype}\ndata: {data}\n\n"

    except asyncio.CancelledError:
        pass
    finally:
        registry.mark_stream_closed(seat.session_id)
        logger.debug("SSE stream closed for seat %s", seat.seat_id)

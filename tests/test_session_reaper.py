"""Unit tests for the session-lifecycle reaper (fix/session-lifecycle-reaper).

Background
----------
``SessionRegistry`` accumulated channel-client sessions forever: a session was
removed only by an explicit ``deregister`` (the graceful DELETE-seat hook). A
Claude Code session that was killed, crashed, or just had its terminal closed
never ran that hook, so its seat orphaned indefinitely — the dashboard showed
100+ stale ``channel-client::…`` sessions aging for days.

The fix ties session liveness to a connection. Two mechanisms:
  * ``reap_stale`` — a periodic sweep deregisters sessions whose ``last_active``
    exceeds a TTL AND which have no live SSE stream. ``main`` is never reaped.
  * an SSE-disconnect hook (in seats.py) deregisters a session when its last
    live stream closes.

These tests cover the reaper directly with a controllable clock + injected
liveness check, so no real threads or sockets are needed.

Run with: ``PYTHONPATH=. .venv/bin/python -m pytest tests/test_session_reaper.py -v``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_registry import (  # noqa: E402
    MAIN_SESSION_ID,
    SessionRegistry,
)


class _Clock:
    """Manually-advanced clock so TTL behavior is deterministic."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _make_registry(clock: _Clock, *, ttl: float = 600.0, live: set[str] | None = None) -> SessionRegistry:
    live = live if live is not None else set()
    # reaper_interval_seconds<=0 keeps the background thread from auto-starting;
    # we drive reap_stale() by hand. liveness_check reads the mutable `live` set.
    return SessionRegistry(
        ttl_seconds=ttl,
        reaper_interval_seconds=0,
        liveness_check=lambda sid: sid in live,
        now=clock,
    )


class TestReaper:
    def test_prunes_stale_session_with_no_live_stream(self):
        clock = _Clock()
        reg = _make_registry(clock, ttl=600.0)
        reg.register(session_id="ghost", participant_id="channel-client::generic")

        # Not yet stale.
        clock.advance(300.0)
        assert reg.reap_stale() == []
        assert reg.get("ghost") is not None

        # Past the TTL with no live stream → reaped.
        clock.advance(400.0)  # total 700s > 600s TTL
        assert reg.reap_stale() == ["ghost"]
        assert reg.get("ghost") is None

    def test_never_reaps_main_even_when_stale(self):
        clock = _Clock()
        reg = _make_registry(clock, ttl=600.0)
        reg.register(session_id=MAIN_SESSION_ID, participant_id="channel-client-pool")

        clock.advance(10_000.0)  # wildly past TTL
        assert reg.reap_stale() == []
        assert reg.get(MAIN_SESSION_ID) is not None

    def test_preserves_live_but_quiet_session(self):
        clock = _Clock()
        live = {"live-seat"}
        reg = _make_registry(clock, ttl=600.0, live=live)
        reg.register(session_id="live-seat", participant_id="channel-client::generic")

        # Idle well past the TTL, but it still holds a live SSE stream.
        clock.advance(5_000.0)
        assert reg.reap_stale() == []
        assert reg.get("live-seat") is not None

        # Once the stream drops, the next sweep reaps it.
        live.discard("live-seat")
        assert reg.reap_stale() == ["live-seat"]
        assert reg.get("live-seat") is None

    def test_touch_resets_the_ttl(self):
        clock = _Clock()
        reg = _make_registry(clock, ttl=600.0)
        reg.register(session_id="chatty", participant_id="channel-client::generic")

        clock.advance(500.0)
        assert reg.touch("chatty") is True  # activity keeps it alive
        clock.advance(500.0)  # 500s since last touch < 600s TTL
        assert reg.reap_stale() == []
        assert reg.get("chatty") is not None

        clock.advance(200.0)  # 700s since touch > TTL, no live stream
        assert reg.reap_stale() == ["chatty"]

    def test_touch_unknown_session_returns_false(self):
        clock = _Clock()
        reg = _make_registry(clock)
        assert reg.touch("nope") is False

    def test_liveness_check_failure_is_treated_as_unknown_not_reaped(self):
        clock = _Clock()

        def _boom(_sid: str) -> bool:
            raise RuntimeError("seat registry unavailable")

        reg = SessionRegistry(ttl_seconds=600.0, reaper_interval_seconds=0, liveness_check=_boom, now=clock)
        reg.register(session_id="risky", participant_id="channel-client::generic")
        clock.advance(1_000.0)
        # A failing liveness check must not cause a reap (fail safe, keep it).
        assert reg.reap_stale() == []
        assert reg.get("risky") is not None

    def test_reaps_multiple_and_keeps_live_and_main(self):
        clock = _Clock()
        live = {"keep-live"}
        reg = _make_registry(clock, ttl=600.0, live=live)
        reg.register(session_id=MAIN_SESSION_ID, participant_id="channel-client-pool")
        reg.register(session_id="keep-live", participant_id="channel-client::generic")
        reg.register(session_id="stale-1", participant_id="channel-client::generic")
        reg.register(session_id="stale-2", participant_id="channel-client::generic")

        clock.advance(700.0)
        reaped = set(reg.reap_stale())
        assert reaped == {"stale-1", "stale-2"}
        assert reg.get(MAIN_SESSION_ID) is not None
        assert reg.get("keep-live") is not None


class TestSeatStreamLiveness:
    """The SSE stream open/close accounting that feeds the reaper's liveness."""

    def test_stream_open_close_drives_idle_callback(self):
        from seats import SeatRegistry

        reg = SeatRegistry()
        idle: list[str] = []
        reg.set_on_session_idle(idle.append)

        reg.mark_stream_open("sess-A")
        reg.mark_stream_open("sess-A")
        assert reg.has_live_stream("sess-A") is True

        # First close: still one stream left, no idle yet.
        reg.mark_stream_closed("sess-A")
        assert reg.has_live_stream("sess-A") is True
        assert idle == []

        # Last close: session goes idle → callback fires once.
        reg.mark_stream_closed("sess-A")
        assert reg.has_live_stream("sess-A") is False
        assert idle == ["sess-A"]

    def test_live_session_ids_tracks_open_streams(self):
        from seats import SeatRegistry

        reg = SeatRegistry()
        reg.mark_stream_open("a")
        reg.mark_stream_open("b")
        reg.mark_stream_closed("a")
        assert reg.live_session_ids() == {"b"}

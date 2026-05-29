"""Tests for SeatRegistry fan-out echo suppression.

Covers:
  * Unit: fan_out with exclude_seat skips the originating seat.
  * Unit: fan_out with no exclude_seat delivers to all seats.
  * Unit: fan_out returns correct delivery count with and without exclusion.
  * Unit: fan_out_all with exclude_seat skips the seat across all sessions.
  * Integration: simulate dashboard-chat POST with seat_id — originator
    does not receive its own broadcast back.

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_seat_fanout.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seats import VALID_CLIENT_TYPES, Seat, SeatRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seat(session_id: str, seat_id: str) -> Seat:
    seat = Seat(
        seat_id=seat_id,
        session_id=session_id,
        client_type="claude-code-channel",
        device_uuid=f"dev-{seat_id}",
    )
    # Attach a real asyncio loop so enqueue works synchronously via put_nowait
    seat.loop = None  # no loop — falls back to put_nowait path
    return seat


def _drain(seat: Seat) -> list[dict]:
    """Pull everything currently in the seat's queue (non-blocking)."""
    items = []
    while True:
        try:
            items.append(seat.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# ---------------------------------------------------------------------------
# Unit: VALID_CLIENT_TYPES
# ---------------------------------------------------------------------------


class TestValidClientTypes:
    def test_contains_rtvi_client(self):
        assert "rtvi-client" in VALID_CLIENT_TYPES

    def test_contains_legacy_types(self):
        assert "claude-code-channel" in VALID_CLIENT_TYPES
        assert "generic" in VALID_CLIENT_TYPES

    def test_is_frozenset(self):
        assert isinstance(VALID_CLIENT_TYPES, frozenset)


# Unit: SeatRegistry.fan_out
# ---------------------------------------------------------------------------


class TestFanOutExcludeSeat:
    def test_no_exclusion_delivers_to_all_seats(self):
        reg = SeatRegistry()
        s1 = _make_seat("sess", "seat-a")
        s2 = _make_seat("sess", "seat-b")
        with reg._lock:
            reg._seats["sess"] = {"seat-a": s1, "seat-b": s2}

        event = {"type": "user_message", "content": "hello"}
        count = reg.fan_out("sess", event)

        assert count == 2
        assert len(_drain(s1)) == 1
        assert len(_drain(s2)) == 1

    def test_exclude_seat_skips_originator(self):
        reg = SeatRegistry()
        s1 = _make_seat("sess", "seat-a")
        s2 = _make_seat("sess", "seat-b")
        with reg._lock:
            reg._seats["sess"] = {"seat-a": s1, "seat-b": s2}

        event = {"type": "assistant_message", "content": "reply"}
        count = reg.fan_out("sess", event, exclude_seat="seat-a")

        # Only seat-b receives it
        assert count == 1
        assert len(_drain(s1)) == 0, "originating seat should not receive its own message"
        assert len(_drain(s2)) == 1

    def test_exclude_seat_nonexistent_is_harmless(self):
        """Excluding a seat_id not in the registry is a no-op — no errors."""
        reg = SeatRegistry()
        s1 = _make_seat("sess", "seat-a")
        with reg._lock:
            reg._seats["sess"] = {"seat-a": s1}

        count = reg.fan_out("sess", {"type": "test"}, exclude_seat="seat-ghost")
        assert count == 1
        assert len(_drain(s1)) == 1

    def test_exclude_seat_none_delivers_all(self):
        """Passing exclude_seat=None is the same as no exclusion."""
        reg = SeatRegistry()
        s1 = _make_seat("sess", "seat-a")
        s2 = _make_seat("sess", "seat-b")
        with reg._lock:
            reg._seats["sess"] = {"seat-a": s1, "seat-b": s2}

        count = reg.fan_out("sess", {"type": "test"}, exclude_seat=None)
        assert count == 2
        assert len(_drain(s1)) == 1
        assert len(_drain(s2)) == 1

    def test_fan_out_returns_zero_for_empty_session(self):
        reg = SeatRegistry()
        count = reg.fan_out("no-such-session", {"type": "test"}, exclude_seat="any")
        assert count == 0


# ---------------------------------------------------------------------------
# Unit: SeatRegistry.fan_out_all
# ---------------------------------------------------------------------------


class TestFanOutAllExcludeSeat:
    def test_exclude_seat_skips_across_sessions(self):
        reg = SeatRegistry()
        s_a1 = _make_seat("sess-1", "seat-x")  # originator
        s_a2 = _make_seat("sess-1", "seat-y")
        s_b1 = _make_seat("sess-2", "seat-x")  # same seat_id in different session
        with reg._lock:
            reg._seats["sess-1"] = {"seat-x": s_a1, "seat-y": s_a2}
            reg._seats["sess-2"] = {"seat-x": s_b1}

        count = reg.fan_out_all({"type": "global"}, exclude_seat="seat-x")

        # seat-x appears in both sessions — both excluded; only seat-y receives
        assert count == 1
        assert len(_drain(s_a1)) == 0
        assert len(_drain(s_a2)) == 1
        assert len(_drain(s_b1)) == 0

    def test_fan_out_all_no_exclusion_reaches_all(self):
        reg = SeatRegistry()
        s1 = _make_seat("sess-1", "seat-a")
        s2 = _make_seat("sess-2", "seat-b")
        with reg._lock:
            reg._seats["sess-1"] = {"seat-a": s1}
            reg._seats["sess-2"] = {"seat-b": s2}

        count = reg.fan_out_all({"type": "ping"})
        assert count == 2
        assert len(_drain(s1)) == 1
        assert len(_drain(s2)) == 1


# ---------------------------------------------------------------------------
# Integration: dashboard-chat POST excludes originating seat
# ---------------------------------------------------------------------------


class TestDashboardChatEchoSuppression:
    """Verify that the /v1/dashboard-chat endpoint does not echo to the sender."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app, base_url="http://localhost:7860")

    @pytest.fixture(autouse=True)
    def _clean_seats(self):
        """Ensure each test starts with a fresh seat registry."""
        from seats import get_seat_registry

        # Reset the singleton by clearing its internal state
        reg = get_seat_registry()
        with reg._lock:
            reg._seats.clear()
        yield
        with reg._lock:
            reg._seats.clear()

    def test_dashboard_chat_does_not_echo_to_sender(self, client):
        """POST /v1/dashboard-chat with seat_id must not deliver to that seat."""
        from seats import get_seat_registry

        reg = get_seat_registry()

        # Register two seats manually
        sender = _make_seat("session-test", "sender-seat")
        observer = _make_seat("session-test", "observer-seat")
        with reg._lock:
            reg._seats["session-test"] = {
                "sender-seat": sender,
                "observer-seat": observer,
            }

        resp = client.post(
            "/v1/dashboard-chat",
            json={
                "text": "hello from assistant",
                "role": "assistant",
                "session_id": "session-test",
                "seat_id": "sender-seat",
            },
        )
        assert resp.status_code == 200

        sender_events = _drain(sender)
        observer_events = _drain(observer)

        assert len(sender_events) == 0, f"Sender received its own broadcast back — echo loop bug: {sender_events}"
        assert len(observer_events) == 1, f"Observer should receive the message: {observer_events}"
        assert observer_events[0]["type"] == "assistant_message"
        assert observer_events[0]["content"] == "hello from assistant"

    def test_dashboard_chat_without_seat_id_broadcasts_all(self, client):
        """POST /v1/dashboard-chat without seat_id delivers to all seats (no exclusion)."""
        from seats import get_seat_registry

        reg = get_seat_registry()

        s1 = _make_seat("session-all", "seat-1")
        s2 = _make_seat("session-all", "seat-2")
        with reg._lock:
            reg._seats["session-all"] = {"seat-1": s1, "seat-2": s2}

        resp = client.post(
            "/v1/dashboard-chat",
            json={
                "text": "broadcast",
                "role": "assistant",
                "session_id": "session-all",
                # no seat_id
            },
        )
        assert resp.status_code == 200

        assert len(_drain(s1)) == 1
        assert len(_drain(s2)) == 1

    def test_session_messages_excludes_sender(self, client):
        """POST /v1/sessions/{id}/messages with seat_id excludes that seat."""
        from seats import get_seat_registry

        reg = get_seat_registry()

        sender = _make_seat("session-msg", "sender-seat")
        receiver = _make_seat("session-msg", "receiver-seat")
        with reg._lock:
            reg._seats["session-msg"] = {
                "sender-seat": sender,
                "receiver-seat": receiver,
            }

        resp = client.post(
            "/v1/sessions/session-msg/messages",
            json={
                "content": "user typed this",
                "input_type": "text",
                "role": "user",
                "seat_id": "sender-seat",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["seats_notified"] == 1

        assert len(_drain(sender)) == 0, "Sender must not receive its own message"
        assert len(_drain(receiver)) == 1

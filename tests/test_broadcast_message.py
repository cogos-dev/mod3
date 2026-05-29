"""Tests for POST /v1/sessions/broadcast-message — dashboard-to-MCP receive path.

Covers:
  * Broadcast fans to ALL seats across ALL sessions when no target is specified.
  * Addressed broadcast (target_session_id) fans only to the specified session.
  * Empty content returns 400.
  * Invalid JSON returns 400.
  * Returns seats_notified count and target label.

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_broadcast_message.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seats import Seat  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seat(session_id: str, seat_id: str) -> Seat:
    return Seat(
        seat_id=seat_id,
        session_id=session_id,
        client_type="claude-code-channel",
        device_uuid=f"dev-{seat_id}",
    )


def _drain(seat: Seat) -> list[dict]:
    items = []
    while True:
        try:
            items.append(seat.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture(autouse=True)
def _clean_seats():
    from seats import get_seat_registry

    reg = get_seat_registry()
    with reg._lock:
        reg._seats.clear()
    yield
    with reg._lock:
        reg._seats.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBroadcastMessage:
    def test_broadcast_fans_to_all_sessions(self, client):
        """Without target_session_id, all seats in all sessions receive the event."""
        from seats import get_seat_registry

        reg = get_seat_registry()
        s1 = _make_seat("session-a", "seat-1")
        s2 = _make_seat("session-b", "seat-2")
        with reg._lock:
            reg._seats["session-a"] = {"seat-1": s1}
            reg._seats["session-b"] = {"seat-2": s2}

        resp = client.post(
            "/v1/sessions/broadcast-message",
            json={"content": "hello channel", "input_type": "text", "role": "user"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["seats_notified"] == 2
        assert data["target"] == "all"

        e1 = _drain(s1)
        e2 = _drain(s2)
        assert len(e1) == 1
        assert len(e2) == 1
        assert e1[0]["type"] == "user_message"
        assert e1[0]["content"] == "hello channel"

    def test_addressed_broadcast_targets_session(self, client):
        """With target_session_id, only that session's seats receive the event."""
        from seats import get_seat_registry

        reg = get_seat_registry()
        s1 = _make_seat("session-a", "seat-1")
        s2 = _make_seat("session-b", "seat-2")
        with reg._lock:
            reg._seats["session-a"] = {"seat-1": s1}
            reg._seats["session-b"] = {"seat-2": s2}

        resp = client.post(
            "/v1/sessions/broadcast-message",
            json={
                "content": "hello session-a only",
                "input_type": "text",
                "role": "user",
                "target_session_id": "session-a",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["seats_notified"] == 1
        assert data["target"] == "session-a"

        assert len(_drain(s1)) == 1, "target session should receive the message"
        assert len(_drain(s2)) == 0, "non-target session should not receive the message"

    def test_empty_content_returns_400(self, client):
        """Empty content field is rejected with 400."""
        resp = client.post(
            "/v1/sessions/broadcast-message",
            json={"content": "", "role": "user"},
        )
        assert resp.status_code == 400

    def test_missing_content_returns_400(self, client):
        """Missing content field is rejected with 400."""
        resp = client.post(
            "/v1/sessions/broadcast-message",
            json={"role": "user"},
        )
        assert resp.status_code == 400

    def test_broadcast_with_no_seats_returns_zero(self, client):
        """When no seats are registered, seats_notified is 0 but status is ok."""
        resp = client.post(
            "/v1/sessions/broadcast-message",
            json={"content": "hello nobody", "role": "user"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["seats_notified"] == 0

    def test_event_shape_is_user_message(self, client):
        """The fanned event has type=user_message with correct fields."""
        from seats import get_seat_registry

        reg = get_seat_registry()
        s = _make_seat("sess", "seat-x")
        with reg._lock:
            reg._seats["sess"] = {"seat-x": s}

        client.post(
            "/v1/sessions/broadcast-message",
            json={"content": "test content", "input_type": "voice", "role": "user"},
        )

        events = _drain(s)
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "user_message"
        assert ev["content"] == "test content"
        assert ev["input_type"] == "voice"
        assert ev["role"] == "user"

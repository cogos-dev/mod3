"""Unit + integration tests for the ADR-082 Phase 1 session registry.

Covers:
  * Voice-pool allocation (greedy + preferred + collision flagging)
  * Device resolution with a stubbed sounddevice-shaped lookup
    — "system-default" live re-query, named-device match, fallback
  * Global serializer round-robin across sessions
  * SessionRegistry submit() auto-creating a "default" session for
    legacy callers
  * HTTP surface: /v1/sessions endpoints smoke-tested via FastAPI TestClient

Run with: ``.venv/bin/python -m pytest tests/test_session_registry.py -v``
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_registry import (  # noqa: E402
    DEFAULT_SESSION_ID,
    VOICE_POOL,
    GlobalSerializer,
    SessionRegistry,
    resolve_output_device,
)

# ---------------------------------------------------------------------------
# Voice-pool allocation
# ---------------------------------------------------------------------------


class TestVoiceAllocation:
    def test_greedy_allocation_walks_the_pool(self):
        reg = SessionRegistry()
        a = reg.register(session_id="s1", participant_id="a").session
        b = reg.register(session_id="s2", participant_id="b").session
        c = reg.register(session_id="s3", participant_id="c").session
        assert a.assigned_voice == VOICE_POOL[0]
        assert b.assigned_voice == VOICE_POOL[1]
        assert c.assigned_voice == VOICE_POOL[2]
        assert not a.voice_conflict
        assert not b.voice_conflict
        assert not c.voice_conflict

    def test_preferred_voice_honored(self):
        reg = SessionRegistry()
        result = reg.register(session_id="s1", participant_id="a", preferred_voice="bf_emma")
        assert result.session.assigned_voice == "bf_emma"
        assert not result.voice_conflict

    def test_preferred_voice_collision_flagged_but_assigned(self):
        reg = SessionRegistry()
        first = reg.register(session_id="s1", participant_id="a", preferred_voice="bm_lewis")
        second = reg.register(session_id="s2", participant_id="b", preferred_voice="bm_lewis")
        # Both get the voice (collision is a flag, not a veto — per ADR)
        assert first.session.assigned_voice == "bm_lewis"
        assert not first.voice_conflict
        assert second.session.assigned_voice == "bm_lewis"
        assert second.voice_conflict

    def test_deregister_returns_voice_to_pool(self):
        reg = SessionRegistry()
        reg.register(session_id="s1", participant_id="a")  # takes bm_lewis
        result = reg.deregister("s1")
        assert result["status"] == "ok"
        assert result["released_voice"] == VOICE_POOL[0]
        # Next registration picks up the released voice
        next_session = reg.register(session_id="s2", participant_id="b").session
        assert next_session.assigned_voice == VOICE_POOL[0]

    def test_out_of_pool_preferred_voice_is_added(self):
        """Using a voice the pool didn't know about still works."""
        reg = SessionRegistry()
        result = reg.register(session_id="s1", participant_id="a", preferred_voice="af_ember")
        assert result.session.assigned_voice == "af_ember"
        assert "af_ember" in reg.voice_pool()


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def _fake_device_list() -> list[dict[str, Any]]:
    """Synthetic device list — stable across calls."""
    return [
        {"name": "MacBook Pro Speakers", "max_output_channels": 2},
        {"name": "DisplayLink Monitor", "max_output_channels": 2},
        {"name": "Realtek USB2.0 Audio", "max_output_channels": 2},
        {"name": "Microphone", "max_output_channels": 0},
    ]


class TestDeviceResolution:
    def test_system_default_live_requery(self):
        """The default must be re-read every call — not cached."""
        current = {"idx": 0}

        def query() -> list[dict[str, Any]]:
            return _fake_device_list()

        def default_idx() -> int:
            return current["idx"]

        r1 = resolve_output_device("system-default", query_devices=query, default_output_index=default_idx)
        assert r1.index == 0
        assert r1.name == "MacBook Pro Speakers"
        assert not r1.fallback

        # User plugs in headphones; the OS default changes.
        current["idx"] = 2
        r2 = resolve_output_device("system-default", query_devices=query, default_output_index=default_idx)
        assert r2.index == 2
        assert r2.name == "Realtek USB2.0 Audio"

    def test_named_device_match(self):
        r = resolve_output_device(
            "Realtek",
            query_devices=_fake_device_list,
            default_output_index=lambda: 0,
        )
        assert r.index == 2
        assert r.name == "Realtek USB2.0 Audio"
        assert not r.fallback

    def test_named_device_missing_falls_back_to_default(self):
        r = resolve_output_device(
            "AirPods Pro",  # not plugged in
            query_devices=_fake_device_list,
            default_output_index=lambda: 1,
        )
        assert r.fallback is True
        assert r.index == 1
        assert r.name == "DisplayLink Monitor"
        assert "fell back" in r.reason.lower()

    def test_numeric_index_match(self):
        r = resolve_output_device(
            "2",
            query_devices=_fake_device_list,
            default_output_index=lambda: 0,
        )
        assert r.index == 2
        assert r.name == "Realtek USB2.0 Audio"

    def test_empty_preferred_treated_as_default(self):
        r = resolve_output_device(
            "",
            query_devices=_fake_device_list,
            default_output_index=lambda: 1,
        )
        assert r.index == 1

    def test_default_index_out_of_range_returns_implicit(self):
        r = resolve_output_device(
            "system-default",
            query_devices=_fake_device_list,
            default_output_index=lambda: 99,
        )
        assert r.index is None
        # Not a "fallback" in the device-fallback sense — OS just couldn't tell us
        assert "portaudio" in r.reason.lower() or "unknown" in r.reason.lower()


# ---------------------------------------------------------------------------
# Global serializer
# ---------------------------------------------------------------------------


class TestGlobalSerializer:
    def _build(self, policy: str = "round-robin") -> tuple[GlobalSerializer, list[tuple[str, Any]]]:
        """Return (serializer, log) where log records (session_id, payload)."""
        log: list[tuple[str, Any]] = []
        barrier = threading.Event()

        def dispatcher(session_id: str, payload: Any) -> None:
            log.append((session_id, payload))
            barrier.set()

        ser = GlobalSerializer(policy=policy, dispatcher=dispatcher)
        return ser, log

    def test_round_robin_interleaves_two_sessions(self):
        reg = SessionRegistry()
        ser = reg.serializer

        events: list[tuple[str, Any]] = []
        completed = threading.Event()
        counter = {"n": 0}

        def dispatcher(session_id: str, payload: Any) -> None:
            events.append((session_id, payload))
            counter["n"] += 1
            if counter["n"] >= 6:
                completed.set()

        ser.attach_dispatcher(dispatcher)
        reg.register(session_id="A", participant_id="a")
        reg.register(session_id="B", participant_id="b")

        # Submit 3 jobs to each before starting the dispatcher so we can
        # observe round-robin ordering instead of racing with a ticking queue.
        for i in range(3):
            reg.submit("A", {"job": f"A{i}"})
        for i in range(3):
            reg.submit("B", {"job": f"B{i}"})

        reg.start()
        assert completed.wait(timeout=2.0), f"serializer stalled: {events}"
        reg.stop()

        sessions = [sid for sid, _ in events]
        # Round-robin starting from A (first session with pending work):
        # the cursor walks A → B alternately.
        expected = ["A", "B", "A", "B", "A", "B"]
        assert sessions == expected, f"Expected round-robin, got {sessions}"

    def test_fifo_global_preserves_arrival_order(self):
        reg = SessionRegistry()
        reg.serializer.set_policy("fifo-global")
        events: list[tuple[str, Any]] = []
        done = threading.Event()

        def dispatcher(session_id: str, payload: Any) -> None:
            events.append((session_id, payload))
            if len(events) == 4:
                done.set()

        reg.serializer.attach_dispatcher(dispatcher)
        reg.register(session_id="A", participant_id="a")
        reg.register(session_id="B", participant_id="b")

        reg.submit("A", "a1")
        reg.submit("B", "b1")
        reg.submit("A", "a2")
        reg.submit("B", "b2")

        reg.start()
        assert done.wait(timeout=2.0)
        reg.stop()

        order = [payload for _, payload in events]
        assert order == ["a1", "b1", "a2", "b2"], f"FIFO broken: {order}"

    def test_priority_policy_drains_higher_priority_first(self):
        reg = SessionRegistry()
        reg.serializer.set_policy("priority")
        events: list[tuple[str, Any]] = []
        done = threading.Event()

        def dispatcher(session_id: str, payload: Any) -> None:
            events.append((session_id, payload))
            if len(events) == 3:
                done.set()

        reg.serializer.attach_dispatcher(dispatcher)
        reg.register(session_id="low", participant_id="low", priority=0)
        reg.register(session_id="hi", participant_id="hi", priority=10)

        reg.submit("low", "L1")
        reg.submit("low", "L2")
        reg.submit("hi", "H1")  # this should drain first under priority

        reg.start()
        assert done.wait(timeout=2.0)
        reg.stop()

        assert events[0] == ("hi", "H1"), f"Priority not honored: {events}"

    def test_submit_on_unregistered_session_raises(self):
        reg = SessionRegistry()
        with pytest.raises(KeyError):
            reg.submit("ghost", "payload", auto_create_default=False)

    def test_submit_without_session_id_auto_creates_default(self):
        reg = SessionRegistry()
        done = threading.Event()
        events: list[tuple[str, Any]] = []

        def dispatcher(session_id: str, payload: Any) -> None:
            events.append((session_id, payload))
            done.set()

        reg.serializer.attach_dispatcher(dispatcher)
        reg.submit(None, "legacy-call")
        reg.start()
        assert done.wait(timeout=1.0)
        reg.stop()

        # Auto-created default session
        assert events[0][0] == DEFAULT_SESSION_ID
        assert reg.get(DEFAULT_SESSION_ID) is not None

    def test_deregister_drops_queued_jobs(self):
        reg = SessionRegistry()
        reg.register(session_id="A", participant_id="a")
        reg.submit("A", "j1")
        reg.submit("A", "j2")
        result = reg.deregister("A")
        assert result["dropped_jobs"] == 2


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


class TestHTTPSessionEndpoints:
    """Smoke-test the HTTP endpoints via FastAPI TestClient.

    We use the live registry module singleton — tests register unique session
    ids with a 'pytest-' prefix and deregister in teardown so we do not leak.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app)

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        # Clean up any pytest- sessions that leaked.
        from session_registry import get_default_registry

        reg = get_default_registry()
        for s in list(reg.list()):
            if s.session_id.startswith("pytest-"):
                reg.deregister(s.session_id)

    def test_register_returns_session_channel(self, client):
        r = client.post(
            "/v1/sessions/register",
            json={
                "session_id": "pytest-s1",
                "participant_id": "pytest-cog",
                "participant_type": "agent",
                "preferred_output_device": "system-default",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == "pytest-s1"
        assert body["participant_id"] == "pytest-cog"
        assert body["assigned_voice"] in VOICE_POOL
        assert body["preferred_output_device"] == "system-default"
        # live-resolved device block present
        assert "output_device" in body
        assert "preferred" in body["output_device"]

    def test_list_includes_registered_session(self, client):
        client.post(
            "/v1/sessions/register",
            json={"session_id": "pytest-list-1", "participant_id": "pytest-sandy"},
        )
        r = client.get("/v1/sessions")
        assert r.status_code == 200
        body = r.json()
        ids = [s["session_id"] for s in body["sessions"]]
        assert "pytest-list-1" in ids
        assert "voice_pool" in body
        assert body["serializer"]["policy"] in ("round-robin", "priority", "fifo-global")

    def test_deregister_releases_voice(self, client):
        client.post(
            "/v1/sessions/register",
            json={
                "session_id": "pytest-dr-1",
                "participant_id": "pytest-cog",
                "preferred_voice": "bf_emma",
            },
        )
        r = client.post("/v1/sessions/pytest-dr-1/deregister")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["released_voice"] == "bf_emma"

    def test_deregister_unknown_returns_404(self, client):
        r = client.post("/v1/sessions/pytest-nonexistent/deregister")
        assert r.status_code == 404

    def test_get_single_session(self, client):
        client.post(
            "/v1/sessions/register",
            json={"session_id": "pytest-get-1", "participant_id": "pytest-user"},
        )
        r = client.get("/v1/sessions/pytest-get-1")
        assert r.status_code == 200
        assert r.json()["session_id"] == "pytest-get-1"

    def test_synthesize_rejects_unknown_session(self, client):
        r = client.post(
            "/v1/synthesize",
            json={
                "text": "hello",
                "session_id": "pytest-ghost",
            },
        )
        assert r.status_code == 404

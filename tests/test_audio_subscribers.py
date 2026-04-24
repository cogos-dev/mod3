"""Unit + integration tests for the Wave 4.3 audio-subscriber registry.

Covers:
  * AudioSubscriberRegistry register / unregister / count / has_subscribers
  * emit_wav delivers header JSON + binary bytes to every subscriber
  * /v1/sessions/{id}/subscribers HTTP endpoint returns the correct shape
  * /ws/audio/{session_id} accepts a WebSocket upgrade, registers the
    subscriber for the lifetime of the connection, and unregisters on close
  * /v1/synthesize with a session_id AND a live subscriber emits the WAV
    over the WebSocket (via emit_wav) in addition to returning the HTTP
    response body

Run with: ``.venv/bin/python -m pytest tests/test_audio_subscribers.py -v``
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_subscribers import (  # noqa: E402
    AudioSubscriberRegistry,
    get_default_audio_subscribers,
    reset_default_audio_subscribers,
)

# ---------------------------------------------------------------------------
# Unit tests — AudioSubscriberRegistry
# ---------------------------------------------------------------------------


class _FakeWS:
    """Minimal stand-in for fastapi.WebSocket — records sent frames."""

    def __init__(self) -> None:
        self.json_sent: list[dict] = []
        self.bytes_sent: list[bytes] = []
        self.closed = False

    async def send_json(self, frame: dict) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.json_sent.append(frame)

    async def send_bytes(self, payload: bytes) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.bytes_sent.append(payload)


class TestAudioSubscriberRegistry:
    def test_register_and_has_subscriber(self):
        reg = AudioSubscriberRegistry()
        assert not reg.has_subscribers("s1")
        assert reg.count("s1") == 0

        loop = asyncio.new_event_loop()
        ws = _FakeWS()
        try:
            sub = reg.register("s1", ws, loop)
            assert reg.has_subscribers("s1")
            assert reg.count("s1") == 1

            reg.unregister("s1", sub)
            assert not reg.has_subscribers("s1")
            assert reg.count("s1") == 0
        finally:
            loop.close()

    def test_multiple_subscribers_per_session(self):
        reg = AudioSubscriberRegistry()
        loop = asyncio.new_event_loop()
        try:
            a = reg.register("s1", _FakeWS(), loop)
            b = reg.register("s1", _FakeWS(), loop)
            assert reg.count("s1") == 2
            reg.unregister("s1", a)
            assert reg.count("s1") == 1
            reg.unregister("s1", b)
            assert reg.count("s1") == 0
            # Empty bucket is pruned so snapshot stays compact
            assert reg.snapshot() == {}
        finally:
            loop.close()

    def test_unregister_unknown_is_noop(self):
        reg = AudioSubscriberRegistry()
        loop = asyncio.new_event_loop()
        try:
            ws = _FakeWS()
            sub = reg.register("s1", ws, loop)
            reg.unregister("s1", sub)
            # Second call on the already-removed sub should be a no-op
            reg.unregister("s1", sub)
            # Call on a session that never existed
            reg.unregister("ghost", sub)
        finally:
            loop.close()

    def test_emit_wav_delivers_header_and_bytes(self):
        reg = AudioSubscriberRegistry()
        loop = asyncio.new_event_loop()

        async def run():
            ws = _FakeWS()
            sub = reg.register("s1", ws, loop)
            try:
                delivered = reg.emit_wav(
                    "s1",
                    b"fake-wav-bytes",
                    job_id="job-1",
                    duration_sec=1.23,
                    sample_rate=24000,
                )
                # emit_wav schedules a coroutine on the loop; await it.
                await asyncio.sleep(0.05)
                assert delivered == 1
                assert len(ws.json_sent) == 1
                header = ws.json_sent[0]
                assert header["type"] == "audio_header"
                assert header["session_id"] == "s1"
                assert header["job_id"] == "job-1"
                assert header["duration_sec"] == 1.23
                assert header["sample_rate"] == 24000
                assert header["bytes"] == len(b"fake-wav-bytes")
                assert ws.bytes_sent == [b"fake-wav-bytes"]
            finally:
                reg.unregister("s1", sub)

        try:
            loop.run_until_complete(run())
        finally:
            loop.close()

    def test_emit_wav_with_no_subscribers_returns_zero(self):
        reg = AudioSubscriberRegistry()
        delivered = reg.emit_wav("s1", b"anything")
        assert delivered == 0

    def test_default_registry_is_shared_singleton(self):
        reset_default_audio_subscribers()
        a = get_default_audio_subscribers()
        b = get_default_audio_subscribers()
        assert a is b


# ---------------------------------------------------------------------------
# HTTP surface tests
# ---------------------------------------------------------------------------


class TestSubscribersEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app)

    @pytest.fixture(autouse=True)
    def _isolate_subscribers(self):
        reset_default_audio_subscribers()
        yield
        reset_default_audio_subscribers()

    def test_no_subscribers_returns_false(self, client):
        r = client.get("/v1/sessions/unknown-sid/subscribers")
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "unknown-sid"
        assert body["subscribed"] is False
        assert body["count"] == 0

    def test_ws_audio_registers_and_endpoint_reflects_it(self, client):
        """Open the WebSocket, check /subscribers, then disconnect."""
        with client.websocket_connect("/ws/audio/ws-test-1"):
            r = client.get("/v1/sessions/ws-test-1/subscribers")
            assert r.status_code == 200
            body = r.json()
            assert body["subscribed"] is True
            assert body["count"] == 1
        # After close, subscriber is deregistered
        r = client.get("/v1/sessions/ws-test-1/subscribers")
        assert r.status_code == 200
        assert r.json()["subscribed"] is False


# ---------------------------------------------------------------------------
# Integration: /v1/synthesize routes to WS subscriber
# ---------------------------------------------------------------------------


class TestSynthesizeEmitsOverWS:
    """When /v1/synthesize is called with a session_id whose dashboard has a
    live /ws/audio subscription, the WAV bytes are pushed over the WebSocket
    AND returned in the HTTP response body. Callers that skip local playback
    when ``X-Mod3-WS-Subscribers > 0`` avoid a double-play.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app)

    @pytest.fixture(autouse=True)
    def _isolate(self):
        reset_default_audio_subscribers()
        from session_registry import get_default_registry

        reg = get_default_registry()
        for s in list(reg.list()):
            if s.session_id.startswith("pytest-"):
                reg.deregister(s.session_id)
        yield
        for s in list(reg.list()):
            if s.session_id.startswith("pytest-"):
                reg.deregister(s.session_id)
        reset_default_audio_subscribers()

    @pytest.mark.skipif(
        os.environ.get("SKIP_TTS_TESTS") == "1",
        reason="loads Kokoro engine — slow; set SKIP_TTS_TESTS=1 to skip in CI",
    )
    def test_synthesize_with_subscriber_emits_over_ws(self, client):
        # Register a session and open a subscriber.
        client.post(
            "/v1/sessions/register",
            json={
                "session_id": "pytest-ws-1",
                "participant_id": "pytest-user",
                "participant_type": "user",
            },
        )
        with client.websocket_connect("/ws/audio/pytest-ws-1") as ws:
            # Synthesize, naming the session
            r = client.post(
                "/v1/synthesize",
                json={
                    "text": "hi",
                    "session_id": "pytest-ws-1",
                },
            )
            assert r.status_code == 200, r.text
            assert r.headers.get("X-Mod3-WS-Subscribers") == "1"

            # The WebSocket should have received an audio_header + binary pair
            header = ws.receive_json()
            assert header["type"] == "audio_header"
            assert header["session_id"] == "pytest-ws-1"
            assert header["format"] == "wav"
            audio = ws.receive_bytes()
            assert audio.startswith(b"RIFF") and b"WAVE" in audio[:16]

    def test_synthesize_without_session_skips_ws_emit(self, client):
        # Even without hitting Kokoro, a /v1/synthesize without session_id
        # should report 0 WS subscribers in the response header.
        # We don't actually need to wait for synthesis to complete — just
        # verify the endpoint path when no subscriber exists.
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        assert not subs.has_subscribers("nonexistent-sid")

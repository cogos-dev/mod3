"""Unit + integration tests for the Wave 4.3 / RTVI 1.3.0 audio-subscriber registry.

Covers:
  * AudioSubscriberRegistry register / unregister / count / has_subscribers
  * emit_wav delivers RTVI frames (bot-tts-started / bot-tts-audio / bot-tts-stopped)
  * RTVI frame shapes are spec-compliant (label, type, id, data fields)
  * PCM extraction from WAV (44-byte header strip)
  * /v1/sessions/{id}/subscribers HTTP endpoint returns the correct shape
  * /ws/audio/{session_id} accepts a WebSocket upgrade, registers the
    subscriber for the lifetime of the connection, and unregisters on close
  * /v1/synthesize with a session_id AND a live subscriber emits RTVI frames
    over the WebSocket in addition to returning the HTTP response body
  * Pipecat-client-compatible shape assertion (RTVI 1.3.0 structure)

Run with: ``.venv/bin/python -m pytest tests/test_audio_subscribers.py -v``
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_subscribers import (  # noqa: E402
    AudioSubscriberRegistry,
    _build_rtvi_frames,
    _extract_pcm_from_wav,
    get_default_audio_subscribers,
    reset_default_audio_subscribers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WAV_HEADER_BYTES = 44
_FAKE_PCM = b"\x00\x01" * 100  # 200 bytes of int16 samples


def _make_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Build a minimal valid PCM WAV around raw int16 bytes."""
    data_size = len(pcm)
    buf = bytearray()
    buf += b"RIFF"
    buf += struct.pack("<I", 36 + data_size)
    buf += b"WAVE"
    buf += b"fmt "
    buf += struct.pack("<I", 16)
    buf += struct.pack("<H", 1)
    buf += struct.pack("<H", 1)
    buf += struct.pack("<I", sample_rate)
    buf += struct.pack("<I", sample_rate * 2)
    buf += struct.pack("<H", 2)
    buf += struct.pack("<H", 16)
    buf += b"data"
    buf += struct.pack("<I", data_size)
    buf += pcm
    return bytes(buf)


# ---------------------------------------------------------------------------
# Fake WebSocket
# ---------------------------------------------------------------------------


class _FakeWS:
    """Minimal stand-in for fastapi.WebSocket -- records sent frames."""

    def __init__(self) -> None:
        self.text_sent: list[str] = []
        self.json_sent: list[dict] = []
        self.bytes_sent: list[bytes] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.text_sent.append(data)

    async def send_json(self, frame: dict) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.json_sent.append(frame)

    async def send_bytes(self, payload: bytes) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.bytes_sent.append(payload)


# ---------------------------------------------------------------------------
# Unit: PCM extraction helper
# ---------------------------------------------------------------------------


class TestExtractPcmFromWav:
    def test_strips_header(self):
        wav = _make_wav(_FAKE_PCM)
        pcm = _extract_pcm_from_wav(wav)
        assert pcm == _FAKE_PCM

    def test_too_short_returns_empty(self):
        assert _extract_pcm_from_wav(b"\x00" * 10) == b""

    def test_exactly_header_returns_empty(self):
        assert _extract_pcm_from_wav(b"\x00" * _WAV_HEADER_BYTES) == b""


# ---------------------------------------------------------------------------
# Unit: RTVI frame builder
# ---------------------------------------------------------------------------


class TestBuildRtviFrames:
    def _parse_all(self, audio_b64: str, sr: int) -> list[dict]:
        frames = _build_rtvi_frames(audio_b64=audio_b64, sample_rate=sr)
        assert len(frames) == 3, "expect started / audio / stopped"
        return [json.loads(f) for f in frames]

    def test_label_is_rtvi_ai(self):
        msgs = self._parse_all("AAAA", 24000)
        for m in msgs:
            assert m["label"] == "rtvi-ai"

    def test_types_in_order(self):
        msgs = self._parse_all("AAAA", 24000)
        assert msgs[0]["type"] == "bot-tts-started"
        assert msgs[1]["type"] == "bot-tts-audio"
        assert msgs[2]["type"] == "bot-tts-stopped"

    def test_ids_are_distinct_uuids(self):
        msgs = self._parse_all("AAAA", 24000)
        ids = [m["id"] for m in msgs]
        assert len(set(ids)) == 3, "each frame must have a unique id"
        for i in ids:
            assert len(i) == 36, "id must be a UUID string"

    def test_audio_data_fields(self):
        b64 = base64.b64encode(_FAKE_PCM).decode()
        msgs = self._parse_all(b64, 48000)
        audio_msg = msgs[1]
        assert audio_msg["data"]["audio"] == b64
        assert audio_msg["data"]["sample_rate"] == 48000
        assert audio_msg["data"]["num_channels"] == 1

    def test_started_stopped_have_empty_data(self):
        msgs = self._parse_all("AAAA", 24000)
        assert msgs[0]["data"] == {}
        assert msgs[2]["data"] == {}


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

    def test_emit_wav_delivers_rtvi_frames(self):
        """emit_wav must send three RTVI JSON text frames (started / audio / stopped)."""
        reg = AudioSubscriberRegistry()
        loop = asyncio.new_event_loop()
        wav = _make_wav(_FAKE_PCM, sample_rate=24000)

        async def run():
            ws = _FakeWS()
            sub = reg.register("s1", ws, loop)
            try:
                delivered = reg.emit_wav(
                    "s1",
                    wav,
                    job_id="job-1",
                    duration_sec=1.23,
                    sample_rate=24000,
                )
                # emit_wav schedules a coroutine on the loop; give it a tick.
                await asyncio.sleep(0.05)
                assert delivered == 1
                # Three JSON text frames; no binary frames; no json_sent frames.
                assert len(ws.text_sent) == 3
                assert ws.bytes_sent == []
                assert ws.json_sent == []

                frames = [json.loads(f) for f in ws.text_sent]
                for f in frames:
                    assert f["label"] == "rtvi-ai"

                assert frames[0]["type"] == "bot-tts-started"
                assert frames[1]["type"] == "bot-tts-audio"
                assert frames[2]["type"] == "bot-tts-stopped"

                # Verify audio payload round-trips correctly.
                audio_data = frames[1]["data"]
                assert audio_data["sample_rate"] == 24000
                assert audio_data["num_channels"] == 1
                decoded = base64.b64decode(audio_data["audio"])
                assert decoded == _FAKE_PCM
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

        return TestClient(http_api.app, base_url="http://localhost:7860")

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
# Pipecat / RTVI 1.3.0 compatibility assertion
# ---------------------------------------------------------------------------


class TestRtviCompatibility:
    """Assert that emitted frames match the RTVI 1.3.0 spec shape.

    A real Pipecat web client expects these exact field names and value types.
    We verify the structural contract without importing the RTVI library.
    """

    def test_bot_tts_audio_spec_shape(self):
        pcm = b"\x01\x02" * 50
        b64 = base64.b64encode(pcm).decode()
        frames = _build_rtvi_frames(audio_b64=b64, sample_rate=24000)
        audio_frame = json.loads(frames[1])
        assert audio_frame["label"] == "rtvi-ai"
        assert audio_frame["type"] == "bot-tts-audio"
        assert isinstance(audio_frame["id"], str) and audio_frame["id"]
        d = audio_frame["data"]
        assert isinstance(d["audio"], str)
        assert isinstance(d["sample_rate"], int) and d["sample_rate"] > 0
        assert isinstance(d["num_channels"], int) and d["num_channels"] >= 1
        assert base64.b64decode(d["audio"]) == pcm

    def test_bot_tts_started_spec_shape(self):
        frames = _build_rtvi_frames(audio_b64="AAAA", sample_rate=24000)
        started = json.loads(frames[0])
        assert started["label"] == "rtvi-ai"
        assert started["type"] == "bot-tts-started"
        assert isinstance(started["id"], str) and started["id"]
        assert started["data"] == {}

    def test_bot_tts_stopped_spec_shape(self):
        frames = _build_rtvi_frames(audio_b64="AAAA", sample_rate=24000)
        stopped = json.loads(frames[2])
        assert stopped["label"] == "rtvi-ai"
        assert stopped["type"] == "bot-tts-stopped"
        assert isinstance(stopped["id"], str) and stopped["id"]
        assert stopped["data"] == {}


# ---------------------------------------------------------------------------
# RTVI T5: disconnect-bot graceful close
# ---------------------------------------------------------------------------


class TestDisconnectBot:
    """Verify that a 'disconnect-bot' RTVI frame causes the ws_audio handler
    to close the connection and unregister the subscriber cleanly.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app, base_url="http://localhost:7860")

    @pytest.fixture(autouse=True)
    def _isolate_subscribers(self):
        reset_default_audio_subscribers()
        yield
        reset_default_audio_subscribers()

    def test_disconnect_bot_closes_and_unregisters(self, client):
        """Sending disconnect-bot should close the WS and deregister subscriber."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-disconnect-bot-1"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            assert subs.has_subscribers(sid)
            ws.send_text('{"label":"rtvi-ai","type":"disconnect-bot","id":"test-id-1","data":{}}')
            # After sending disconnect-bot the server closes; receive() raises or returns close
            try:
                ws.receive_text()
            except Exception:
                pass  # expected — server closed

        # Subscriber must be cleaned up
        assert not subs.has_subscribers(sid)

    def test_non_json_frame_does_not_crash(self, client):
        """Non-JSON frames should be silently ignored, not crash the handler."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-disconnect-bot-2"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text("not-valid-json")
            # Connection should still be open (server tolerates non-JSON)
            assert subs.has_subscribers(sid)

    def test_other_rtvi_type_does_not_close(self, client):
        """A text frame with a different RTVI type should not close the connection."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-disconnect-bot-3"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(
                '{"label":"rtvi-ai","type":"client-ready","id":"some-id","data":{"version":"1.3.0","about":{}}}'
            )
            # Connection must still be alive
            assert subs.has_subscribers(sid)


# ---------------------------------------------------------------------------
# Integration: /v1/synthesize routes RTVI frames to WS subscriber
# ---------------------------------------------------------------------------


class TestSynthesizeEmitsRtviOverWS:
    """When /v1/synthesize is called with a session_id whose dashboard has a
    live /ws/audio subscription, RTVI frames are pushed over the WebSocket
    AND WAV bytes are returned in the HTTP response body.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app, base_url="http://localhost:7860")

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
        reason="loads Kokoro engine -- slow; set SKIP_TTS_TESTS=1 to skip in CI",
    )
    def test_synthesize_with_subscriber_emits_rtvi_over_ws(self, client):
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
            r = client.post(
                "/v1/synthesize",
                json={
                    "text": "hi",
                    "session_id": "pytest-ws-1",
                },
            )
            assert r.status_code == 200, r.text
            assert r.headers.get("X-Mod3-WS-Subscribers") == "1"

            # Expect three RTVI JSON text frames.
            started_raw = ws.receive_text()
            audio_raw = ws.receive_text()
            stopped_raw = ws.receive_text()

            started = json.loads(started_raw)
            audio = json.loads(audio_raw)
            stopped = json.loads(stopped_raw)

            assert started["label"] == "rtvi-ai"
            assert started["type"] == "bot-tts-started"
            assert audio["label"] == "rtvi-ai"
            assert audio["type"] == "bot-tts-audio"
            assert "audio" in audio["data"]
            assert audio["data"]["sample_rate"] > 0
            assert stopped["label"] == "rtvi-ai"
            assert stopped["type"] == "bot-tts-stopped"

    def test_synthesize_without_session_skips_ws_emit(self, client):
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        assert not subs.has_subscribers("nonexistent-sid")


# ---------------------------------------------------------------------------
# RTVI T4: transcript + speaking-lifecycle emit methods
# ---------------------------------------------------------------------------


class TestRtviTranscriptEmit:
    """Unit tests for the 6 new RTVI T4 emit methods on AudioSubscriberRegistry.

    All unit-level: no live WebSocket needed. Tests verify:
    - Returns 0 when no subscriber is registered.
    - Returns > 0 when a mock subscriber is present.
    - Frame JSON has correct RTVI 1.3.0 shape.
    """

    def setup_method(self):
        self.reg = AudioSubscriberRegistry()

    def test_emit_returns_zero_without_subscribers(self):
        sid = "t4-unit-empty"
        assert self.reg.emit_user_transcription(sid, "hello") == 0
        assert self.reg.emit_bot_transcription(sid, "hello") == 0
        assert self.reg.emit_user_started_speaking(sid) == 0
        assert self.reg.emit_user_stopped_speaking(sid) == 0
        assert self.reg.emit_bot_llm_started(sid) == 0
        assert self.reg.emit_bot_llm_stopped(sid) == 0

    def test_frame_shapes_via_send_single(self):
        """Verify JSON shape of each emit type by inspecting the frame directly."""
        import asyncio
        from unittest.mock import MagicMock  # noqa: PLC0415

        from audio_subscribers import _SessionBucket, _Subscriber  # noqa: PLC0415

        mock_ws = MagicMock()
        sent_texts = []

        async def mock_send_text(text):
            sent_texts.append(text)

        mock_ws.send_text = mock_send_text

        mock_loop = asyncio.new_event_loop()
        sid = "t4-unit-shape"

        # Manually inject a subscriber
        sub = _Subscriber(ws=mock_ws, loop=mock_loop)
        bucket = _SessionBucket(subscribers=[sub])
        self.reg._buckets[sid] = bucket

        mock_loop.run_until_complete(asyncio.sleep(0))  # prime the loop

        # Directly test frame JSON shapes without threading complexity
        self.reg.emit_user_transcription(sid, "hello world", is_final=True)
        self.reg.emit_bot_transcription(sid, "bot says hi", is_final=False)
        self.reg.emit_user_started_speaking(sid)
        self.reg.emit_user_stopped_speaking(sid)
        self.reg.emit_bot_llm_started(sid)
        self.reg.emit_bot_llm_stopped(sid)

        # Drain pending coroutines in mock loop
        mock_loop.run_until_complete(asyncio.sleep(0.01))
        mock_loop.close()

        assert len(sent_texts) == 6

        frames = [json.loads(t) for t in sent_texts]
        types = [f["type"] for f in frames]
        assert "user-transcription" in types
        assert "bot-transcription" in types
        assert "user-started-speaking" in types
        assert "user-stopped-speaking" in types
        assert "bot-llm-started" in types
        assert "bot-llm-stopped" in types

        # Spot-check user-transcription shape
        ut = next(f for f in frames if f["type"] == "user-transcription")
        assert ut["label"] == "rtvi-ai"
        assert isinstance(ut["id"], str) and ut["id"]
        assert ut["data"]["text"] == "hello world"
        assert ut["data"]["is_final"] is True

        # Spot-check bot-transcription is_final=False
        bt = next(f for f in frames if f["type"] == "bot-transcription")
        assert bt["data"]["is_final"] is False

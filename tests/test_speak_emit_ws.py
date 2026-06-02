"""Tests for emit_wav on the /v1/speak drain path.

Three required proofs:
  (a) speak with subscriber → frames on /ws/audio
  (b) speak without subscriber → no WS emit, local play intact
  (c) barge-in/stop still flushes correctly (bot-tts-stopped sent)

These tests exercise:
  1. The new AudioSubscriberRegistry streaming methods
     (emit_tts_started / emit_tts_audio_chunk / emit_tts_stopped)
  2. The _run_speech_job wiring that drives those methods

Run with:
    .venv/bin/python -m pytest tests/test_speak_emit_ws.py -v
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
import unittest.mock as mock
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_subscribers import (  # noqa: E402
    AudioSubscriberRegistry,
    get_default_audio_subscribers,
    reset_default_audio_subscribers,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 24000
_CHUNK_SAMPLES = 2400  # 0.1 s of 24 kHz mono float32


def _make_chunk_samples(n: int = _CHUNK_SAMPLES) -> np.ndarray:
    """Sine-wave float32 samples in [-1, 1]."""
    t = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


class _FakeWS:
    """Stand-in for fastapi.WebSocket — records sent frames."""

    def __init__(self) -> None:
        self.text_sent: list[str] = []
        self.closed = False

    async def send_text(self, data: str) -> None:
        if self.closed:
            raise RuntimeError("socket closed")
        self.text_sent.append(data)


def _run_coroutine_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_registry_with_subscriber(session_id: str) -> tuple[AudioSubscriberRegistry, _FakeWS, asyncio.AbstractEventLoop]:
    """Return a registry that has one live subscriber for session_id."""
    reg = AudioSubscriberRegistry()
    ws = _FakeWS()
    loop = asyncio.new_event_loop()
    reg.register(session_id, ws, loop)
    return reg, ws, loop


def _drain_loop(ws: _FakeWS, loop: asyncio.AbstractEventLoop, n: int) -> list[dict]:
    """Run the event loop until at least n frames have been sent, then stop.

    Returns the parsed frames.
    """
    async def _wait():
        deadline = 2.0  # seconds
        import time
        t0 = time.monotonic()
        while len(ws.text_sent) < n and (time.monotonic() - t0) < deadline:
            await asyncio.sleep(0.01)

    loop.run_until_complete(_wait())
    return [json.loads(f) for f in ws.text_sent]


# ---------------------------------------------------------------------------
# Unit tests: new emit_tts_* methods on AudioSubscriberRegistry
# ---------------------------------------------------------------------------


class TestEmitTtsStarted:
    def test_sends_bot_tts_started_frame(self):
        sid = "test-started"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        try:
            reg.emit_tts_started(sid)
            frames = _drain_loop(ws, loop, 1)
            assert len(frames) == 1
            f = frames[0]
            assert f["label"] == "rtvi-ai"
            assert f["type"] == "bot-tts-started"
            assert "id" in f
            assert f["data"] == {}
        finally:
            loop.close()

    def test_returns_delivered_count(self):
        sid = "test-started-count"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        try:
            n = reg.emit_tts_started(sid)
            assert n == 1
            _drain_loop(ws, loop, 1)  # let the coroutine run so no RuntimeWarning
        finally:
            loop.close()

    def test_returns_zero_without_subscriber(self):
        reg = AudioSubscriberRegistry()
        assert reg.emit_tts_started("no-sub") == 0


class TestEmitTtsAudioChunk:
    def test_sends_bot_tts_audio_frame(self):
        sid = "test-chunk"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        pcm = (np.zeros(100, dtype=np.float32) + 0.5).astype(np.int16).tobytes()
        try:
            reg.emit_tts_audio_chunk(sid, pcm, sample_rate=24000)
            frames = _drain_loop(ws, loop, 1)
            assert len(frames) == 1
            f = frames[0]
            assert f["type"] == "bot-tts-audio"
            assert f["data"]["sample_rate"] == 24000
            assert f["data"]["num_channels"] == 1
            # audio is base64 of the pcm bytes we passed
            decoded = base64.b64decode(f["data"]["audio"])
            assert decoded == pcm
        finally:
            loop.close()

    def test_default_sample_rate_24000(self):
        sid = "test-chunk-sr"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        pcm = b"\x00\x01" * 100
        try:
            reg.emit_tts_audio_chunk(sid, pcm)  # no sample_rate kwarg
            frames = _drain_loop(ws, loop, 1)
            assert frames[0]["data"]["sample_rate"] == 24000
        finally:
            loop.close()

    def test_returns_zero_without_subscriber(self):
        reg = AudioSubscriberRegistry()
        assert reg.emit_tts_audio_chunk("no-sub", b"\x00\x01" * 50) == 0


class TestEmitTtsStopped:
    def test_sends_bot_tts_stopped_frame(self):
        sid = "test-stopped"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        try:
            reg.emit_tts_stopped(sid)
            frames = _drain_loop(ws, loop, 1)
            f = frames[0]
            assert f["type"] == "bot-tts-stopped"
            assert f["data"] == {}
        finally:
            loop.close()

    def test_returns_zero_without_subscriber(self):
        reg = AudioSubscriberRegistry()
        assert reg.emit_tts_stopped("no-sub") == 0


# ---------------------------------------------------------------------------
# Integration: streaming triplet — started → N audio → stopped
# ---------------------------------------------------------------------------


class TestStreamingTriplet:
    """emit_tts_started + N emit_tts_audio_chunk + emit_tts_stopped produces
    a well-formed RTVI streaming sequence: one started, N audio, one stopped."""

    def _make_pcm(self) -> bytes:
        samples = _make_chunk_samples()
        return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def test_three_chunk_sequence(self):
        sid = "stream-test"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        n_chunks = 3
        try:
            reg.emit_tts_started(sid)
            for _ in range(n_chunks):
                reg.emit_tts_audio_chunk(sid, self._make_pcm(), sample_rate=_SAMPLE_RATE)
            reg.emit_tts_stopped(sid)

            frames = _drain_loop(ws, loop, 1 + n_chunks + 1)
            types = [f["type"] for f in frames]

            assert types[0] == "bot-tts-started"
            assert types[-1] == "bot-tts-stopped"
            assert all(t == "bot-tts-audio" for t in types[1:-1])
            assert len([t for t in types if t == "bot-tts-audio"]) == n_chunks
        finally:
            loop.close()

    def test_ids_are_unique(self):
        sid = "stream-ids"
        reg, ws, loop = _make_registry_with_subscriber(sid)
        try:
            reg.emit_tts_started(sid)
            reg.emit_tts_audio_chunk(sid, self._make_pcm())
            reg.emit_tts_stopped(sid)
            frames = _drain_loop(ws, loop, 3)
            ids = [f["id"] for f in frames]
            assert len(set(ids)) == 3, "all frame ids must be distinct UUIDs"
        finally:
            loop.close()

    def test_no_subscriber_emits_nothing(self):
        reg = AudioSubscriberRegistry()
        sid = "no-sub"
        assert reg.emit_tts_started(sid) == 0
        assert reg.emit_tts_audio_chunk(sid, b"\x00" * 100) == 0
        assert reg.emit_tts_stopped(sid) == 0


# ---------------------------------------------------------------------------
# Integration: _run_speech_job wiring via mock
# ---------------------------------------------------------------------------


class _FakeAudioChunk:
    """Mimics engine.AudioChunk (samples, sample_rate, metadata)."""

    def __init__(self, n: int = _CHUNK_SAMPLES, sr: int = _SAMPLE_RATE, is_final: bool = False):
        self.samples = _make_chunk_samples(n)
        self.sample_rate = sr
        self.metadata = {"is_final": is_final}


def _make_speak_entry(session_id: str | None = None) -> dict:
    return {
        "job_id": "job-test",
        "text": "hello world",
        "voice": "test-voice",
        "stream": True,
        "streaming_interval": 0.1,
        "speed": 1.0,
        "emotion": 0.5,
        "ref_audio": None,
        "session_id": session_id,
    }


def _build_run_speech_job_mocks(
    chunks: list[_FakeAudioChunk],
    audio_subs: AudioSubscriberRegistry,
    monkeypatch,
) -> tuple[dict, dict]:
    """Patch all server-level globals and return (jobs_dict, entry)."""
    import server  # noqa: PLC0415

    jobs: dict = {}
    entry = _make_speak_entry(session_id="sess-abc" if audio_subs.has_subscribers("sess-abc") else None)
    jobs[entry["job_id"]] = {"status": "queued"}

    # Patch module globals that _run_speech_job reads/writes
    monkeypatch.setattr(server, "_jobs", jobs, raising=False)
    monkeypatch.setattr(server, "_last_metrics", None, raising=False)
    monkeypatch.setattr(server, "_current_player", None, raising=False)
    monkeypatch.setattr(server, "_current_player_lock", threading.Lock(), raising=False)

    # Patch engine loader
    fake_engine = mock.MagicMock()
    fake_engine.generate_audio.return_value = iter(chunks)
    fake_engine.get_model.return_value = mock.MagicMock(sample_rate=_SAMPLE_RATE)
    monkeypatch.setattr(server, "_engine_module", lambda: fake_engine, raising=False)

    # Patch AdaptivePlayer
    class _FakePlayer:
        def __init__(self, **kw): pass
        def queue_audio(self, samples, chunk_meta=None): pass
        def flush(self): pass
        def mark_done(self): pass
        def get_progress(self): return (0.0, 1.0)
        def wait(self, timeout=None):
            m = mock.MagicMock()
            m.to_dict.return_value = {}
            return m

    monkeypatch.setattr(server, "_adaptive_player_class", lambda: _FakePlayer, raising=False)

    # Patch device resolution
    monkeypatch.setattr(server, "_resolve_device_for_entry", lambda e: (None, None), raising=False)

    # Patch voice resolution
    monkeypatch.setattr(server, "_resolve_voice_via_bus", lambda v: ("test-engine", v), raising=False)

    # Patch speaking lock (always acquired, never lost)
    monkeypatch.setattr(server, "_acquire_speaking_lock", lambda jid, txt: True, raising=False)
    monkeypatch.setattr(server, "_i_own_speaking_lock", lambda jid: True, raising=False)
    monkeypatch.setattr(server, "_release_speaking_lock", lambda jid: None, raising=False)

    # Patch pipeline state
    fake_pipeline = mock.MagicMock()
    monkeypatch.setattr(server, "pipeline_state", fake_pipeline, raising=False)

    # Patch bus state
    monkeypatch.setattr(server, "_set_bus_voice_state", mock.MagicMock(), raising=False)

    # Patch audio_subscribers module to use our controlled registry
    monkeypatch.setattr(
        "audio_subscribers._default_registry",
        audio_subs,
    )

    return jobs, entry


class TestRunSpeechJobWsEmit:
    """(a) speak with subscriber → frames on /ws/audio"""

    def test_speak_with_subscriber_emits_rtvi_frames(self, monkeypatch):
        reset_default_audio_subscribers()
        sid = "sess-abc"
        reg, ws, loop = _make_registry_with_subscriber(sid)

        chunks = [_FakeAudioChunk() for _ in range(2)]
        jobs, entry = _build_run_speech_job_mocks(chunks, reg, monkeypatch)
        entry["session_id"] = sid

        import server  # noqa: PLC0415
        server._run_speech_job(entry)

        # Drain the event loop to pick up all scheduled coroutines
        frames = _drain_loop(ws, loop, 4)  # started + 2 audio + stopped
        types = [f["type"] for f in frames]

        assert types[0] == "bot-tts-started"
        assert types[-1] == "bot-tts-stopped"
        assert types.count("bot-tts-audio") == 2
        loop.close()

    def test_speak_audio_frame_contains_valid_pcm(self, monkeypatch):
        reset_default_audio_subscribers()
        sid = "sess-abc"
        reg, ws, loop = _make_registry_with_subscriber(sid)

        chunks = [_FakeAudioChunk()]
        jobs, entry = _build_run_speech_job_mocks(chunks, reg, monkeypatch)
        entry["session_id"] = sid

        import server  # noqa: PLC0415
        server._run_speech_job(entry)

        frames = _drain_loop(ws, loop, 3)
        audio_frame = next(f for f in frames if f["type"] == "bot-tts-audio")

        decoded = base64.b64decode(audio_frame["data"]["audio"])
        # Must be non-empty int16 PCM bytes (even number of bytes)
        assert len(decoded) > 0
        assert len(decoded) % 2 == 0
        assert audio_frame["data"]["sample_rate"] == _SAMPLE_RATE
        assert audio_frame["data"]["num_channels"] == 1
        loop.close()


class TestRunSpeechJobNoSubscriber:
    """(b) speak without subscriber → no WS emit, local play intact"""

    def test_speak_without_subscriber_no_ws_emit(self, monkeypatch):
        reset_default_audio_subscribers()
        # Empty registry — no subscribers
        reg = AudioSubscriberRegistry()

        chunks = [_FakeAudioChunk() for _ in range(2)]
        jobs, entry = _build_run_speech_job_mocks(chunks, reg, monkeypatch)
        entry["session_id"] = "sess-abc"  # session key, but no sub registered

        # Track queue_audio calls to verify local play still happens
        player_calls = []

        class _TrackingPlayer:
            def __init__(self, **kw): pass
            def queue_audio(self, samples, chunk_meta=None):
                player_calls.append(len(samples))
            def flush(self): pass
            def mark_done(self): pass
            def get_progress(self): return (0.0, 1.0)
            def wait(self, timeout=None):
                m = mock.MagicMock()
                m.to_dict.return_value = {}
                return m

        import server  # noqa: PLC0415
        monkeypatch.setattr(server, "_adaptive_player_class", lambda: _TrackingPlayer, raising=False)

        server._run_speech_job(entry)

        # No frames emitted to any WS
        assert reg.snapshot() == {}  # no subscribers ever

        # Local player received all chunks
        assert len(player_calls) == 2, "local play must still happen without WS subscriber"


class TestRunSpeechJobBargein:
    """(c) barge-in/stop still flushes correctly — bot-tts-stopped sent"""

    def test_stopped_sent_on_bargein_break(self, monkeypatch):
        """When loop breaks early (lock lost = simulated barge-in), stopped fires."""
        reset_default_audio_subscribers()
        sid = "sess-abc"
        reg, ws, loop = _make_registry_with_subscriber(sid)

        # Three chunks, but speaking lock is "lost" after the first
        chunks = [_FakeAudioChunk() for _ in range(3)]

        lock_call_count = [0]

        def _i_own(jid):
            lock_call_count[0] += 1
            return lock_call_count[0] <= 1  # own on first check, lose on second

        jobs, entry = _build_run_speech_job_mocks(chunks, reg, monkeypatch)
        entry["session_id"] = sid

        import server  # noqa: PLC0415
        monkeypatch.setattr(server, "_i_own_speaking_lock", _i_own, raising=False)

        server._run_speech_job(entry)

        frames = _drain_loop(ws, loop, 3)  # started + 1 audio + stopped (break after 1st)
        types = [f["type"] for f in frames]

        # Must have started and stopped — even if interrupted mid-stream
        assert "bot-tts-started" in types
        assert "bot-tts-stopped" in types
        loop.close()

    def test_stopped_sent_on_synthesis_exception(self, monkeypatch):
        """If synthesis raises, bot-tts-stopped must still be sent (via finally)."""
        reset_default_audio_subscribers()
        sid = "sess-abc"
        reg, ws, loop = _make_registry_with_subscriber(sid)

        # First chunk succeeds, then the generator raises
        def _failing_gen(*a, **kw):
            yield _FakeAudioChunk()
            raise RuntimeError("synthesis error")

        chunks_dummy = [_FakeAudioChunk()]  # not used, overriding generate_audio
        jobs, entry = _build_run_speech_job_mocks(chunks_dummy, reg, monkeypatch)
        entry["session_id"] = sid

        import server  # noqa: PLC0415
        fake_engine = mock.MagicMock()
        fake_engine.generate_audio.side_effect = None
        fake_engine.generate_audio = _failing_gen
        fake_engine.get_model.return_value = mock.MagicMock(sample_rate=_SAMPLE_RATE)
        monkeypatch.setattr(server, "_engine_module", lambda: fake_engine, raising=False)

        server._run_speech_job(entry)

        frames = _drain_loop(ws, loop, 2)  # at least started + stopped
        types = [f["type"] for f in frames]

        assert "bot-tts-started" in types, "started must fire before exception"
        assert "bot-tts-stopped" in types, "stopped must fire via finally after exception"
        loop.close()

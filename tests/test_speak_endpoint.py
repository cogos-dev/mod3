"""Tests for POST /v1/speak — queue-aware HTTP speak endpoint.

Covers:
  - queue_position=0 on first call (playing immediately)
  - queue_position=1 on second concurrent call (queued behind first)
  - empty text rejected with 400
  - schema defaults and field passthrough

The tests mock _start_speech from server.py so no TTS engines or audio
hardware are required.

Run: python3 -m pytest tests/test_speak_endpoint.py -v
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

# Ensure the project root is on the path so imports resolve when running
# standalone (python3 -m pytest tests/test_speak_endpoint.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Schema unit tests (no HTTP layer)
# ---------------------------------------------------------------------------


class TestSpeakRequestSchema:
    def test_defaults(self):
        from schemas.http.synthesize import SpeakRequest

        req = SpeakRequest(text="hello")
        assert req.voice == "bm_lewis"
        assert req.stream is True
        assert req.speed == 1.25
        assert req.emotion == 0.5
        assert req.session_id == ""
        assert req.ref_audio == ""

    def test_custom_fields_accepted(self):
        from schemas.http.synthesize import SpeakRequest

        req = SpeakRequest(
            text="hello",
            voice="af_bella",
            speed=1.5,
            emotion=0.8,
            session_id="cs-abc",
            ref_audio="/tmp/ref.wav",
        )
        assert req.voice == "af_bella"
        assert req.speed == 1.5
        assert req.emotion == 0.8
        assert req.session_id == "cs-abc"
        assert req.ref_audio == "/tmp/ref.wav"

    def test_exported_from_package(self):
        from schemas.http import SpeakRequest

        assert SpeakRequest is not None


# ---------------------------------------------------------------------------
# HTTP endpoint tests via FastAPI TestClient with mocked _start_speech
# ---------------------------------------------------------------------------


class TestSpeakEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app, base_url="http://localhost:7860")

    def test_first_call_returns_queue_position_zero(self, client):
        """First enqueued job returns queue_position=0 and status=speaking."""
        import server

        original = server._start_speech
        call_count = [0]

        def fake_start_speech(
            text, voice, stream=True, speed=1.0, emotion=0.5, session_id=None, ref_audio=None, **kwargs
        ):
            n = call_count[0]
            call_count[0] += 1
            job_id = f"job-test-{n:04d}"
            position = 0  # first call, no queue
            return job_id, position

        server._start_speech = fake_start_speech
        try:
            r = client.post("/v1/speak", json={"text": "hello world"})
        finally:
            server._start_speech = original

        assert r.status_code == 200, r.text
        body = r.json()
        assert "job_id" in body
        assert body["queue_position"] == 0
        assert body["status"] == "speaking"

    def test_second_call_returns_queue_position_one(self, client):
        """Second concurrent call returns queue_position=1 and status=queued."""
        import server

        original = server._start_speech
        call_count = [0]

        def fake_start_speech(
            text, voice, stream=True, speed=1.0, emotion=0.5, session_id=None, ref_audio=None, **kwargs
        ):
            n = call_count[0]
            call_count[0] += 1
            job_id = f"job-test-{n:04d}"
            position = n  # 0 for first, 1 for second, etc.
            return job_id, position

        server._start_speech = fake_start_speech
        try:
            r1 = client.post("/v1/speak", json={"text": "first"})
            r2 = client.post("/v1/speak", json={"text": "second"})
        finally:
            server._start_speech = original

        assert r1.status_code == 200, r1.text
        assert r2.status_code == 200, r2.text

        body1 = r1.json()
        body2 = r2.json()

        assert body1["queue_position"] == 0
        assert body1["status"] == "speaking"

        assert body2["queue_position"] == 1
        assert body2["status"] == "queued"

    def test_empty_text_rejected_with_400(self, client):
        """Empty or whitespace-only text is rejected before hitting the queue."""
        r = client.post("/v1/speak", json={"text": "   "})
        assert r.status_code == 400, r.text
        body = r.json()
        assert "error" in body
        assert "text" in body["error"].lower()

    def test_completely_empty_text_rejected(self, client):
        """Zero-length text is rejected with 400."""
        r = client.post("/v1/speak", json={"text": ""})
        assert r.status_code == 400, r.text

    def test_response_shape(self, client):
        """Response always contains job_id, queue_position, and status."""
        import server

        original = server._start_speech

        def fake_start(text, voice, **kwargs):
            return "job-shape-test", 0

        server._start_speech = fake_start
        try:
            r = client.post("/v1/speak", json={"text": "shape test"})
        finally:
            server._start_speech = original

        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) >= {"job_id", "queue_position", "status"}
        assert isinstance(body["job_id"], str)
        assert isinstance(body["queue_position"], int)
        assert body["status"] in ("speaking", "queued")

    def test_voice_param_forwarded(self, client):
        """Voice parameter is forwarded to _start_speech."""
        import server

        original = server._start_speech
        received = {}

        def fake_start(text, voice, **kwargs):
            received["text"] = text
            received["voice"] = voice
            return "job-voice-test", 0

        server._start_speech = fake_start
        try:
            r = client.post("/v1/speak", json={"text": "hi", "voice": "af_bella"})
        finally:
            server._start_speech = original

        assert r.status_code == 200
        assert received["voice"] == "af_bella"

    def test_session_id_forwarded(self, client):
        """Non-empty session_id is forwarded as a non-None value."""
        import server

        original = server._start_speech
        received = {}

        def fake_start(text, voice, stream=True, speed=1.0, emotion=0.5, session_id=None, ref_audio=None, **kwargs):
            received["session_id"] = session_id
            return "job-sess-test", 0

        server._start_speech = fake_start
        try:
            r = client.post("/v1/speak", json={"text": "hi", "session_id": "cs-test"})
        finally:
            server._start_speech = original

        assert r.status_code == 200
        assert received["session_id"] == "cs-test"

    def test_empty_session_id_forwarded_as_none(self, client):
        """Empty session_id string is converted to None before forwarding."""
        import server

        original = server._start_speech
        received = {}

        def fake_start(text, voice, stream=True, speed=1.0, emotion=0.5, session_id=None, ref_audio=None, **kwargs):
            received["session_id"] = session_id
            return "job-no-sess", 0

        server._start_speech = fake_start
        try:
            r = client.post("/v1/speak", json={"text": "hi", "session_id": ""})
        finally:
            server._start_speech = original

        assert r.status_code == 200
        assert received["session_id"] is None


# ---------------------------------------------------------------------------
# Queue position semantics — no HTTP layer, direct _start_speech calls
# ---------------------------------------------------------------------------


class TestQueuePositionSemantics:
    """Verify queue_position=0 on first enqueue, position=1 on second enqueue.

    These tests call server._start_speech directly (with a patched queue) to
    validate the semantics without HTTP overhead. The HTTP endpoint is just a
    thin wrapper — if _start_speech returns the right positions, the endpoint
    will too (confirmed by TestSpeakEndpoint above).
    """

    def test_first_enqueue_gets_position_zero(self):
        """First enqueued job returns position 0 (plays immediately)."""
        from server import SpeechQueue

        q = SpeechQueue()
        q._draining = True  # prevent auto-drain during test
        pos = q.enqueue("job-q-first", {"text": "first", "voice": "bm_lewis", "stream": True})
        assert pos == 0, f"Expected position 0, got {pos}"

    def test_second_enqueue_gets_position_one(self):
        """Second enqueued job returns position 1 (queued behind first)."""
        from server import SpeechQueue

        q = SpeechQueue()
        q._draining = True
        q.enqueue("job-q-0", {"text": "first", "voice": "bm_lewis", "stream": True})
        pos = q.enqueue("job-q-1", {"text": "second", "voice": "bm_lewis", "stream": True})
        assert pos == 1, f"Expected position 1, got {pos}"

    def test_concurrent_enqueue_produces_distinct_positions(self):
        """Concurrent enqueues get distinct positions — no two calls get the same slot."""
        from server import SpeechQueue

        q = SpeechQueue()
        q._draining = True
        n = 3
        positions = []
        barrier = threading.Barrier(n)

        def enqueue_at_barrier(i):
            barrier.wait()  # all goroutines start at the same moment
            pos = q.enqueue(f"job-concurrent-{i}", {"text": f"msg {i}", "voice": "bm_lewis", "stream": True})
            positions.append(pos)

        threads = [threading.Thread(target=enqueue_at_barrier, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(positions) == n
        assert sorted(positions) == list(range(n)), (
            f"Expected positions {{0,1,2}}, got {sorted(positions)} — "
            "concurrent enqueues did NOT produce distinct slots"
        )

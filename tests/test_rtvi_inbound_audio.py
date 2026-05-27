"""Tests for RTVI 1.3.0 inbound raw-audio routing in /ws/audio/{session_id}.

Covers (B+ workstream T3):
  * raw-audio JSON frame is accepted and does not crash the handler
  * raw-audio-batch JSON frame is accepted (same handler path)
  * Non-JSON frame is tolerated (no crash)
  * user-started-speaking and user-stopped-speaking frames are accepted silently
  * Handler stays alive after receiving raw-audio frames

Note: full STT integration is not tested here — that requires a live VAD/Whisper
stack. These tests verify the dispatch path (frame parsing, channel routing) and
that the handler stays alive. Functional STT output is covered by the inbound.py
unit tests and E2E integration (T6).

Run with: ``.venv/bin/python -m pytest tests/test_rtvi_inbound_audio.py -v``
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_audio_frame(*, duration_samples: int = 1600, sample_rate: int = 16000) -> str:
    """Build a minimal RTVI raw-audio JSON frame with synthetic int16 PCM."""
    pcm = np.zeros(duration_samples, dtype=np.int16)
    audio_b64 = base64.b64encode(pcm.tobytes()).decode()
    return json.dumps(
        {
            "label": "rtvi-ai",
            "type": "raw-audio",
            "id": "test-raw-audio-id",
            "data": {
                "audio": audio_b64,
                "sample_rate": sample_rate,
                "num_channels": 1,
            },
        }
    )


def _make_vad_frame(event_type: str) -> str:
    """Build a user-started/stopped-speaking RTVI frame."""
    return json.dumps(
        {
            "label": "rtvi-ai",
            "type": event_type,
            "id": f"test-{event_type}-id",
            "data": {},
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture(autouse=True)
def _isolate_subscribers():
    from audio_subscribers import reset_default_audio_subscribers

    reset_default_audio_subscribers()
    yield
    reset_default_audio_subscribers()


# ---------------------------------------------------------------------------
# Inbound audio routing tests
# ---------------------------------------------------------------------------


class TestRawAudioFrame:
    def test_raw_audio_frame_does_not_crash_handler(self, client):
        """Sending a raw-audio frame should be silently processed; connection stays open."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-raw-audio-1"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(_make_raw_audio_frame())
            # Connection must remain alive
            assert subs.has_subscribers(sid)

    def test_raw_audio_batch_frame_does_not_crash_handler(self, client):
        """raw-audio-batch is handled identically to raw-audio."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-raw-audio-batch-1"

        frame = _make_raw_audio_frame()
        parsed = json.loads(frame)
        parsed["type"] = "raw-audio-batch"
        batch_frame = json.dumps(parsed)

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(batch_frame)
            assert subs.has_subscribers(sid)

    def test_multiple_raw_audio_frames_do_not_crash(self, client):
        """Sending several audio frames in sequence is stable."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-raw-audio-multi"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            for _ in range(3):
                ws.send_text(_make_raw_audio_frame())
            assert subs.has_subscribers(sid)

    def test_empty_audio_field_is_tolerated(self, client):
        """A raw-audio frame with empty audio field should not crash the handler."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-empty-audio"

        frame = json.dumps(
            {
                "label": "rtvi-ai",
                "type": "raw-audio",
                "id": "empty-audio-id",
                "data": {"audio": "", "sample_rate": 16000, "num_channels": 1},
            }
        )

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(frame)
            assert subs.has_subscribers(sid)


class TestVadSignals:
    def test_user_started_speaking_is_accepted(self, client):
        """user-started-speaking frame should be accepted without crashing."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-vad-started"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(_make_vad_frame("user-started-speaking"))
            assert subs.has_subscribers(sid)

    def test_user_stopped_speaking_is_accepted(self, client):
        """user-stopped-speaking frame should be accepted without crashing."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-vad-stopped"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(_make_vad_frame("user-stopped-speaking"))
            assert subs.has_subscribers(sid)


class TestInboundTolerance:
    def test_non_json_text_frame_is_ignored(self, client):
        """Non-JSON text frames must be tolerated; connection stays open."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-t3-nojson"

        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text("this is definitely not json")
            assert subs.has_subscribers(sid)

    def test_missing_data_field_is_tolerated(self, client):
        """raw-audio frame missing data should not crash."""
        from audio_subscribers import get_default_audio_subscribers

        subs = get_default_audio_subscribers()
        sid = "test-t3-nodata"

        frame = json.dumps({"label": "rtvi-ai", "type": "raw-audio", "id": "no-data-id"})
        with client.websocket_connect(f"/ws/audio/{sid}") as ws:
            ws.send_text(frame)
            assert subs.has_subscribers(sid)

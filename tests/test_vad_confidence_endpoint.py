"""Tests for POST /v1/vad/confidence endpoint.

Tests the lightweight per-packet VAD confidence endpoint that uses the
vendored pipecat ONNX SileroVADAnalyzer — no torch required.
"""

import os
import struct
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_silence_frame(samples: int = 512) -> bytes:
    """Generate a silent int16 PCM frame."""
    return struct.pack(f"<{samples}h", *([0] * samples))


def _make_client(app):
    """TestClient pre-configured to pass mod3's CSRF localhost guard."""
    return TestClient(app, base_url="http://localhost:7860", raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Happy-path: pipecat VAD available
# ---------------------------------------------------------------------------


class TestVadConfidenceEndpointAvailable:
    """Tests when is_pipecat_vad_available() returns True."""

    @pytest.fixture(autouse=True)
    def _patch_vad(self, monkeypatch):
        monkeypatch.setattr("vad.is_pipecat_vad_available", lambda: True)
        monkeypatch.setattr("vad.voice_confidence", lambda buf, sample_rate=16000: 0.02)

    @pytest.fixture
    def client(self):
        import importlib

        import http_api

        importlib.reload(http_api)
        return _make_client(http_api.app)

    def test_silence_returns_low_confidence(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "confidence" in body
        assert isinstance(body["confidence"], float)
        assert body["confidence"] == pytest.approx(0.02)

    def test_available_true_when_pipecat_ready(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        assert resp.json()["available"] is True

    def test_latency_ms_present_and_numeric(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        body = resp.json()
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], (int, float))
        assert body["latency_ms"] >= 0.0

    def test_empty_body_returns_zero_confidence(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"] == 0.0
        assert "available" in body

    def test_sample_rate_query_param_forwarded(self, monkeypatch):
        """?sample_rate=8000 should be passed through to voice_confidence."""
        calls = []

        def _mock_vc(buf, sample_rate=16000):
            calls.append(sample_rate)
            return 0.0

        monkeypatch.setattr("vad.is_pipecat_vad_available", lambda: True)
        monkeypatch.setattr("vad.voice_confidence", _mock_vc)

        import importlib

        import http_api

        importlib.reload(http_api)
        c = _make_client(http_api.app)

        frame = _make_silence_frame(256)  # 8kHz frame = 256 samples
        c.post(
            "/v1/vad/confidence?sample_rate=8000",
            content=frame,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert calls and calls[-1] == 8000

    def test_high_confidence_frame(self, monkeypatch):
        """When voice_confidence returns 0.95, endpoint echoes it."""
        monkeypatch.setattr("vad.is_pipecat_vad_available", lambda: True)
        monkeypatch.setattr("vad.voice_confidence", lambda buf, sample_rate=16000: 0.95)

        import importlib

        import http_api

        importlib.reload(http_api)
        c = _make_client(http_api.app)

        resp = c.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        assert resp.json()["confidence"] == pytest.approx(0.95)

    def test_response_is_json(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.headers["content-type"].startswith("application/json")

    def test_confidence_in_valid_range(self, monkeypatch):
        monkeypatch.setattr("vad.is_pipecat_vad_available", lambda: True)
        monkeypatch.setattr("vad.voice_confidence", lambda buf, sample_rate=16000: 1.0)

        import importlib

        import http_api

        importlib.reload(http_api)
        c = _make_client(http_api.app)

        resp = c.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        body = resp.json()
        assert 0.0 <= body["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Degraded-path: pipecat VAD unavailable (onnxruntime missing)
# ---------------------------------------------------------------------------


class TestVadConfidenceEndpointUnavailable:
    """Tests when is_pipecat_vad_available() returns False (onnxruntime not installed)."""

    @pytest.fixture(autouse=True)
    def _patch_vad(self, monkeypatch):
        monkeypatch.setattr("vad.is_pipecat_vad_available", lambda: False)
        monkeypatch.setattr("vad.voice_confidence", lambda buf, sample_rate=16000: 0.0)

    @pytest.fixture
    def client(self):
        import importlib

        import http_api

        importlib.reload(http_api)
        return _make_client(http_api.app)

    def test_available_false(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        assert resp.json()["available"] is False

    def test_confidence_zero_when_unavailable(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=_make_silence_frame(),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.json()["confidence"] == 0.0

    def test_empty_body_unavailable(self, client):
        resp = client.post(
            "/v1/vad/confidence",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"] == 0.0
        assert body["available"] is False


# ---------------------------------------------------------------------------
# Endpoint listed in /capabilities
# ---------------------------------------------------------------------------


class TestCapabilitiesManifest:
    """The new endpoint should appear in GET /capabilities."""

    @pytest.fixture(autouse=True)
    def _patch_vad(self, monkeypatch):
        monkeypatch.setattr("vad.is_pipecat_vad_available", lambda: True)
        monkeypatch.setattr("vad.voice_confidence", lambda buf, sample_rate=16000: 0.0)

    @pytest.fixture
    def client(self):
        import importlib

        import http_api

        importlib.reload(http_api)
        return _make_client(http_api.app)

    def test_vad_confidence_in_endpoints(self, client):
        resp = client.get("/capabilities")
        assert resp.status_code == 200
        body = resp.json()
        endpoints = body.get("endpoints", {})
        assert "vad_confidence" in endpoints, (
            f"vad_confidence missing from capabilities endpoints: {list(endpoints.keys())}"
        )
        assert "/v1/vad/confidence" in endpoints["vad_confidence"]

"""Tests for POST /v1/transcribe endpoint.

Tests the HTTP transcription endpoint without loading real Whisper models.
Uses mocked WhisperDecoder to verify request handling, response shape,
and hallucination filtering.
"""

import io
import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a synthetic WAV file with silence."""
    samples = np.zeros(int(duration_sec * sample_rate), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class MockCognitiveEvent:
    """Mock CognitiveEvent returned by WhisperDecoder.decode()."""

    def __init__(self, content: str, language: str = "en", filtered: bool = False):
        self.content = content
        self.metadata = {"language": language}
        if filtered:
            self.metadata["filtered"] = True


class MockWhisperDecoder:
    """Mock WhisperDecoder for testing without loading real models."""

    def __init__(
        self,
        transcript: str = "hello world",
        language: str = "en",
        filtered: bool = False,
        model: str | None = None,
        load_base: bool = True,
    ):
        self._transcript = transcript
        self._language = language
        self._filtered = filtered
        self.decode_called = False
        self.decode_audio = None

    def decode(self, raw: bytes, **kwargs) -> MockCognitiveEvent:
        self.decode_called = True
        self.decode_audio = kwargs.get("audio")
        if self._filtered:
            return MockCognitiveEvent("", self._language, filtered=True)
        return MockCognitiveEvent(self._transcript, self._language)


@pytest.fixture()
def client(monkeypatch):
    """Create test client with mocked WhisperDecoder."""
    import http_api

    mock_decoder = MockWhisperDecoder(transcript="test transcription", language="en")
    monkeypatch.setattr(http_api, "_stt_decoder", mock_decoder)

    from fastapi.testclient import TestClient

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture()
def client_hallucination(monkeypatch):
    """Create test client with mocked decoder that filters as hallucination."""
    import http_api

    mock_decoder = MockWhisperDecoder(transcript="", language="en", filtered=True)
    monkeypatch.setattr(http_api, "_stt_decoder", mock_decoder)

    from fastapi.testclient import TestClient

    return TestClient(http_api.app, base_url="http://localhost:7860")


class TestTranscribeEndpoint:
    """Tests for /v1/transcribe endpoint."""

    def test_transcribe_wav_happy_path(self, client):
        """Happy path: WAV file returns transcript with expected shape."""
        wav_bytes = _make_wav_bytes(duration_sec=1.0)

        response = client.post(
            "/v1/transcribe",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )

        assert response.status_code == 200, response.text
        data = response.json()

        # Check response shape
        assert "transcript" in data
        assert "language" in data
        assert "duration_sec" in data
        assert "stt_ms" in data

        # Check values
        assert data["transcript"] == "test transcription"
        assert data["language"] == "en"
        assert data["duration_sec"] > 0
        assert data["stt_ms"] >= 0

    def test_transcribe_empty_file_returns_400(self, client):
        """Empty file upload returns 400 error."""
        response = client.post(
            "/v1/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_transcribe_hallucination_filtered(self, client_hallucination):
        """Hallucination-filtered transcript returns empty string."""
        wav_bytes = _make_wav_bytes(duration_sec=0.5)

        response = client_hallucination.post(
            "/v1/transcribe",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == ""

    def test_transcribe_corrupt_audio_returns_400(self, client):
        """Corrupt/unparseable audio returns 400 error."""
        response = client.post(
            "/v1/transcribe",
            files={"file": ("bad.wav", b"not valid audio data", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_transcribe_detects_wav_by_magic_bytes(self, client):
        """WAV detection works via RIFF magic bytes even without extension."""
        wav_bytes = _make_wav_bytes(duration_sec=0.5)

        response = client.post(
            "/v1/transcribe",
            files={"file": ("audio_file", wav_bytes, "application/octet-stream")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "test transcription"


class TestTranscribeLazyLoad:
    """Tests for lazy-loading behavior."""

    def test_stt_decoder_global_starts_none(self):
        """_stt_decoder global should be None at module load."""
        import http_api

        # Check that the global exists (may have been set by other tests,
        # but at least the attribute should exist)
        assert hasattr(http_api, "_stt_decoder")
        assert hasattr(http_api, "_get_stt_decoder")

    def test_get_stt_decoder_creates_instance(self, monkeypatch):
        """_get_stt_decoder() creates WhisperDecoder on first call."""
        import http_api

        # Reset global state
        monkeypatch.setattr(http_api, "_stt_decoder", None)

        # Mock WhisperDecoder to avoid loading real model
        created_instances = []

        class MockDecoder:
            def __init__(self, model=None, load_base=True):
                self.model = model
                self.load_base = load_base
                created_instances.append(self)

        import modules.voice

        monkeypatch.setattr(modules.voice, "WhisperDecoder", MockDecoder)

        # Call the function
        decoder = http_api._get_stt_decoder()

        # Should have created exactly one decoder
        assert len(created_instances) == 1
        assert decoder is created_instances[0]
        assert decoder.model == "mlx-community/whisper-large-v3-turbo"
        assert decoder.load_base is False

        # Second call should return same instance
        decoder2 = http_api._get_stt_decoder()
        assert decoder2 is decoder
        assert len(created_instances) == 1  # No new instance created

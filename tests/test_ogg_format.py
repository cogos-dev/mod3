"""Tests for OGG/Opus output format.

Covers:
- SynthesizeRequest accepts format="ogg", rejects invalid values
- SpeakRequest has no format field (removed in 0807712 -- was dead/silent)
- encode_ogg() returns valid OGG/Opus bytes (OggS magic header)
- /v1/synthesize returns OGG bytes + correct Content-Type when format="ogg"
- MCP tool (mod3_speak via channel_client.build_mcp_server) passthrough
  returns base64 OGG when skip_playback=True and format="ogg"

Run with: .venv/bin/python -m pytest tests/test_ogg_format.py -v
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSynthesizeRequestFormat:
    """SynthesizeRequest must accept ogg and reject unknown formats."""

    def test_ogg_accepted(self):
        from schemas.http.synthesize import SynthesizeRequest

        req = SynthesizeRequest(text="hello", format="ogg")
        assert req.format == "ogg"

    def test_wav_default_unchanged(self):
        from schemas.http.synthesize import SynthesizeRequest

        req = SynthesizeRequest(text="hello")
        assert req.format == "wav"

    def test_pcm_still_accepted(self):
        from schemas.http.synthesize import SynthesizeRequest

        req = SynthesizeRequest(text="hello", format="pcm")
        assert req.format == "pcm"

    def test_mp3_rejected(self):
        from schemas.http.synthesize import SynthesizeRequest

        with pytest.raises(Exception):
            SynthesizeRequest(text="hello", format="mp3")

    def test_empty_format_rejected(self):
        from schemas.http.synthesize import SynthesizeRequest

        with pytest.raises(Exception):
            SynthesizeRequest(text="hello", format="")

    def test_uppercase_ogg_rejected(self):
        """Pattern is case-sensitive; OGG is not valid."""
        from schemas.http.synthesize import SynthesizeRequest

        with pytest.raises(Exception):
            SynthesizeRequest(text="hello", format="OGG")


class TestSpeakRequestHasNoFormatField:
    """SpeakRequest must NOT have a format field.

    /v1/speak is a playback-queue endpoint; format selection is only
    meaningful on /v1/synthesize. The field was removed in 0807712 to
    avoid silent wrong-format behaviour.
    """

    def test_no_format_declared_field(self):
        from schemas.http.synthesize import SpeakRequest

        SpeakRequest(text="hello")  # noqa: F841
        # format must not be a declared model field (check on class, not instance)
        assert "format" not in SpeakRequest.model_fields, (
            "SpeakRequest must not have a declared 'format' model field — "
            "it was removed because /v1/speak always produces WAV for the drain thread."
        )

    def test_format_kwarg_not_surfaced_as_declared_attribute(self):
        """Even if extra='allow' passes the kwarg through, the field must not
        be listed in model_fields (i.e. it's not a first-class field).
        """
        from schemas.http.synthesize import SpeakRequest

        req = SpeakRequest(**{"text": "hello", "format": "ogg"})  # type: ignore[arg-type]
        assert "format" not in SpeakRequest.model_fields


# ---------------------------------------------------------------------------
# encode_ogg unit test
# ---------------------------------------------------------------------------

_OGG_MAGIC = b"OggS"


class TestEncodeOgg:
    """encode_ogg() must return valid OGG/Opus bytes."""

    def test_returns_bytes(self):
        from http_api import encode_ogg

        samples = np.zeros(24000, dtype=np.float32)  # 1 s silence at 24 kHz
        result = encode_ogg(samples, 24000)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_oggs_magic_header(self):
        """All OGG streams start with the 'OggS' capture pattern."""
        from http_api import encode_ogg

        samples = np.zeros(24000, dtype=np.float32)
        result = encode_ogg(samples, 24000)
        assert result[:4] == _OGG_MAGIC, (
            f"Expected OggS at offset 0, got {result[:4]!r}"
        )

    def test_standard_rate_24khz(self):
        """24000 Hz is natively Opus-compatible — no resampling path."""
        from http_api import encode_ogg

        samples = np.random.default_rng(42).uniform(-0.1, 0.1, 24000).astype(np.float32)
        result = encode_ogg(samples, 24000)
        assert result[:4] == _OGG_MAGIC

    def test_standard_rate_48khz(self):
        """48000 Hz is natively Opus-compatible."""
        from http_api import encode_ogg

        samples = np.zeros(48000, dtype=np.float32)
        result = encode_ogg(samples, 48000)
        assert result[:4] == _OGG_MAGIC

    def test_nonstandard_rate_resampled(self):
        """22050 Hz is not Opus-valid; encode_ogg must resample to 24000."""
        from http_api import encode_ogg

        samples = np.zeros(22050, dtype=np.float32)
        result = encode_ogg(samples, 22050)
        assert result[:4] == _OGG_MAGIC

    def test_output_size_reasonable(self):
        """1 second of silence should produce at least a header (>100 bytes)."""
        from http_api import encode_ogg

        samples = np.zeros(24000, dtype=np.float32)
        result = encode_ogg(samples, 24000)
        assert len(result) > 100


# ---------------------------------------------------------------------------
# /v1/synthesize HTTP endpoint
# ---------------------------------------------------------------------------


def _make_fake_audio_chunk(sample_rate: int = 24000, duration_secs: float = 0.1):
    """Return a minimal AudioChunk-like object for mocking generate_audio."""
    from engine import AudioChunk

    n = int(sample_rate * duration_secs)
    samples = np.zeros(n, dtype=np.float32)
    return AudioChunk(samples=samples, sample_rate=sample_rate, metadata={"engine": "mock"})


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


class TestSynthesizeOggEndpoint:
    """Integration: /v1/synthesize returns OGG bytes + correct headers."""

    @pytest.fixture(autouse=True)
    def _patch_voice_resolver(self):
        """Bypass the bus voice-resolver to avoid needing a live ModalityBus."""
        import http_api

        original = http_api._resolve_voice_via_bus
        http_api._resolve_voice_via_bus = lambda v: v
        yield
        http_api._resolve_voice_via_bus = original

    @pytest.fixture(autouse=True)
    def _patch_generate_audio(self):
        """Replace generate_audio with a fast no-op that returns one chunk."""
        import http_api

        chunk = _make_fake_audio_chunk()
        original = http_api.generate_audio
        http_api.generate_audio = lambda *a, **kw: iter([chunk])
        yield
        http_api.generate_audio = original

    def test_ogg_format_returns_200(self, client):
        r = client.post("/v1/synthesize", json={"text": "hello", "format": "ogg"})
        assert r.status_code == 200, r.text

    def test_ogg_content_type(self, client):
        r = client.post("/v1/synthesize", json={"text": "hello", "format": "ogg"})
        assert r.status_code == 200, r.text
        ct = r.headers.get("content-type", "")
        assert "audio/ogg" in ct, f"Expected audio/ogg in Content-Type, got {ct!r}"
        assert "codecs=opus" in ct, (
            f"Expected codecs=opus in Content-Type per RFC 7845 §9, got {ct!r}"
        )

    def test_ogg_body_has_magic_header(self, client):
        r = client.post("/v1/synthesize", json={"text": "hello", "format": "ogg"})
        assert r.status_code == 200, r.text
        assert r.content[:4] == _OGG_MAGIC, (
            f"Expected OggS magic at start of response body, got {r.content[:4]!r}"
        )

    def test_wav_format_still_works(self, client):
        """Regression: WAV path must be unaffected."""
        r = client.post("/v1/synthesize", json={"text": "hello", "format": "wav"})
        assert r.status_code == 200, r.text
        ct = r.headers.get("content-type", "")
        assert "audio/wav" in ct

    def test_invalid_format_rejected_at_schema(self, client):
        r = client.post("/v1/synthesize", json={"text": "hello", "format": "mp3"})
        assert r.status_code == 422, (
            f"Expected 422 Unprocessable Entity for invalid format, got {r.status_code}"
        )


# ---------------------------------------------------------------------------
# MCP tool passthrough (channel_client.build_mcp_server / mod3_speak)
# ---------------------------------------------------------------------------


def _make_fake_ogg_bytes() -> bytes:
    """Minimal valid OGG/Opus bytes — 0.1 s silence at 48 kHz."""
    from http_api import encode_ogg

    samples = np.zeros(4800, dtype=np.float32)
    return encode_ogg(samples, 48000)


def _build_mcp_mod3_speak(server_url: str = "http://localhost:7860"):
    """Instantiate build_mcp_server with a fake client and extract mod3_speak.fn."""
    from clients.channel_client import build_mcp_server

    fake_client = SimpleNamespace(
        server_url=server_url,
        session_id="test-session",
        seat_id=None,
        token=None,
    )
    mcp = build_mcp_server(fake_client)  # type: ignore[arg-type]
    tool = mcp._tool_manager._tools["mod3_speak"]
    return tool.fn  # the underlying async coroutine function


class TestMcpSpeakOggPassthrough:
    """mod3_speak MCP tool with skip_playback=True, format="ogg" must
    call /v1/synthesize with format=ogg and return base64-encoded OGG audio.
    """

    @pytest.mark.asyncio
    async def test_skip_playback_ogg_returns_base64(self):
        fake_ogg = _make_fake_ogg_bytes()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.content = fake_ogg
        fake_response.headers = {
            "content-type": "audio/ogg; codecs=opus",
            "X-Mod3-Duration-Sec": "0.1",
            "X-Mod3-Sample-Rate": "48000",
        }
        fake_response.raise_for_status = MagicMock()

        fake_http = AsyncMock()
        fake_http.post = AsyncMock(return_value=fake_response)
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=None)

        mod3_speak = _build_mcp_mod3_speak()

        with patch("clients.channel_client.httpx.AsyncClient", return_value=fake_http):
            result = await mod3_speak(
                text="hello",
                skip_playback=True,
                format="ogg",
            )

        assert "audio_base64" in result
        decoded = base64.b64decode(result["audio_base64"])
        assert decoded[:4] == _OGG_MAGIC, (
            f"Decoded audio must start with OggS, got {decoded[:4]!r}"
        )

    @pytest.mark.asyncio
    async def test_skip_playback_ogg_media_type(self):
        fake_ogg = _make_fake_ogg_bytes()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.content = fake_ogg
        fake_response.headers = {
            "content-type": "audio/ogg; codecs=opus",
            "X-Mod3-Duration-Sec": "0.1",
            "X-Mod3-Sample-Rate": "48000",
        }
        fake_response.raise_for_status = MagicMock()

        fake_http = AsyncMock()
        fake_http.post = AsyncMock(return_value=fake_response)
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=None)

        mod3_speak = _build_mcp_mod3_speak()

        with patch("clients.channel_client.httpx.AsyncClient", return_value=fake_http):
            result = await mod3_speak(
                text="hello",
                skip_playback=True,
                format="ogg",
            )

        assert result.get("format") == "ogg"
        # channel_client strips the codecs= parameter from the content-type header
        assert result["media_type"] == "audio/ogg", (
            f"Expected media_type='audio/ogg', got {result['media_type']!r}"
        )

    @pytest.mark.asyncio
    async def test_skip_playback_ogg_calls_synthesize_not_speak(self):
        """With skip_playback=True, mod3_speak must POST to /v1/synthesize,
        NOT /v1/speak.
        """
        fake_ogg = _make_fake_ogg_bytes()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.content = fake_ogg
        fake_response.headers = {
            "content-type": "audio/ogg; codecs=opus",
            "X-Mod3-Duration-Sec": "0.1",
            "X-Mod3-Sample-Rate": "48000",
        }
        fake_response.raise_for_status = MagicMock()

        posted_urls: list[str] = []

        async def fake_post(url: str, **kwargs):
            posted_urls.append(url)
            return fake_response

        fake_http = AsyncMock()
        fake_http.post = fake_post
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=None)

        mod3_speak = _build_mcp_mod3_speak()

        with patch("clients.channel_client.httpx.AsyncClient", return_value=fake_http):
            await mod3_speak(text="hello", skip_playback=True, format="ogg")

        assert any("/v1/synthesize" in url for url in posted_urls), (
            f"Expected a POST to /v1/synthesize, got: {posted_urls}"
        )
        assert not any("/v1/speak" in url for url in posted_urls), (
            f"skip_playback=True must NOT call /v1/speak, got: {posted_urls}"
        )

    @pytest.mark.asyncio
    async def test_skip_playback_ogg_format_in_request_body(self):
        """mod3_speak must forward format='ogg' in the synthesize request body."""
        fake_ogg = _make_fake_ogg_bytes()

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.content = fake_ogg
        fake_response.headers = {
            "content-type": "audio/ogg; codecs=opus",
            "X-Mod3-Duration-Sec": "0.1",
            "X-Mod3-Sample-Rate": "48000",
        }
        fake_response.raise_for_status = MagicMock()

        captured_bodies: list[dict] = []

        async def fake_post(url: str, json: dict | None = None, **kwargs):
            if json is not None:
                captured_bodies.append(json)
            return fake_response

        fake_http = AsyncMock()
        fake_http.post = fake_post
        fake_http.__aenter__ = AsyncMock(return_value=fake_http)
        fake_http.__aexit__ = AsyncMock(return_value=None)

        mod3_speak = _build_mcp_mod3_speak()

        with patch("clients.channel_client.httpx.AsyncClient", return_value=fake_http):
            await mod3_speak(text="hello world", skip_playback=True, format="ogg")

        assert captured_bodies, "Expected at least one POST body"
        body = captured_bodies[-1]
        assert body.get("format") == "ogg", (
            f"Expected format='ogg' in synthesize request body, got: {body}"
        )

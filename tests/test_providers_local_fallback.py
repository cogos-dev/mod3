"""Tests for CogOSProvider's local-inference guarantees.

Covers the three routing fixes in providers.py:

  * FIX #1 (primary): when the preferred local LAN model (Eclipse) is
    unreachable, voice routes to the local Ollama E4B floor — NOT to the
    kernel (whose cloud-first fallback chain would pick a paid provider).
  * FIX #2 (medium): a kernel error (429 / 5xx / connection failure) falls
    back to local Ollama rather than killing the turn.
  * FIX #3 (low): _make_traceparent's trace_id is exactly 32 hex chars.

Run: python -m pytest tests/test_providers_local_fallback.py -v
"""

from __future__ import annotations

import os
import re
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from providers import CogOSProvider, ProviderResponse, ToolCall  # noqa: E402

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingOllama:
    """Stand-in OllamaProvider that records whether it was invoked."""

    name = "ollama/gemma4:e4b"

    def __init__(self) -> None:
        self.called = False
        self.last_args: tuple | None = None

    async def chat(self, messages, tools=None, system=""):
        self.called = True
        self.last_args = (messages, tools, system)
        return ProviderResponse(
            tool_calls=[ToolCall(name="output", arguments={"text": "from ollama"})],
            text="from ollama",
            raw={"served_by": "ollama"},
        )


class _MockAsyncClient:
    """Mock httpx.AsyncClient: GET drives the Eclipse probe, POST the kernel.

    Construct with callables for ``get_handler`` / ``post_handler`` that take
    the URL and return either a fake response or raise.
    """

    def __init__(self, *, get_handler=None, post_handler=None, **_kwargs):
        self._get_handler = get_handler
        self._post_handler = post_handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, url, *args, **kwargs):
        return self._get_handler(url)

    async def post(self, url, *args, **kwargs):
        return self._post_handler(url)


class _FakeResponse:
    """Minimal httpx-like response with raise_for_status + json."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.request = httpx.Request("POST", "http://localhost:6931/v1/chat/completions")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self):
        return self._payload


# A kernel response shaped like the OpenAI-compat /v1/chat/completions body.
_KERNEL_OK_PAYLOAD = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "lmstudio-eclipse",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "from kernel",
                "tool_calls": [],
            },
            "finish_reason": "stop",
        }
    ],
}


def _patch_client(monkeypatch, *, get_handler, post_handler):
    def _factory(*args, **kwargs):
        return _MockAsyncClient(get_handler=get_handler, post_handler=post_handler, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)


# ---------------------------------------------------------------------------
# FIX #3 — traceparent is 32 hex chars
# ---------------------------------------------------------------------------


class TestTraceparent:
    def test_trace_id_is_32_hex_chars(self):
        traceparent, trace_id = CogOSProvider._make_traceparent()
        assert len(trace_id) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", trace_id)

    def test_traceparent_header_shape(self):
        traceparent, trace_id = CogOSProvider._make_traceparent()
        # 00-<32hex>-<16hex>-01
        m = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-01", traceparent)
        assert m is not None
        assert m.group(1) == trace_id


# ---------------------------------------------------------------------------
# FIX #1 — Eclipse down → local Ollama, never the kernel/cloud
# ---------------------------------------------------------------------------


class TestEclipseDownRoutesLocal:
    @pytest.mark.asyncio
    async def test_eclipse_unreachable_lands_on_ollama_not_kernel(self, monkeypatch):
        ollama = _RecordingOllama()
        kernel_called = {"v": False}

        def _get(url):
            # Eclipse probe: simulate connection refused.
            raise httpx.ConnectError("connection refused")

        def _post(url):
            kernel_called["v"] = True
            return _FakeResponse(200, _KERNEL_OK_PAYLOAD)

        _patch_client(monkeypatch, get_handler=_get, post_handler=_post)

        provider = CogOSProvider(
            eclipse_probe_url="http://192.168.10.191:1234/v1/models",
            ollama_fallback=ollama,
        )
        resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert ollama.called, "voice must route to local Ollama when Eclipse is down"
        assert not kernel_called["v"], "must NOT delegate to the kernel (cloud-first chain)"
        assert resp.text == "from ollama"

    @pytest.mark.asyncio
    async def test_eclipse_reachable_uses_kernel(self, monkeypatch):
        ollama = _RecordingOllama()
        kernel_called = {"v": False}

        def _get(url):
            return _FakeResponse(200, {})  # Eclipse /v1/models healthy

        def _post(url):
            kernel_called["v"] = True
            return _FakeResponse(200, _KERNEL_OK_PAYLOAD)

        _patch_client(monkeypatch, get_handler=_get, post_handler=_post)

        provider = CogOSProvider(
            eclipse_probe_url="http://192.168.10.191:1234/v1/models",
            ollama_fallback=ollama,
        )
        resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert kernel_called["v"], "Eclipse up → delegate to kernel"
        assert not ollama.called
        assert resp.text == "from kernel"

    @pytest.mark.asyncio
    async def test_empty_probe_url_disables_preflight(self, monkeypatch):
        ollama = _RecordingOllama()
        kernel_called = {"v": False}

        def _get(url):
            raise AssertionError("probe should be skipped when URL is empty")

        def _post(url):
            kernel_called["v"] = True
            return _FakeResponse(200, _KERNEL_OK_PAYLOAD)

        _patch_client(monkeypatch, get_handler=_get, post_handler=_post)

        provider = CogOSProvider(eclipse_probe_url="", ollama_fallback=ollama)
        resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert kernel_called["v"]
        assert resp.text == "from kernel"


# ---------------------------------------------------------------------------
# FIX #2 — kernel error → local Ollama, never raises
# ---------------------------------------------------------------------------


class TestKernelErrorFallsBackLocal:
    @pytest.mark.asyncio
    async def test_kernel_429_falls_back_to_ollama(self, monkeypatch):
        ollama = _RecordingOllama()

        def _get(url):
            return _FakeResponse(200, {})  # Eclipse healthy → kernel is tried

        def _post(url):
            return _FakeResponse(429, {"error": "rate limited"})  # cloud rate-limit

        _patch_client(monkeypatch, get_handler=_get, post_handler=_post)

        provider = CogOSProvider(
            eclipse_probe_url="http://eclipse/v1/models",
            ollama_fallback=ollama,
        )
        resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert ollama.called, "kernel 429 must fall back to local Ollama"
        assert resp.text == "from ollama"

    @pytest.mark.asyncio
    async def test_kernel_connection_error_falls_back_to_ollama(self, monkeypatch):
        ollama = _RecordingOllama()

        def _get(url):
            return _FakeResponse(200, {})

        def _post(url):
            raise httpx.ConnectError("connection refused")

        _patch_client(monkeypatch, get_handler=_get, post_handler=_post)

        provider = CogOSProvider(
            eclipse_probe_url="http://eclipse/v1/models",
            ollama_fallback=ollama,
        )
        resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert ollama.called
        assert resp.text == "from ollama"

    @pytest.mark.asyncio
    async def test_kernel_500_falls_back_to_ollama(self, monkeypatch):
        ollama = _RecordingOllama()

        def _get(url):
            return _FakeResponse(200, {})

        def _post(url):
            return _FakeResponse(503, {"error": "unavailable"})

        _patch_client(monkeypatch, get_handler=_get, post_handler=_post)

        provider = CogOSProvider(
            eclipse_probe_url="http://eclipse/v1/models",
            ollama_fallback=ollama,
        )
        resp = await provider.chat([{"role": "user", "content": "hi"}])

        assert ollama.called
        assert resp.text == "from ollama"

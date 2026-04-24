"""Tests for cogos_agent_bridge — POST body shape + response fan-out."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bus_bridge import BusEnvelope  # noqa: E402
from cogos_agent_bridge import (  # noqa: E402
    _extract_response_text,
    _extract_session_id,
    post_user_message,
    run_response_bridge,
)


class _FakeSubscriber:
    def __init__(self, envelopes: list[BusEnvelope]) -> None:
        self._envelopes = envelopes

    async def stream(self):
        for env in self._envelopes:
            yield env


def _env(payload: dict, event_id: str = "r1") -> BusEnvelope:
    return BusEnvelope(
        raw={"type": "bus.event", "data": payload},
        kind="bus.event",
        payload=payload,
        ts="2026-04-17T00:00:00Z",
        event_id=event_id,
    )


def test_extract_response_text_handles_content_wrapped_json():
    # Kernel wraps the sent `message` string inside {"content": "<str>"}.
    inner = {"type": "agent_response", "text": "hi there", "ts": "2026-04-17T00:00:00Z"}
    payload = {"content": json.dumps(inner)}
    assert _extract_response_text(payload) == "hi there"


def test_extract_response_text_handles_plain_content_string():
    assert _extract_response_text({"content": "hello"}) == "hello"


def test_extract_response_text_skips_unparseable():
    assert _extract_response_text({"foo": "bar"}) is None
    assert _extract_response_text({}) is None


def test_post_user_message_body_shape():
    captured: dict = {}

    class _FakeResp:
        status_code = 200
        text = ""

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):  # noqa: A002 — matches httpx signature
            captured["url"] = url
            captured["body"] = json
            return _FakeResp()

    with patch("cogos_agent_bridge.httpx.AsyncClient", _FakeClient):
        ok = asyncio.run(post_user_message("hello agent", session_id="mod3:browser:abc"))

    assert ok is True
    body = captured["body"]
    assert body["bus_id"] == "bus_dashboard_chat"
    assert body["type"] == "user_message"
    assert body["from"] == "mod3-dashboard"
    # `message` is a JSON-encoded event dict — parse and check shape.
    event = json.loads(body["message"])
    assert event["type"] == "user_message"
    assert event["text"] == "hello agent"
    assert event["session_id"] == "mod3:browser:abc"
    assert "ts" in event


def test_run_response_bridge_fans_out_to_broadcast():
    # Envelope in the shape the kernel emits: payload = {"content": "<json>"}
    inner = json.dumps({"type": "agent_response", "text": "reply one"})
    envelopes = [
        _env({"content": inner}, "r1"),
        _env({"content": "free-form string reply"}, "r2"),
        _env({"foo": "bar"}, "r3"),  # no text — should be skipped
    ]
    sub = _FakeSubscriber(envelopes)

    with (
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_text") as mock_text,
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_complete") as mock_done,
    ):
        asyncio.run(run_response_bridge(sub))

    texts = [c.args[0] for c in mock_text.call_args_list]
    assert texts == ["reply one", "free-form string reply"]
    # One completion frame per forwarded text — never zero, never doubled.
    assert mock_done.call_count == 2
    # Payload shape: provider tag plus any kernel-supplied timing/ids.
    for call in mock_done.call_args_list:
        metrics = call.args[0]
        assert metrics["provider"] == "cogos-agent"


# ---------------------------------------------------------------------------
# Session-id extraction & forwarding (Codex review #4 / Fix 4)
#
# The kernel-side change includes session_id in reply payloads so mod3 can
# route to the originating BrowserChannel. When session_id is missing
# (older kernel, non-session-scoped event), broadcast_response_text falls
# back to broadcasting — preserving backward compat.
# ---------------------------------------------------------------------------


def test_extract_session_id_from_top_level():
    assert _extract_session_id({"session_id": "mod3:browser:abc"}) == "mod3:browser:abc"


def test_extract_session_id_from_content_wrapped_json():
    inner = {"type": "agent_response", "text": "hi", "session_id": "mod3:browser:xyz"}
    payload = {"content": json.dumps(inner)}
    assert _extract_session_id(payload) == "mod3:browser:xyz"


def test_extract_session_id_returns_none_when_absent():
    assert _extract_session_id({"text": "hi"}) is None
    assert _extract_session_id({"content": json.dumps({"text": "hi"})}) is None
    assert _extract_session_id({"content": "free-form string"}) is None
    assert _extract_session_id({}) is None


def test_run_response_bridge_forwards_session_id_when_present():
    """When the kernel reply includes session_id, it must reach broadcast."""
    inner = json.dumps({"type": "agent_response", "text": "scoped reply", "session_id": "mod3:browser:abc"})
    envelopes = [_env({"content": inner}, "r1")]
    sub = _FakeSubscriber(envelopes)

    with (
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_text") as mock_text,
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_complete") as mock_done,
    ):
        asyncio.run(run_response_bridge(sub))

    assert mock_text.call_count == 1
    text_call = mock_text.call_args_list[0]
    assert text_call.args[0] == "scoped reply"
    # session_id passed as keyword
    assert text_call.kwargs.get("session_id") == "mod3:browser:abc"
    # Completion frame routes to the same session so the originating
    # channel's spinner clears — not a broadcast.
    assert mock_done.call_count == 1
    done_call = mock_done.call_args_list[0]
    assert done_call.kwargs.get("session_id") == "mod3:browser:abc"


def test_run_response_bridge_falls_back_to_broadcast_when_no_session_id():
    """Old-kernel reply (no session_id) -> broadcast_response_text(session_id=None)."""
    inner = json.dumps({"type": "agent_response", "text": "broadcast reply"})
    envelopes = [_env({"content": inner}, "r1")]
    sub = _FakeSubscriber(envelopes)

    with (
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_text") as mock_text,
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_complete") as mock_done,
    ):
        asyncio.run(run_response_bridge(sub))

    assert mock_text.call_count == 1
    text_call = mock_text.call_args_list[0]
    assert text_call.args[0] == "broadcast reply"
    assert text_call.kwargs.get("session_id") is None
    # Completion frame also broadcasts (matches the text-frame routing).
    assert mock_done.call_count == 1
    assert mock_done.call_args_list[0].kwargs.get("session_id") is None


def test_run_response_bridge_skips_complete_when_text_missing():
    """Events with no recoverable text must not emit a completion frame.

    Holding the 1:1 pairing keeps the UI's turn counter honest: we only
    mark a turn done when we actually rendered something for it.
    """
    envelopes = [_env({"foo": "bar"}, "r1"), _env({}, "r2")]
    sub = _FakeSubscriber(envelopes)

    with (
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_text") as mock_text,
        patch("cogos_agent_bridge.BrowserChannel.broadcast_response_complete") as mock_done,
    ):
        asyncio.run(run_response_bridge(sub))

    assert mock_text.call_count == 0
    assert mock_done.call_count == 0


def test_post_user_message_uses_runtime_endpoint(monkeypatch):
    """post_user_message must POST to the URL derived from COGOS_ENDPOINT
    at call time — not a stale module-import-time value."""
    monkeypatch.setenv("COGOS_ENDPOINT", "http://kernel.test:9000")

    captured: dict = {}

    class _FakeResp:
        status_code = 200
        text = ""

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):  # noqa: A002
            captured["url"] = url
            return _FakeResp()

    with patch("cogos_agent_bridge.httpx.AsyncClient", _FakeClient):
        ok = asyncio.run(post_user_message("hi", session_id="mod3:browser:abc"))

    assert ok is True
    assert captured["url"] == "http://kernel.test:9000/v1/bus/send"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

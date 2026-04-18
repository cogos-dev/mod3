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

    with patch("cogos_agent_bridge.BrowserChannel.broadcast_response_text") as mock_bcast:
        asyncio.run(run_response_bridge(sub))

    texts = [c.args[0] for c in mock_bcast.call_args_list]
    assert texts == ["reply one", "free-form string reply"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for BrowserChannel.broadcast_response_text session routing.

Covers Codex review #4 / Fix 4: kernel replies that include a session_id
should be delivered ONLY to the matching BrowserChannel; replies without a
session_id should fall back to broadcast (preserves backward compatibility
with older kernels that don't yet include the session field).

The session_id format is the convention from cogos_agent_bridge.post_user_message:
``mod3:<channel_id>``. The leading ``mod3:`` is stripped to match the
``BrowserChannel.channel_id`` value (e.g. ``browser:abc12345``).

We don't spin up a real WebSocket; instead we register lightweight stand-ins
on ``BrowserChannel._active_channels`` (a class-level set) that record the
frames they receive.

Run: python -m pytest tests/test_browser_channel_routing.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from channels import BrowserChannel  # noqa: E402


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, frame: dict) -> None:
        self.sent.append(frame)


class _FakeChannel:
    """Mimics enough of BrowserChannel for broadcast_response_text to use it.

    BrowserChannel.broadcast_response_text iterates ``_active_channels`` and
    calls ``asyncio.run_coroutine_threadsafe(ch.ws.send_json(frame), ch._loop)``.
    We replace run_coroutine_threadsafe with a synchronous shim that just
    runs the coroutine on the current loop, so we don't need a separate
    background loop per fake channel.
    """

    def __init__(self, channel_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.channel_id = channel_id
        self.ws = _FakeWS()
        self._loop = loop
        self._active = True


@pytest.fixture(autouse=True)
def _isolate_active_channels():
    """Snapshot and restore BrowserChannel._active_channels around each test."""
    snapshot = set(BrowserChannel._active_channels)
    BrowserChannel._active_channels.clear()
    yield
    BrowserChannel._active_channels.clear()
    BrowserChannel._active_channels.update(snapshot)


def _patched_run(coro, _loop):
    """Drive the awaitable to completion on the current event loop."""
    asyncio.get_event_loop().run_until_complete(coro)

    class _Done:
        def result(self, timeout: float = 0) -> Any:  # noqa: ARG002
            return None

    return _Done()


def _broadcast_with_loop(text: str, session_id: str | None = None) -> None:
    """Run broadcast_response_text with run_coroutine_threadsafe stubbed.

    Uses a fresh event loop so the fake channels' ws.send_json coroutines
    actually run.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with patch("channels.asyncio.run_coroutine_threadsafe", _patched_run):
            BrowserChannel.broadcast_response_text(text, session_id=session_id)
    finally:
        loop.close()


def _broadcast_complete_with_loop(metrics: dict | None = None, session_id: str | None = None) -> None:
    """Sibling of _broadcast_with_loop for the response_complete frame."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with patch("channels.asyncio.run_coroutine_threadsafe", _patched_run):
            BrowserChannel.broadcast_response_complete(metrics, session_id=session_id)
    finally:
        loop.close()


def test_broadcast_with_no_session_id_fans_out_to_all_active_channels():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    b = _FakeChannel("browser:bbb", loop)
    c = _FakeChannel("browser:ccc", loop)
    BrowserChannel._active_channels.update({a, b, c})

    _broadcast_with_loop("hello everyone")

    for ch in (a, b, c):
        assert len(ch.ws.sent) == 1
        assert ch.ws.sent[0] == {"type": "response_text", "text": "hello everyone"}

    loop.close()


def test_broadcast_with_session_id_routes_to_only_matching_channel():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    b = _FakeChannel("browser:bbb", loop)
    c = _FakeChannel("browser:ccc", loop)
    BrowserChannel._active_channels.update({a, b, c})

    _broadcast_with_loop("just for B", session_id="mod3:browser:bbb")

    assert a.ws.sent == []
    assert b.ws.sent == [{"type": "response_text", "text": "just for B"}]
    assert c.ws.sent == []

    loop.close()


def test_broadcast_skips_inactive_channels():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    a._active = False
    b = _FakeChannel("browser:bbb", loop)
    BrowserChannel._active_channels.update({a, b})

    _broadcast_with_loop("only active wins")

    assert a.ws.sent == []
    assert b.ws.sent == [{"type": "response_text", "text": "only active wins"}]

    loop.close()


def test_broadcast_session_id_without_mod3_prefix_falls_back_to_broadcast():
    """Defensive: if a malformed session_id arrives, don't lose the message."""
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    b = _FakeChannel("browser:bbb", loop)
    BrowserChannel._active_channels.update({a, b})

    # No "mod3:" prefix -> expected_channel stays None -> broadcast
    _broadcast_with_loop("legacy session id", session_id="browser:aaa")

    assert len(a.ws.sent) == 1
    assert len(b.ws.sent) == 1
    assert a.ws.sent[0]["text"] == "legacy session id"

    loop.close()


def test_broadcast_session_id_with_no_match_drops_silently():
    """Routed session for a channel that's not connected -> no delivery."""
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    BrowserChannel._active_channels.add(a)

    _broadcast_with_loop("for a ghost", session_id="mod3:browser:zzz")

    assert a.ws.sent == []

    loop.close()


# ---------------------------------------------------------------------------
# response_complete routing (kernel-path turn-done signal)
#
# Paired with broadcast_response_text in cogos_agent_bridge.run_response_bridge;
# must follow the same session-scoped routing so the complete-frame lands on
# the originating dashboard channel (otherwise multi-client setups see
# cross-talk or hang on a non-matching spinner).
# ---------------------------------------------------------------------------


def test_broadcast_complete_with_no_session_id_fans_out_to_all_active_channels():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    b = _FakeChannel("browser:bbb", loop)
    BrowserChannel._active_channels.update({a, b})

    _broadcast_complete_with_loop({"provider": "cogos-agent"})

    for ch in (a, b):
        assert len(ch.ws.sent) == 1
        frame = ch.ws.sent[0]
        assert frame["type"] == "response_complete"
        assert frame["metrics"] == {"provider": "cogos-agent"}

    loop.close()


def test_broadcast_complete_routes_to_matching_session_only():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    b = _FakeChannel("browser:bbb", loop)
    BrowserChannel._active_channels.update({a, b})

    _broadcast_complete_with_loop(
        {"provider": "cogos-agent", "event_id": "r42"},
        session_id="mod3:browser:bbb",
    )

    assert a.ws.sent == []
    assert len(b.ws.sent) == 1
    assert b.ws.sent[0]["type"] == "response_complete"
    assert b.ws.sent[0]["metrics"]["event_id"] == "r42"

    loop.close()


def test_broadcast_complete_defaults_to_empty_metrics():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    BrowserChannel._active_channels.add(a)

    _broadcast_complete_with_loop()

    assert a.ws.sent == [{"type": "response_complete", "metrics": {}}]

    loop.close()


def test_broadcast_complete_skips_inactive_channels():
    loop = asyncio.new_event_loop()
    a = _FakeChannel("browser:aaa", loop)
    a._active = False
    b = _FakeChannel("browser:bbb", loop)
    BrowserChannel._active_channels.update({a, b})

    _broadcast_complete_with_loop({"provider": "cogos-agent"})

    assert a.ws.sent == []
    assert len(b.ws.sent) == 1

    loop.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

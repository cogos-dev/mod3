"""Smoke tests for the symmetric dashboard chat round-trip (Path B).

Tests the /ws/dashboard-chat WebSocket endpoint and the mod3_dashboard_post
MCP tool via the HTTP-layer FastAPI app.

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_dashboard_chat.py -v
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import HAS_MCP  # noqa: E402

# Marker for tests that call @mcp.tool()-decorated functions on server.
needs_mcp = pytest.mark.skipif(
    not HAS_MCP, reason="mcp package required for @mcp.tool-decorated server functions"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


# ---------------------------------------------------------------------------
# /ws/dashboard-chat — subscriber receives broadcast
# ---------------------------------------------------------------------------


class TestDashboardChatWs:
    def test_connect_and_receive_broadcast(self, client):
        """Connecting to /ws/dashboard-chat and broadcasting should deliver the frame."""
        from server import _dashboard_chat_broadcast

        received = []

        def _subscriber():
            with client.websocket_connect("/ws/dashboard-chat") as ws:
                # Signal ready
                ready.set()
                # Wait for one message with a 3s timeout
                import queue as _q

                msg_q = _q.Queue()
                import threading as _t

                def _recv():
                    try:
                        raw = ws.receive_text()
                        msg_q.put(raw)
                    except Exception as exc:
                        msg_q.put(exc)

                recv_thread = _t.Thread(target=_recv, daemon=True)
                recv_thread.start()
                try:
                    item = msg_q.get(timeout=3.0)
                    received.append(item)
                except Exception:
                    pass

        ready = threading.Event()
        t = threading.Thread(target=_subscriber, daemon=True)
        t.start()

        # Wait for subscriber to connect
        assert ready.wait(timeout=3.0), "subscriber did not connect in time"
        time.sleep(0.05)  # small grace period for queue registration

        # Broadcast a message
        delivered = _dashboard_chat_broadcast(
            {"type": "chat", "role": "assistant", "text": "hello from Claude", "session_id": "test-123"}
        )

        t.join(timeout=4.0)

        assert delivered >= 1, f"expected at least 1 delivery, got {delivered}"
        assert len(received) == 1, f"expected 1 received message, got {received}"

        frame = json.loads(received[0])
        assert frame["type"] == "chat"
        assert frame["role"] == "assistant"
        assert frame["text"] == "hello from Claude"
        assert frame["session_id"] == "test-123"


# ---------------------------------------------------------------------------
# mod3_dashboard_post MCP tool — via server module directly
# ---------------------------------------------------------------------------


@needs_mcp
class TestMod3DashboardPostTool:
    def test_returns_ok_with_no_subscribers(self):
        """mod3_dashboard_post returns ok even when no WS subscribers are connected."""
        from server import mod3_dashboard_post

        raw = mod3_dashboard_post(text="hello from Claude")
        result = json.loads(raw)
        assert result["status"] == "ok"
        assert result["role"] == "assistant"
        assert "hello" in result["text_preview"]
        # delivered_to may be 0 if no subscribers are connected
        assert isinstance(result["delivered_to"], int)

    def test_empty_text_returns_error(self):
        """mod3_dashboard_post rejects empty text."""
        from server import mod3_dashboard_post

        raw = mod3_dashboard_post(text="   ")
        result = json.loads(raw)
        assert result["status"] == "error"

    def test_custom_role(self):
        """mod3_dashboard_post respects custom role parameter."""
        from server import mod3_dashboard_post

        raw = mod3_dashboard_post(text="user typed this", role="user", session_id="sid-abc")
        result = json.loads(raw)
        assert result["status"] == "ok"
        assert result["role"] == "user"

    def test_round_trip_via_ws(self, client):
        """Full round-trip: call mod3_dashboard_post, subscriber receives the frame."""
        from server import mod3_dashboard_post

        received = []
        ready = threading.Event()

        def _subscriber():
            with client.websocket_connect("/ws/dashboard-chat") as ws:
                ready.set()
                import queue as _q
                import threading as _t

                msg_q = _q.Queue()

                def _recv():
                    try:
                        raw = ws.receive_text()
                        msg_q.put(raw)
                    except Exception as exc:
                        msg_q.put(exc)

                recv_thread = _t.Thread(target=_recv, daemon=True)
                recv_thread.start()
                try:
                    item = msg_q.get(timeout=3.0)
                    received.append(item)
                except Exception:
                    pass

        t = threading.Thread(target=_subscriber, daemon=True)
        t.start()

        assert ready.wait(timeout=3.0), "subscriber did not connect in time"
        time.sleep(0.05)

        raw = mod3_dashboard_post(
            text="hello from Claude",
            session_id="smoke-test",
            role="assistant",
        )
        result = json.loads(raw)
        assert result["status"] == "ok"
        assert result["delivered_to"] >= 1

        t.join(timeout=4.0)

        assert len(received) == 1
        frame = json.loads(received[0])
        assert frame["type"] == "chat"
        assert frame["text"] == "hello from Claude"
        assert frame["session_id"] == "smoke-test"

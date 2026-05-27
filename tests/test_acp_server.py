"""Integration tests for the /ws/acp endpoint.

Uses FastAPI TestClient (synchronous WebSocket test) to exercise the full
JSON-RPC handshake without loading actual MLX models.

Run with: PYTHONPATH=. .venv/bin/python -m pytest tests/test_acp_server.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


# ---------------------------------------------------------------------------
# Helper to build ACP messages
# ---------------------------------------------------------------------------


def _req(method: str, params: dict, id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params})


def _notif(method: str, params: dict) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params})


def _parse(raw: str) -> dict:
    return json.loads(raw)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestAcpInitialize:
    def test_initialize_returns_capabilities(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(
                _req(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {"fs": {}, "terminal": False},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                    id=0,
                )
            )
            resp = _parse(ws.receive_text())

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 0
        assert "result" in resp
        caps = resp["result"]["agentCapabilities"]
        assert caps["promptCapabilities"]["audio"] is False
        assert caps["promptCapabilities"]["image"] is False
        assert caps["promptCapabilities"]["embeddedContext"] is False
        # session/list + session/resume advertised; loadSession advertised
        assert caps["sessionCapabilities"]["list"] is True
        assert caps["sessionCapabilities"]["resume"] is True
        assert caps["loadSession"] is True

    def test_initialize_handles_string_id(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {"protocolVersion": 1}, id="init-1"))
            resp = _parse(ws.receive_text())
        assert resp["id"] == "init-1"


# ---------------------------------------------------------------------------
# session/new
# ---------------------------------------------------------------------------


class TestAcpSessionNew:
    def test_session_new_returns_session_id(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()  # consume initialize response

            ws.send_text(_req("session/new", {"cwd": "/", "mcpServers": []}, id=1))
            resp = _parse(ws.receive_text())

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "sessionId" in resp["result"]
        assert resp["result"]["sessionId"].startswith("mod3-")

    def test_session_id_is_unique_per_call(self, client):
        ids = []
        for i in range(2):
            with client.websocket_connect("/ws/acp") as ws:
                ws.send_text(_req("initialize", {}, id=0))
                ws.receive_text()
                ws.send_text(_req("session/new", {}, id=1))
                resp = _parse(ws.receive_text())
                ids.append(resp["result"]["sessionId"])
        assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# session/prompt
# ---------------------------------------------------------------------------


class TestAcpSessionPrompt:
    def test_session_prompt_empty_text_returns_error(self, client):
        """Prompt with no text blocks should return an InvalidParams error."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()
            ws.send_text(_req("session/new", {}, id=1))
            new_resp = _parse(ws.receive_text())
            session_id = new_resp["result"]["sessionId"]

            ws.send_text(
                _req(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": []},
                    id=2,
                )
            )
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_session_prompt_returns_error_when_no_seats(self, client):
        """When no channel-client seats are attached, session/prompt returns
        a structured error explaining how to connect the channel client.
        """
        with patch("seats.SeatRegistry.fan_out", return_value=0):
            with client.websocket_connect("/ws/acp") as ws:
                ws.send_text(_req("initialize", {}, id=0))
                ws.receive_text()
                ws.send_text(_req("session/new", {}, id=1))
                new_resp = _parse(ws.receive_text())
                session_id = new_resp["result"]["sessionId"]

                ws.send_text(
                    _req(
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "Hello?"}],
                        },
                        id=2,
                    )
                )

                resp = _parse(ws.receive_text())

        assert resp.get("id") == 2
        assert "error" in resp
        assert resp["error"]["code"] == -32000
        assert "channel-client" in resp["error"]["message"].lower()

    def test_session_prompt_fans_to_seats_and_resolves(self, client):
        """When seats are attached, session/prompt fans the message and resolves
        immediately with end_turn (responses flow via the channel client tools).
        """
        fanned: list[dict] = []

        def _fake_fan_out(session_id: str, payload: dict) -> int:
            fanned.append({"session_id": session_id, "payload": payload})
            return 1  # one seat notified

        with patch("seats.SeatRegistry.fan_out", side_effect=_fake_fan_out):
            with client.websocket_connect("/ws/acp") as ws:
                ws.send_text(_req("initialize", {}, id=0))
                ws.receive_text()
                ws.send_text(_req("session/new", {}, id=1))
                new_resp = _parse(ws.receive_text())
                session_id = new_resp["result"]["sessionId"]

                ws.send_text(
                    _req(
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "Hello from ACP"}],
                        },
                        id=2,
                    )
                )

                resp = _parse(ws.receive_text())

        assert resp.get("id") == 2
        assert "result" in resp
        assert resp["result"]["stopReason"] == "end_turn"
        # Verify the fan-out carried the right content
        assert len(fanned) == 1
        assert fanned[0]["payload"]["content"] == "Hello from ACP"
        assert fanned[0]["payload"]["type"] == "user_message"


# ---------------------------------------------------------------------------
# session/cancel
# ---------------------------------------------------------------------------


class TestAcpSessionCancel:
    def test_session_cancel_notification_no_response(self, client):
        """session/cancel is a notification -- no response should be sent."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {}, id=0))
            ws.receive_text()
            ws.send_text(_req("session/new", {}, id=1))
            new_resp = _parse(ws.receive_text())
            session_id = new_resp["result"]["sessionId"]

            # Send cancel notification (no id).
            ws.send_text(_notif("session/cancel", {"sessionId": session_id}))

            # Immediately send another request to confirm the connection is still alive.
            ws.send_text(_req("session/new", {}, id=2))
            resp = _parse(ws.receive_text())
            assert resp["id"] == 2
            assert "sessionId" in resp["result"]


# ---------------------------------------------------------------------------
# Unknown method
# ---------------------------------------------------------------------------


class TestAcpUnknownMethod:
    def test_unknown_method_returns_error(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("bogus/method", {}, id=99))
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32601
        assert resp["id"] == 99

    def test_parse_error_on_invalid_json(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text("not-json{{")
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32700

    def test_invalid_jsonrpc_version(self, client):
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(json.dumps({"jsonrpc": "1.0", "id": 1, "method": "initialize"}))
            resp = _parse(ws.receive_text())
        assert "error" in resp

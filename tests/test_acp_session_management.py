"""Tests for the optional ACP session-management methods.

Covers:
  session/list     — lists registered TTS sessions
  session/load     — retrieves state of a specific session
  session/resume   — binds the ACP connection to a named session
  authenticate     — no-op handshake (authMethods is empty)

Cross-method composition: list -> load -> resume produces consistent state.

Run with:
  PYTHONPATH=. pytest tests/test_acp_session_management.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


def _req(method: str, params: dict, id: int = 1) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": params})


def _parse(raw: str) -> dict:
    return json.loads(raw)


def _do_initialize(ws) -> None:
    ws.send_text(
        _req(
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {"fs": {}, "terminal": False}},
            id=0,
        )
    )
    ws.receive_text()  # consume initialize response


# ---------------------------------------------------------------------------
# authenticate
# ---------------------------------------------------------------------------


class TestAcpAuthenticate:
    def test_authenticate_returns_success(self, client):
        """authenticate should return {success: true} immediately (authMethods is [])."""
        with client.websocket_connect("/ws/acp") as ws:
            _do_initialize(ws)
            ws.send_text(_req("authenticate", {"methodId": ""}, id=1))
            resp = _parse(ws.receive_text())

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["success"] is True

    def test_authenticate_with_any_method_id_succeeds(self, client):
        """Any methodId should succeed because authMethods is [] (no real check)."""
        with client.websocket_connect("/ws/acp") as ws:
            _do_initialize(ws)
            ws.send_text(_req("authenticate", {"methodId": "bearer"}, id=2))
            resp = _parse(ws.receive_text())

        assert "result" in resp
        assert resp["result"]["success"] is True


# ---------------------------------------------------------------------------
# session/list
# ---------------------------------------------------------------------------


class TestAcpSessionList:
    def test_session_list_returns_sessions_key(self, client):
        """session/list must return {sessions: [...]}."""
        with client.websocket_connect("/ws/acp") as ws:
            _do_initialize(ws)
            ws.send_text(_req("session/list", {}, id=1))
            resp = _parse(ws.receive_text())

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert "sessions" in resp["result"]
        assert isinstance(resp["result"]["sessions"], list)

    def test_session_list_includes_registered_session(self, client):
        """A session registered in the TTS registry should appear in session/list."""

        # Build a minimal mock registry with one session
        mock_session = {
            "session_id": "test-tts-session",
            "state": "idle",
            "participant_id": "chaz",
            "participant_type": "human",
        }

        with patch("session_registry.get_default_registry") as mock_get_reg:
            mock_reg = mock_get_reg.return_value
            mock_reg.list_serialized.return_value = [mock_session]

            with client.websocket_connect("/ws/acp") as ws:
                _do_initialize(ws)
                ws.send_text(_req("session/list", {}, id=1))
                resp = _parse(ws.receive_text())

        sessions = resp["result"]["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["sessionId"] == "test-tts-session"
        assert sessions[0]["state"] == "idle"
        assert sessions[0]["participantId"] == "chaz"
        assert sessions[0]["participantType"] == "human"

    def test_session_list_empty_when_no_sessions_registered(self, client):
        """session/list with no registered sessions returns sessions: []."""
        with patch("session_registry.get_default_registry") as mock_get_reg:
            mock_get_reg.return_value.list_serialized.return_value = []
            with client.websocket_connect("/ws/acp") as ws:
                _do_initialize(ws)
                ws.send_text(_req("session/list", {}, id=1))
                resp = _parse(ws.receive_text())

        assert resp["result"]["sessions"] == []

    def test_initialize_advertises_list_capability(self, client):
        """initialize response must advertise sessionCapabilities.list: true."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {"protocolVersion": 1}, id=0))
            resp = _parse(ws.receive_text())

        sc = resp["result"]["agentCapabilities"]["sessionCapabilities"]
        assert sc["list"] is True


# ---------------------------------------------------------------------------
# session/load
# ---------------------------------------------------------------------------


class TestAcpSessionLoad:
    def test_session_load_returns_state(self, client):
        """session/load must return {sessionId, state: {...}}."""
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.to_dict.return_value = {
            "session_id": "tts-abc",
            "state": "idle",
            "participant_id": "chaz",
            "participant_type": "human",
            "assigned_voice": "eng_uk_m_davids",
        }

        with patch("session_registry.get_default_registry") as mock_get_reg:
            mock_get_reg.return_value.get.return_value = mock_session

            with client.websocket_connect("/ws/acp") as ws:
                _do_initialize(ws)
                ws.send_text(_req("session/load", {"sessionId": "tts-abc"}, id=1))
                resp = _parse(ws.receive_text())

        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["sessionId"] == "tts-abc"
        assert resp["result"]["state"]["participant_id"] == "chaz"

    def test_session_load_missing_session_id_returns_error(self, client):
        """session/load without sessionId should return InvalidParams error."""
        with client.websocket_connect("/ws/acp") as ws:
            _do_initialize(ws)
            ws.send_text(_req("session/load", {}, id=1))
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_session_load_unknown_session_returns_error(self, client):
        """session/load for an unknown sessionId should return InternalError."""
        with patch("session_registry.get_default_registry") as mock_get_reg:
            mock_get_reg.return_value.get.return_value = None

            with client.websocket_connect("/ws/acp") as ws:
                _do_initialize(ws)
                ws.send_text(_req("session/load", {"sessionId": "nonexistent-xyz"}, id=1))
                resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32603
        assert "nonexistent-xyz" in resp["error"]["message"]

    def test_initialize_advertises_load_session_capability(self, client):
        """initialize response must advertise loadSession: true."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {"protocolVersion": 1}, id=0))
            resp = _parse(ws.receive_text())

        assert resp["result"]["agentCapabilities"]["loadSession"] is True


# ---------------------------------------------------------------------------
# session/resume
# ---------------------------------------------------------------------------


class TestAcpSessionResume:
    def test_session_resume_returns_session_id(self, client):
        """session/resume must return {sessionId} echoing the requested id."""
        with client.websocket_connect("/ws/acp") as ws:
            _do_initialize(ws)
            ws.send_text(_req("session/resume", {"sessionId": "default"}, id=1))
            resp = _parse(ws.receive_text())

        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["sessionId"] == "default"

    def test_session_resume_missing_session_id_returns_error(self, client):
        """session/resume without sessionId should return InvalidParams error."""
        with client.websocket_connect("/ws/acp") as ws:
            _do_initialize(ws)
            ws.send_text(_req("session/resume", {}, id=1))
            resp = _parse(ws.receive_text())

        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_session_resume_binds_connection_for_prompt(self, client):
        """After session/resume, session/prompt fans to the resumed session_id."""
        fanned: list[dict] = []

        def _fake_fan_out(session_id: str, payload: dict) -> int:
            fanned.append({"session_id": session_id, "payload": payload})
            return 1

        with patch("seats.SeatRegistry.fan_out", side_effect=_fake_fan_out):
            with client.websocket_connect("/ws/acp") as ws:
                _do_initialize(ws)
                # Resume the named session
                ws.send_text(_req("session/resume", {"sessionId": "my-session"}, id=1))
                ws.receive_text()  # consume resume response

                # Send a prompt — should fan to my-session
                ws.send_text(
                    _req(
                        "session/prompt",
                        {
                            "sessionId": "my-session",
                            "prompt": [{"type": "text", "text": "resumed prompt"}],
                        },
                        id=2,
                    )
                )
                resp = _parse(ws.receive_text())

        assert resp["id"] == 2
        assert resp.get("result", {}).get("stopReason") == "end_turn"
        assert fanned[0]["session_id"] == "my-session"
        assert fanned[0]["payload"]["content"] == "resumed prompt"

    def test_initialize_advertises_resume_capability(self, client):
        """initialize response must advertise sessionCapabilities.resume: true."""
        with client.websocket_connect("/ws/acp") as ws:
            ws.send_text(_req("initialize", {"protocolVersion": 1}, id=0))
            resp = _parse(ws.receive_text())

        sc = resp["result"]["agentCapabilities"]["sessionCapabilities"]
        assert sc["resume"] is True


# ---------------------------------------------------------------------------
# Cross-method composition: list -> load -> resume
# ---------------------------------------------------------------------------


class TestAcpSessionComposition:
    def test_list_load_resume_consistent_session_id(self, client):
        """list returns a session; load retrieves its state; resume binds it.

        All three operations must agree on the session_id.
        """
        from unittest.mock import MagicMock

        session_id = "compose-test-session"
        mock_session = MagicMock()
        mock_session.to_dict.return_value = {
            "session_id": session_id,
            "state": "idle",
            "participant_id": "chaz",
            "participant_type": "human",
            "assigned_voice": "eng_uk_m_davids",
        }
        mock_list_item = {
            "session_id": session_id,
            "state": "idle",
            "participant_id": "chaz",
            "participant_type": "human",
        }

        with patch("session_registry.get_default_registry") as mock_get_reg:
            mock_reg = mock_get_reg.return_value
            mock_reg.list_serialized.return_value = [mock_list_item]
            mock_reg.get.return_value = mock_session

            with client.websocket_connect("/ws/acp") as ws:
                _do_initialize(ws)

                # Step 1: list
                ws.send_text(_req("session/list", {}, id=1))
                list_resp = _parse(ws.receive_text())
                listed_id = list_resp["result"]["sessions"][0]["sessionId"]
                assert listed_id == session_id

                # Step 2: load the session we just found
                ws.send_text(_req("session/load", {"sessionId": listed_id}, id=2))
                load_resp = _parse(ws.receive_text())
                assert load_resp["result"]["sessionId"] == session_id
                assert load_resp["result"]["state"]["session_id"] == session_id

                # Step 3: resume it
                ws.send_text(_req("session/resume", {"sessionId": listed_id}, id=3))
                resume_resp = _parse(ws.receive_text())
                assert resume_resp["result"]["sessionId"] == session_id

"""Tests for the 'main' session auto-create on server startup.

Fix 1 in feat/session-auto-create-and-acp-wire-fix:

channel_client.py uses _DEFAULT_SESSION_ID = "main" (clients/channel_client.py:79).
Before this fix, mod3 did not create the 'main' session on startup, so:
  - GET /v1/sessions returned {sessions: []} even with channel clients running.
  - POST /v1/sessions/main/seats succeeded (SeatRegistry auto-creates the bucket),
    but the session was not visible in the voice-TTS session_registry.
  - The dashboard showed "No active sessions" despite active channel clients.

After the fix, the _lifespan startup hook calls session_registry.register('main')
before yielding, so /v1/sessions returns the session immediately and the dashboard
shows it correctly.

Run with: PYTHONPATH=. .venv/bin/python -m pytest tests/test_session_auto_create.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """FastAPI TestClient that exercises the full app lifespan (startup + shutdown)."""
    from fastapi.testclient import TestClient

    import http_api

    # TestClient wraps the app's lifespan, so startup hooks run on __enter__.
    with TestClient(http_api.app, base_url="http://localhost:7860") as c:
        yield c


# ---------------------------------------------------------------------------
# Test: GET /v1/sessions returns 'main' after startup
# ---------------------------------------------------------------------------


class TestMainSessionAutoCreate:
    def test_get_sessions_returns_main_after_startup(self, client):
        """After startup, GET /v1/sessions must include the 'main' session."""
        resp = client.get("/v1/sessions")
        assert resp.status_code == 200
        data = resp.json()
        session_ids = [s["session_id"] for s in data["sessions"]]
        assert "main" in session_ids, f"Expected 'main' in sessions after startup, got: {session_ids}"

    def test_get_session_main_returns_200(self, client):
        """GET /v1/sessions/main must return 200 (not 404) after startup."""
        resp = client.get("/v1/sessions/main")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "main"

    def test_post_seat_to_main_succeeds(self, client):
        """POST /v1/sessions/main/seats must succeed after startup.

        Before the fix this would silently succeed at the SeatRegistry level
        but the main session would not appear in /v1/sessions. After the fix
        the session is pre-registered and both endpoints work consistently.

        access.py allows localhost unconditionally (policy=self). TestClient
        sets the peer to testclient (127.0.0.1), which is_allowed() treats as
        localhost-allowed. We patch is_allowed to True to avoid filesystem
        access during test runs.
        """
        from unittest.mock import patch

        with patch("access.is_allowed", return_value=True):
            resp = client.post(
                "/v1/sessions/main/seats",
                json={"client_type": "generic", "device_uuid": "test-device-001"},
            )
        # 200 or 201 are both acceptable success shapes
        assert resp.status_code in (200, 201), (
            f"Expected success registering seat in 'main', got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data.get("session_id") == "main"
        assert "seat_id" in data

    def test_auto_create_is_idempotent(self, client):
        """Re-creating 'main' (simulating server restart) must not fail or
        change the session identity."""
        from session_registry import get_default_registry

        reg = get_default_registry()
        before = reg.get("main")
        assert before is not None, "main session must exist after startup"

        # Re-register — same as what would happen on a restart with an existing
        # in-memory registry (e.g. reload in dev mode).
        result = reg.register(
            session_id="main",
            participant_id="channel-client-pool",
            participant_type="agent",
        )
        assert result.created is False, "Re-registering 'main' must return created=False (idempotent)"
        assert result.session.session_id == "main"

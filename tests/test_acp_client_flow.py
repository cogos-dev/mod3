"""E2E-level flow test for the ACP-client session-browser pattern.

Tests the spawn → seat-attach sequence at the HTTP layer without actually
starting a Claude Code subprocess (mocked out). The test validates that:

  1. POST /v1/claude-code/spawn proxies to the kernel and returns the kernel
     response verbatim.
  2. GET /dashboard/sessions.html is served from the dashboard directory.
  3. The spawn proxy surfaces kernel errors as 503 when the kernel is
     unreachable.

The channel-client seat-attach loop (seat registers via
POST /v1/sessions/{id}/seats → SSE stream opens → events arrive) is covered
by the existing seat-registry tests. This file covers the spawn proxy layer
and the dashboard static-file delivery, which are the new code paths
introduced in PR #43 (Sessions browser).

Run with:
  PYTHONPATH=. pytest tests/test_acp_client_flow.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixture — FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import http_api  # noqa: PLC0415

    return TestClient(http_api.app, base_url="http://localhost:7860")


# ---------------------------------------------------------------------------
# Dashboard static delivery
# ---------------------------------------------------------------------------


class TestDashboardStatic:
    def test_sessions_html_served(self, client):
        """GET /dashboard/sessions.html should return 200 with HTML content."""
        r = client.get("/dashboard/sessions.html")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        # Confirm it is the sessions page (not the main index)
        assert "Sessions" in r.text

    def test_index_html_served(self, client):
        """GET /dashboard still returns the main index.html."""
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_path_traversal_rejected(self, client):
        """GET /dashboard/../../etc/passwd should return 400."""
        r = client.get("/dashboard/../../etc/passwd")
        assert r.status_code in (400, 404)

    def test_missing_file_404(self, client):
        """GET /dashboard/nonexistent.html should return 404."""
        r = client.get("/dashboard/nonexistent.html")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Spawn proxy — kernel reachable
# ---------------------------------------------------------------------------


class TestSpawnProxy:
    def test_spawn_proxies_to_kernel_and_returns_response(self, client):
        """POST /v1/claude-code/spawn should proxy to the kernel and return its body."""
        kernel_response = {
            "process_id": "test-proc-abc123",
            "session_id": "sess-xyz",
            "project": "-Users-slowbro",
            "status": "spawned",
            "spawned_at": "2026-05-14T22:00:00Z",
        }

        # Mock httpx.AsyncClient to avoid real network call to localhost:6931
        mock_response = AsyncMock()
        mock_response.content = json.dumps(kernel_response).encode()
        mock_response.status_code = 201

        async def _mock_post(*args, **kwargs):
            return mock_response

        mock_client_instance = AsyncMock()
        mock_client_instance.post = _mock_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            r = client.post(
                "/v1/claude-code/spawn",
                json={"project": "-Users-slowbro", "session_id": "sess-xyz"},
            )

        assert r.status_code == 201
        body = r.json()
        assert body["process_id"] == "test-proc-abc123"
        assert body["status"] == "spawned"

    def test_spawn_without_session_id_proxies_fresh_mount(self, client):
        """POST /v1/claude-code/spawn without session_id should proxy a fresh mount."""
        kernel_response = {
            "process_id": "proc-fresh-001",
            "status": "spawned",
            "spawned_at": "2026-05-14T22:01:00Z",
        }

        mock_response = AsyncMock()
        mock_response.content = json.dumps(kernel_response).encode()
        mock_response.status_code = 201

        async def _mock_post(*args, **kwargs):
            return mock_response

        mock_client_instance = AsyncMock()
        mock_client_instance.post = _mock_post
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            r = client.post("/v1/claude-code/spawn", json={})

        assert r.status_code == 201
        assert r.json()["status"] == "spawned"


# ---------------------------------------------------------------------------
# Spawn proxy — kernel unreachable
# ---------------------------------------------------------------------------


class TestSpawnProxyKernelDown:
    def test_spawn_returns_503_when_kernel_unreachable(self, client):
        """POST /v1/claude-code/spawn should return 503 when the kernel is down."""
        with patch(
            "httpx.AsyncClient",
            side_effect=lambda **kw: _connect_error_client(),
        ):
            r = client.post(
                "/v1/claude-code/spawn",
                json={"project": "-Users-slowbro"},
            )

        assert r.status_code == 503
        body = r.json()
        assert "error" in body
        assert body["error"]["type"] == "kernel_unavailable"


def _connect_error_client():
    """Returns a mock AsyncClient that raises ConnectError on post."""
    import httpx

    class _ErrClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("connection refused")

    return _ErrClient()


# ---------------------------------------------------------------------------
# Seat-attach registry unit smoke (without HTTP layer)
# ---------------------------------------------------------------------------


class TestSeatRegistrySmoke:
    """Validate the seat registry directly (no HTTP).

    The seat-registration HTTP endpoint is an async FastAPI handler that
    constructs asyncio.Queue objects — calling it from a synchronous
    TestClient can trigger loop lifecycle issues on Python 3.10.  The
    underlying SeatRegistry is synchronous and testable directly.

    This test confirms that after a spawn call returns a session_id, that
    session_id is a valid key for seat registration — the registry accepts
    it without preconditions, which is the contract the ACP-client pattern
    depends on.
    """

    def test_registry_accepts_spawn_derived_session_id(self):
        """SeatRegistry.register should accept any session_id without precondition."""
        from seats import SeatRegistry

        registry = SeatRegistry()
        # The session_id that the spawn endpoint would return
        session_id = "acp-test-session-001"
        seat = registry.register(
            session_id=session_id,
            client_type="claude-code-channel",
            device_uuid="test-device-acp-001",
        )
        assert seat.seat_id
        assert seat.session_id == session_id
        assert seat.client_type == "claude-code-channel"

        # Revoke
        removed = registry.revoke(session_id, seat.seat_id)
        assert removed is True
        # Confirm it's gone
        assert registry.get(session_id, seat.seat_id) is None

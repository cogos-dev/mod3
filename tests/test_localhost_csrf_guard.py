"""Tests for the localhost CSRF / DNS-rebinding guard middleware.

Covers:
  * State-changing requests (POST) with a disallowed Origin are rejected 403.
  * State-changing requests (POST) with an allowed localhost Origin pass.
  * State-changing requests with NO Origin header pass (non-browser / same-origin).
  * State-changing requests with a disallowed Host header (DNS rebinding) are rejected 403.
  * State-changing requests with an allowed localhost Host pass.
  * Read-only GET requests are NOT gated (pass regardless of Origin / Host).

FastAPI's TestClient sends "Host: testserver" by default. Tests that exercise
Origin checking must also supply an allowed Host header so the Host check does
not pre-empt the Origin check.  Tests that exercise Host checking supply an
explicit Host header to override "testserver".

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_localhost_csrf_guard.py -v
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
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clean_seats():
    """Keep seat registry clean between tests."""
    from seats import get_seat_registry

    reg = get_seat_registry()
    with reg._lock:
        reg._seats.clear()
    yield
    with reg._lock:
        reg._seats.clear()


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

_MUTATING_ENDPOINT = "/v1/sessions/broadcast-message"
_MUTATING_BODY = {"content": "hello", "role": "user"}

# A Host header that passes the localhost check.
_GOOD_HOST = "localhost:7860"

_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:7860",
    "http://127.0.0.1",
    "http://127.0.0.1:7860",
]

_DISALLOWED_ORIGINS = [
    "https://evil.com",
    "http://evil.com",
    "http://evil.example",
    "http://not-localhost",
    "null",
]


# ---------------------------------------------------------------------------
# Origin / CSRF tests
# (always set a good Host so Host check does not pre-empt Origin check)
# ---------------------------------------------------------------------------


class TestOriginGuard:
    def test_disallowed_origin_is_rejected(self, client):
        """A cross-origin POST with a disallowed Origin returns 403."""
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            headers={"Host": _GOOD_HOST, "Origin": "https://evil.com"},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert data["error"] == "forbidden"
        # The detail must mention the disallowed origin.
        assert "evil.com" in data["detail"] or "Origin" in data["detail"]

    @pytest.mark.parametrize("disallowed", _DISALLOWED_ORIGINS)
    def test_various_disallowed_origins_rejected(self, client, disallowed):
        """All known bad origins are rejected with 403."""
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            headers={"Host": _GOOD_HOST, "Origin": disallowed},
        )
        assert resp.status_code == 403, (
            f"Expected 403 for Origin={disallowed!r}, got {resp.status_code}"
        )

    @pytest.mark.parametrize("allowed", _ALLOWED_ORIGINS)
    def test_allowed_localhost_origin_passes(self, client, allowed):
        """Requests from localhost origins (dashboard) are accepted by the guard."""
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            headers={"Host": _GOOD_HOST, "Origin": allowed},
        )
        # 200 = ok, 400 = bad payload — either means the guard passed.
        # 403 would mean the guard wrongly blocked it.
        assert resp.status_code != 403, (
            f"Guard incorrectly rejected allowed Origin={allowed!r}: {resp.status_code}"
        )

    def test_no_origin_header_passes(self, client):
        """Requests without an Origin header (non-browser / same-origin) are not blocked.

        This covers the channel_client (httpx) use-case: programmatic
        non-browser clients do not send Origin and must not be blocked.
        """
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            headers={"Host": _GOOD_HOST},
            # No Origin header.
        )
        assert resp.status_code != 403, (
            f"Guard incorrectly blocked request with no Origin: {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# Host (DNS-rebinding) tests
# ---------------------------------------------------------------------------


class TestHostGuard:
    def test_disallowed_host_is_rejected(self, client):
        """A POST with a non-localhost Host header is rejected 403."""
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            headers={"Host": "evil.com"},
        )
        assert resp.status_code == 403
        data = resp.json()
        assert data["error"] == "forbidden"
        assert "Host" in data["detail"] or "localhost" in data["detail"]

    def test_testserver_host_is_rejected(self, client):
        """The TestClient default host 'testserver' is not a localhost address."""
        # This test sends the raw TestClient default (no Host override) to
        # confirm the guard correctly rejects it.
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            # TestClient injects "Host: testserver"
        )
        assert resp.status_code == 403

    @pytest.mark.parametrize(
        "good_host",
        [
            "localhost",
            "localhost:7860",
            "127.0.0.1",
            "127.0.0.1:7860",
        ],
    )
    def test_localhost_host_passes(self, client, good_host):
        """Requests with a localhost Host header are not blocked by the Host check."""
        resp = client.post(
            _MUTATING_ENDPOINT,
            json=_MUTATING_BODY,
            headers={"Host": good_host},
        )
        assert resp.status_code != 403, (
            f"Guard incorrectly rejected Host={good_host!r}: {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# Read-only paths are not gated
# ---------------------------------------------------------------------------


class TestReadOnlyPaths:
    def test_get_health_passes_bad_origin(self, client):
        """GET /health is never gated."""
        resp = client.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code == 200

    def test_get_voices_passes_bad_origin(self, client):
        """GET /v1/voices is not gated."""
        resp = client.get(
            "/v1/voices",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code != 403

    def test_get_sessions_passes_bad_origin(self, client):
        """GET /v1/sessions is not gated."""
        resp = client.get(
            "/v1/sessions",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code != 403

    def test_get_jobs_passes_bad_origin(self, client):
        """GET /v1/jobs is not gated."""
        resp = client.get(
            "/v1/jobs",
            headers={"Origin": "https://evil.com"},
        )
        assert resp.status_code != 403

    def test_get_health_passes_bad_host(self, client):
        """GET /health passes even with a bad Host header."""
        resp = client.get(
            "/health",
            headers={"Host": "evil.com"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Seat registration endpoint
# ---------------------------------------------------------------------------


class TestSeatRegistration:
    def test_seat_register_disallowed_origin_rejected(self, client):
        """POST /v1/sessions/{id}/seats from a disallowed origin is blocked."""
        resp = client.post(
            "/v1/sessions/main/seats",
            json={"client_type": "generic", "device_uuid": "test-device"},
            headers={"Host": _GOOD_HOST, "Origin": "https://evil.com"},
        )
        assert resp.status_code == 403

    def test_seat_register_no_origin_localhost_host_passes_guard(self, client):
        """POST /v1/sessions/{id}/seats with no Origin and localhost Host passes the
        CSRF guard (access.py may then apply its own policy — separate layer)."""
        resp = client.post(
            "/v1/sessions/main/seats",
            json={"client_type": "generic", "device_uuid": "test-device"},
            headers={"Host": _GOOD_HOST},
            # No Origin — simulates channel_client (httpx)
        )
        # The CSRF guard must NOT return 403; other layers may return their own status.
        assert resp.status_code != 403, (
            f"CSRF guard wrongly blocked seat registration with no Origin: {resp.status_code}"
        )

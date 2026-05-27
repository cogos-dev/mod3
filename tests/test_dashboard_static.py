"""Regression tests for dashboard static route serving.

Guards against the trailing-slash 404 bug: GET /dashboard/ must return the
same index.html content as GET /dashboard and GET /dashboard/index.html.

Also covers that named subpages (sessions.html, console.html, voice-lab.html)
continue to be served correctly.

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_dashboard_static.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860")


# ---------------------------------------------------------------------------
# /dashboard trailing-slash parity
# ---------------------------------------------------------------------------


class TestDashboardTrailingSlash:
    def test_dashboard_no_slash_returns_200(self, client):
        """GET /dashboard returns 200."""
        resp = client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 200, f"expected 200, got {resp.status_code}"

    def test_dashboard_trailing_slash_returns_200(self, client):
        """GET /dashboard/ returns 200 (regression: was 404)."""
        resp = client.get("/dashboard/", follow_redirects=False)
        assert resp.status_code == 200, f"expected 200, got {resp.status_code}"

    def test_dashboard_index_html_returns_200(self, client):
        """GET /dashboard/index.html returns 200."""
        resp = client.get("/dashboard/index.html", follow_redirects=False)
        assert resp.status_code == 200, f"expected 200, got {resp.status_code}"

    def test_dashboard_slash_and_no_slash_same_body(self, client):
        """/dashboard and /dashboard/ serve identical content."""
        resp_no_slash = client.get("/dashboard")
        resp_slash = client.get("/dashboard/")
        assert resp_no_slash.status_code == 200
        assert resp_slash.status_code == 200
        assert resp_no_slash.content == resp_slash.content, "/dashboard and /dashboard/ returned different bodies"

    # -----------------------------------------------------------------------
    # Subpages — must still work
    # -----------------------------------------------------------------------

    def test_sessions_html_still_served(self, client):
        """GET /dashboard/sessions.html continues to return 200."""
        resp = client.get("/dashboard/sessions.html")
        assert resp.status_code == 200

    def test_console_html_still_served(self, client):
        """GET /dashboard/console.html continues to return 200."""
        resp = client.get("/dashboard/console.html")
        assert resp.status_code == 200

    def test_voice_lab_html_still_served(self, client):
        """GET /dashboard/voice-lab.html continues to return 200."""
        resp = client.get("/dashboard/voice-lab.html")
        assert resp.status_code == 200

    def test_path_traversal_rejected(self, client):
        """Path traversal attempts do not leak files outside dashboard/.

        The HTTP layer normalizes /dashboard/../etc/passwd to /etc/passwd before
        the route handler sees it, so it never matches /dashboard/{filename} and
        returns 404. Either 400 (explicit rejection) or 404 (no route match) is
        acceptable — the important invariant is not-200.
        """
        resp = client.get("/dashboard/../etc/passwd", follow_redirects=False)
        assert resp.status_code in (400, 404), f"expected 400 or 404 for path traversal, got {resp.status_code}"

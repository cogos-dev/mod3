"""pytest configuration — shared fixtures for the mod3 test suite.

The ``client`` fixture here creates a ``TestClient`` with
``base_url="http://localhost:7860"`` so that all test requests carry
``Host: localhost:7860``.  This is required because the localhost CSRF guard
rejects requests whose Host header is not a loopback address, and Starlette's
``TestClient`` defaults to ``Host: testserver`` which is correctly refused.

Individual test modules that define their own ``client`` fixture will shadow
this one for their own tests — that is intentional for tests that need to
exercise the guard directly (e.g. ``test_localhost_csrf_guard.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def client():
    """TestClient pre-configured with a localhost base_url.

    Sends ``Host: localhost:7860`` on every request so the localhost
    CSRF guard does not reject test traffic as a DNS-rebinding attempt.
    """
    from fastapi.testclient import TestClient

    import http_api

    return TestClient(http_api.app, base_url="http://localhost:7860", raise_server_exceptions=False)

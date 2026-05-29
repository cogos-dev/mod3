"""Tests for seat-registration → SessionRegistry mirror.

Before this fix, mod3 only seeded the legacy ``"main"`` session into the
voice-TTS ``SessionRegistry`` at startup. Channel clients that registered
seats under a *different* session_id (e.g. a real Claude Code session UUID
per PR #103's binding) appeared at ``GET /v1/sessions/{id}/seats`` but were
absent from the top-level ``GET /v1/sessions`` roster. Symptom: dashboard
sidebar reported "No active sessions" while a live Claude Code channel
client was attached.

After this fix, ``POST /v1/sessions/{id}/seats`` mirrors the session into
``SessionRegistry`` after the seat is registered. The mirror is idempotent
(``SessionRegistry.register`` preserves existing voice allocation), so
multiple seats under the same session don't reshuffle the voice.

Run with: ``PYTHONPATH=. .venv/bin/python -m pytest tests/test_seat_session_mirror.py -v``
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    with TestClient(http_api.app, base_url="http://localhost:7860") as c:
        yield c


class TestSeatSessionMirror:
    def test_seat_register_mirrors_session_into_registry(self, client):
        """POST /v1/sessions/<uuid>/seats must populate /v1/sessions with the
        same session_id, so the dashboard sidebar enumerates it."""
        sid = str(uuid.uuid4())

        with patch("access.is_allowed", return_value=True):
            resp = client.post(
                f"/v1/sessions/{sid}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": sid},
            )
        assert resp.status_code in (200, 201), resp.text

        listing = client.get("/v1/sessions").json()
        ids = [s["session_id"] for s in listing["sessions"]]
        assert sid in ids, f"seat-bearing session {sid} must appear in /v1/sessions, got {ids}"

        detail = client.get(f"/v1/sessions/{sid}").json()
        assert detail["session_id"] == sid
        assert detail["participant_type"] == "agent"
        assert detail["participant_id"].startswith("channel-client::")

    def test_mirror_is_idempotent_across_multiple_seats(self, client):
        """Multiple seats under the same session_id must not reshuffle the
        assigned voice (SessionRegistry.register is idempotent on re-register)."""
        sid = str(uuid.uuid4())

        with patch("access.is_allowed", return_value=True):
            r1 = client.post(
                f"/v1/sessions/{sid}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": sid},
            )
            assert r1.status_code in (200, 201)
            voice_after_first = client.get(f"/v1/sessions/{sid}").json()["assigned_voice"]

            r2 = client.post(
                f"/v1/sessions/{sid}/seats",
                json={"client_type": "generic", "device_uuid": "second-device"},
            )
            assert r2.status_code in (200, 201)
            voice_after_second = client.get(f"/v1/sessions/{sid}").json()["assigned_voice"]

        assert voice_after_first == voice_after_second, (
            "voice must be stable across repeat seat-registrations under the same session_id"
        )

    def test_main_session_unaffected(self, client):
        """The startup-seeded 'main' session must continue to appear regardless
        of any other seat-bearing sessions."""
        sid = str(uuid.uuid4())
        with patch("access.is_allowed", return_value=True):
            client.post(
                f"/v1/sessions/{sid}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": sid},
            )

        ids = [s["session_id"] for s in client.get("/v1/sessions").json()["sessions"]]
        assert "main" in ids
        assert sid in ids

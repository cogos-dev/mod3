"""Tests for Primitive 2: multi-identity harness binding.

Covers:
  * Schema: assistant_iss/assistant_sub fields on SessionRegisterRequest
  * Schema: backward compat — legacy iss/sub-only callers still work
  * Schema: anonymous registration (no identity fields) — no error
  * Seat dataclass: to_dict() includes all four identity fields
  * SeatRegistry.register(): assistant_iss/sub stored on Seat
  * SeatRegistry.register(): legacy iss/sub args map to user_iss/user_sub
  * HTTP POST /v1/sessions/{session_id}/seats: agentic registration
    with all four fields → Seat.assistant_sub set + presence.started event
    includes both pairs
  * HTTP POST /v1/sessions/{session_id}/seats: user-only (iss/sub, no assistant)
    → backward compat, assistant fields None, event includes None assistant fields
  * HTTP POST /v1/sessions/{session_id}/seats: anonymous (no identity claims)
    → no error, no presence.started event
  * Non-agentic seat (client_type="generic") with assistant claims → claims stored
    but seat is generic (no enforcement; policy is declarative not prescriptive in v1)

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_multi_identity_binding.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seats import Seat, SeatRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain(seat: Seat) -> list[dict]:
    """Pull all pending events from a seat's queue (non-blocking)."""
    items: list[dict] = []
    while True:
        try:
            items.append(seat.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# ---------------------------------------------------------------------------
# Schema: SessionRegisterRequest
# ---------------------------------------------------------------------------


class TestSessionRegisterRequestMultiIdentity:
    """Verify the schema accepts / defaults the new assistant_* fields."""

    def test_assistant_fields_default_none(self):
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(session_id="s1", participant_id="cog")
        assert req.assistant_iss is None
        assert req.assistant_sub is None

    def test_assistant_fields_set(self):
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(
            session_id="s1",
            participant_id="chaz",
            iss="cogos-dev",
            sub="chaz",
            assistant_iss="cogos-dev",
            assistant_sub="cog",
        )
        assert req.iss == "cogos-dev"
        assert req.sub == "chaz"
        assert req.assistant_iss == "cogos-dev"
        assert req.assistant_sub == "cog"

    def test_legacy_user_only_no_error(self):
        """Pre-Primitive-2 callers that pass only iss/sub get no error."""
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(
            session_id="s1",
            participant_id="chaz",
            iss="cogos-dev",
            sub="chaz",
        )
        assert req.iss == "cogos-dev"
        assert req.sub == "chaz"
        assert req.assistant_iss is None
        assert req.assistant_sub is None

    def test_anonymous_registration_no_error(self):
        """Callers with no identity claims whatsoever are backward compatible."""
        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(session_id="s1", participant_id="cog")
        assert req.iss is None
        assert req.sub is None
        assert req.assistant_iss is None
        assert req.assistant_sub is None

    def test_all_four_fields_in_json(self):
        import json

        from schemas.http.sessions import SessionRegisterRequest

        req = SessionRegisterRequest(
            session_id="s1",
            participant_id="chaz",
            iss="cogos-dev",
            sub="chaz",
            assistant_iss="cogos-dev",
            assistant_sub="cog",
        )
        d = json.loads(req.model_dump_json())
        assert d["iss"] == "cogos-dev"
        assert d["sub"] == "chaz"
        assert d["assistant_iss"] == "cogos-dev"
        assert d["assistant_sub"] == "cog"


# ---------------------------------------------------------------------------
# Seat dataclass: to_dict()
# ---------------------------------------------------------------------------


class TestSeatToDict:
    def test_to_dict_includes_all_four_identity_fields_when_set(self):
        seat = Seat(
            seat_id="s",
            session_id="sess",
            client_type="claude-code-channel",
            device_uuid="dev",
            user_iss="cogos-dev",
            user_sub="chaz",
            assistant_iss="cogos-dev",
            assistant_sub="cog",
        )
        d = seat.to_dict()
        assert d["user_iss"] == "cogos-dev"
        assert d["user_sub"] == "chaz"
        assert d["assistant_iss"] == "cogos-dev"
        assert d["assistant_sub"] == "cog"

    def test_to_dict_includes_all_four_fields_as_none_when_absent(self):
        """All four identity keys are always present in the dict (None when unset)."""
        seat = Seat(
            seat_id="s",
            session_id="sess",
            client_type="generic",
            device_uuid="dev",
        )
        d = seat.to_dict()
        assert "user_iss" in d
        assert "user_sub" in d
        assert "assistant_iss" in d
        assert "assistant_sub" in d
        assert d["user_iss"] is None
        assert d["assistant_sub"] is None

    def test_to_dict_includes_base_fields(self):
        seat = Seat(
            seat_id="seat-abc",
            session_id="sess-xyz",
            client_type="generic",
            device_uuid="dev-001",
        )
        d = seat.to_dict()
        assert d["seat_id"] == "seat-abc"
        assert d["session_id"] == "sess-xyz"
        assert d["client_type"] == "generic"
        assert d["device_uuid"] == "dev-001"
        assert "created_at" in d


# ---------------------------------------------------------------------------
# SeatRegistry.register()
# ---------------------------------------------------------------------------


class TestSeatRegistryRegisterIdentity:
    def test_agentic_registration_stores_all_four_fields(self):
        reg = SeatRegistry()
        seat = reg.register(
            session_id="sess",
            client_type="claude-code-channel",
            device_uuid="dev-cc",
            user_iss="cogos-dev",
            user_sub="chaz",
            assistant_iss="cogos-dev",
            assistant_sub="cog",
        )
        assert seat.user_iss == "cogos-dev"
        assert seat.user_sub == "chaz"
        assert seat.assistant_iss == "cogos-dev"
        assert seat.assistant_sub == "cog"

    def test_legacy_iss_sub_maps_to_user_fields(self):
        """Wave-6b callers passing iss/sub have those mapped to user_iss/user_sub."""
        reg = SeatRegistry()
        seat = reg.register(
            session_id="sess",
            client_type="generic",
            device_uuid="dev",
            iss="cogos-dev",
            sub="chaz",
        )
        assert seat.user_iss == "cogos-dev"
        assert seat.user_sub == "chaz"
        assert seat.assistant_iss is None
        assert seat.assistant_sub is None

    def test_anonymous_registration_no_error(self):
        reg = SeatRegistry()
        seat = reg.register(
            session_id="sess",
            client_type="generic",
            device_uuid="dev",
        )
        assert seat.user_iss is None
        assert seat.user_sub is None
        assert seat.assistant_iss is None
        assert seat.assistant_sub is None

    def test_assistant_sub_none_for_non_agentic_no_claims(self):
        """A generic seat with no assistant claims has None assistant fields."""
        reg = SeatRegistry()
        seat = reg.register(
            session_id="sess",
            client_type="generic",
            device_uuid="dev",
            user_iss="cogos-dev",
            user_sub="chaz",
            # no assistant_iss / assistant_sub
        )
        assert seat.assistant_iss is None
        assert seat.assistant_sub is None


# ---------------------------------------------------------------------------
# HTTP integration: POST /v1/sessions/{session_id}/seats
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with access control bypassed (localhost is not the TestClient host)."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    import http_api

    # access.py's is_allowed() rejects TestClient connections because the
    # TestClient host is "testclient", not "127.0.0.1". Patch the import so
    # these tests focus on identity binding, not access policy.
    with patch.dict("sys.modules", {"access": None}):
        yield TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture(autouse=True)
def _clean_seats():
    """Ensure each test starts with a fresh seat registry singleton."""
    from seats import get_seat_registry

    reg = get_seat_registry()
    with reg._lock:
        reg._seats.clear()
    yield
    with reg._lock:
        reg._seats.clear()


class TestSeatRegisterEndpointMultiIdentity:
    def test_agentic_registration_sets_assistant_sub(self, client):
        """POST with all four identity fields → Seat.assistant_sub is set."""
        from seats import get_seat_registry

        resp = client.post(
            "/v1/sessions/sess-agent/seats",
            json={
                "client_type": "claude-code-channel",
                "device_uuid": "cc-session-abc",
                "user_iss": "cogos-dev",
                "user_sub": "chaz",
                "assistant_iss": "cogos-dev",
                "assistant_sub": "cog",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        seat_id = data["seat_id"]

        reg = get_seat_registry()
        seat = reg.get("sess-agent", seat_id)
        assert seat is not None
        assert seat.assistant_sub == "cog"
        assert seat.user_sub == "chaz"

    def test_agentic_registration_emits_presence_started_with_both_pairs(self, client):
        """POST with all four fields → presence.started event contains both pairs."""
        from seats import get_seat_registry

        # Register a second seat first so it receives the presence.started event.
        observer = get_seat_registry().register(
            session_id="sess-agent",
            client_type="generic",
            device_uuid="observer",
        )

        resp = client.post(
            "/v1/sessions/sess-agent/seats",
            json={
                "client_type": "claude-code-channel",
                "device_uuid": "cc-session-abc",
                "user_iss": "cogos-dev",
                "user_sub": "chaz",
                "assistant_iss": "cogos-dev",
                "assistant_sub": "cog",
            },
        )
        assert resp.status_code == 200

        events = _drain(observer)
        presence_events = [e for e in events if e.get("type") == "presence.started"]
        assert len(presence_events) == 1, f"Expected 1 presence.started, got: {events}"

        ev = presence_events[0]
        assert ev["user_sub"] == "chaz"
        assert ev["assistant_sub"] == "cog"
        assert ev["user_iss"] == "cogos-dev"
        assert ev["assistant_iss"] == "cogos-dev"

    def test_legacy_user_only_backward_compat(self, client):
        """POST with only iss/sub (no assistant fields) → no crash, assistant fields None."""
        from seats import get_seat_registry

        resp = client.post(
            "/v1/sessions/sess-legacy/seats",
            json={
                "client_type": "generic",
                "device_uuid": "dev-legacy",
                "iss": "cogos-dev",
                "sub": "chaz",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        seat_id = data["seat_id"]

        reg = get_seat_registry()
        seat = reg.get("sess-legacy", seat_id)
        assert seat is not None
        assert seat.user_sub == "chaz"
        assert seat.assistant_iss is None
        assert seat.assistant_sub is None

    def test_legacy_user_only_presence_started_no_assistant_fields(self, client):
        """User-only presence.started event includes assistant fields as None."""
        from seats import get_seat_registry

        observer = get_seat_registry().register(
            session_id="sess-legacy",
            client_type="generic",
            device_uuid="observer",
        )

        client.post(
            "/v1/sessions/sess-legacy/seats",
            json={
                "client_type": "generic",
                "device_uuid": "dev",
                "iss": "cogos-dev",
                "sub": "chaz",
            },
        )

        events = _drain(observer)
        presence_events = [e for e in events if e.get("type") == "presence.started"]
        assert len(presence_events) == 1
        ev = presence_events[0]
        assert ev["user_sub"] == "chaz"
        assert ev["assistant_iss"] is None
        assert ev["assistant_sub"] is None

    def test_anonymous_registration_no_presence_started(self, client):
        """POST with no identity claims → no presence.started event emitted."""
        from seats import get_seat_registry

        observer = get_seat_registry().register(
            session_id="sess-anon",
            client_type="generic",
            device_uuid="observer",
        )

        resp = client.post(
            "/v1/sessions/sess-anon/seats",
            json={
                "client_type": "generic",
                "device_uuid": "anon-dev",
            },
        )
        assert resp.status_code == 200

        events = _drain(observer)
        presence_events = [e for e in events if e.get("type") == "presence.started"]
        assert len(presence_events) == 0, f"Unexpected presence.started on anonymous seat: {presence_events}"

    def test_generic_client_type_with_no_assistant_claims(self, client):
        """Non-agentic seat registered with no assistant_iss/sub → assistant fields remain None."""
        from seats import get_seat_registry

        resp = client.post(
            "/v1/sessions/sess-generic/seats",
            json={
                "client_type": "generic",
                "device_uuid": "dev-generic",
                "iss": "cogos-dev",
                "sub": "chaz",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        seat_id = data["seat_id"]

        reg = get_seat_registry()
        seat = reg.get("sess-generic", seat_id)
        assert seat is not None
        assert seat.client_type == "generic"
        assert seat.assistant_iss is None
        assert seat.assistant_sub is None

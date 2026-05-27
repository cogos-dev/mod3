"""Tests for Fix A — mod3 seat-registration → kernel ChannelSessionRegistry callback.

After this fix, ``POST /v1/sessions/{id}/seats`` fires a best-effort
``POST {COGOS_KERNEL_URL}/v1/channel-sessions/register`` callback so the
kernel's ChannelSessionRegistry stays authoritative (closes the kernel:1 /
mod3:99 skew from cog://mem/semantic/insights/cross-session-tapestry-2026-05-26).

Symmetrically, ``DELETE /v1/sessions/{id}/seats/{seat_id}`` fires
``POST {kernel}/v1/channel-sessions/{session_id}/deregister`` so the kernel
drops its record on seat revoke.

Both callbacks are best-effort (wrapped in try/except); a kernel failure must
never block the seat operation.

Run with::

    PYTHONPATH=. python -m pytest tests/test_kernel_session_callback.py -v
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    with TestClient(http_api.app) as c:
        yield c


class TestKernelRegisterCallback:
    """POST /v1/sessions/{id}/seats → kernel /v1/channel-sessions/register."""

    def test_seat_register_fires_kernel_callback(self, client):
        """A successful seat registration must fire a POST to the kernel's
        /v1/channel-sessions/register endpoint, passing the EXISTING session_id
        so the kernel does not re-mint."""
        session_id = str(uuid.uuid4())

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("access.is_allowed", return_value=True), \
             patch("httpx.post", return_value=mock_response) as mock_post:
            resp = client.post(
                f"/v1/sessions/{session_id}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": session_id},
            )

        assert resp.status_code in (200, 201), resp.text

        # The kernel callback must have been called.
        assert mock_post.called, "httpx.post must be called to notify kernel"

        # Find the kernel register call (not any other httpx call).
        kernel_calls = [
            call for call in mock_post.call_args_list
            if "/v1/channel-sessions/register" in str(call)
        ]
        assert len(kernel_calls) >= 1, (
            f"Expected at least one call to /v1/channel-sessions/register, "
            f"got calls: {mock_post.call_args_list}"
        )

        # The payload must pass the EXISTING session_id — never let the kernel re-mint.
        call_kwargs = kernel_calls[0].kwargs
        payload = call_kwargs.get("json", {})
        assert payload.get("session_id") == session_id, (
            f"kernel callback must pass the existing session_id={session_id!r}, "
            f"got payload: {payload}"
        )
        assert payload.get("participant_id"), "participant_id must be set in kernel callback"

    def test_kernel_callback_failure_does_not_block_seat_register(self, client):
        """If the kernel is unreachable, seat registration must still succeed."""
        session_id = str(uuid.uuid4())

        with patch("access.is_allowed", return_value=True), \
             patch("httpx.post", side_effect=Exception("kernel unreachable")):
            resp = client.post(
                f"/v1/sessions/{session_id}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": session_id},
            )

        # Seat registration must succeed regardless of kernel callback failure.
        assert resp.status_code in (200, 201), (
            f"seat register must succeed even when kernel is down; got {resp.status_code}: {resp.text}"
        )
        data = resp.json()
        assert data.get("seat_id"), "response must include seat_id"
        assert data.get("session_id") == session_id

    def test_kernel_callback_passes_iss_sub_when_present(self, client):
        """Identity claims (user_iss/user_sub) must be forwarded to the kernel as
        iss/sub so the ChannelSessionRecord carries the WHO."""
        session_id = str(uuid.uuid4())

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("access.is_allowed", return_value=True), \
             patch("httpx.post", return_value=mock_response) as mock_post:
            resp = client.post(
                f"/v1/sessions/{session_id}/seats",
                json={
                    "client_type": "claude-code-channel",
                    "device_uuid": session_id,
                    "user_iss": "https://cogos.local",
                    "user_sub": "chaz",
                },
            )

        assert resp.status_code in (200, 201), resp.text

        kernel_calls = [
            call for call in mock_post.call_args_list
            if "/v1/channel-sessions/register" in str(call)
        ]
        assert len(kernel_calls) >= 1

        payload = kernel_calls[0].kwargs.get("json", {})
        assert payload.get("iss") == "https://cogos.local", (
            f"user_iss must be forwarded as 'iss', got payload: {payload}"
        )
        assert payload.get("sub") == "chaz", (
            f"user_sub must be forwarded as 'sub', got payload: {payload}"
        )

    def test_kernel_callback_omits_iss_sub_when_absent(self, client):
        """When no identity claims are present the iss/sub keys must be absent
        from the kernel payload (clean JSON, not None/null)."""
        session_id = str(uuid.uuid4())

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("access.is_allowed", return_value=True), \
             patch("httpx.post", return_value=mock_response) as mock_post:
            resp = client.post(
                f"/v1/sessions/{session_id}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": session_id},
            )

        assert resp.status_code in (200, 201), resp.text

        kernel_calls = [
            call for call in mock_post.call_args_list
            if "/v1/channel-sessions/register" in str(call)
        ]
        assert len(kernel_calls) >= 1

        payload = kernel_calls[0].kwargs.get("json", {})
        assert "iss" not in payload, f"iss must be absent when no identity, got: {payload}"
        assert "sub" not in payload, f"sub must be absent when no identity, got: {payload}"


class TestKernelDeregisterCallback:
    """DELETE /v1/sessions/{id}/seats/{seat_id} → kernel /v1/channel-sessions/{id}/deregister."""

    def test_seat_revoke_fires_kernel_deregister_callback(self, client):
        """A successful seat revoke must fire a POST to the kernel's
        /v1/channel-sessions/{session_id}/deregister endpoint."""
        session_id = str(uuid.uuid4())

        # First register a seat.
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("access.is_allowed", return_value=True), \
             patch("httpx.post", return_value=mock_response):
            reg = client.post(
                f"/v1/sessions/{session_id}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": session_id},
            )
        assert reg.status_code in (200, 201), reg.text
        seat_id = reg.json()["seat_id"]

        # Now revoke and capture the kernel deregister callback.
        with patch("httpx.post", return_value=mock_response) as mock_post:
            del_resp = client.delete(f"/v1/sessions/{session_id}/seats/{seat_id}")

        assert del_resp.status_code == 200, del_resp.text

        deregister_calls = [
            call for call in mock_post.call_args_list
            if f"/v1/channel-sessions/{session_id}/deregister" in str(call)
        ]
        assert len(deregister_calls) >= 1, (
            f"Expected kernel deregister callback, got calls: {mock_post.call_args_list}"
        )

    def test_kernel_deregister_failure_does_not_block_seat_revoke(self, client):
        """If the kernel is unreachable during deregister, seat revoke must still succeed."""
        session_id = str(uuid.uuid4())

        mock_response = MagicMock()
        mock_response.status_code = 200

        # Register seat (mock kernel callback as success).
        with patch("access.is_allowed", return_value=True), \
             patch("httpx.post", return_value=mock_response):
            reg = client.post(
                f"/v1/sessions/{session_id}/seats",
                json={"client_type": "claude-code-channel", "device_uuid": session_id},
            )
        assert reg.status_code in (200, 201), reg.text
        seat_id = reg.json()["seat_id"]

        # Revoke with kernel callback failing.
        with patch("httpx.post", side_effect=Exception("kernel unreachable")):
            del_resp = client.delete(f"/v1/sessions/{session_id}/seats/{seat_id}")

        assert del_resp.status_code == 200, (
            f"seat revoke must succeed even when kernel deregister callback fails; "
            f"got {del_resp.status_code}: {del_resp.text}"
        )
        data = del_resp.json()
        assert data.get("status") == "revoked"

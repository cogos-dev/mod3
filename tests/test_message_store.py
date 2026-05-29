"""Tests for the per-session chat history ring buffer + GET endpoint.

The store is in-memory, bounded per session. Three behaviors matter to callers:
- appending a message returns the stored entry with id/ts/session_id set
- get(limit) honors insertion order and respects ``limit``
- the bucket is bounded — once full, oldest messages are dropped first

Plus end-to-end via FastAPI TestClient: POST /v1/sessions/<id>/messages and
POST /v1/dashboard-chat must populate the store, and GET /v1/sessions/<id>/messages
must surface what they appended.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _isolate_store():
    """Reset the module-level singleton between tests so they don't share state."""
    from message_store import reset_default_store_for_tests

    reset_default_store_for_tests()
    yield
    reset_default_store_for_tests()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import http_api

    with TestClient(http_api.app, base_url="http://localhost:7860") as c:
        yield c


class TestMessageStoreUnit:
    def test_append_returns_entry_with_canonical_fields(self):
        from message_store import MessageStore

        store = MessageStore()
        entry = store.append(session_id="s1", role="user", content="hello")
        assert entry["session_id"] == "s1"
        assert entry["role"] == "user"
        assert entry["content"] == "hello"
        assert entry["input_type"] == "text"
        assert "id" in entry and len(entry["id"]) == 8
        assert isinstance(entry["ts"], float)

    def test_get_returns_in_insertion_order(self):
        from message_store import MessageStore

        store = MessageStore()
        for i in range(5):
            store.append(session_id="s1", role="user", content=f"msg-{i}")
        msgs = store.get("s1")
        assert [m["content"] for m in msgs] == ["msg-0", "msg-1", "msg-2", "msg-3", "msg-4"]

    def test_get_with_limit_returns_most_recent(self):
        from message_store import MessageStore

        store = MessageStore()
        for i in range(10):
            store.append(session_id="s1", role="user", content=f"msg-{i}")
        recent = store.get("s1", limit=3)
        assert [m["content"] for m in recent] == ["msg-7", "msg-8", "msg-9"]

    def test_buckets_are_session_scoped(self):
        from message_store import MessageStore

        store = MessageStore()
        store.append(session_id="s1", role="user", content="for s1")
        store.append(session_id="s2", role="assistant", content="for s2")
        assert [m["content"] for m in store.get("s1")] == ["for s1"]
        assert [m["content"] for m in store.get("s2")] == ["for s2"]

    def test_ring_buffer_bound(self):
        from message_store import MessageStore

        store = MessageStore(max_per_session=3)
        for i in range(5):
            store.append(session_id="s1", role="user", content=f"msg-{i}")
        msgs = store.get("s1")
        assert [m["content"] for m in msgs] == ["msg-2", "msg-3", "msg-4"]

    def test_role_must_be_user_or_assistant(self):
        from message_store import MessageStore

        store = MessageStore()
        with pytest.raises(ValueError):
            store.append(session_id="s1", role="system", content="oops")


class TestHistoryEndpoint:
    def test_session_message_post_populates_history(self, client):
        sid = str(uuid.uuid4())
        with patch("access.is_allowed", return_value=True):
            # Register a seat so the session exists end-to-end (mirrors the
            # channel-client flow that PR #103/111 wires up).
            client.post(f"/v1/sessions/{sid}/seats", json={"client_type": "claude-code-channel", "device_uuid": sid})

        r = client.post(
            f"/v1/sessions/{sid}/messages", json={"content": "hello there", "role": "user", "input_type": "text"}
        )
        assert r.status_code == 200, r.text

        hist = client.get(f"/v1/sessions/{sid}/messages").json()
        assert hist["session_id"] == sid
        assert hist["count"] == 1
        assert hist["messages"][0]["content"] == "hello there"
        assert hist["messages"][0]["role"] == "user"

    def test_dashboard_chat_post_persists_assistant_reply(self, client):
        sid = str(uuid.uuid4())
        r = client.post("/v1/dashboard-chat", json={"text": "hi from agent", "role": "assistant", "session_id": sid})
        assert r.status_code == 200, r.text

        hist = client.get(f"/v1/sessions/{sid}/messages").json()
        assert hist["count"] == 1
        assert hist["messages"][0]["role"] == "assistant"
        assert hist["messages"][0]["content"] == "hi from agent"

    def test_broadcast_message_with_target_persists_to_target(self, client):
        sid = str(uuid.uuid4())
        r = client.post("/v1/sessions/broadcast-message", json={"content": "targeted hello", "target_session_id": sid})
        assert r.status_code == 200, r.text

        hist = client.get(f"/v1/sessions/{sid}/messages").json()
        assert hist["count"] == 1
        assert hist["messages"][0]["content"] == "targeted hello"

    def test_broadcast_without_target_persists_under_main(self, client):
        r = client.post("/v1/sessions/broadcast-message", json={"content": "fanout hello"})
        assert r.status_code == 200, r.text

        hist = client.get("/v1/sessions/main/messages").json()
        contents = [m["content"] for m in hist["messages"]]
        assert "fanout hello" in contents

    def test_get_history_empty_session_returns_zero(self, client):
        sid = str(uuid.uuid4())
        hist = client.get(f"/v1/sessions/{sid}/messages").json()
        assert hist["count"] == 0
        assert hist["messages"] == []

    def test_get_history_clamps_limit(self, client):
        sid = str(uuid.uuid4())
        for i in range(5):
            client.post(f"/v1/sessions/{sid}/messages", json={"content": f"m{i}", "role": "user"})

        # Negative / zero / oversize clamp to default 100
        for bad in (0, -10, 99999):
            hist = client.get(f"/v1/sessions/{sid}/messages?limit={bad}").json()
            assert hist["count"] == 5

        # Honors explicit small limit
        hist = client.get(f"/v1/sessions/{sid}/messages?limit=2").json()
        assert hist["count"] == 2
        assert [m["content"] for m in hist["messages"]] == ["m3", "m4"]

"""Tests for barge-in gate retry-with-backoff + escalation.

Covers the fix for flight-review.md §5 fix 6: voice output was requested 3x
during a 12+ hour flight and held 3x on the "user is speaking" gate, with no
retry and no escalation signal — the operator never heard any mod3 output
for the entire flight because a stuck/false-positive VAD signal silently
suppressed every speak() call.

check_bargein_gate() (server.py) now:
  - retries the barge-in signal file check up to MOD3_GATE_RETRY_ATTEMPTS
    times (default 3) with MOD3_GATE_RETRY_INTERVAL_SEC between checks
    (default 3.0s, ~10s total budget) before declaring the caller blocked.
  - tracks a per-streak-key consecutive-hold counter so repeated re-sends of
    a held request accumulate a visible "held N times" signal instead of each
    looking like an unremarkable, independent "held" reply.
  - is exercised both by the legacy speak() MCP tool and by /v1/speak
    (speak_enqueue in http_api.py), which is the endpoint mod3_speak (the
    channel_client.py tool Claude Code actually calls) hits.

Tests monkeypatch server._MOD3_GATE_RETRY_ATTEMPTS /
server._MOD3_GATE_RETRY_INTERVAL_SEC directly (not the env var) to keep the
retry loop fast, and monkeypatch server._BARGEIN_SIGNAL to a tmp_path so real
mod3 state on the dev machine can never be read or written.

Run: python3 -m pytest tests/test_gate_escalation.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_retries(monkeypatch, tmp_path):
    """Point the gate at an isolated signal path and shrink the retry budget.

    3 attempts, 0.01s apart — fast enough for a unit test while still
    exercising the real retry loop shape (not a single check).
    """
    import server

    signal_path = str(tmp_path / "mod3-barge-in.json")
    monkeypatch.setattr(server, "_BARGEIN_SIGNAL", signal_path)
    monkeypatch.setattr(server, "_MOD3_GATE_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(server, "_MOD3_GATE_RETRY_INTERVAL_SEC", 0.01)
    # Hold streaks are process-global; reset so tests don't see leakage
    # from whichever test ran before them.
    server._gate_hold_streak.clear()
    return signal_path


def _write_signal(path: str, event: str) -> None:
    with open(path, "w") as f:
        json.dump({"event": event, "source": "test", "timestamp": "2026-07-03T00:00:00Z"}, f)


# ---------------------------------------------------------------------------
# check_bargein_gate — unit tests
# ---------------------------------------------------------------------------


class TestCheckBargeinGate:
    def test_not_blocked_when_no_signal_file(self, fast_retries):
        """No signal file at all => idle => not blocked, single attempt."""
        import server

        result = server.check_bargein_gate(streak_key="s1")
        assert result["blocked"] is False
        assert result["attempts"] == 1
        assert result["held_count"] == 0

    def test_not_blocked_when_signal_says_end(self, fast_retries):
        """user_speaking_end (or any non-start event) reads as idle."""
        import server

        _write_signal(fast_retries, "user_speaking_end")
        result = server.check_bargein_gate(streak_key="s1")
        assert result["blocked"] is False
        assert result["held_count"] == 0

    def test_blocked_after_exhausting_retries(self, fast_retries):
        """Signal pinned to user_speaking_start for the whole retry window => blocked."""
        import server

        _write_signal(fast_retries, "user_speaking_start")
        result = server.check_bargein_gate(streak_key="s1")
        assert result["blocked"] is True
        assert result["attempts"] == server._MOD3_GATE_RETRY_ATTEMPTS
        assert result["held_count"] == 1

    def test_gate_clears_mid_retry_is_not_blocked(self, fast_retries, monkeypatch):
        """If the signal clears partway through the retry budget, gate succeeds.

        This is the core fix: a real utterance is a few seconds, so the retry
        loop must give it a real chance to clear rather than failing on the
        first read.
        """
        import server

        _write_signal(fast_retries, "user_speaking_start")

        calls = {"n": 0}
        real_sleep = server.time.sleep

        def _clear_after_first_check(seconds):
            calls["n"] += 1
            if calls["n"] == 1:
                _write_signal(fast_retries, "user_speaking_end")
            real_sleep(0)  # keep it fast; just yield

        monkeypatch.setattr(server.time, "sleep", _clear_after_first_check)

        result = server.check_bargein_gate(streak_key="s1")
        assert result["blocked"] is False
        assert result["attempts"] == 2  # first check blocked, second cleared
        assert result["held_count"] == 0

    def test_held_count_accumulates_across_calls(self, fast_retries):
        """Repeated blocked calls with the same streak_key accumulate held_count.

        This is the escalation signal itself: an agent re-sending speak()
        after being told "held" should see the count climb, not repeat
        the same unremarkable message forever.
        """
        import server

        _write_signal(fast_retries, "user_speaking_start")

        r1 = server.check_bargein_gate(streak_key="s1")
        r2 = server.check_bargein_gate(streak_key="s1")
        r3 = server.check_bargein_gate(streak_key="s1")

        assert [r1["held_count"], r2["held_count"], r3["held_count"]] == [1, 2, 3]

    def test_held_count_resets_after_gate_clears(self, fast_retries):
        """A successful (unblocked) check resets the streak back to 0."""
        import server

        _write_signal(fast_retries, "user_speaking_start")
        r1 = server.check_bargein_gate(streak_key="s1")
        assert r1["held_count"] == 1

        _write_signal(fast_retries, "user_speaking_end")
        r2 = server.check_bargein_gate(streak_key="s1")
        assert r2["blocked"] is False

        _write_signal(fast_retries, "user_speaking_start")
        r3 = server.check_bargein_gate(streak_key="s1")
        assert r3["held_count"] == 1, "streak should have reset, not continued from 1"

    def test_held_count_is_independent_per_streak_key(self, fast_retries):
        """Different streak_keys (sessions) track independent hold counters."""
        import server

        _write_signal(fast_retries, "user_speaking_start")
        r_a1 = server.check_bargein_gate(streak_key="session-a")
        r_b1 = server.check_bargein_gate(streak_key="session-b")
        r_a2 = server.check_bargein_gate(streak_key="session-a")

        assert r_a1["held_count"] == 1
        assert r_b1["held_count"] == 1
        assert r_a2["held_count"] == 2

    def test_corrupt_signal_file_treated_as_idle(self, fast_retries):
        """Malformed JSON in the signal file must not raise — treat as idle."""
        import server

        with open(fast_retries, "w") as f:
            f.write("{not valid json")

        result = server.check_bargein_gate(streak_key="s1")
        assert result["blocked"] is False


# ---------------------------------------------------------------------------
# speak() MCP tool — end-to-end gate escalation behavior
# ---------------------------------------------------------------------------


class TestSpeakToolGateEscalation:
    def test_speak_returns_held_with_escalation_after_repeated_blocks(self, fast_retries):
        """After 2+ consecutive holds, speak() surfaces a non-null escalation message.

        This is the structured "audio held N times — VAD may be
        false-positive" signal called for by flight-review.md §5 fix 6, so a
        stuck gate is visible to the caller instead of silently dropping
        audio forever.
        """
        import server

        _write_signal(fast_retries, "user_speaking_start")

        first = json.loads(server.speak(text="hello once"))
        assert first["status"] == "held"
        assert first["held_count"] == 1
        assert first.get("escalation") is None

        second = json.loads(server.speak(text="hello twice"))
        assert second["status"] == "held"
        assert second["held_count"] == 2
        assert second["escalation"] is not None
        assert "held 2 time" in second["escalation"]

    def test_speak_proceeds_once_gate_clears(self, fast_retries):
        """Once the barge-in signal clears, speak() enqueues normally."""
        import server

        # No signal file => idle => should proceed to _start_speech.
        original = server._start_speech
        called = {"n": 0}

        def fake_start_speech(text, voice, **kwargs):
            called["n"] += 1
            return "job-abc", 0

        server._start_speech = fake_start_speech
        try:
            result = json.loads(server.speak(text="hello"))
        finally:
            server._start_speech = original

        assert result["status"] == "speaking"
        assert called["n"] == 1

    def test_speak_gate_check_includes_retries(self, fast_retries, monkeypatch):
        """speak() must actually retry, not just check once and give up.

        Regression guard for the original defect: the old code read the
        signal file exactly once and returned a terminal "held" reply.
        """
        import server

        _write_signal(fast_retries, "user_speaking_start")
        sleep_calls = []
        real_sleep = server.time.sleep
        monkeypatch.setattr(server.time, "sleep", lambda s: (sleep_calls.append(s), real_sleep(0))[0])

        result = json.loads(server.speak(text="hello"))

        assert result["status"] == "held"
        assert result["gate_attempts"] == server._MOD3_GATE_RETRY_ATTEMPTS
        # 3 attempts => 2 sleeps between them
        assert len(sleep_calls) == server._MOD3_GATE_RETRY_ATTEMPTS - 1


# ---------------------------------------------------------------------------
# /v1/speak (speak_enqueue) — the endpoint mod3_speak actually calls
# ---------------------------------------------------------------------------


class TestSpeakEnqueueGateEscalation:
    """POST /v1/speak is what clients/channel_client.py's mod3_speak MCP tool
    hits. Previously this path had NO gate check at all — only the legacy
    stdio speak() tool checked the barge-in signal. Since mod3_speak is the
    interface Claude Code actually uses, the gate (with retry + escalation)
    must be enforced here too.
    """

    @pytest.fixture
    def client(self, fast_retries):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app, base_url="http://localhost:7860")

    def test_held_when_gate_blocked(self, client, fast_retries):
        _write_signal(fast_retries, "user_speaking_start")

        r = client.post("/v1/speak", json={"text": "hello", "session_id": "sess-1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "held"
        assert body["user_state"] == "recording"
        assert body["held_count"] == 1

    def test_escalation_present_after_repeated_holds(self, client, fast_retries):
        _write_signal(fast_retries, "user_speaking_start")

        client.post("/v1/speak", json={"text": "first", "session_id": "sess-2"})
        r2 = client.post("/v1/speak", json={"text": "second", "session_id": "sess-2"})

        body2 = r2.json()
        assert body2["held_count"] == 2
        assert body2["escalation"] is not None

    def test_proceeds_to_start_speech_when_gate_clears(self, client, fast_retries):
        """No signal file => gate clears => /v1/speak enqueues via _start_speech."""
        import server

        original = server._start_speech

        def fake_start_speech(text, voice, **kwargs):
            return "job-xyz", 0

        server._start_speech = fake_start_speech
        try:
            r = client.post("/v1/speak", json={"text": "hello", "session_id": "sess-3"})
        finally:
            server._start_speech = original

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "speaking"
        assert body["job_id"] == "job-xyz"

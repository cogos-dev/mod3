"""Tests for the barge-in provider registry + shared consumer helper.

Covers:
  * BargeinProvider lifecycle (start/stop, thread dies on stop)
  * Registry event dispatch: user_speaking_start -> pipeline_state.interrupt
  * user_speaking_end does NOT call interrupt (end is a no-op for now)
  * handle_bargein_start returns None when nothing is speaking
  * Subscribers fire after the consumer helper, and their exceptions don't
    break other subscribers
  * start_from_env respects the env var (empty default, unknown names warned)

Run: python -m pytest tests/test_bargein_provider_registry.py -v
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bargein import BargeinRegistry, handle_bargein_start  # noqa: E402
from bargein.providers.base import BargeinEvent, BargeinProvider  # noqa: E402
from pipeline_state import PipelineState  # noqa: E402

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeProvider(BargeinProvider):
    """Minimal provider: ``_run`` blocks on stop; tests call ``trigger`` to emit."""

    source = "browser_vad"  # existing literal — doesn't require the new "superwhisper"

    def __init__(self, on_event):
        super().__init__(on_event)
        self.run_called = threading.Event()
        self.run_returned = threading.Event()

    def _run(self) -> None:
        self.run_called.set()
        # Wait for stop; the registry / test drives via trigger().
        while not self._stop.is_set():
            self._stop.wait(0.05)
        self.run_returned.set()

    def trigger(self, event_type: str, **metadata) -> None:
        """Drive an event through the callback synchronously."""
        self._emit(event_type, metadata)


class _FakePlayer:
    """Minimal player stub for PipelineState: flush() is all interrupt() calls."""

    def __init__(self):
        self.flushed = False

    def flush(self) -> None:
        self.flushed = True


def _speaking_state(text: str = "Hello world how are you") -> PipelineState:
    """Build a PipelineState already in the speaking state."""
    state = PipelineState()
    state.start_speaking(text, _FakePlayer())
    # Pretend ~30% delivered so interrupt() has a non-zero spoken_pct
    state.update_position(samples_played=30, total_samples=100)
    return state


# ---------------------------------------------------------------------------
# handle_bargein_start (the shared consumer helper)
# ---------------------------------------------------------------------------


def test_handle_bargein_start_interrupts_speaking_state():
    state = _speaking_state("The capital of France is Paris")
    assert state.is_speaking

    info = handle_bargein_start(state, source="superwhisper", metadata={"folder": "1234"})

    assert info is not None
    assert info.reason == "barge_in"
    assert 0.0 < info.spoken_pct <= 1.0
    assert state.is_speaking is False
    # Interrupt is recorded for downstream BargeinContext consumption
    assert state.last_interrupt is info


def test_handle_bargein_start_noop_when_silent():
    state = PipelineState()
    assert state.is_speaking is False

    info = handle_bargein_start(state, source="superwhisper")

    assert info is None
    assert state.last_interrupt is None


# ---------------------------------------------------------------------------
# BargeinProvider lifecycle
# ---------------------------------------------------------------------------


def test_provider_start_and_stop():
    events: list[BargeinEvent] = []
    p = FakeProvider(on_event=events.append)

    p.start()
    assert p.run_called.wait(timeout=1.0), "_run was not entered"
    assert p.is_running

    p.stop(timeout=1.0)
    assert p.run_returned.wait(timeout=1.0), "_run did not return after stop()"
    assert not p.is_running


def test_provider_start_is_idempotent():
    p = FakeProvider(on_event=lambda _e: None)
    p.start()
    first_thread = p._thread
    p.start()  # should not spawn a second thread
    assert p._thread is first_thread
    p.stop()


def test_provider_emits_events_through_callback():
    events: list[BargeinEvent] = []
    p = FakeProvider(on_event=events.append)
    p.start()
    try:
        p.trigger("user_speaking_start", folder="1234")
        p.trigger("user_speaking_end", folder="1234")
    finally:
        p.stop()

    assert len(events) == 2
    assert events[0].event_type == "user_speaking_start"
    assert events[0].source == "browser_vad"
    assert events[0].metadata == {"folder": "1234"}
    assert events[1].event_type == "user_speaking_end"


def test_callback_exception_does_not_kill_provider():
    """A raising callback must not propagate out of _emit."""

    def boom(_event):
        raise RuntimeError("consumer exploded")

    p = FakeProvider(on_event=boom)
    p.start()
    try:
        # Should not raise
        p.trigger("user_speaking_start")
    finally:
        p.stop()
    # If we got here, _emit swallowed the exception.


# ---------------------------------------------------------------------------
# BargeinRegistry
# ---------------------------------------------------------------------------


def test_registry_routes_start_event_into_consumer_helper():
    state = _speaking_state()
    registry = BargeinRegistry(state)
    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()

    try:
        p.trigger("user_speaking_start", folder="1234")
        # Dispatch is synchronous on the triggering thread
        assert state.is_speaking is False
        assert state.last_interrupt is not None
        assert state.last_interrupt.reason == "barge_in"
    finally:
        registry.stop_all()


def test_registry_end_event_does_not_interrupt():
    state = _speaking_state()
    registry = BargeinRegistry(state)
    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()

    try:
        p.trigger("user_speaking_end", folder="1234")
        # End events must not interrupt ongoing playback
        assert state.is_speaking is True
        assert state.last_interrupt is None
    finally:
        registry.stop_all()


def test_registry_subscribers_fire_after_consumer():
    state = _speaking_state()
    registry = BargeinRegistry(state)
    seen: list[BargeinEvent] = []
    registry.subscribe(seen.append)

    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()

    try:
        p.trigger("user_speaking_start")
        p.trigger("user_speaking_end")
    finally:
        registry.stop_all()

    # Subscriber sees both events, in order
    assert [e.event_type for e in seen] == ["user_speaking_start", "user_speaking_end"]


def test_registry_subscriber_exception_isolated():
    """One subscriber raising must not prevent later subscribers from firing."""
    state = PipelineState()  # not speaking — consumer is a no-op
    registry = BargeinRegistry(state)

    def raiser(_e):
        raise RuntimeError("nope")

    other: list[BargeinEvent] = []
    registry.subscribe(raiser)
    registry.subscribe(other.append)

    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()
    try:
        p.trigger("user_speaking_start")
    finally:
        registry.stop_all()

    assert len(other) == 1


def test_registry_stop_all_stops_every_provider():
    state = PipelineState()
    registry = BargeinRegistry(state)

    providers = [FakeProvider(on_event=registry._dispatch) for _ in range(3)]
    for p in providers:
        registry.register(p)
    registry.start_all()

    # Wait until they're all actually running
    for p in providers:
        assert p.run_called.wait(timeout=1.0)

    registry.stop_all(timeout=1.0)

    for p in providers:
        assert p.run_returned.wait(timeout=1.0)
        assert not p.is_running


# ---------------------------------------------------------------------------
# start_from_env
# ---------------------------------------------------------------------------


def test_start_from_env_empty_by_default(monkeypatch):
    monkeypatch.delenv("MOD3_BARGEIN_PROVIDERS", raising=False)
    registry = BargeinRegistry(PipelineState())
    started = registry.start_from_env()
    assert started == []


def test_start_from_env_ignores_unknown_names(monkeypatch, caplog):
    monkeypatch.setenv("MOD3_BARGEIN_PROVIDERS", "definitely_not_a_provider")
    registry = BargeinRegistry(PipelineState())
    started = registry.start_from_env()
    assert started == []


def test_start_from_env_instantiates_known_provider(monkeypatch):
    """Happy path: 'superwhisper' in env -> SuperWhisperProvider registered.

    We don't wait for it to find recordings (that directory may not exist on
    the test host) — we only verify construction + registration + shutdown.
    """
    monkeypatch.setenv("MOD3_BARGEIN_PROVIDERS", "superwhisper")
    # Point it at a directory we know doesn't exist so its _run returns fast
    monkeypatch.setenv("SW_RECORDINGS_DIR", "/tmp/mod3-bargein-test-nonexistent")

    registry = BargeinRegistry(PipelineState())
    started = registry.start_from_env()
    try:
        assert started == ["superwhisper"]
        # One provider is registered
        assert len(registry._providers) == 1
        from bargein.providers.superwhisper import SuperWhisperProvider

        assert isinstance(registry._providers[0], SuperWhisperProvider)
    finally:
        registry.stop_all(timeout=1.0)


# ---------------------------------------------------------------------------
# wait_for_event — used by await_voice_input() to block until in-process
# providers fire (replaces the old file-only poll). Regression target for
# Codex review #4 / Fix 1.
# ---------------------------------------------------------------------------


def test_wait_for_event_returns_matching_event():
    """wait_for_event returns the event when an in-process provider emits it."""
    registry = BargeinRegistry(PipelineState())
    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()

    fired = threading.Event()
    captured: list[BargeinEvent | None] = []

    def _waiter():
        evt = registry.wait_for_event("user_speaking_end", timeout=2.0)
        captured.append(evt)
        fired.set()

    threading.Thread(target=_waiter, daemon=True).start()
    # Give the waiter a tick to subscribe before we trigger
    time.sleep(0.05)
    p.trigger("user_speaking_end", folder="42")

    assert fired.wait(timeout=2.0), "wait_for_event did not return after emit"
    registry.stop_all()
    assert len(captured) == 1
    assert captured[0] is not None
    assert captured[0].event_type == "user_speaking_end"
    assert captured[0].metadata == {"folder": "42"}


def test_wait_for_event_filters_by_event_type():
    """A start event must NOT satisfy a wait for end."""
    registry = BargeinRegistry(PipelineState())
    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()

    captured: list[BargeinEvent | None] = []
    done = threading.Event()

    def _waiter():
        captured.append(registry.wait_for_event("user_speaking_end", timeout=0.4))
        done.set()

    threading.Thread(target=_waiter, daemon=True).start()
    time.sleep(0.05)
    # Wrong event type — the waiter should ignore this and time out
    p.trigger("user_speaking_start")

    assert done.wait(timeout=2.0)
    registry.stop_all()
    assert captured == [None]


def test_wait_for_event_filters_by_source():
    """source=... narrows the wait to a specific provider."""
    registry = BargeinRegistry(PipelineState())
    p_browser = FakeProvider(on_event=registry._dispatch)

    # Subclass with a different source value
    class _SuperFake(FakeProvider):
        source = "superwhisper"

    p_sw = _SuperFake(on_event=registry._dispatch)
    registry.register(p_browser)
    registry.register(p_sw)
    registry.start_all()

    captured: list[BargeinEvent | None] = []
    done = threading.Event()

    def _waiter():
        captured.append(registry.wait_for_event("user_speaking_end", source="superwhisper", timeout=2.0))
        done.set()

    threading.Thread(target=_waiter, daemon=True).start()
    time.sleep(0.05)
    # Browser-VAD end first — should be ignored by source filter
    p_browser.trigger("user_speaking_end", folder="b1")
    time.sleep(0.05)
    # Then SuperWhisper end — should satisfy
    p_sw.trigger("user_speaking_end", folder="sw1")

    assert done.wait(timeout=2.0)
    registry.stop_all()
    assert captured[0] is not None
    assert captured[0].source == "superwhisper"
    assert captured[0].metadata == {"folder": "sw1"}


def test_wait_for_event_times_out_when_silent():
    """No event emitted -> wait_for_event returns None within timeout."""
    registry = BargeinRegistry(PipelineState())
    t0 = time.monotonic()
    result = registry.wait_for_event("user_speaking_end", timeout=0.2)
    elapsed = time.monotonic() - t0
    assert result is None
    assert 0.15 < elapsed < 1.0


def test_wait_for_event_unsubscribes_on_completion(monkeypatch):
    """The temporary waiter subscriber must not leak after the wait returns."""
    registry = BargeinRegistry(PipelineState())
    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()

    starting = len(registry._subscribers)

    def _do_wait():
        registry.wait_for_event("user_speaking_end", timeout=0.5)

    t = threading.Thread(target=_do_wait, daemon=True)
    t.start()
    time.sleep(0.05)
    # While waiting, the subscriber count should be elevated
    assert len(registry._subscribers) == starting + 1
    p.trigger("user_speaking_end")
    t.join(timeout=2.0)
    registry.stop_all()
    # Cleaned up
    assert len(registry._subscribers) == starting


# ---------------------------------------------------------------------------
# make_file_mirror_subscriber — bridges in-process events to the legacy
# /tmp/mod3-barge-in.json signal file so out-of-process pollers (mcp_shim)
# keep working alongside the new registry.
# ---------------------------------------------------------------------------


def test_file_mirror_subscriber_writes_event_to_path(tmp_path):
    import json as _json

    from bargein import make_file_mirror_subscriber

    signal_path = str(tmp_path / "mod3-barge-in.json")
    registry = BargeinRegistry(PipelineState())
    registry.subscribe(make_file_mirror_subscriber(signal_path))

    p = FakeProvider(on_event=registry._dispatch)
    registry.register(p)
    registry.start_all()
    try:
        p.trigger("user_speaking_end", folder="abc")
    finally:
        registry.stop_all()

    with open(signal_path) as f:
        written = _json.load(f)

    assert written["event"] == "user_speaking_end"
    assert written["source"] == "browser_vad"  # FakeProvider's source
    assert written["folder"] == "abc"
    assert written["via"] == "bargein_registry"
    assert "timestamp" in written


# ---------------------------------------------------------------------------
# await_voice_input — end-to-end regression test for Codex review #4 / Fix 1.
# The unit tests above cover wait_for_event + make_file_mirror_subscriber in
# isolation; these lock down the actual mod3 tool function, which is what the
# original regression (in-process user_speaking_end never waking the tool)
# was about.
# ---------------------------------------------------------------------------


def test_await_voice_input_returns_when_registry_emits_end(monkeypatch, tmp_path):
    """Regression: await_voice_input() must return when an in-process provider
    dispatches user_speaking_end through the registry.

    This is exactly the bug Fix 1 addressed — before the registry-aware wait,
    await_voice_input only watched the legacy signal file and never saw
    events from in-process providers like SuperWhisperProvider.
    """
    import json as _json

    import server  # noqa: E402

    # Isolate the file signal so we don't race with any existing /tmp state
    signal_path = str(tmp_path / "mod3-barge-in.json")
    monkeypatch.setattr(server, "_BARGEIN_SIGNAL", signal_path)
    monkeypatch.setattr(server, "_bargein_last_mtime", 0.0)

    result_box: list[str] = []
    t0 = time.monotonic()

    def _caller():
        result_box.append(server.await_voice_input(timeout_sec=5.0))

    caller = threading.Thread(target=_caller, daemon=True)
    caller.start()

    # Let await_voice_input subscribe before we dispatch
    time.sleep(0.2)

    server._bargein_registry._dispatch(
        BargeinEvent(
            source="superwhisper",
            event_type="user_speaking_end",
            metadata={"folder": "42"},
        )
    )

    caller.join(timeout=3.0)
    elapsed = time.monotonic() - t0

    assert not caller.is_alive(), "await_voice_input did not return after registry dispatch"
    # 5.0s timeout would mean we missed the event; anything under ~2s means we caught it
    assert elapsed < 2.5, f"took {elapsed:.2f}s — likely timed out rather than catching the event"
    assert len(result_box) == 1

    result = _json.loads(result_box[0])
    # status may be "ok" (if SuperWhisper recordings exist) or "error" (no transcript
    # to read in this test env), but MUST NOT be "timeout" — that is the regression.
    assert result["status"] != "timeout", f"timed out despite registry event: {result}"


def test_await_voice_input_returns_on_legacy_file_write(monkeypatch, tmp_path):
    """Backward-compat: out-of-process producers (e.g. integrations/bargein-producer.py)
    write ``user_speaking_end`` to ``/tmp/mod3-barge-in.json``. await_voice_input()
    must still wake on that path after the Fix 2 refactor.
    """
    import json as _json

    import server  # noqa: E402

    signal_path = str(tmp_path / "mod3-barge-in.json")
    monkeypatch.setattr(server, "_BARGEIN_SIGNAL", signal_path)
    monkeypatch.setattr(server, "_bargein_last_mtime", 0.0)

    result_box: list[str] = []
    t0 = time.monotonic()

    def _caller():
        result_box.append(server.await_voice_input(timeout_sec=5.0))

    caller = threading.Thread(target=_caller, daemon=True)
    caller.start()

    # Give await_voice_input a tick to enter its wait
    time.sleep(0.2)

    # Simulate the legacy producer writing to the signal file
    with open(signal_path, "w") as f:
        _json.dump(
            {
                "event": "user_speaking_end",
                "source": "superwhisper",
                "timestamp": "2026-04-19T00:00:00Z",
            },
            f,
        )

    caller.join(timeout=3.0)
    elapsed = time.monotonic() - t0

    assert not caller.is_alive(), "await_voice_input did not return after file write"
    assert elapsed < 2.5, f"took {elapsed:.2f}s — likely timed out rather than reading file"
    assert len(result_box) == 1

    result = _json.loads(result_box[0])
    assert result["status"] != "timeout", f"timed out despite file write: {result}"


def test_await_voice_input_times_out_when_no_signal(monkeypatch, tmp_path):
    """If neither source fires, await_voice_input() must actually time out.

    This is the negative control for the two regression tests above — if it
    always returned quickly, they wouldn't be proving anything.
    """
    import json as _json

    import server  # noqa: E402

    signal_path = str(tmp_path / "mod3-barge-in.json")
    monkeypatch.setattr(server, "_BARGEIN_SIGNAL", signal_path)
    monkeypatch.setattr(server, "_bargein_last_mtime", 0.0)

    t0 = time.monotonic()
    raw = server.await_voice_input(timeout_sec=0.4)
    elapsed = time.monotonic() - t0

    assert 0.3 < elapsed < 2.0, f"timeout path ran for {elapsed:.2f}s"
    result = _json.loads(raw)
    assert result["status"] == "timeout"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

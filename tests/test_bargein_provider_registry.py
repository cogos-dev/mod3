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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Tests for InboundPipeline mic idle-release.

Covers the fix for the second flight-review-adjacent defect: the mod3 server
process pinned the microphone open continuously (observed uptime 3.9 days,
vad:true, mic held the entire time — the OS mic indicator never went dark).
InboundPipeline.start() acquired AudioCapture once and nothing ever released
it; only a full daemon restart cleared the OS-level "in use" state.

InboundPipeline now:
  - tracks _last_activity_monotonic, bumped by mark_activity() whenever VAD
    detects real speech (both the composed VADStage and the legacy inline
    tick path call it).
  - runs a background watcher thread (_mic_idle_watcher_loop) that releases
    the AudioCapture device (via _release_capture) once idle exceeds
    MOD3_MIC_IDLE_RELEASE_SECONDS (default 300s / 5 min).
  - transparently re-acquires the device on demand in _tick() the next time
    it needs audio (via _acquire_capture) — no behavior change from the
    caller's perspective beyond the mic no longer being held while idle.
  - mic_idle_release_sec <= 0 disables the watcher entirely (restores the
    historical always-open behavior), matching engine.py's
    MOD3_TTS_IDLE_UNLOAD_SECONDS convention (opt-out via non-positive value).

Run: python3 -m pytest tests/test_mic_idle_release.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_server():
    if "server" not in sys.modules:
        fake_server = ModuleType("server")
        fake_server.emit_channel_event = MagicMock()
        fake_server.emit_permission_verdict = MagicMock()
        sys.modules["server"] = fake_server


@pytest.fixture(autouse=True)
def _restore_real_server_module():
    """Undo _stub_server()'s sys.modules stubbing after each test.

    _stub_server() (mirrored from tests/test_intentional_stages.py and
    tests/test_channel_pipeline_graph.py) installs a fake 'server' module into
    sys.modules only when one isn't already present — a real import of
    server.py is heavy (starts background threads, builds the FastAPI app)
    and unnecessary for exercising InboundPipeline in isolation. But if this
    test file is the first in the process to import 'inbound' (and therefore
    the first to call _stub_server()), the fake stub is left behind in
    sys.modules for every test file that runs afterward in the same pytest
    session. A plain `import server` in a later file (e.g.
    tests/test_speak_endpoint.py) then silently receives the fake stub
    instead of the real module — missing _start_speech, check_bargein_gate,
    etc. — causing failures with no visible connection to this file.

    Snapshot/restore sys.modules['server'] around every test here so this
    file can never leak a fake stub to tests that run after it.
    """
    had_server = "server" in sys.modules
    original = sys.modules.get("server")
    try:
        yield
    finally:
        if had_server:
            sys.modules["server"] = original
        else:
            sys.modules.pop("server", None)


def _make_pipeline(mock_capture=None, **kwargs):
    """Instantiate InboundPipeline with a mocked AudioCapture.

    The mock tracks is_active as real state (not just a MagicMock return
    value) so start()/stop()/is_active() interact the way the real
    AudioCapture would.
    """
    _stub_server()
    if "inbound" in sys.modules:
        del sys.modules["inbound"]

    if mock_capture is None:
        mock_capture = _StatefulFakeCapture()

    with patch("capture.AudioCapture", return_value=mock_capture):
        mock_bus = MagicMock()
        mock_state = MagicMock()
        mock_state.is_speaking = False

        from inbound import InboundPipeline

        pipeline = InboundPipeline(bus=mock_bus, pipeline_state=mock_state, **kwargs)
        return pipeline, mock_capture


class _StatefulFakeCapture:
    """Minimal AudioCapture stand-in with real is_active()/start()/stop() state.

    Records call counts so tests can assert on acquire/release cycling
    without touching real audio hardware.
    """

    def __init__(self):
        self._active = False
        self.start_calls = 0
        self.stop_calls = 0

    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        self.start_calls += 1
        self._active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._active = False

    def get_audio(self, duration_sec: float):
        return None  # no accumulated audio; _tick will just wait and retry


# ---------------------------------------------------------------------------
# Idle-release configuration resolution
# ---------------------------------------------------------------------------


class TestMicIdleReleaseConfig:
    def test_default_enabled_with_default_window(self):
        """Default (no override) is enabled at the 300s class default."""
        pipeline, _ = _make_pipeline()
        assert pipeline._mic_idle_release_enabled is True
        assert pipeline._mic_idle_release_sec == pytest.approx(300.0)

    def test_constructor_override(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=42.0)
        assert pipeline._mic_idle_release_sec == pytest.approx(42.0)
        assert pipeline._mic_idle_release_enabled is True

    def test_zero_disables(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=0)
        assert pipeline._mic_idle_release_enabled is False

    def test_negative_disables(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=-5.0)
        assert pipeline._mic_idle_release_enabled is False

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("MOD3_MIC_IDLE_RELEASE_SECONDS", "60")
        pipeline, _ = _make_pipeline()
        assert pipeline._mic_idle_release_sec == pytest.approx(60.0)

    def test_env_var_zero_disables(self, monkeypatch):
        monkeypatch.setenv("MOD3_MIC_IDLE_RELEASE_SECONDS", "0")
        pipeline, _ = _make_pipeline()
        assert pipeline._mic_idle_release_enabled is False

    def test_env_var_non_numeric_falls_back_to_default(self, monkeypatch, caplog):
        import logging

        monkeypatch.setenv("MOD3_MIC_IDLE_RELEASE_SECONDS", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="mod3.inbound"):
            pipeline, _ = _make_pipeline()
        assert pipeline._mic_idle_release_sec == pytest.approx(300.0)
        assert "MOD3_MIC_IDLE_RELEASE_SECONDS" in caplog.text

    def test_constructor_arg_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv("MOD3_MIC_IDLE_RELEASE_SECONDS", "999")
        pipeline, _ = _make_pipeline(mic_idle_release_sec=5.0)
        assert pipeline._mic_idle_release_sec == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# start()/stop() lifecycle
# ---------------------------------------------------------------------------


class TestStartStopLifecycle:
    def test_start_acquires_capture(self):
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0)
        pipeline.start()
        try:
            assert capture.is_active() is True
            assert capture.start_calls == 1
        finally:
            pipeline.stop()

    def test_stop_releases_capture(self):
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0)
        pipeline.start()
        pipeline.stop()
        assert capture.is_active() is False
        assert capture.stop_calls == 1

    def test_start_launches_idle_watcher_thread_when_enabled(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=60.0)
        pipeline.start()
        try:
            assert pipeline._mic_idle_watcher_thread is not None
            assert pipeline._mic_idle_watcher_thread.is_alive()
        finally:
            pipeline.stop()

    def test_start_does_not_launch_watcher_when_disabled(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=0)
        pipeline.start()
        try:
            assert pipeline._mic_idle_watcher_thread is None
        finally:
            pipeline.stop()

    def test_stop_joins_idle_watcher_thread(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=60.0)
        pipeline.start()
        watcher = pipeline._mic_idle_watcher_thread
        pipeline.stop()
        assert pipeline._mic_idle_watcher_thread is None
        assert not watcher.is_alive()

    def test_stop_is_idempotent(self):
        """Calling stop() when not running must not raise or double-count."""
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0)
        pipeline.stop()  # never started
        assert capture.stop_calls == 0


# ---------------------------------------------------------------------------
# mark_activity() / mic_is_open
# ---------------------------------------------------------------------------


class TestActivityTracking:
    def test_mark_activity_resets_clock(self):
        pipeline, _ = _make_pipeline(mic_idle_release_sec=60.0)
        pipeline._last_activity_monotonic = time.monotonic() - 1000
        pipeline.mark_activity()
        assert time.monotonic() - pipeline._last_activity_monotonic < 1.0

    def test_mic_is_open_reflects_capture_state(self):
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0)
        assert pipeline.mic_is_open is False
        capture.start()
        assert pipeline.mic_is_open is True

    def test_vad_stage_calls_mark_activity_on_speech(self):
        """The composed VADStage.process() resets the idle clock on speech onset."""
        # Import VADStage / VADResult AFTER _make_pipeline() — _make_pipeline
        # deletes and re-imports the 'inbound' module, so importing the class
        # beforehand would bind to a stale module object that
        # patch("inbound.detect_speech", ...) below would not affect.
        pipeline, _ = _make_pipeline(mic_idle_release_sec=60.0)
        from inbound import VADStage
        from vad import VADResult

        pipeline._last_activity_monotonic = time.monotonic() - 1000

        stage = VADStage()
        stage.configure(pipeline)

        vad_result = VADResult(
            has_speech=True,
            confidence=0.9,
            speech_ratio=0.8,
            num_segments=1,
            total_speech_sec=1.0,
            total_audio_sec=2.0,
        )
        chunk = np.zeros(1600, dtype=np.float32)

        with patch("inbound.detect_speech", return_value=vad_result):
            stage.process({"chunk": chunk})

        assert time.monotonic() - pipeline._last_activity_monotonic < 1.0

    def test_vad_stage_does_not_mark_activity_on_silence(self):
        # See note above: import AFTER _make_pipeline() for the patch to bind
        # to the same module object the stage instance uses.
        pipeline, _ = _make_pipeline(mic_idle_release_sec=60.0)
        from inbound import VADStage
        from vad import VADResult

        stale = time.monotonic() - 1000
        pipeline._last_activity_monotonic = stale

        stage = VADStage()
        stage.configure(pipeline)

        vad_result = VADResult(
            has_speech=False,
            confidence=0.0,
            speech_ratio=0.0,
            num_segments=0,
            total_speech_sec=0.0,
            total_audio_sec=2.0,
        )
        chunk = np.zeros(1600, dtype=np.float32)

        with patch("inbound.detect_speech", return_value=vad_result):
            stage.process({"chunk": chunk})

        assert pipeline._last_activity_monotonic == stale


# ---------------------------------------------------------------------------
# Idle watcher behavior (real thread, tiny window, no real audio)
# ---------------------------------------------------------------------------


class TestMicIdleWatcherBehavior:
    def test_idle_check_releases_capture_past_the_window(self):
        """_mic_idle_check_once releases an active, past-window-idle capture.

        Calls the release-decision method directly rather than waiting on
        the real watcher thread's poll interval (which has a 5s floor —
        see _mic_idle_watcher_poll_interval — so a timing-based test would
        need to either wait 5s+ or be flaky under load).
        """
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0.05)
        capture.start()
        pipeline._last_activity_monotonic = time.monotonic() - 10.0  # well past 0.05s

        released = pipeline._mic_idle_check_once()

        assert released is True
        assert capture.is_active() is False
        assert capture.stop_calls == 1

    def test_idle_check_leaves_recently_active_capture_open(self):
        """_mic_idle_check_once must not release a capture that is not yet idle."""
        pipeline, capture = _make_pipeline(mic_idle_release_sec=60.0)
        capture.start()
        pipeline._last_activity_monotonic = time.monotonic()  # just active

        released = pipeline._mic_idle_check_once()

        assert released is False
        assert capture.is_active() is True
        assert capture.stop_calls == 0

    def test_idle_check_is_noop_when_already_released(self):
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0.05)
        assert capture.is_active() is False
        pipeline._last_activity_monotonic = time.monotonic() - 10.0

        released = pipeline._mic_idle_check_once()

        assert released is False
        assert capture.stop_calls == 0

    def test_watcher_thread_releases_capture_after_idle_window(self):
        """End-to-end: the real background thread releases an idle mic.

        Uses a poll interval fast enough to observe within the test timeout
        by monkeypatching _mic_idle_watcher_poll_interval (bypassing the 5s
        floor) rather than waiting on the production floor value.
        """
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0.05)
        pipeline._mic_idle_watcher_poll_interval = lambda: 0.02
        pipeline.start()
        try:
            assert capture.is_active() is True
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and capture.is_active():
                time.sleep(0.01)
            assert capture.is_active() is False, "mic was not idle-released within the window"
            assert capture.stop_calls >= 1
        finally:
            pipeline.stop()

    def test_watcher_does_not_release_capture_recently_marked_active(self):
        """If activity keeps getting marked, the watcher must not release the mic."""
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0.2)
        pipeline.start()
        try:
            end = time.monotonic() + 0.6
            while time.monotonic() < end:
                pipeline.mark_activity()
                time.sleep(0.02)
            assert capture.is_active() is True, "mic was released despite continuous activity"
        finally:
            pipeline.stop()

    def test_tick_reacquires_capture_after_idle_release(self):
        """_tick() must transparently re-acquire the mic if it finds it released."""
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0)
        # Simulate: watcher already released the mic (or it was never opened).
        capture._active = False
        pipeline._tick()
        assert capture.is_active() is True
        assert capture.start_calls >= 1

    def test_disabled_release_never_stops_capture(self):
        """mic_idle_release_sec=0 must never release the mic once acquired."""
        pipeline, capture = _make_pipeline(mic_idle_release_sec=0)
        pipeline.start()
        try:
            time.sleep(0.2)
            assert capture.is_active() is True
            assert capture.stop_calls == 0
        finally:
            pipeline.stop()

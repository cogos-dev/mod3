"""Tests for the adaptive pre-playback buffer (operator two-loop policy).

Covers the pure policy math in adaptive_player.py (deficit_ema_to_buffer_ms,
probe_latency_to_buffer_ms, compute_initial_buffer_ms, grow_target_buffer_ms),
the cross-job deficit-EMA telemetry, the best-effort LMS probe, and the thin
AdaptivePlayer wiring (starvation counting, rebuffering, recorded metrics).

No real audio hardware or network access is required — AdaptivePlayer's
_callback is pure Python/numpy given a preallocated output array, and the
LMS probe is exercised via a monkeypatched httpx.Client.

Run: python3 -m pytest tests/test_buffer_policy.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the project root is on the path so imports resolve when running
# standalone (python3 -m pytest tests/test_buffer_policy.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_player import (  # noqa: E402
    ELEVATED_BUFFER_MS,
    HEAVY_BUFFER_MS_CAP,
    PROBE_BUSY_MS,
    PROBE_TIMEOUT_MS,
    QUIET_BUFFER_MS,
    STARVATION_GROWTH_MS,
    AdaptivePlayer,
    compute_initial_buffer_ms,
    deficit_ema_to_buffer_ms,
    get_deficit_ema,
    grow_target_buffer_ms,
    probe_latency_to_buffer_ms,
    probe_lms_contention,
    record_chunk_deficit,
)

# ---------------------------------------------------------------------------
# Feedforward: deficit EMA -> buffer_ms
# ---------------------------------------------------------------------------


class TestDeficitEmaToBufferMs:
    def test_quiet_at_or_below_threshold(self):
        assert deficit_ema_to_buffer_ms(1.0) == QUIET_BUFFER_MS
        assert deficit_ema_to_buffer_ms(1.05) == QUIET_BUFFER_MS

    def test_elevated_anchor_maps_exactly(self):
        assert deficit_ema_to_buffer_ms(1.3) == ELEVATED_BUFFER_MS

    def test_measured_contention_falls_between_quiet_and_elevated(self):
        """1.26x is the measured wall-clock cost of one concurrent LMS call."""
        buf = deficit_ema_to_buffer_ms(1.26)
        assert QUIET_BUFFER_MS < buf < ELEVATED_BUFFER_MS

    def test_heavy_caps_at_and_beyond_2x(self):
        assert deficit_ema_to_buffer_ms(2.0) == HEAVY_BUFFER_MS_CAP
        assert deficit_ema_to_buffer_ms(5.0) == HEAVY_BUFFER_MS_CAP

    def test_monotonic_nondecreasing(self):
        xs = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 3.0]
        ys = [deficit_ema_to_buffer_ms(x) for x in xs]
        assert ys == sorted(ys)


# ---------------------------------------------------------------------------
# Feedforward: LMS probe latency -> buffer_ms
# ---------------------------------------------------------------------------


class TestProbeLatencyToBufferMs:
    def test_fast_reply_is_quiet(self):
        assert probe_latency_to_buffer_ms(1.0) == QUIET_BUFFER_MS
        assert probe_latency_to_buffer_ms(PROBE_BUSY_MS) == QUIET_BUFFER_MS

    def test_slow_reply_scales_to_heavy_cap_at_timeout_boundary(self):
        assert probe_latency_to_buffer_ms(PROBE_TIMEOUT_MS) == HEAVY_BUFFER_MS_CAP

    def test_midpoint_is_between_elevated_and_cap(self):
        mid = (PROBE_BUSY_MS + PROBE_TIMEOUT_MS) / 2
        val = probe_latency_to_buffer_ms(mid)
        assert ELEVATED_BUFFER_MS < val < HEAVY_BUFFER_MS_CAP


# ---------------------------------------------------------------------------
# Feedforward: combined policy (primary + optional secondary)
# ---------------------------------------------------------------------------


class TestComputeInitialBufferMs:
    def test_primary_signal_only(self):
        assert compute_initial_buffer_ms(deficit_ema=1.0) == QUIET_BUFFER_MS

    def test_probe_timeout_means_assume_quiet(self):
        val = compute_initial_buffer_ms(deficit_ema=1.0, probe_latency_ms=None, probe_timed_out=True)
        assert val == QUIET_BUFFER_MS

    def test_probe_never_lowers_the_primary_estimate(self):
        """Primary already reads heavy; a fast (quiet) probe must not pull it down."""
        val = compute_initial_buffer_ms(deficit_ema=2.0, probe_latency_ms=1.0, probe_timed_out=False)
        assert val == HEAVY_BUFFER_MS_CAP

    def test_probe_can_raise_a_quiet_primary_estimate(self):
        val = compute_initial_buffer_ms(deficit_ema=1.0, probe_latency_ms=PROBE_TIMEOUT_MS, probe_timed_out=False)
        assert val == HEAVY_BUFFER_MS_CAP

    def test_result_never_exceeds_the_cap(self):
        val = compute_initial_buffer_ms(deficit_ema=10.0, probe_latency_ms=1000.0, probe_timed_out=False)
        assert val == HEAVY_BUFFER_MS_CAP


# ---------------------------------------------------------------------------
# Feedback: starvation growth
# ---------------------------------------------------------------------------


class TestGrowTargetBufferMs:
    def test_grows_by_the_fixed_increment(self):
        assert grow_target_buffer_ms(150.0) == 150.0 + STARVATION_GROWTH_MS

    def test_caps_at_heavy_buffer_ms(self):
        assert grow_target_buffer_ms(HEAVY_BUFFER_MS_CAP) == HEAVY_BUFFER_MS_CAP
        assert grow_target_buffer_ms(HEAVY_BUFFER_MS_CAP - 10) == HEAVY_BUFFER_MS_CAP

    def test_result_never_decreases(self):
        assert grow_target_buffer_ms(500.0) >= 500.0


# ---------------------------------------------------------------------------
# Cross-job chunk-deficit telemetry (process-wide EMA)
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_deficit_ema():
    """Save/restore the process-wide EMA so tests don't leak state into each other."""
    import adaptive_player

    original = adaptive_player._deficit_ema
    adaptive_player._deficit_ema = 1.0
    yield
    adaptive_player._deficit_ema = original


class TestDeficitEmaTelemetry:
    def test_default_is_exactly_realtime(self, reset_deficit_ema):
        assert get_deficit_ema() == 1.0

    def test_recording_a_slow_chunk_moves_the_ema_toward_its_ratio(self, reset_deficit_ema):
        record_chunk_deficit(synth_time_sec=2.52, audio_duration_sec=2.0)  # ratio 1.26
        ema = get_deficit_ema()
        assert 1.0 < ema < 1.26, "EMA should move toward the ratio, not jump straight to it"

    def test_zero_duration_chunk_is_ignored(self, reset_deficit_ema):
        record_chunk_deficit(synth_time_sec=1.0, audio_duration_sec=0.0)
        assert get_deficit_ema() == 1.0


# ---------------------------------------------------------------------------
# LMS contention probe (network I/O, mocked)
# ---------------------------------------------------------------------------


class _FakeSuccessClient:
    def __init__(self, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get(self, url):
        return None


def _raising_client_factory(exc: Exception):
    """Build an httpx.Client-shaped stand-in whose get() always raises `exc`."""

    class _FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def get(self, url):
            raise exc

    return _FakeClient


class TestProbeLmsContention:
    def test_success_reports_latency_not_timed_out(self, monkeypatch):
        import adaptive_player

        monkeypatch.setattr(adaptive_player.httpx, "Client", _FakeSuccessClient)
        latency_ms, timed_out = probe_lms_contention()
        assert timed_out is False
        assert latency_ms is not None
        assert latency_ms >= 0

    def test_timeout_is_reported_as_assume_quiet(self, monkeypatch):
        import httpx

        import adaptive_player

        monkeypatch.setattr(adaptive_player.httpx, "Client", _raising_client_factory(httpx.TimeoutException("slow")))
        latency_ms, timed_out = probe_lms_contention()
        assert timed_out is True
        assert latency_ms is None

    def test_connection_error_is_reported_as_assume_quiet(self, monkeypatch):
        """LMS not running at all must not fail the job — same as a timeout."""
        import adaptive_player

        monkeypatch.setattr(adaptive_player.httpx, "Client", _raising_client_factory(ConnectionError("refused")))
        latency_ms, timed_out = probe_lms_contention()
        assert timed_out is True
        assert latency_ms is None


# ---------------------------------------------------------------------------
# AdaptivePlayer wiring: thin, but exercised end to end without hardware
# ---------------------------------------------------------------------------


def _silent_player(initial_buffer_ms: float = 100.0) -> AdaptivePlayer:
    """A player pretending its stream is already running, without opening one."""
    player = AdaptivePlayer(sample_rate=24_000, initial_buffer_ms=initial_buffer_ms)
    player._playing = True
    return player


class TestAdaptivePlayerBufferWiring:
    def test_initial_buffer_ms_defaults_to_quiet(self):
        player = AdaptivePlayer(sample_rate=24_000)
        assert player.initial_buffer_ms == QUIET_BUFFER_MS
        assert player._target_buffer_ms == QUIET_BUFFER_MS

    def test_initial_buffer_ms_is_honored(self):
        player = AdaptivePlayer(sample_rate=24_000, initial_buffer_ms=500.0)
        assert player.initial_buffer_ms == 500.0
        assert player._needed_samples() == int(24_000 * 0.5)

    def test_starvation_increments_count_and_grows_target(self):
        player = _silent_player(initial_buffer_ms=100.0)
        outdata = np.zeros((256, 1), dtype=np.float32)

        player._callback(outdata, 256, None, None)  # buffer empty, generation not done

        assert player._starvation_count == 1
        assert player._target_buffer_ms == 100.0 + STARVATION_GROWTH_MS
        assert player._rebuffering is True

    def test_starvation_is_not_double_counted_while_already_rebuffering(self):
        player = _silent_player(initial_buffer_ms=100.0)
        outdata = np.zeros((256, 1), dtype=np.float32)

        player._callback(outdata, 256, None, None)
        player._callback(outdata, 256, None, None)
        player._callback(outdata, 256, None, None)

        assert player._starvation_count == 1, "one drought = one event, not one per callback tick"

    def test_rebuffering_holds_output_silent_until_target_rebuilt(self):
        player = _silent_player(initial_buffer_ms=100.0)
        outdata = np.zeros((256, 1), dtype=np.float32)
        player._callback(outdata, 256, None, None)  # trigger starvation -> rebuffering
        assert player._rebuffering is True

        # A small chunk arrives — not enough to satisfy the grown target yet.
        player.queue_audio(np.ones(50, dtype=np.float32))
        outdata2 = np.zeros((256, 1), dtype=np.float32)
        player._callback(outdata2, 256, None, None)
        assert player._rebuffering is True, "should still be rebuilding the cushion"
        assert player._samples_played == 0, "output must stay silent while rebuffering"

        # Enough audio arrives to rebuild the (grown) target buffer.
        needed = player._needed_samples()
        player.queue_audio(np.full(needed, 0.5, dtype=np.float32))
        outdata3 = np.zeros((300, 1), dtype=np.float32)
        player._callback(outdata3, 300, None, None)
        assert player._rebuffering is False
        assert player._samples_played == 300, "playback resumed once the cushion rebuilt"

    def test_target_buffer_never_shrinks_after_growth(self):
        player = _silent_player(initial_buffer_ms=150.0)
        outdata = np.zeros((64, 1), dtype=np.float32)
        player._callback(outdata, 64, None, None)  # starvation -> grows
        grown = player._target_buffer_ms
        assert grown == 150.0 + STARVATION_GROWTH_MS

        player.queue_audio(np.full(50_000, 0.1, dtype=np.float32))
        for _ in range(5):
            out = np.zeros((64, 1), dtype=np.float32)
            player._callback(out, 64, None, None)

        assert player._target_buffer_ms == grown, "target must never shrink mid-utterance"

    def test_starvation_growth_is_capped(self):
        player = _silent_player(initial_buffer_ms=HEAVY_BUFFER_MS_CAP - 100)
        outdata = np.zeros((64, 1), dtype=np.float32)

        player._callback(outdata, 64, None, None)
        assert player._target_buffer_ms == HEAVY_BUFFER_MS_CAP

        player._rebuffering = False  # simulate a second, independent drought
        player._callback(outdata, 64, None, None)
        assert player._target_buffer_ms == HEAVY_BUFFER_MS_CAP
        assert player._starvation_count == 2

    def test_build_metrics_reports_buffer_fields(self):
        player = AdaptivePlayer(sample_rate=24_000, initial_buffer_ms=150.0)
        player._starvation_count = 2
        player._target_buffer_ms = 750.0

        result = player._build_metrics().to_dict()

        assert result["buffer"]["initial_buffer_ms"] == 150.0
        assert result["buffer"]["final_buffer_ms"] == 750.0
        assert result["buffer"]["starvation_count"] == 2

    def test_queue_audio_feeds_chunk_deficit_telemetry(self, reset_deficit_ema):
        # _playing=True so queue_audio's startup gate never fires — this test
        # only cares about the telemetry side effect, not real playback.
        player = _silent_player(initial_buffer_ms=150.0)
        before = get_deficit_ema()

        # 1 second of audio that took 1.5s to synthesize -> ratio 1.5, above realtime.
        player.queue_audio(
            np.zeros(24_000, dtype=np.float32),
            chunk_meta={"gen_time_sec": 1.5, "samples": 24_000},
        )

        assert get_deficit_ema() > before

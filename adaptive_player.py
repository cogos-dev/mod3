"""Adaptive audio player with a two-loop pre-playback buffer and per-session metrics.

Adapted from mlx_audio's AudioPlayer but with full instrumentation:
- Underrun counting (empty buffer during active playback)
- Per-callback buffer depth tracking
- TTFA measurement (time from first queued audio to first audible output)
- Structured PlaybackMetrics returned on completion

Buffer policy (operator-designed, two loops):
- FEEDFORWARD: initial_buffer_ms is computed once at job start from load
  signals — primarily mod3's own recent chunk-deficit telemetry (an EMA of
  synth_time/audio_duration across recently generated chunks, process-wide),
  optionally corroborated by a timeboxed probe of the LMS lane (GPU
  contention proxy: Metal is shared between local TTS and a local LLM).
  See compute_initial_buffer_ms().
- FEEDBACK: during playback, a true starvation event (buffer empty while
  the generator is still producing) grows the target buffer for the rest
  of the utterance and holds output silent until the cushion rebuilds.
  The target never shrinks mid-utterance. See grow_target_buffer_ms().

The policy math above is pure (inputs -> buffer_ms); AdaptivePlayer's
runtime wiring only gathers inputs and calls it.
"""

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock

import httpx
import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Buffer policy — pure functions (inputs -> buffer_ms). Unit-tested in
# tests/test_buffer_policy.py; nothing here touches audio hardware or state.
# ---------------------------------------------------------------------------

# Defaults per the operator's design: quiet/elevated/heavy anchors.
QUIET_BUFFER_MS = 150.0
ELEVATED_BUFFER_MS = 500.0
HEAVY_BUFFER_MS_CAP = 2000.0

# Chunk-deficit EMA anchors (ratio of synth_time to audio_duration). At/below
# 1.05x synthesis is keeping pace with real time (quiet). 1.3x brackets the
# measured 1.26x wall-clock cost of one concurrent LMS call sharing Metal
# with TTS (elevated). At/above 2.0x is clamped at the cap (heavy).
_QUIET_DEFICIT = 1.05
_ELEVATED_DEFICIT = 1.3
_HEAVY_DEFICIT = 2.0

# LMS contention probe (GET /v1/models) — a fast reply means the lane is
# idle; a slow one corroborates contention already inferred from the
# deficit EMA. The probe is timeboxed to PROBE_TIMEOUT_MS; a timeout means
# "assume quiet" (no LMS running is not contention).
PROBE_BUSY_MS = 50.0
PROBE_TIMEOUT_MS = 100.0
_LMS_MODELS_URL = "http://localhost:1234/v1/models"

# Feedback growth increment per starvation event (mid-utterance, monotonic).
STARVATION_GROWTH_MS = 300.0


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linearly interpolate y at x within [x0, x1], clamped to [y0, y1]."""
    if x1 <= x0:
        return y1
    t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
    return y0 + t * (y1 - y0)


def deficit_ema_to_buffer_ms(deficit_ema: float) -> float:
    """Map the chunk-deficit EMA to a target startup buffer (primary signal).

    Piecewise-linear over the three anchors above: quiet at/under 1.05x
    realtime, elevated at 1.3x, heavy at/beyond 2.0x.
    """
    if deficit_ema <= _QUIET_DEFICIT:
        return QUIET_BUFFER_MS
    if deficit_ema <= _ELEVATED_DEFICIT:
        return _lerp(deficit_ema, _QUIET_DEFICIT, _ELEVATED_DEFICIT, QUIET_BUFFER_MS, ELEVATED_BUFFER_MS)
    if deficit_ema <= _HEAVY_DEFICIT:
        return _lerp(deficit_ema, _ELEVATED_DEFICIT, _HEAVY_DEFICIT, ELEVATED_BUFFER_MS, HEAVY_BUFFER_MS_CAP)
    return HEAVY_BUFFER_MS_CAP


def probe_latency_to_buffer_ms(probe_latency_ms: float) -> float:
    """Map a corroborating LMS-probe latency to a buffer floor (secondary signal)."""
    if probe_latency_ms <= PROBE_BUSY_MS:
        return QUIET_BUFFER_MS
    return _lerp(probe_latency_ms, PROBE_BUSY_MS, PROBE_TIMEOUT_MS, ELEVATED_BUFFER_MS, HEAVY_BUFFER_MS_CAP)


def compute_initial_buffer_ms(
    deficit_ema: float,
    probe_latency_ms: float | None = None,
    probe_timed_out: bool = False,
) -> float:
    """Feedforward policy: initial_buffer_ms from load signals at job start.

    The primary signal is mod3's own recent chunk-deficit EMA. The optional
    LMS probe only ever raises the estimate — a timeout (or no probe at all)
    means "assume quiet", contributing nothing beyond the primary signal.
    """
    buffer_ms = deficit_ema_to_buffer_ms(deficit_ema)
    if probe_timed_out or probe_latency_ms is None:
        return buffer_ms
    probe_component = probe_latency_to_buffer_ms(probe_latency_ms)
    return min(HEAVY_BUFFER_MS_CAP, max(buffer_ms, probe_component))


def grow_target_buffer_ms(current_target_ms: float) -> float:
    """Feedback policy: on drain starvation, grow the target buffer.

    Monotonic non-decreasing (never shrinks mid-utterance), capped at
    HEAVY_BUFFER_MS_CAP.
    """
    return min(HEAVY_BUFFER_MS_CAP, current_target_ms + STARVATION_GROWTH_MS)


def probe_lms_contention(
    url: str = _LMS_MODELS_URL, timeout_sec: float = PROBE_TIMEOUT_MS / 1000.0
) -> tuple[float | None, bool]:
    """Best-effort GET latency to the LMS lane, as a contention proxy.

    Returns (latency_ms, timed_out). Any failure — timeout, connection
    refused, LMS not running — is treated as timed_out=True/latency_ms=None,
    which compute_initial_buffer_ms() reads as "assume quiet".
    """
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            client.get(url)
        return (time.perf_counter() - start) * 1000.0, False
    except Exception:  # noqa: BLE001 — probe is best-effort, never fails the job
        return None, True


# ---------------------------------------------------------------------------
# Cross-job chunk-deficit telemetry — process-wide, feeds the feedforward
# signal above. Updated once per generated chunk (see AdaptivePlayer.
# queue_audio); read at the *next* job's start, before any of that job's
# own chunks have arrived.
# ---------------------------------------------------------------------------

_DEFICIT_EMA_ALPHA = 0.3
_deficit_ema = 1.0  # 1.0 = synthesis exactly at realtime; no deficit
_deficit_ema_lock = Lock()


def record_chunk_deficit(synth_time_sec: float, audio_duration_sec: float) -> None:
    """Update the process-wide chunk-deficit EMA. Called once per chunk."""
    global _deficit_ema
    if audio_duration_sec <= 0:
        return
    ratio = synth_time_sec / audio_duration_sec
    with _deficit_ema_lock:
        _deficit_ema = _DEFICIT_EMA_ALPHA * ratio + (1 - _DEFICIT_EMA_ALPHA) * _deficit_ema


def get_deficit_ema() -> float:
    """Current process-wide chunk-deficit EMA (read at job start)."""
    with _deficit_ema_lock:
        return _deficit_ema


@dataclass
class PlaybackMetrics:
    """Frozen snapshot of a single playback session."""

    # Audio
    duration_sec: float = 0.0
    total_samples: int = 0
    sample_rate: int = 24_000

    # Timing
    ttfa_sec: float = 0.0  # first queue_audio → first audible output
    total_wall_sec: float = 0.0
    overall_rtf: float = 0.0  # duration_sec / total_wall_sec

    # Chunks (from generator)
    chunk_count: int = 0
    per_chunk: list[dict] = field(default_factory=list)

    # Buffer health
    startup_delay_sec: float = 0.0
    peak_buffer_samples: int = 0
    min_buffer_samples: int = 0
    underrun_count: int = 0

    # Adaptive pre-playback buffer (two-loop policy — see module docstring)
    initial_buffer_ms: float = 0.0
    final_buffer_ms: float = 0.0
    starvation_count: int = 0

    # Memory
    peak_memory_gb: float = 0.0

    # Mode
    mode: str = "streaming"

    def to_dict(self) -> dict:
        return {
            "status": "ok",
            "mode": self.mode,
            "audio": {
                "duration_sec": round(self.duration_sec, 2),
                "total_samples": self.total_samples,
            },
            "timing": {
                "ttfa_sec": round(self.ttfa_sec, 3),
                "total_wall_sec": round(self.total_wall_sec, 2),
                "overall_rtf": round(self.overall_rtf, 2),
            },
            "chunks": {
                "count": self.chunk_count,
                "per_chunk": self.per_chunk,
            },
            "buffer": {
                "startup_delay_sec": round(self.startup_delay_sec, 3),
                "peak_samples": self.peak_buffer_samples,
                "min_samples": self.min_buffer_samples,
                "underruns": self.underrun_count,
                "initial_buffer_ms": round(self.initial_buffer_ms, 1),
                "final_buffer_ms": round(self.final_buffer_ms, 1),
                "starvation_count": self.starvation_count,
            },
            "memory_peak_gb": round(self.peak_memory_gb, 2),
        }


class AdaptivePlayer:
    """Callback-based audio player with a two-loop adaptive pre-playback buffer.

    Usage:
        player = AdaptivePlayer(sample_rate=24000, initial_buffer_ms=150)
        # In a background thread:
        for chunk in generate(...):
            player.queue_audio(chunk_audio, chunk_meta={...})
        player.mark_done()
        # In the foreground:
        metrics = player.wait()
    """

    def __init__(
        self,
        sample_rate: int = 24_000,
        buffer_size: int = 2048,
        device: int | str | None = None,
        initial_buffer_ms: float | None = None,
    ):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device = device  # sounddevice output device index or name

        # Buffer
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = Lock()

        # Stream
        self._stream: sd.OutputStream | None = None
        self._playing = False
        self._drain_event = Event()
        self._stream_finished = Event()  # set by sounddevice finished_callback
        self._generation_done = False

        # Adaptive pre-playback buffer (operator two-loop policy — see module
        # docstring). Feedforward: sized once here from recent load signals
        # (defaults to QUIET_BUFFER_MS if the caller has no signal yet).
        # Feedback: _target_buffer_ms only ever grows during playback — see
        # _callback() and grow_target_buffer_ms().
        self.initial_buffer_ms = QUIET_BUFFER_MS if initial_buffer_ms is None else initial_buffer_ms
        self._target_buffer_ms = self.initial_buffer_ms
        self._starvation_count = 0
        self._rebuffering = False  # True while holding output silent to rebuild the cushion

        # Metrics accumulators
        self._first_queue_time: float | None = None
        self._first_pull_time: float | None = None
        self._total_queued_samples = 0
        self._peak_buffer = 0
        self._min_buffer = sys.maxsize
        self._underruns = 0
        self._chunk_metrics: list[dict] = []
        self._startup_delay = 0.0
        self._peak_memory_gb = 0.0

        # Progress tracking (for PipelineState position updates)
        self._samples_played = 0

        # Synchronization: set when mark_done() is called
        self._done_event = Event()

    def _needed_samples(self) -> int:
        """Sample-count threshold for the current (possibly grown) target buffer."""
        return int(self.sample_rate * self._target_buffer_ms / 1000.0)

    # ------------------------------------------------------------------
    # Callback (runs in audio thread)
    # ------------------------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        outdata.fill(0)
        filled = 0

        with self._buffer_lock:
            current_buffer = sum(map(len, self._buffer))

            # Feedback loop, part 1: if we're rebuilding the cushion after a
            # starvation event, hold off draining until it's back at target
            # (or generation finished, so there's nothing left to wait for).
            if self._rebuffering and (current_buffer >= self._needed_samples() or self._generation_done):
                self._rebuffering = False

            if not self._rebuffering:
                while filled < frames and self._buffer:
                    buf = self._buffer[0]
                    to_copy = min(frames - filled, len(buf))
                    outdata[filled : filled + to_copy, 0] = buf[:to_copy]
                    filled += to_copy

                    if to_copy == len(buf):
                        self._buffer.popleft()
                    else:
                        self._buffer[0] = buf[to_copy:]

                current_buffer = sum(map(len, self._buffer))

            # Feedback loop, part 2: true starvation — buffer empty while the
            # generator is still producing (not the expected end-of-stream
            # drain). Grow the target (monotonic — never shrinks
            # mid-utterance) and start rebuffering instead of continuing to
            # trickle out chunks one small gap at a time.
            if filled == 0 and self._playing and not self._generation_done and not self._rebuffering:
                self._starvation_count += 1
                self._target_buffer_ms = grow_target_buffer_ms(self._target_buffer_ms)
                self._rebuffering = True

        # Progress tracking (lock-free; only written here in audio thread)
        self._samples_played += filled

        # Metrics
        if filled > 0 and self._first_pull_time is None:
            self._first_pull_time = time.perf_counter()

        if self._playing:
            if current_buffer < self._min_buffer:
                self._min_buffer = current_buffer
            if current_buffer > self._peak_buffer:
                self._peak_buffer = current_buffer

        if filled == 0 and self._playing:
            self._underruns += 1

        # Stop only when buffer is empty AND generation is done
        if current_buffer == 0 and filled < frames and self._generation_done:
            self._drain_event.set()
            raise sd.CallbackStop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    LEAD_SILENCE_SEC = 0.2  # silence before first audio to let device settle

    def queue_audio(self, samples: np.ndarray, chunk_meta: dict | None = None):
        """Queue audio samples for playback. Called from generator thread."""
        samples = np.asarray(samples, dtype=np.float32)
        if len(samples) == 0:
            return

        now = time.perf_counter()
        if self._first_queue_time is None:
            self._first_queue_time = now
            # Prepend silence so the audio device is settled before speech starts
            silence = np.zeros(int(self.sample_rate * self.LEAD_SILENCE_SEC), dtype=np.float32)
            with self._buffer_lock:
                self._buffer.append(silence)
                self._total_queued_samples += len(silence)

        with self._buffer_lock:
            self._buffer.append(samples)
            self._total_queued_samples += len(samples)
            current_buffer = sum(map(len, self._buffer))

        if current_buffer > self._peak_buffer:
            self._peak_buffer = current_buffer

        # Record per-chunk metrics
        if chunk_meta is not None:
            chunk_meta["buffer_depth"] = current_buffer
            self._chunk_metrics.append(chunk_meta)
            mem = chunk_meta.get("peak_memory_gb", 0.0)
            if mem > self._peak_memory_gb:
                self._peak_memory_gb = mem
            # Feed this chunk's synth-vs-realtime ratio into the process-wide
            # deficit EMA — the primary feedforward signal for the *next*
            # job's initial_buffer_ms (see compute_initial_buffer_ms).
            gen_time_sec = chunk_meta.get("gen_time_sec")
            if gen_time_sec is not None and len(samples):
                record_chunk_deficit(gen_time_sec, len(samples) / self.sample_rate)

        # Adaptive startup: buffer at least _target_buffer_ms of audio
        # content before starting playback. The target was sized at
        # construction (feedforward) and may have grown since (feedback).
        if not self._playing and current_buffer >= self._needed_samples():
            self._startup_delay = now - self._first_queue_time
            self._start_stream()

    def mark_done(self):
        """Signal that the generator has finished producing audio."""
        self._generation_done = True
        self._done_event.set()
        # Nothing was generated — unblock wait() immediately
        if self._total_queued_samples == 0:
            self._drain_event.set()
            return
        # If we never hit the buffer threshold (very short text), start now
        if not self._playing:
            if self._first_queue_time is not None:
                self._startup_delay = time.perf_counter() - self._first_queue_time
            self._start_stream()

    def get_progress(self) -> tuple[int, int]:
        """Return (samples_played, total_samples_queued) for position tracking.

        Called by PipelineState to compute spoken_pct. The samples_played
        counter is updated in the audio callback; total_samples_queued is
        updated in queue_audio(). Both are monotonically increasing.
        """
        return self._samples_played, self._total_queued_samples

    def wait(self, timeout: float = 120.0) -> PlaybackMetrics:
        """Block until playback finishes. Returns metrics."""
        # Wait for generation to at least finish before checking state
        self._done_event.wait(timeout=timeout)
        # Wait for buffer to drain (callback raises CallbackStop)
        self._drain_event.wait(timeout=timeout)
        # Wait for sounddevice to fully flush the device buffer
        self._stream_finished.wait(timeout=5.0)
        self._stop_stream()
        return self._build_metrics()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_device(self):
        """Resolve the output device, falling back to system default if unavailable."""
        if self.device is None:
            return None  # sounddevice uses system default

        try:
            devices = sd.query_devices()
            if isinstance(self.device, int):
                if self.device < len(devices):
                    info = devices[self.device]
                    if info["max_output_channels"] > 0:
                        return self.device
            elif isinstance(self.device, str):
                for i, d in enumerate(devices):
                    if self.device in d["name"] and d["max_output_channels"] > 0:
                        return i
        except Exception:
            pass

        # Device unavailable — fall back to system default.
        return None

    def _start_stream(self):
        self._stream_finished.clear()
        resolved = self._resolve_device()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=resolved,
            callback=self._callback,
            finished_callback=self._on_stream_finished,
            blocksize=self.buffer_size,
        )
        self._stream.start()
        self._playing = True
        self._drain_event.clear()

    def _on_stream_finished(self):
        """Called by sounddevice after stream fully stops (all audio flushed)."""
        self._stream_finished.set()

    def _stop_stream(self):
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            self._playing = False

    def flush(self):
        """Discard everything and stop playback immediately."""
        with self._buffer_lock:
            self._buffer.clear()
        self._generation_done = True
        self._stop_stream()
        self._drain_event.set()
        self._stream_finished.set()
        self._done_event.set()

    def _build_metrics(self) -> PlaybackMetrics:
        duration = self._total_queued_samples / self.sample_rate
        now = time.perf_counter()
        wall = (now - self._first_queue_time) if self._first_queue_time else 0.0

        ttfa = 0.0
        if self._first_pull_time and self._first_queue_time:
            ttfa = self._first_pull_time - self._first_queue_time

        return PlaybackMetrics(
            duration_sec=duration,
            total_samples=self._total_queued_samples,
            sample_rate=self.sample_rate,
            ttfa_sec=ttfa,
            total_wall_sec=wall,
            overall_rtf=duration / wall if wall > 0 else 0.0,
            chunk_count=len(self._chunk_metrics),
            per_chunk=self._chunk_metrics,
            startup_delay_sec=self._startup_delay,
            peak_buffer_samples=self._peak_buffer,
            min_buffer_samples=self._min_buffer if self._min_buffer != sys.maxsize else 0,
            underrun_count=self._underruns,
            peak_memory_gb=self._peak_memory_gb,
            mode="streaming" if len(self._chunk_metrics) > 1 else "batch",
            initial_buffer_ms=self.initial_buffer_ms,
            final_buffer_ms=self._target_buffer_ms,
            starvation_count=self._starvation_count,
        )

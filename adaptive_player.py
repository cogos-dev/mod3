"""Adaptive audio player with EMA buffering and per-session metrics.

Adapted from mlx_audio's AudioPlayer but with full instrumentation:
- Underrun counting (empty buffer during active playback)
- Per-callback buffer depth tracking
- TTFA measurement (time from first queued audio to first audible output)
- Structured PlaybackMetrics returned on completion
"""

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock

import numpy as np
import sounddevice as sd


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
            },
            "memory_peak_gb": round(self.peak_memory_gb, 2),
        }


class AdaptivePlayer:
    """Callback-based audio player with EMA-adaptive startup buffering.

    Usage:
        player = AdaptivePlayer(sample_rate=24000)
        # In a background thread:
        for chunk in generate(...):
            player.queue_audio(chunk_audio, chunk_meta={...})
        player.mark_done()
        # In the foreground:
        metrics = player.wait()
    """

    # EMA parameters (same as mlx_audio AudioPlayer)
    EMA_ALPHA = 0.25
    MEASURE_WINDOW = 0.25  # seconds between rate measurements
    MIN_BUFFER_SECONDS = 1.5  # startup threshold = arrival_rate * this

    def __init__(self, sample_rate: int = 24_000, buffer_size: int = 2048, device: int | str | None = None):
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

        # EMA arrival rate tracking
        self._window_sample_count = 0
        self._window_start = time.perf_counter()
        self._arrival_rate = float(sample_rate)  # assume realtime initially

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

    # ------------------------------------------------------------------
    # Callback (runs in audio thread)
    # ------------------------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status):
        outdata.fill(0)
        filled = 0

        with self._buffer_lock:
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

        # EMA arrival rate
        self._window_sample_count += len(samples)
        if now - self._window_start >= self.MEASURE_WINDOW:
            elapsed = now - self._window_start
            inst_rate = self._window_sample_count / elapsed
            self._arrival_rate = self.EMA_ALPHA * inst_rate + (1 - self.EMA_ALPHA) * self._arrival_rate
            self._window_sample_count = 0
            self._window_start = now

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

        # Adaptive startup
        needed = int(self._arrival_rate * self.MIN_BUFFER_SECONDS)
        if not self._playing and current_buffer >= needed:
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
        )

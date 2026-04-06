"""Continuous microphone audio capture for the Mod3 input pipeline.

Provides a ring-buffer-backed AudioCapture class that streams from
any sounddevice input device and exposes thread-safe reads of the
most recent N seconds.  Audio is always delivered as float32 mono
at the configured sample rate (default 16 kHz).

No side effects on import.
"""

from __future__ import annotations

import threading
from typing import Union

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def list_input_devices() -> list[dict]:
    """List available audio input devices.

    Returns a list of dicts, each with:
        index   - device index for sounddevice
        name    - human-readable device name
        channels - max input channels
        default  - True if this is the system default input
    """
    default_input = sd.default.device[0]
    results: list[dict] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            results.append(
                {
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "default": i == default_input,
                }
            )
    return results


def _resolve_device(device: Union[int, str, None]) -> Union[int, None]:
    """Resolve a device specifier to a sounddevice index (or None for default).

    Args:
        device: None (system default), int (index), or str (name substring).

    Returns:
        Integer device index, or None to use the system default.

    Raises:
        ValueError: If a string doesn't match any input device.
        ValueError: If an integer index doesn't correspond to an input device.
    """
    if device is None:
        return None

    if isinstance(device, int):
        info = sd.query_devices(device)
        if info["max_input_channels"] < 1:
            raise ValueError(f"Device {device} ({info['name']}) has no input channels")
        return device

    if isinstance(device, str):
        needle = device.lower()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and needle in d["name"].lower():
                return i
        raise ValueError(f"No input device matching '{device}'. Available: {[d['name'] for d in list_input_devices()]}")

    raise TypeError(f"device must be int, str, or None — got {type(device)}")


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class _RingBuffer:
    """Fixed-capacity ring buffer for 1-D float32 audio samples.

    Thread-safe: one writer (the stream callback) and one reader
    (get_audio) can operate concurrently without corruption.
    """

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0  # next write index (mod capacity)
        self._samples_written = 0  # total samples ever written
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Number of valid samples currently in the buffer."""
        with self._lock:
            return min(self._samples_written, self._capacity)

    def write(self, data: np.ndarray) -> None:
        """Append samples.  Overwrites oldest data when full."""
        n = len(data)
        if n == 0:
            return

        with self._lock:
            if n >= self._capacity:
                # More data than buffer can hold — keep the tail
                self._buf[:] = data[-self._capacity :]
                self._write_pos = 0
                self._samples_written += n
                return

            end = self._write_pos + n
            if end <= self._capacity:
                self._buf[self._write_pos : end] = data
            else:
                first = self._capacity - self._write_pos
                self._buf[self._write_pos :] = data[:first]
                self._buf[: n - first] = data[first:]

            self._write_pos = end % self._capacity
            self._samples_written += n

    def read_last(self, n_samples: int) -> np.ndarray | None:
        """Return the most recent *n_samples* as a contiguous copy.

        Returns None if fewer than *n_samples* are available.
        """
        with self._lock:
            avail = min(self._samples_written, self._capacity)
            if n_samples > avail:
                return None

            start = (self._write_pos - n_samples) % self._capacity
            if start + n_samples <= self._capacity:
                return self._buf[start : start + n_samples].copy()
            else:
                first = self._capacity - start
                return np.concatenate(
                    [
                        self._buf[start:],
                        self._buf[: n_samples - first],
                    ]
                ).copy()


# ---------------------------------------------------------------------------
# AudioCapture
# ---------------------------------------------------------------------------


class AudioCapture:
    """Continuous microphone capture with ring-buffer storage.

    Args:
        sample_rate:       Target sample rate (default 16 000 Hz).
        channels:          Number of capture channels (default 1 / mono).
        chunk_duration_ms: Size of each callback chunk in milliseconds.
        device:            Input device — None (default), int (index),
                           or str (name substring match).
        buffer_duration_sec: Ring buffer capacity in seconds (default 60).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 500,
        device: Union[int, str, None] = None,
        buffer_duration_sec: int = 60,
    ):
        self._target_sr = sample_rate
        self._channels = channels
        self._chunk_ms = chunk_duration_ms
        self._device_spec = device
        self._device_index: int | None = None

        # Ring buffer sized for the requested duration
        buf_samples = sample_rate * buffer_duration_sec
        self._ring = _RingBuffer(buf_samples)

        self._stream: sd.InputStream | None = None
        self._active = False

        # If the hardware sample rate differs from the target, we resample
        self._native_sr: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start recording from the microphone."""
        if self._active:
            return

        self._device_index = _resolve_device(self._device_spec)

        # Query native sample rate of the chosen device
        if self._device_index is not None:
            info = sd.query_devices(self._device_index)
        else:
            info = sd.query_devices(sd.default.device[0])
        self._native_sr = int(info["default_samplerate"])

        # Determine the stream sample rate.  sounddevice will use the
        # device's native rate unless we ask for something specific.
        # If the device supports our target rate natively, use it directly
        # to avoid resampling overhead.  Otherwise, capture at native rate
        # and resample in the callback.
        need_resample = self._native_sr != self._target_sr
        stream_sr = self._native_sr if need_resample else self._target_sr

        blocksize = int(stream_sr * self._chunk_ms / 1000)

        self._stream = sd.InputStream(
            samplerate=stream_sr,
            blocksize=blocksize,
            device=self._device_index,
            channels=self._channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._active = True

    def stop(self) -> None:
        """Stop recording."""
        if not self._active:
            return
        self._active = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio(self, duration_sec: float) -> np.ndarray | None:
        """Return the last *duration_sec* seconds from the ring buffer.

        Returns:
            Float32 numpy array of shape (samples,), or None if not
            enough audio has been captured yet.
        """
        n_samples = int(self._target_sr * duration_sec)
        return self._ring.read_last(n_samples)

    def is_active(self) -> bool:
        """Whether capture is currently running."""
        return self._active

    @property
    def device_info(self) -> dict:
        """Info dict for the current (or selected) input device."""
        idx = self._device_index
        if idx is None:
            idx = _resolve_device(self._device_spec)
            if idx is None:
                idx = sd.default.device[0]
        info = sd.query_devices(idx)
        return {
            "index": idx,
            "name": info["name"],
            "channels": info["max_input_channels"],
            "default_samplerate": info["default_samplerate"],
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,  # noqa: ANN001 — PaCallbackTimeInfo
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice stream callback — runs in a separate thread."""
        if status:
            # Input overflow / underflow — not fatal, just skip bad data
            pass

        # indata shape: (frames, channels) — squeeze to 1-D for mono
        audio = indata[:, 0] if indata.ndim > 1 else indata.ravel()

        # Resample to target rate if needed
        if self._native_sr is not None and self._native_sr != self._target_sr:
            audio = self._resample(audio, self._native_sr, self._target_sr)

        self._ring.write(audio)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear-interpolation resample.

        Good enough for voice capture where the ratio is typically
        close to 1 (e.g. 44100 -> 16000 or 48000 -> 16000).
        For higher-fidelity resampling, swap in scipy.signal.resample
        or soxr.
        """
        if orig_sr == target_sr:
            return audio
        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        # np.interp is fast and allocation-light
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

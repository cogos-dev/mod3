"""Shared pipeline state for Mod3 reflex arc.

Thread-safe bridge between inbound (VAD/microphone) and outbound (TTS/playback)
pipelines. Enables sub-50ms interrupt: VAD detects user speech -> interrupt() ->
player.flush() -> silence, without waiting for the LLM round-trip.

Usage:
    state = PipelineState()

    # Outbound side (TTS thread)
    state.start_speaking("Hello world", player)
    state.update_position(samples_played, total_samples)
    state.stop_speaking()

    # Inbound side (VAD thread)
    if state.is_speaking:
        info = state.interrupt(reason="vad_reflex")
"""

import threading
import time
from dataclasses import dataclass


@dataclass
class InterruptInfo:
    """Record of a playback interruption."""

    timestamp: float
    spoken_pct: float  # 0.0 - 1.0, how much was delivered
    delivered_text: str  # text that was actually spoken
    full_text: str  # original full text
    reason: str  # "vad_reflex", "manual_stop", etc.


class PipelineState:
    """Thread-safe shared state between inbound and outbound pipelines.

    The outbound side (TTS player) registers when it starts/stops speaking.
    The inbound side (VAD) checks if speech is happening and triggers interrupt.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._speaking = False
        self._player = None  # AdaptivePlayer reference
        self._text = ""  # full text being spoken
        self._samples_played = 0
        self._total_samples = 0
        self._last_interrupt: InterruptInfo | None = None

    # ------------------------------------------------------------------
    # Outbound side calls these
    # ------------------------------------------------------------------

    def start_speaking(self, text: str, player) -> None:
        """Called when TTS playback begins. Records the player reference and text."""
        with self._lock:
            self._speaking = True
            self._player = player
            self._text = text
            self._samples_played = 0
            self._total_samples = 0

    def stop_speaking(self) -> None:
        """Called when TTS playback finishes normally."""
        with self._lock:
            self._speaking = False
            self._player = None
            self._text = ""
            self._samples_played = 0
            self._total_samples = 0

    def update_position(self, samples_played: int, total_samples: int) -> None:
        """Called periodically by the player to track progress."""
        with self._lock:
            self._samples_played = samples_played
            self._total_samples = total_samples

    # ------------------------------------------------------------------
    # Inbound side calls these
    # ------------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is currently playing audio."""
        with self._lock:
            return self._speaking

    def interrupt(self, reason: str = "vad_reflex") -> InterruptInfo | None:
        """Halt current playback immediately. Returns interrupt info, or None if not speaking.

        This is the reflex arc: VAD fires -> interrupt() -> player.flush() -> silence.
        Must complete in < 50ms.
        """
        with self._lock:
            if not self._speaking:
                return None

            # Snapshot state before we tear it down
            player = self._player
            text = self._text
            pct = self._samples_played / self._total_samples if self._total_samples > 0 else 0.0

            # Clear speaking state immediately (inside lock)
            self._speaking = False
            self._player = None

        # Call flush outside the lock -- flush() has its own internal locking
        # and we don't want to hold our state lock while blocking on audio teardown.
        if player is not None:
            player.flush()

        info = InterruptInfo(
            timestamp=time.time(),
            spoken_pct=pct,
            delivered_text=self.delivered_text(text, pct),
            full_text=text,
            reason=reason,
        )

        with self._lock:
            self._last_interrupt = info
            self._text = ""
            self._samples_played = 0
            self._total_samples = 0

        return info

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def last_interrupt(self) -> InterruptInfo | None:
        """Most recent interruption, if any."""
        with self._lock:
            return self._last_interrupt

    @property
    def spoken_pct(self) -> float:
        """Current delivery progress (0.0-1.0). 0 if not speaking."""
        with self._lock:
            if not self._speaking or self._total_samples == 0:
                return 0.0
            return self._samples_played / self._total_samples

    @staticmethod
    def delivered_text(full_text: str, pct: float) -> str:
        """Estimate the text that was actually spoken given a percentage.

        Splits on word boundaries near the percentage point so we never
        cut a word in half.
        """
        if pct <= 0.0:
            return ""
        if pct >= 1.0:
            return full_text

        # Target character position
        target = int(pct * len(full_text))

        # Find the nearest word boundary at or before the target.
        # Walk backward from target to find the end of the last complete word.
        if target >= len(full_text):
            return full_text

        # If we're already at a space or end of text, trim trailing space
        if full_text[target] == " ":
            return full_text[:target].rstrip()

        # We're in the middle of a word -- find where this word started
        # and cut just before it (keeping only fully-spoken words).
        boundary = full_text.rfind(" ", 0, target)
        if boundary == -1:
            # We're inside the very first word. If we delivered more than
            # half of it, include it; otherwise return empty.
            if target >= len(full_text.split()[0]) / 2:
                return full_text.split()[0]
            return ""

        return full_text[:boundary].rstrip()

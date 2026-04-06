"""Inbound voice pipeline — continuous mic → VAD → STT → channel notification.

Runs a background thread that listens to the microphone via AudioCapture,
gates on Silero VAD to avoid waking Whisper on silence, accumulates speech
until an utterance boundary (silence window), then sends the complete
utterance through ModalityBus.perceive() for STT and BoH filtering.
Transcripts are emitted as MCP channel notifications to Claude Code.

Reflex arc: if TTS is playing when the user speaks, the pipeline calls
PipelineState.interrupt() to flush playback within ~50ms — no LLM
round-trip needed.

No side effects on import.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time

import numpy as np

from bus import ModalityBus
from capture import AudioCapture
from pipeline_state import PipelineState
from server import emit_channel_event, emit_permission_verdict
from vad import VADResult, detect_speech

# Matches verbal permission verdicts like "yes abcde", "n fghij" (case insensitive).
# The request_id is 5 lowercase letters (a-z excluding 'l').
PERMISSION_VERDICT_RE = re.compile(r"^\s*(y|yes|n|no)\s+([a-km-z]{5})\s*$", re.IGNORECASE)

logger = logging.getLogger("mod3.inbound")


class InboundPipeline:
    """Continuous voice input: mic → VAD → STT → channel notification.

    Runs in a background thread. Uses AudioCapture for mic input,
    ModalityBus.perceive() for the VAD→STT→BoH pipeline, and
    emit_channel_event() to send notifications to Claude Code.
    """

    def __init__(
        self,
        bus: ModalityBus,
        pipeline_state: PipelineState,
        capture: AudioCapture | None = None,
        chunk_duration_sec: float = 2.0,
        vad_threshold: float = 0.5,
        speaker: str = "user",
        sample_rate: int = 16000,
        min_silence_duration_sec: float = 0.5,
        loop_sleep_sec: float = 0.05,
    ):
        self._bus = bus
        self._pipeline_state = pipeline_state
        self._capture = capture or AudioCapture(sample_rate=sample_rate)
        self._chunk_sec = chunk_duration_sec
        self._vad_threshold = vad_threshold
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._min_silence_sec = min_silence_duration_sec
        self._loop_sleep_sec = loop_sleep_sec

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the listening loop in a background thread."""
        if self._running:
            return

        if not self._capture.is_active():
            self._capture.start()

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="inbound-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("inbound pipeline started")

    def stop(self) -> None:
        """Stop the listening loop and mic capture."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._capture.stop()
        self._running = False
        logger.info("inbound pipeline stopped")

    @property
    def is_running(self) -> bool:
        """Whether the listening loop is currently active."""
        return self._running and not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Listening loop (runs in background thread)
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Main loop: chunk → VAD pre-check → accumulate → STT → notify."""
        logger.debug("listen loop entered")

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("error in listen loop tick")
                # Brief pause on error to avoid tight spin
                self._stop_event.wait(0.5)

        logger.debug("listen loop exited")

    def _tick(self) -> None:
        """Single iteration of the listen loop."""

        # 1. Grab a chunk of audio from the ring buffer
        chunk = self._capture.get_audio(self._chunk_sec)
        if chunk is None:
            # Not enough data accumulated yet — wait and retry
            self._stop_event.wait(self._loop_sleep_sec)
            return

        # 2. Fast VAD pre-check (Silero, no Whisper)
        vad_result = detect_speech(
            chunk,
            sample_rate=self._sample_rate,
            threshold=self._vad_threshold,
        )

        if not vad_result.has_speech:
            # No speech — sleep briefly and loop
            self._stop_event.wait(self._loop_sleep_sec)
            return

        # 3. Speech detected — reflex arc: interrupt TTS if speaking
        if self._pipeline_state.is_speaking:
            interrupt_info = self._pipeline_state.interrupt("vad_reflex")
            if interrupt_info is not None:
                logger.info(
                    "reflex interrupt: spoken_pct=%.1f%% reason=%s",
                    interrupt_info.spoken_pct * 100,
                    interrupt_info.reason,
                )

        # 4. Accumulate audio until utterance boundary (silence window)
        utterance, final_vad = self._accumulate_utterance(chunk, vad_result)
        if utterance is None:
            return

        # 5. Send complete utterance through the bus pipeline (Gate → Whisper → BoH)
        audio_bytes = utterance.astype(np.float32).tobytes()
        event = self._bus.perceive(
            audio_bytes,
            modality="voice",
            channel="mod3-voice",
        )

        if event is None:
            # Gate rejected or hallucination filtered
            logger.debug("utterance filtered by bus pipeline")
            return

        # 6. Emit channel notification to Claude Code
        logger.info("transcript: %s (confidence=%.2f)", event.content[:80], event.confidence)
        self._emit_notification(event, final_vad)

    # ------------------------------------------------------------------
    # Speech accumulation
    # ------------------------------------------------------------------

    def _accumulate_utterance(
        self,
        initial_chunk: np.ndarray,
        initial_vad: VADResult,
    ) -> tuple[np.ndarray | None, VADResult]:
        """Read chunks until silence exceeds the silence window.

        Starts with the initial chunk that triggered speech detection,
        then keeps reading while VAD still reports speech. Once silence
        persists for min_silence_duration_sec, considers the utterance
        complete.

        Returns:
            (accumulated_audio, last_vad_result) or (None, last_vad) if
            the pipeline was stopped during accumulation.
        """
        chunks: list[np.ndarray] = [initial_chunk]
        last_speech_time = time.monotonic()
        last_vad = initial_vad

        while not self._stop_event.is_set():
            # Brief pause before grabbing the next chunk
            self._stop_event.wait(self._loop_sleep_sec)
            if self._stop_event.is_set():
                return None, last_vad

            chunk = self._capture.get_audio(self._chunk_sec)
            if chunk is None:
                continue

            vad_result = detect_speech(
                chunk,
                sample_rate=self._sample_rate,
                threshold=self._vad_threshold,
            )
            last_vad = vad_result

            if vad_result.has_speech:
                chunks.append(chunk)
                last_speech_time = time.monotonic()
            else:
                # Silence detected — check if we've exceeded the silence window
                silence_elapsed = time.monotonic() - last_speech_time
                if silence_elapsed >= self._min_silence_sec:
                    # Utterance boundary reached
                    break
                # Still within the grace period — keep accumulating
                # (include the silent tail so Whisper has context)
                chunks.append(chunk)

        if self._stop_event.is_set():
            return None, last_vad

        utterance = np.concatenate(chunks)
        duration = len(utterance) / self._sample_rate
        logger.debug(
            "utterance accumulated: %.1fs, %d chunks",
            duration,
            len(chunks),
        )
        return utterance, last_vad

    # ------------------------------------------------------------------
    # Notification delivery
    # ------------------------------------------------------------------

    def _emit_notification(self, event, vad_result: VADResult) -> None:
        """Send the transcript to Claude Code as a channel notification.

        If the transcript matches a permission verdict pattern (e.g. "yes abcde"),
        emits a permission verdict notification instead of a normal channel event.

        emit_channel_event() / emit_permission_verdict() are async; we run
        them synchronously from the background thread via asyncio.run().
        """
        # Check if this transcript is a permission verdict
        match = PERMISSION_VERDICT_RE.match(event.content)
        if match:
            request_id = match.group(2).lower()
            behavior = "allow" if match.group(1).lower().startswith("y") else "deny"
            logger.info(
                "permission verdict detected: %s %s (from: %r)",
                behavior,
                request_id,
                event.content,
            )
            try:
                asyncio.run(emit_permission_verdict(request_id, behavior))
            except RuntimeError as exc:
                logger.warning("failed to emit permission verdict: %s", exc)
            except Exception:
                logger.exception("unexpected error emitting permission verdict")
            return

        # Normal channel notification path
        try:
            asyncio.run(
                emit_channel_event(
                    content=event.content,
                    meta={
                        "source": "mod3-voice",
                        "speaker": self._speaker,
                        "confidence": str(round(event.confidence, 2)),
                        "speech_ratio": str(round(vad_result.speech_ratio, 2)),
                    },
                )
            )
        except RuntimeError as exc:
            # MCP session not active — log but don't crash the loop
            logger.warning("failed to emit channel event: %s", exc)
        except Exception:
            logger.exception("unexpected error emitting channel event")

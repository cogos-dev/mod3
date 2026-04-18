"""Browser channel — WebSocket adapter for the Mod³ dashboard.

Wraps a FastAPI WebSocket connection as a ChannelDescriptor on the bus.
Knows the WebSocket protocol (binary PCM / JSON control frames),
knows nothing about LLMs or agent logic.

Includes three-tier adaptive STT scheduler:
  T1 (Whisper Base, ~31ms): per-chunk during speech
  T2 (Whisper Large, ~470ms): on natural pause
  T3 (Whisper Large, ~470ms): on end-of-utterance (final)

Server→client WebSocket message types:
  audio, response_text, response_complete, interrupted,
  partial_transcript, transcript,
  trace_event  — kernel cycle-trace events (ADR-083), fanned out via
                 BrowserChannel.broadcast_trace_event().
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from bus import ModalityBus
from modality import CognitiveEvent, EncodedOutput, ModalityType
from modules.voice import WhisperDecoder
from pipeline_state import PipelineState

logger = logging.getLogger("mod3.channels")


class BrowserChannel:
    """WebSocket-backed channel for the browser dashboard."""

    # Registry of currently-active dashboard channels. Used by
    # broadcast_trace_event() to fan kernel cycle-trace events out to every
    # connected dashboard client (see ADR-083). Populated in __init__,
    # pruned in _cleanup.
    _active_channels: "set[BrowserChannel]" = set()

    def __init__(
        self,
        ws: WebSocket,
        bus: ModalityBus,
        pipeline_state: PipelineState,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[CognitiveEvent], Awaitable[None]] | None = None,
    ):
        self.ws = ws
        self.bus = bus
        self.pipeline_state = pipeline_state
        self._loop = loop
        self._on_event = on_event
        self.channel_id = f"browser:{uuid.uuid4().hex[:8]}"
        self.config: dict[str, Any] = {
            "voice": "bm_lewis",
            "speed": 1.25,
            "model": "kokoro",
        }
        self._audio_buffer = bytearray()
        self._active = True

        # Three-tier STT state
        self._streaming_decoder = WhisperDecoder(load_base=True)
        self._streaming_audio = bytearray()  # Growing buffer for streaming STT
        self._last_t1_time = 0.0  # Last T1 transcription time
        self._last_speech_time = 0.0  # Last time we received speech audio
        self._t1_interval = 0.3  # Run T1 every 300ms
        self._t2_pause_threshold = 0.6  # Run T2 after 600ms pause
        self._is_speaking = False  # Whether user is currently speaking
        self._t2_scheduled = False  # Whether T2 is already scheduled

        # Register on the bus with a delivery callback
        bus.register_channel(
            self.channel_id,
            modalities=[ModalityType.VOICE, ModalityType.TEXT],
            deliver=self._deliver_sync,
        )
        BrowserChannel._active_channels.add(self)
        logger.info("BrowserChannel registered: %s", self.channel_id)

    # ------------------------------------------------------------------
    # Delivery (bus → browser)
    # ------------------------------------------------------------------

    def _deliver_sync(self, output: EncodedOutput) -> None:
        """Called from the sync OutputQueue drain thread. Bridges to async."""
        if not self._active:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._deliver_async(output), self._loop)
            future.result(timeout=10.0)
        except (WebSocketDisconnect, RuntimeError, TimeoutError):
            logger.debug("deliver failed (client disconnected?), deactivating channel")
            self._active = False

    async def _deliver_async(self, output: EncodedOutput) -> None:
        """Send encoded output over the WebSocket."""
        import base64

        logger.info(
            "deliver: modality=%s format=%s bytes=%d duration=%.1fs",
            output.modality.value if output.modality else "none",
            output.format,
            len(output.data) if output.data else 0,
            output.duration_sec,
        )

        if output.modality == ModalityType.VOICE and output.data:
            # Send audio as base64 JSON (avoids binary frame issues)
            audio_b64 = base64.b64encode(output.data).decode("ascii")
            logger.info("deliver: sending base64 audio JSON (%d chars)", len(audio_b64))
            await self.ws.send_json(
                {
                    "type": "audio",
                    "data": audio_b64,
                    "format": output.format or "wav",
                    "duration_sec": round(output.duration_sec, 2),
                    "sample_rate": output.metadata.get("sample_rate", 24000),
                }
            )
            logger.info("deliver: audio sent OK")
        elif output.modality == ModalityType.TEXT:
            text = output.data.decode("utf-8") if isinstance(output.data, bytes) else str(output.data)
            logger.info("deliver: sending text response (%d chars)", len(text))
            await self.ws.send_json({"type": "response_text", "text": text})
        else:
            logger.warning("deliver: unhandled modality %s, dropping", output.modality)

    # ------------------------------------------------------------------
    # Receive loop (browser → server)
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main receive loop — runs until WebSocket disconnects."""
        try:
            while True:
                message = await self.ws.receive()
                msg_type = message.get("type", "")
                if msg_type == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"]:
                    self._handle_audio(message["bytes"])
                elif "text" in message and message["text"]:
                    await self._handle_json(json.loads(message["text"]))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("BrowserChannel error: %s", e)
        finally:
            self._cleanup()

    def _handle_audio(self, pcm_bytes: bytes) -> None:
        """Binary frame: raw Int16 PCM at 16kHz from browser Silero VAD.

        A5: Receives streaming audio during speech (from onFrameProcessed)
        AND the final complete buffer (from onSpeechEnd). Both accumulate
        for the final T3 utterance processing.

        During speech, audio also accumulates in _streaming_audio for T1/T2
        partial transcription.
        """
        self._audio_buffer.extend(pcm_bytes)
        self._streaming_audio.extend(pcm_bytes)
        self._last_speech_time = time.monotonic()
        self._is_speaking = True

        # T1: Fast Whisper Base transcription every _t1_interval
        now = time.monotonic()
        if now - self._last_t1_time >= self._t1_interval and len(self._streaming_audio) > 6400:
            self._last_t1_time = now
            asyncio.ensure_future(self._run_t1())

        # Schedule T2 check on pause detection
        if not self._t2_scheduled:
            asyncio.ensure_future(self._schedule_t2_on_pause())

    async def _handle_json(self, msg: dict) -> None:
        """JSON frame: control message dispatch."""
        msg_type = msg.get("type", "")
        logger.info("Received JSON: type=%s", msg_type)

        if msg_type == "end_of_speech":
            await self._process_utterance()
        elif msg_type == "text_message":
            text = msg.get("text", "").strip()
            if text:
                await self._process_text(text)
        elif msg_type == "interrupt":
            await self._handle_interrupt()
        elif msg_type == "config":
            for key in ("model", "voice", "speed"):
                if key in msg:
                    self.config[key] = msg[key]

    # ------------------------------------------------------------------
    # Three-Tier STT
    # ------------------------------------------------------------------

    async def _run_t1(self) -> None:
        """T1: Fast Whisper Base transcription on growing audio buffer (~31ms).

        Runs every ~300ms during speech. Emits partial_transcript with
        confirmed/tentative text at 30% opacity.
        """
        if not self._streaming_audio:
            return

        pcm_data = bytes(self._streaming_audio)

        def _transcribe_t1():
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 4800:  # <300ms
                return None
            return self._streaming_decoder.decode_streaming(audio, tier="t1")

        try:
            result = await asyncio.to_thread(_transcribe_t1)
            if result and result.get("changed") and not result.get("filtered"):
                await self.ws.send_json(
                    {
                        "type": "partial_transcript",
                        "confirmed": result["confirmed"],
                        "tentative": result["tentative"],
                        "tier": "t1",
                        "elapsed_ms": result["elapsed_ms"],
                    }
                )
        except Exception as e:
            logger.debug("T1 error: %s", e)

    async def _run_t2(self) -> None:
        """T2: Large model transcription on natural pause (~470ms).

        Runs when speech pauses for >600ms but hasn't ended. Emits
        partial_transcript with higher confidence (60% opacity).
        """
        if not self._streaming_audio:
            return

        pcm_data = bytes(self._streaming_audio)

        def _transcribe_t2():
            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) < 8000:  # <500ms
                return None
            return self._streaming_decoder.decode_streaming(audio, tier="t2")

        try:
            result = await asyncio.to_thread(_transcribe_t2)
            if result and not result.get("filtered"):
                await self.ws.send_json(
                    {
                        "type": "partial_transcript",
                        "confirmed": result["confirmed"],
                        "tentative": result["tentative"],
                        "tier": "t2",
                        "elapsed_ms": result["elapsed_ms"],
                    }
                )
        except Exception as e:
            logger.debug("T2 error: %s", e)
        finally:
            self._t2_scheduled = False

    async def _schedule_t2_on_pause(self) -> None:
        """Check if speech has paused long enough for T2."""
        await asyncio.sleep(self._t2_pause_threshold)
        if not self._is_speaking:
            return
        # Check if there's been a pause since last audio
        silence = time.monotonic() - self._last_speech_time
        if silence >= self._t2_pause_threshold and not self._t2_scheduled:
            self._t2_scheduled = True
            await self._run_t2()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def _process_utterance(self) -> None:
        """T3: PCM audio buffer → WhisperDecoder STT → CognitiveEvent → agent loop.

        This is the final tier — end-of-utterance. Uses the Large model for
        maximum accuracy. Everything is confirmed (100% opacity).

        Skips the server-side VoiceGate (Silero VAD) because the browser
        already ran Silero VAD client-side — no need to validate again,
        and it avoids the torchaudio dependency for resampling.
        """
        pcm_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        # Reset streaming state
        self._streaming_audio.clear()
        self._streaming_decoder.reset_streaming()
        self._is_speaking = False

        if len(pcm_data) < 6400:  # <200ms at 16kHz Int16
            return

        t0 = time.perf_counter()

        # Transcribe via mlx_whisper — needs a temp WAV file
        def _transcribe():
            import io
            import os
            import struct
            import tempfile

            import mlx_whisper
            import numpy as np

            from vad import is_hallucination

            audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Skip silence
            if len(audio) < 16000 * 0.3:
                return None
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 0.005:
                return None

            # mlx_whisper needs a file path — write temp WAV
            buf = io.BytesIO()
            buf.write(b"RIFF")
            buf.write(struct.pack("<I", 36 + len(pcm_data)))
            buf.write(b"WAVE")
            buf.write(b"fmt ")
            buf.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16))
            buf.write(b"data")
            buf.write(struct.pack("<I", len(pcm_data)))
            buf.write(pcm_data)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(buf.getvalue())
                tmp_path = f.name

            try:
                result = mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    language="en",
                )
                transcript = result.get("text", "").strip()
                logger.info("STT: '%s' (%.1fs, rms=%.3f)", transcript[:80], len(audio) / 16000, rms)

                if not transcript or is_hallucination(transcript):
                    return None

                return CognitiveEvent(
                    modality=ModalityType.VOICE,
                    content=transcript,
                    source_channel=self.channel_id,
                    confidence=1.0,
                )
            except Exception as e:
                logger.error("STT failed: %s", e)
                return None
            finally:
                os.unlink(tmp_path)

        event = await asyncio.to_thread(_transcribe)

        stt_ms = (time.perf_counter() - t0) * 1000

        if event and event.content:
            # Send transcript to browser
            await self.ws.send_json(
                {
                    "type": "transcript",
                    "text": event.content,
                    "stt_ms": round(stt_ms, 1),
                    "source": "voice",
                }
            )
            # Forward to agent loop
            event.metadata["stt_ms"] = stt_ms
            if self._on_event:
                await self._on_event(event)

    async def _process_text(self, text: str) -> None:
        """Text message → CognitiveEvent → agent loop."""
        event = CognitiveEvent(
            modality=ModalityType.TEXT,
            content=text,
            source_channel=self.channel_id,
            confidence=1.0,
        )
        await self.ws.send_json(
            {
                "type": "transcript",
                "text": text,
                "source": "text",
            }
        )
        if self._on_event:
            await self._on_event(event)

    async def _handle_interrupt(self) -> None:
        """Interrupt in-flight speech."""
        if self.pipeline_state.is_speaking:
            self.pipeline_state.interrupt(reason="browser_interrupt")
        await self.ws.send_json({"type": "interrupted"})

    # ------------------------------------------------------------------
    # Helper methods (called by agent loop)
    # ------------------------------------------------------------------

    async def send_response_text(self, text: str) -> None:
        """Send response text for display in chat panel."""
        if self._active:
            try:
                logger.info("send_response_text: %s", text[:100])
                await self.ws.send_json({"type": "response_text", "text": text})
            except Exception:
                self._active = False

    async def send_response_complete(self, metrics: dict | None = None) -> None:
        """Signal response is complete."""
        if self._active:
            try:
                await self.ws.send_json(
                    {
                        "type": "response_complete",
                        "metrics": metrics or {},
                    }
                )
            except Exception:
                self._active = False

    # ------------------------------------------------------------------
    # Trace event broadcast (kernel cycle-trace → dashboards)
    # ------------------------------------------------------------------

    @classmethod
    def broadcast_trace_event(cls, event: dict) -> None:
        """Fan a kernel cycle-trace event out to every connected dashboard.

        Per ADR-083, `event` is a pre-parsed CycleEvent dict
        (id, ts, source, cycle_id, kind, payload). Wrapped in the
        `{"type": "trace_event", "event": ...}` envelope and sent to each
        active BrowserChannel's WebSocket. Clients whose send fails are
        skipped silently (they will be pruned by their own disconnect path).
        """
        frame = {"type": "trace_event", "event": event}
        for ch in list(cls._active_channels):
            if not ch._active:
                continue
            try:
                asyncio.run_coroutine_threadsafe(ch.ws.send_json(frame), ch._loop)
            except Exception as exc:  # noqa: BLE001 — disconnected clients are expected
                logger.debug("trace_event send failed for %s: %s", ch.channel_id, exc)

    @classmethod
    def broadcast_response_text(cls, text: str, session_id: str | None = None) -> None:
        """Push an agent-reply text frame to dashboard WebSocket clients.

        Used by the MOD3_USE_COGOS_AGENT response bridge (see
        `cogos_agent_bridge.run_response_bridge`). The frame matches the
        existing text-response shape emitted by `_deliver_async` and
        `send_response_text`: `{"type": "response_text", "text": <text>}`.

        If `session_id` is None (default) the frame is broadcast to every
        active dashboard channel. When provided, only channels whose
        `channel_id` matches the `mod3:<channel_id>` convention from
        `cogos_agent_bridge.post_user_message` receive the frame — this is
        how future multi-user routing will land, but for v1 a None
        broadcast is the common case (only one dashboard attached).

        Thread-safe: dispatches each WS send via `run_coroutine_threadsafe`
        on the channel's own loop, matching `broadcast_trace_event`.
        """
        frame = {"type": "response_text", "text": text}
        expected_channel = None
        if session_id and session_id.startswith("mod3:"):
            expected_channel = session_id[len("mod3:"):]
        for ch in list(cls._active_channels):
            if not ch._active:
                continue
            if expected_channel and ch.channel_id != expected_channel:
                continue
            try:
                asyncio.run_coroutine_threadsafe(ch.ws.send_json(frame), ch._loop)
            except Exception as exc:  # noqa: BLE001 — disconnected clients are expected
                logger.debug("response_text send failed for %s: %s", ch.channel_id, exc)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Deactivate channel and cancel pending TTS jobs on disconnect."""
        self._active = False
        BrowserChannel._active_channels.discard(self)
        ch = self.bus._channels.get(self.channel_id)
        if ch:
            ch.active = False
        cancelled = self.bus._queue_manager.cancel_channel(self.channel_id)
        logger.info(
            "BrowserChannel disconnected: %s (cancelled %d pending jobs)",
            self.channel_id,
            cancelled,
        )

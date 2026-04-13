"""Browser channel — WebSocket adapter for the Mod³ dashboard.

Wraps a FastAPI WebSocket connection as a ChannelDescriptor on the bus.
Knows the WebSocket protocol (binary PCM / JSON control frames),
knows nothing about LLMs or agent logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

from bus import ModalityBus
from modality import CognitiveEvent, EncodedOutput, ModalityType
from pipeline_state import PipelineState

logger = logging.getLogger("mod3.channels")


class BrowserChannel:
    """WebSocket-backed channel for the browser dashboard."""

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

        # Register on the bus with a delivery callback
        bus.register_channel(
            self.channel_id,
            modalities=[ModalityType.VOICE, ModalityType.TEXT],
            deliver=self._deliver_sync,
        )
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
        """Binary frame: raw Int16 PCM at 16kHz from browser Silero VAD."""
        self._audio_buffer.extend(pcm_bytes)

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
    # Processing
    # ------------------------------------------------------------------

    async def _process_utterance(self) -> None:
        """PCM audio buffer → WhisperDecoder STT → CognitiveEvent → agent loop.

        Skips the server-side VoiceGate (Silero VAD) because the browser
        already ran Silero VAD client-side — no need to validate again,
        and it avoids the torchaudio dependency for resampling.
        """
        pcm_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

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
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Deactivate channel and cancel pending TTS jobs on disconnect."""
        self._active = False
        ch = self.bus._channels.get(self.channel_id)
        if ch:
            ch.active = False
        cancelled = self.bus._queue_manager.cancel_channel(self.channel_id)
        logger.info(
            "BrowserChannel disconnected: %s (cancelled %d pending jobs)",
            self.channel_id,
            cancelled,
        )

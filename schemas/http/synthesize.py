"""HTTP schema — POST /v1/synthesize and POST /v1/audio/speech."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class SynthesizeRequest(_Base):
    """POST /v1/synthesize — text to audio (WAV, PCM, or OGG/Opus)."""

    text: str
    voice: str = Field(default="bm_lewis")
    speed: float = Field(default=1.25)
    emotion: float = Field(default=0.5)
    format: str = Field(default="wav", pattern="^(wav|pcm|ogg)$")
    # ADR-082 Phase 1: optional session routing. When present, the
    # session's assigned_voice overrides ``voice`` (unless an explicit
    # non-default was passed), and the session is advanced in the global
    # serializer's round-robin.
    session_id: str | None = Field(default=None)
    # Path to a reference WAV for zero-shot voice cloning. Honored by the
    # chatterbox engine (24 kHz, mono). Other engines ignore it.
    ref_audio: str | None = Field(default=None)


class SpeechRequest(_Base):
    """POST /v1/audio/speech — OpenAI-compatible TTS endpoint."""

    model: str = Field(default="kokoro")
    input: str
    voice: str = Field(default="af_heart")
    response_format: str = Field(default="mp3")
    speed: float = Field(default=1.0)
    # ADR-082 Phase 1 extension — not part of the OpenAI schema but harmless
    # to accept. When absent, behavior is identical to before Phase 1.
    session_id: str | None = Field(default=None)


class SpeakRequest(_Base):
    """POST /v1/speak — queue-aware speak endpoint.

    Wraps _start_speech from server.py. Returns immediately after enqueue
    with {job_id, queue_position, status}. Mod3's drain thread owns all
    audio playback — callers do NOT manage afplay/aplay.
    """

    text: str
    voice: str = Field(default="bm_lewis")
    stream: bool = Field(default=True)
    speed: float = Field(default=1.25)
    emotion: float = Field(default=0.5)
    session_id: str = Field(default="")
    ref_audio: str = Field(default="")
    format: str = Field(default="wav", pattern="^(wav|pcm|ogg)$")


__all__ = ["SpeakRequest", "SpeechRequest", "SynthesizeRequest"]

"""Voice modality module — the first non-trivial modality.

Gate:    Silero VAD (is there speech?)
Decoder: WhisperDecoder — mlx_whisper STT with BoH hallucination filter
Encoder: Mod³ TTS engines (Kokoro, Voxtral, Chatterbox, Spark)

The encoder wraps engine.py and adaptive_player.py for local speaker output,
or returns raw audio bytes for channel delivery (Discord, HTTP).
"""

from __future__ import annotations

import io
import logging
import struct
import time

import numpy as np

from modality import (
    CognitiveEvent,
    CognitiveIntent,
    Decoder,
    EncodedOutput,
    Encoder,
    Gate,
    GateResult,
    ModalityModule,
    ModalityType,
    ModuleState,
    ModuleStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate: Silero VAD
# ---------------------------------------------------------------------------


class VoiceGate(Gate):
    """Voice activity detection gate using Silero VAD."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def check(self, raw: bytes, **kwargs) -> GateResult:
        from vad import detect_speech

        sample_rate = kwargs.get("sample_rate", 16000)
        sample_width = kwargs.get("sample_width", 2)

        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(raw, dtype=np.float32)

        result = detect_speech(audio, sample_rate=sample_rate, threshold=self.threshold)

        return GateResult(
            passed=result.has_speech,
            confidence=result.confidence,
            reason=f"speech_ratio={result.speech_ratio} segments={result.num_segments}",
            metadata={
                "speech_ratio": result.speech_ratio,
                "num_segments": result.num_segments,
                "total_speech_sec": result.total_speech_sec,
                "total_audio_sec": result.total_audio_sec,
            },
        )


# ---------------------------------------------------------------------------
# Decoder: WhisperDecoder — mlx_whisper STT
# ---------------------------------------------------------------------------


class WhisperDecoder(Decoder):
    """Speech-to-text decoder using mlx_whisper on Apple Silicon.

    Accepts PCM float32 bytes at 16kHz or a numpy float32 array directly.
    Lazy-loads the model on first call; subsequent calls reuse it.
    Applies BoH hallucination filter to transcripts.

    Supports two models:
    - Large (whisper-large-v3-turbo): high-quality, used for T2/T3 tiers (~470ms)
    - Base (whisper-base-mlx): fast, used for T1 tier (~31ms)
    """

    # Downgraded from whisper-large-v3-turbo to base to reduce MLX Metal
    # pressure (Gemma + Kokoro + Whisper concurrent load segfaults).
    DEFAULT_MODEL = "mlx-community/whisper-base-mlx"
    BASE_MODEL = "mlx-community/whisper-base-mlx"

    def __init__(self, model: str | None = None, load_base: bool = True):
        self._model = model or self.DEFAULT_MODEL
        self._loaded = False
        self._base_loaded = False
        self._load_base = load_base
        # Streaming state: last transcript for diff-based partial detection
        self._last_streaming_text: str = ""

    def _ensure_model(self) -> None:
        """Trigger model download/load on first use."""
        if not self._loaded:
            import mlx_whisper

            logger.info("WhisperDecoder: loading model %s (first call)", self._model)
            mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32),  # 1 s of silence
                path_or_hf_repo=self._model,
            )
            self._loaded = True
            logger.info("WhisperDecoder: model ready")

    def _ensure_base_model(self) -> None:
        """Load Whisper Base model for T1 fast transcription."""
        if not self._base_loaded:
            import mlx_whisper

            logger.info("WhisperDecoder: loading base model %s", self.BASE_MODEL)
            mlx_whisper.transcribe(
                np.zeros(16000, dtype=np.float32),
                path_or_hf_repo=self.BASE_MODEL,
            )
            self._base_loaded = True
            logger.info("WhisperDecoder: base model ready")

    def decode_streaming(
        self,
        audio: np.ndarray,
        tier: str = "t1",
        **kwargs,
    ) -> dict:
        """Chunked re-transcription with LocalAgreement-2 diff.

        Re-runs mlx_whisper.transcribe() on the growing audio buffer,
        diffs consecutive outputs to produce confirmed vs tentative text.

        Args:
            audio: Growing float32 audio buffer at 16kHz.
            tier: "t1" (Base, fast), "t2" (Large, on pause), "t3" (Large, final).

        Returns:
            dict with keys:
              - confirmed: str — text stable across 2+ consecutive runs
              - tentative: str — new text not yet confirmed
              - full_text: str — complete transcript from this run
              - tier: str — which tier was used
              - changed: bool — whether output differs from last run
        """
        import mlx_whisper

        from vad import is_hallucination

        # Select model based on tier
        if tier == "t1":
            self._ensure_base_model()
            model_path = self.BASE_MODEL
        else:
            self._ensure_model()
            model_path = self._model

        t0 = time.time()
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_path,
            language="en",
        )
        elapsed_ms = (time.time() - t0) * 1000

        transcript: str = result.get("text", "").strip()

        if is_hallucination(transcript):
            return {
                "confirmed": "",
                "tentative": "",
                "full_text": "",
                "tier": tier,
                "changed": False,
                "elapsed_ms": round(elapsed_ms, 1),
                "filtered": True,
            }

        # LocalAgreement-2 diff: find longest common prefix with last run
        prev = self._last_streaming_text
        changed = transcript != prev

        # Confirmed = common prefix (stable across consecutive runs)
        confirmed = ""
        min_len = min(len(prev), len(transcript))
        for i in range(min_len):
            if prev[i] == transcript[i]:
                confirmed = transcript[: i + 1]
            else:
                break

        # Snap to word boundary
        if confirmed and not confirmed.endswith(" "):
            last_space = confirmed.rfind(" ")
            if last_space > 0:
                confirmed = confirmed[:last_space]

        # Tentative = remainder after confirmed prefix
        tentative = transcript[len(confirmed) :].strip()

        # T3 = end-of-utterance, everything is confirmed
        if tier == "t3":
            confirmed = transcript
            tentative = ""

        self._last_streaming_text = transcript

        return {
            "confirmed": confirmed.strip(),
            "tentative": tentative,
            "full_text": transcript,
            "tier": tier,
            "changed": changed,
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def reset_streaming(self) -> None:
        """Reset streaming state between utterances."""
        self._last_streaming_text = ""

    def validate_tts_output(self, audio_samples: np.ndarray, source_text: str, sample_rate: int = 24000) -> dict:
        """Whisper validation loop: run TTS audio through Whisper Base and compare.

        After TTS generates an audio chunk, run it through Whisper Base (~31ms)
        and compare transcript to source text. Flag mismatches.

        Args:
            audio_samples: Float32 audio samples from TTS.
            source_text: The original text that was synthesized.
            sample_rate: Sample rate of the TTS audio.

        Returns:
            dict with keys:
              - match: bool — whether transcript matches source
              - transcript: str — what Whisper heard
              - source: str — original text
              - similarity: float — 0.0-1.0 word overlap ratio
              - elapsed_ms: float
        """
        import mlx_whisper

        self._ensure_base_model()

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple linear resampling
            ratio = 16000 / sample_rate
            new_len = int(len(audio_samples) * ratio)
            indices = np.linspace(0, len(audio_samples) - 1, new_len)
            audio_16k = np.interp(indices, np.arange(len(audio_samples)), audio_samples).astype(np.float32)
        else:
            audio_16k = audio_samples

        t0 = time.time()
        result = mlx_whisper.transcribe(
            audio_16k,
            path_or_hf_repo=self.BASE_MODEL,
            language="en",
        )
        elapsed_ms = (time.time() - t0) * 1000

        transcript = result.get("text", "").strip().lower()
        source_clean = source_text.strip().lower()

        # Word-level similarity
        source_words = set(source_clean.split())
        transcript_words = set(transcript.split())

        if source_words:
            overlap = len(source_words & transcript_words)
            similarity = overlap / len(source_words)
        else:
            similarity = 1.0 if not transcript_words else 0.0

        # Match if similarity >= 0.7 (TTS output may have minor variations)
        match = similarity >= 0.7

        return {
            "match": match,
            "transcript": transcript,
            "source": source_text,
            "similarity": round(similarity, 3),
            "elapsed_ms": round(elapsed_ms, 1),
        }

    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent:
        import mlx_whisper

        from vad import is_hallucination

        # Accept a numpy array directly via kwarg, or convert raw bytes.
        audio: np.ndarray | None = kwargs.get("audio")
        if audio is None:
            audio = np.frombuffer(raw, dtype=np.float32)

        self._ensure_model()

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model,
        )

        transcript: str = result.get("text", "").strip()
        language: str = result.get("language", "")

        # Confidence heuristic: average segment-level no_speech_prob inverted.
        segments = result.get("segments", [])
        if segments:
            avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
            confidence = round(1.0 - avg_no_speech, 4)
        else:
            confidence = 0.0

        if is_hallucination(transcript):
            return CognitiveEvent(
                modality=ModalityType.VOICE,
                content="",
                confidence=0.0,
                metadata={"filtered": True, "reason": "hallucination", "original": transcript},
            )

        return CognitiveEvent(
            modality=ModalityType.VOICE,
            content=transcript,
            source_channel=kwargs.get("channel", ""),
            confidence=confidence,
            metadata={
                "language": language,
                "num_segments": len(segments),
            },
        )


# ---------------------------------------------------------------------------
# Decoder: PlaceholderDecoder — legacy pre-transcribed text path
# ---------------------------------------------------------------------------


class PlaceholderDecoder(Decoder):
    """Accepts pre-transcribed text and wraps it as a CognitiveEvent.

    Retained for reference and as a fallback when mlx_whisper is unavailable.
    """

    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent:
        transcript = kwargs.get("transcript")
        if transcript is None:
            transcript = raw.decode("utf-8", errors="replace")

        from vad import is_hallucination

        if is_hallucination(transcript):
            return CognitiveEvent(
                modality=ModalityType.VOICE,
                content="",
                confidence=0.0,
                metadata={"filtered": True, "reason": "hallucination", "original": transcript},
            )

        return CognitiveEvent(
            modality=ModalityType.VOICE,
            content=transcript,
            source_channel=kwargs.get("channel", ""),
            confidence=kwargs.get("confidence", 0.9),
        )


# ---------------------------------------------------------------------------
# Encoder: Mod³ TTS
# ---------------------------------------------------------------------------


def _encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 samples as 16-bit PCM WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    data_size = len(pcm) * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HH", 1, 1))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))
    buf.write(struct.pack("<HH", 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


class VoiceEncoder(Encoder):
    """TTS encoder using Mod³ engine (Kokoro, Voxtral, Chatterbox, Spark)."""

    def __init__(self, default_voice: str = "bm_lewis", default_speed: float = 1.25):
        self.default_voice = default_voice
        self.default_speed = default_speed
        self._state = ModuleState()

    @property
    def state(self) -> ModuleState:
        return self._state

    def encode(self, intent: CognitiveIntent) -> EncodedOutput:
        from engine import synthesize

        voice = intent.metadata.get("voice", self.default_voice)
        speed = intent.metadata.get("speed", self.default_speed)
        emotion = intent.metadata.get("emotion", 0.5)

        self._state.status = ModuleStatus.ENCODING
        self._state.current_text = intent.content[:100]
        self._state.last_activity = time.time()

        try:
            samples, sample_rate = synthesize(
                intent.content,
                voice=voice,
                speed=speed,
                emotion=emotion,
            )

            wav_bytes = _encode_wav(samples, sample_rate)
            duration = len(samples) / sample_rate

            self._state.status = ModuleStatus.IDLE
            self._state.last_output_text = intent.content[:100]
            self._state.progress = 1.0

            return EncodedOutput(
                modality=ModalityType.VOICE,
                data=wav_bytes,
                format="wav",
                duration_sec=duration,
                metadata={
                    "voice": voice,
                    "speed": speed,
                    "sample_rate": sample_rate,
                    "total_samples": len(samples),
                },
            )
        except Exception as e:
            self._state.status = ModuleStatus.ERROR
            self._state.error = str(e)
            raise


# ---------------------------------------------------------------------------
# Voice module
# ---------------------------------------------------------------------------


class VoiceModule(ModalityModule):
    """Voice modality — VAD gate, STT decoder, TTS encoder."""

    def __init__(
        self,
        decoder: Decoder | None = None,
        default_voice: str = "bm_lewis",
        default_speed: float = 1.25,
        vad_threshold: float = 0.5,
    ):
        self._gate = VoiceGate(threshold=vad_threshold)
        self._decoder = decoder or WhisperDecoder()
        self._encoder = VoiceEncoder(default_voice=default_voice, default_speed=default_speed)

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.VOICE

    @property
    def gate(self) -> Gate:
        return self._gate

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def state(self) -> ModuleState:
        return self._encoder.state

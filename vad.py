"""Silero VAD — Voice Activity Detection for the Mod³ input pipeline.

Detects whether an audio segment contains speech before sending it
to STT. Prevents Whisper hallucinations on silence/noise.

Also includes a Bag of Hallucinations (BoH) post-filter for known
phantom transcription phrases.
"""

import threading
from dataclasses import dataclass

import numpy as np
import torch

_model = None
_model_lock = threading.Lock()
_utils = None


def _get_model():
    """Load Silero VAD model (lazy, thread-safe)."""
    global _model, _utils
    if _model is None:
        with _model_lock:
            if _model is None:
                m, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                )
                _model = m
                _utils = utils
    return _model, _utils


def is_model_loaded() -> bool:
    return _model is not None


@dataclass
class VADResult:
    has_speech: bool
    confidence: float
    speech_ratio: float
    num_segments: int
    total_speech_sec: float
    total_audio_sec: float


def detect_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
) -> VADResult:
    """Check if audio contains speech.

    Args:
        audio: Float32 numpy array, mono.
        sample_rate: Sample rate of audio (will resample to 16kHz if needed).
        threshold: Speech probability threshold (0-1). Higher = stricter.
        min_speech_duration_ms: Minimum speech segment length to count.
        min_silence_duration_ms: Minimum silence between speech segments.

    Returns:
        VADResult with speech detection details.
    """
    model, utils = _get_model()
    assert utils is not None, "VAD utils not loaded"
    get_speech_timestamps = utils[0]

    # Silero VAD expects 16kHz mono
    if sample_rate != 16000:
        import torchaudio.functional as F

        tensor = torch.from_numpy(audio).float()
        tensor = F.resample(tensor, orig_freq=sample_rate, new_freq=16000)
    else:
        tensor = torch.from_numpy(audio).float()

    # Ensure 1D
    if tensor.dim() > 1:
        tensor = tensor.mean(dim=0)

    total_audio_sec = len(tensor) / 16000

    timestamps = get_speech_timestamps(
        tensor,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=False,
        sampling_rate=16000,
    )

    total_speech_samples = sum(ts["end"] - ts["start"] for ts in timestamps)
    total_speech_sec = total_speech_samples / 16000
    speech_ratio = total_speech_sec / total_audio_sec if total_audio_sec > 0 else 0.0

    # Confidence: max speech probability across segments, or 0 if no speech
    if timestamps:
        # Run a quick pass to get the max probability
        confidence = min(1.0, speech_ratio * 2)  # Heuristic from ratio
    else:
        confidence = 0.0

    return VADResult(
        has_speech=len(timestamps) > 0,
        confidence=round(confidence, 3),
        speech_ratio=round(speech_ratio, 3),
        num_segments=len(timestamps),
        total_speech_sec=round(total_speech_sec, 3),
        total_audio_sec=round(total_audio_sec, 3),
    )


def detect_speech_file(file_path: str, threshold: float = 0.5) -> VADResult:
    """Run VAD on a WAV file."""
    import wave

    with wave.open(file_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return detect_speech(audio, sample_rate=sample_rate, threshold=threshold)


# ---------------------------------------------------------------------------
# Bag of Hallucinations (BoH) post-filter
# ---------------------------------------------------------------------------

# Common Whisper phantom phrases generated from silence/noise
# Source: arxiv:2501.11378 + community reports
HALLUCINATION_PHRASES = frozenset(
    {
        "thank you",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "thanks for listening",
        "thank you for listening",
        "please subscribe",
        "subscribe",
        "like and subscribe",
        "see you next time",
        "bye",
        "goodbye",
        "you",
        "the end",
        "i'll see you in the next one",
        "i'll see you in the next video",
        "music",
        "applause",
        "laughter",
        "...",
        "",
    }
)


def is_hallucination(text: str) -> bool:
    """Check if transcription is a known Whisper hallucination."""
    cleaned = text.strip().lower().rstrip(".!?,")
    return cleaned in HALLUCINATION_PHRASES

"""Mod³ inference core — model registry, loading, and audio generation.

No MCP or playback dependencies. Takes text + params, yields numpy audio chunks.
"""

import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "voxtral": {
        "id": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "voices": [
            "casual_male",
            "casual_female",
            "cheerful_female",
            "neutral_male",
            "neutral_female",
            "fr_male",
            "fr_female",
            "es_male",
            "es_female",
            "de_male",
            "de_female",
            "it_male",
            "it_female",
            "pt_male",
            "pt_female",
            "nl_male",
            "nl_female",
            "ar_male",
            "hi_male",
            "hi_female",
        ],
        "default_voice": "casual_male",
    },
    "kokoro": {
        "id": "mlx-community/Kokoro-82M-bf16",
        "voices": [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "af_sky",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ],
        "default_voice": "af_heart",
        "supports_speed": True,
    },
    "chatterbox": {
        "id": "mlx-community/chatterbox-4bit",
        "voices": ["chatterbox"],
        "default_voice": "chatterbox",
        "supports_exaggeration": True,
    },
    "spark": {
        "id": "mlx-community/Spark-TTS-0.5B-bf16",
        "voices": ["spark_male", "spark_female"],
        "default_voice": "spark_male",
        "supports_pitch": True,
        "supports_speed": True,
    },
}

_models: dict = {}
_model_lock = threading.Lock()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using pysbd."""
    sentences = _segmenter.segment(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def resolve_model(voice: str) -> tuple[str, str]:
    """Given a voice name, return (engine_name, voice) or raise."""
    for engine, cfg in MODELS.items():
        if voice in cfg["voices"]:
            return engine, voice
    raise ValueError(f"Unknown voice '{voice}'. Use list_voices() to see options.")


def get_model(engine: str):
    """Load and cache an engine's model. Thread-safe."""
    if engine not in _models:
        with _model_lock:
            if engine not in _models:
                from mlx_audio.tts import load

                _models[engine] = load(MODELS[engine]["id"])
    return _models[engine]


def get_loaded_engines() -> list[str]:
    """Return names of currently loaded engines."""
    return list(_models.keys())


# ---------------------------------------------------------------------------
# Audio chunk
# ---------------------------------------------------------------------------


@dataclass
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    metadata: dict


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_audio(
    text: str,
    voice: str = "bm_lewis",
    speed: float = 1.25,
    emotion: float = 0.5,
    stream: bool = True,
    streaming_interval: float = 1.0,
) -> Iterator[AudioChunk]:
    """Yield AudioChunks for the given text. Core generation pipeline."""
    engine, voice = resolve_model(voice)
    model = get_model(engine)
    sample_rate = model.sample_rate
    sentences = split_sentences(text)
    feather = int(sample_rate * 0.02)

    for si, sentence in enumerate(sentences):
        gen_kwargs: dict[str, object] = dict(text=sentence, verbose=False)
        cfg = MODELS[engine]
        if engine == "chatterbox":
            gen_kwargs["exaggeration"] = emotion
            gen_kwargs["stream"] = stream
            gen_kwargs["streaming_interval"] = streaming_interval
        elif engine == "spark":
            gen_kwargs["gender"] = "female" if voice == "spark_female" else "male"
            gen_kwargs["speed"] = speed
        else:
            gen_kwargs["voice"] = voice
            if cfg.get("supports_speed"):
                gen_kwargs["speed"] = speed
            else:
                gen_kwargs["stream"] = stream
                gen_kwargs["streaming_interval"] = streaming_interval

        for result in model.generate(**gen_kwargs):
            audio = np.array(result.audio).flatten().astype(np.float32)
            metadata = {
                "gen_time_sec": round(result.processing_time_seconds, 4),
                "rtf": round(result.real_time_factor, 2),
                "samples": int(result.samples),
                "tokens": result.token_count,
                "is_final": result.is_final_chunk,
                "sentence": si,
                "peak_memory_gb": round(result.peak_memory_usage, 2),
            }

            if result.is_final_chunk and len(audio) > feather:
                audio = audio.copy()
                audio[-feather:] *= np.linspace(1, 0, feather, dtype=np.float32)

            yield AudioChunk(samples=audio, sample_rate=sample_rate, metadata=metadata)

        # Adaptive sentence gap
        if si < len(sentences) - 1:
            gap_sec = min(0.2, 0.05 + len(sentence) * 0.001)
            gap = np.zeros(int(sample_rate * gap_sec), dtype=np.float32)
            yield AudioChunk(samples=gap, sample_rate=sample_rate, metadata={})


def synthesize(
    text: str,
    voice: str = "bm_lewis",
    speed: float = 1.25,
    emotion: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Generate complete audio. Returns (concatenated_samples, sample_rate)."""
    chunks = list(generate_audio(text, voice=voice, speed=speed, emotion=emotion, stream=False))
    if not chunks:
        return np.array([], dtype=np.float32), 24000
    sample_rate = chunks[0].sample_rate
    all_samples = np.concatenate([c.samples for c in chunks])
    return all_samples, sample_rate

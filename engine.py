"""Mod³ inference core — model registry, loading, and audio generation.

No MCP or playback dependencies. Takes text + params, yields numpy audio chunks.
"""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice profile registry (lazy — imported only on first use)
# ---------------------------------------------------------------------------

_profile_registry = None


def _get_profile_registry():
    """Lazy-init the VoiceProfileRegistry to avoid an import cycle at module load."""
    global _profile_registry
    if _profile_registry is None:
        try:
            from voice_profiles import VoiceProfileRegistry  # noqa: PLC0415

            _profile_registry = VoiceProfileRegistry()
        except Exception as exc:
            _log.warning("voice profile registry unavailable: %s", exc)
            return None
    return _profile_registry


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
        "supports_cloning": True,
    },
    "chatterbox-turbo": {
        "id": "mlx-community/chatterbox-turbo-fp16",
        "voices": ["chatterbox-turbo"],
        "default_voice": "chatterbox-turbo",
        "supports_exaggeration": True,
        "supports_cloning": True,
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

# ---------------------------------------------------------------------------
# Idle-unload subsystem (MOD3_TTS_IDLE_UNLOAD_SECONDS)
# ---------------------------------------------------------------------------
# When MOD3_TTS_IDLE_UNLOAD_SECONDS > 0 a daemon thread watches _last_use_ts.
# After the idle threshold passes with no synthesize activity the loaded models
# are evicted from _models, gc.collect() is called, and MLX Metal memory is
# cleared.  The next get_model() / generate_audio() / synthesize() call
# transparently reloads from disk.
#
# Default is 0 (disabled) so existing behaviour is unchanged.
# ---------------------------------------------------------------------------

_last_use_ts: float = 0.0  # epoch seconds; updated inside _model_lock on each use
_idle_unload_enabled: bool = False
_idle_unload_seconds: float = 0.0


def _parse_idle_unload_seconds() -> float:
    raw = os.environ.get("MOD3_TTS_IDLE_UNLOAD_SECONDS", "0").strip()
    try:
        val = float(raw)
    except ValueError:
        _log.warning(
            "MOD3_TTS_IDLE_UNLOAD_SECONDS=%r is not a number; idle-unload disabled",
            raw,
        )
        return 0.0
    return max(0.0, val)


def _free_mlx_memory() -> None:
    """Best-effort release of MLX / Metal GPU cache."""
    try:
        import mlx.core as mx  # type: ignore[import-untyped]

        mx.metal.clear_cache()
    except Exception:
        pass
    try:
        import mlx.core as mx  # type: ignore[import-untyped]

        mx.clear_memory_pool()
    except Exception:
        pass


def _unload_all_models() -> list[str]:
    """Drop all loaded model references under the lock and free memory.

    Returns the list of engine names that were evicted.
    """
    global _models
    with _model_lock:
        if not _models:
            return []
        evicted = list(_models.keys())
        _models = {}
    gc.collect()
    _free_mlx_memory()
    return evicted


def _idle_unload_watcher(threshold_sec: float) -> None:
    """Background daemon: unload models once idle exceeds *threshold_sec*."""
    _log.info(
        "idle-unload watcher started (threshold=%.0fs)",
        threshold_sec,
    )
    poll = max(5.0, threshold_sec / 10.0)
    while True:
        time.sleep(poll)
        if not _models:
            continue
        with _model_lock:
            last = _last_use_ts
        if last <= 0.0:
            continue
        idle = time.time() - last
        if idle >= threshold_sec:
            evicted = _unload_all_models()
            if evicted:
                _log.info(
                    "idle-unload: evicted %s after %.0fs idle (threshold=%.0fs)",
                    evicted,
                    idle,
                    threshold_sec,
                )


def _start_idle_unload_watcher_if_enabled() -> None:
    """Called once at module import time to launch the watcher daemon."""
    global _idle_unload_enabled, _idle_unload_seconds
    _idle_unload_seconds = _parse_idle_unload_seconds()
    if _idle_unload_seconds <= 0.0:
        return
    _idle_unload_enabled = True
    t = threading.Thread(
        target=_idle_unload_watcher,
        args=(_idle_unload_seconds,),
        daemon=True,
        name="tts-idle-unload",
    )
    t.start()


_start_idle_unload_watcher_if_enabled()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using pysbd."""
    sentences = _segmenter.segment(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _resolve_voice_uri(uri: str) -> str | None:
    """Resolve a cog://voices/* URI to a registry profile name, or None.

    When a cog://voices/<name> URI arrives (e.g. from an identity projection),
    extract the name segment and check whether it is a registered profile in
    the local voice registry. Returns the bare name if found, None otherwise.

    Only the cog://voices/<name> form (one path segment) maps to a profile name.
    The sub-path form cog://voices/<name>/ecapa-embedding is a discriminative
    head reference and does not resolve to a TTS profile.
    """
    if not (uri.startswith("cog://voices/") or uri.startswith("cog:voices/")):
        return None
    # Strip the namespace prefix to get the remainder.
    for prefix in ("cog://voices/", "cog:voices/"):
        if uri.startswith(prefix):
            remainder = uri[len(prefix) :]
            break
    else:
        return None

    # Only bare-name URIs (no sub-path) map to a TTS profile.
    if "/" in remainder:
        return None

    name = remainder.strip()
    if not name:
        return None

    registry = _get_profile_registry()
    if registry is not None and registry.get(name) is not None:
        return name
    return None


def resolve_model(voice: str) -> tuple[str, str]:
    """Given a voice name or cog://voices/* URI, return (engine_name, voice) or raise.

    Resolution order:
    1. If voice is a cog://voices/<name> URI, resolve the name via the local
       voice registry (cog:// URI resolver, Primitive 3). Falls through to the
       bare name resolution if the URI resolves to a known profile.
    2. Check the voice profile registry for a registered cloned profile by name.
    3. Fall through to the built-in MODELS voice list.

    Raises ValueError if no engine can handle the voice.
    """
    # Primitive 3: resolve cog://voices/* URIs to profile names
    if voice.startswith("cog:"):
        resolved_name = _resolve_voice_uri(voice)
        if resolved_name is not None:
            voice = resolved_name
        else:
            raise ValueError(
                f"Unknown voice URI '{voice}': no matching profile in local registry. "
                "Ensure the voice has been enrolled via the voice enrollment skill."
            )

    registry = _get_profile_registry()
    if registry is not None:
        profile = registry.get(voice)
        if profile is not None:
            return profile.engine, voice

    for engine, cfg in MODELS.items():
        if voice in cfg["voices"]:
            return engine, voice
    raise ValueError(f"Unknown voice '{voice}'. Use list_voices() to see options.")


def get_model(engine: str):
    """Load and cache an engine's model. Thread-safe.

    Stamps _last_use_ts so the idle-unload watcher knows when synthesis last
    happened. On a reload after eviction the load path runs again transparently.
    """
    with _model_lock:
        if engine not in _models:
            from mlx_audio.tts import load

            _models[engine] = load(MODELS[engine]["id"])
        if _idle_unload_enabled:
            global _last_use_ts
            _last_use_ts = time.time()
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


# Spark's mlx_audio backend implements speed (and pitch) as a lookup into a
# fixed map of {0.0, 0.5, 1.0, 1.5, 2.0} -> {very_low, low, moderate, high,
# very_high}. Any other value raises KeyError. Mod³'s default speak() speed
# of 1.25 (tuned for Kokoro) hit this on every Spark call. Snap to the
# nearest discrete value before dispatch.
_SPARK_DISCRETE_SPEEDS = (0.0, 0.5, 1.0, 1.5, 2.0)


def _snap_to_spark_discrete(speed: float) -> float:
    return min(_SPARK_DISCRETE_SPEEDS, key=lambda v: abs(v - speed))


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
    ref_audio: str | None = None,
    engine: str | None = None,
) -> Iterator[AudioChunk]:
    """Yield AudioChunks for the given text. Core generation pipeline.

    ref_audio is a path to a reference WAV for zero-shot voice cloning.
    Currently honored only on the chatterbox engine, which clones the
    reference speaker via prepare_conditionals → generate(audio_prompt=...).

    engine is an optional override for voice-driven engine resolution.
    When None (default), the voice name determines the engine via
    resolve_model. When set, that engine is used directly and the voice
    is passed through unchanged; useful for pinning a specific backend
    without renaming the voice.
    """
    if engine is None:
        engine, voice = resolve_model(voice)
    elif engine not in MODELS:
        raise ValueError(f"unknown engine: {engine!r} (available: {sorted(MODELS.keys())})")
    model = get_model(engine)
    sample_rate = model.sample_rate
    sentences = split_sentences(text)
    feather = int(sample_rate * 0.02)

    for si, sentence in enumerate(sentences):
        gen_kwargs: dict[str, object] = dict(text=sentence, verbose=False)
        cfg = MODELS[engine]
        if engine in ("chatterbox", "chatterbox-turbo"):
            gen_kwargs["exaggeration"] = emotion
            gen_kwargs["stream"] = stream
            gen_kwargs["streaming_interval"] = streaming_interval

            # Profile-aware: if voice is a registered profile, prefer its cached
            # Conditionals over re-encoding ref_audio.
            registry = _get_profile_registry()
            profile_conds = registry.get_conditionals(voice) if registry is not None and registry.get(voice) else None
            if profile_conds is not None:
                if engine == "chatterbox-turbo":
                    # ChatterboxTurboTTS.generate() accepts only **kwargs for unknown
                    # arguments and reads conditioning from self._conds, which gets
                    # overwritten on every prepare_conditionals() call. Mutate the
                    # singleton model's _conds before generation so the cloned voice
                    # is actually used. The speech queue serializes calls, so this
                    # mutation is race-safe in practice.
                    model._conds = profile_conds
                else:
                    gen_kwargs["conds"] = profile_conds
            elif ref_audio:
                gen_kwargs["ref_audio"] = ref_audio
        elif engine == "spark":
            gen_kwargs["gender"] = "female" if voice == "spark_female" else "male"
            gen_kwargs["speed"] = _snap_to_spark_discrete(speed)
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

"""
Mod³ TTS MCP Server — gives Claude a voice via multiple TTS engines on Apple Silicon.

Multi-model support: Voxtral, Kokoro, Chatterbox, Spark.
Voice presets are resolved to the correct engine automatically.

Tools:
  speak(text, voice, stream) — synthesize and play, returns structured metrics
  list_voices()              — list available voice presets
  diagnostics()              — engine state + last generation metrics
"""

import json
import threading
import time
import uuid

import numpy as np
import pysbd
from mcp.server.fastmcp import FastMCP

from adaptive_player import AdaptivePlayer

_segmenter = pysbd.Segmenter(language="en", clean=False)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using pysbd."""
    sentences = _segmenter.segment(text.strip())
    return [s.strip() for s in sentences if s.strip()]

mcp = FastMCP(
    "tts",
    instructions=(
        "Mod³ TTS server with multi-model support (Voxtral, Kokoro, Chatterbox, Spark) "
        "running locally on Apple Silicon. "
        "Use the `speak` tool to say something out loud through the user's speakers. "
        "Use `list_voices` to see available voice presets. "
        "Keep spoken text conversational and concise — this is voice, not a document. "
        "The speak tool returns structured JSON metrics including timing, buffer health, "
        "and per-chunk generation stats — use these to diagnose audio quality issues."
    ),
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "voxtral": {
        "id": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
        "voices": [
            "casual_male", "casual_female", "cheerful_female",
            "neutral_male", "neutral_female",
            "fr_male", "fr_female", "es_male", "es_female",
            "de_male", "de_female", "it_male", "it_female",
            "pt_male", "pt_female", "nl_male", "nl_female",
            "ar_male", "hi_male", "hi_female",
        ],
        "default_voice": "casual_male",
    },
    "kokoro": {
        "id": "mlx-community/Kokoro-82M-bf16",
        "voices": [
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael",
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis",
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
_last_metrics: dict | None = None
_output_device: int | str | None = None  # audio output device (None = system default)

# Job tracking for non-blocking speech
_jobs: dict[str, dict] = {}  # id -> {status, metrics, error, start_time}


def _resolve_model(voice: str) -> tuple[str, str]:
    """Given a voice name, return (engine_name, voice) or raise."""
    for engine, cfg in MODELS.items():
        if voice in cfg["voices"]:
            return engine, voice
    raise ValueError(f"Unknown voice '{voice}'. Use list_voices() to see options.")


def _get_model(engine: str):
    if engine not in _models:
        with _model_lock:
            if engine not in _models:
                from mlx_audio.tts import load
                _models[engine] = load(MODELS[engine]["id"])
    return _models[engine]


# ---------------------------------------------------------------------------
# Adaptive playback
# ---------------------------------------------------------------------------

def _start_speech(
    text: str,
    voice: str,
    stream: bool = True,
    streaming_interval: float = 1.0,
    speed: float = 1.0,
) -> str:
    """Start non-blocking speech generation. Returns job ID immediately."""
    global _last_metrics
    engine, voice = _resolve_model(voice)
    model = _get_model(engine)
    player = AdaptivePlayer(sample_rate=model.sample_rate, device=_output_device)

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "status": "speaking",
        "engine": engine,
        "voice": voice,
        "text": text[:100],
        "start_time": time.time(),
        "metrics": None,
        "error": None,
    }

    def _run():
        try:
            _generate_sentences(model, engine, voice, text, player, stream, streaming_interval, speed, sample_rate=model.sample_rate)
        except Exception as e:
            _jobs[job_id]["error"] = str(e)
        finally:
            player.mark_done()

        # Wait for playback to finish, then collect metrics
        metrics = player.wait(timeout=120.0)
        result = metrics.to_dict()
        result["engine"] = engine
        result["voice"] = voice
        _jobs[job_id]["metrics"] = result
        _jobs[job_id]["status"] = "error" if _jobs[job_id]["error"] else "done"
        _last_metrics = result

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def _generate_sentences(model, engine, voice, text, player, stream, streaming_interval, speed, *, sample_rate: int):
    """Generate audio sentence-by-sentence into the adaptive player."""
    sentences = _split_sentences(text)
    feather = int(sample_rate * 0.02)
    gap = np.zeros(int(sample_rate * 0.1), dtype=np.float32)

    for si, sentence in enumerate(sentences):
        chunks_in_sentence = []
        gen_kwargs = dict(text=sentence, verbose=False)
        cfg = MODELS[engine]
        if engine == "chatterbox":
            gen_kwargs["exaggeration"] = speed
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
            chunk_meta = {
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

            player.queue_audio(audio, chunk_meta=chunk_meta)
            chunks_in_sentence.append(True)

        if si < len(sentences) - 1 and chunks_in_sentence:
            player.queue_audio(gap)


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def speak(
    text: str,
    voice: str = "bm_lewis",
    stream: bool = True,
    speed: float = 1.25,
) -> str:
    """Synthesize text to speech and play it through the user's speakers.

    Non-blocking: returns immediately with a job ID while audio plays in the
    background. Use speech_status(id) to check completion and get metrics.

    Args:
        text: The text to speak aloud. Keep it conversational.
        voice: Voice preset. Use list_voices() to see options.
               Defaults to "bm_lewis" (Kokoro).
        stream: If True, plays audio chunks as they generate (lower latency).
                If False, generates all audio first then plays (better prosody).
        speed: Speed multiplier (engines with speed support). Default 1.25.
    """
    if not text.strip():
        return json.dumps({"status": "error", "error": "Nothing to say"})

    try:
        job_id = _start_speech(text, voice, stream=stream, speed=speed)
        return json.dumps({"status": "speaking", "job_id": job_id})
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def speech_status(job_id: str = "") -> str:
    """Check status of a speech job, or get the most recent result.

    Args:
        job_id: The job ID returned by speak(). If empty, returns the latest job.
    """
    if not job_id:
        if not _jobs:
            return json.dumps({"status": "idle", "message": "No speech jobs"})
        job_id = max(_jobs, key=lambda k: _jobs[k]["start_time"])

    job = _jobs.get(job_id)
    if not job:
        return json.dumps({"status": "error", "error": f"Unknown job '{job_id}'"})

    result = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "speaking":
        result["elapsed_sec"] = round(time.time() - job["start_time"], 1)
    if job["metrics"]:
        result["metrics"] = job["metrics"]
    if job["error"]:
        result["error"] = job["error"]
    return json.dumps(result)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def list_voices() -> str:
    """List all available voice presets grouped by engine."""
    lines = []
    for engine, cfg in MODELS.items():
        extras = []
        if cfg.get("supports_speed"):
            extras.append("speed control")
        tag = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"  {engine}{tag}: {', '.join(cfg['voices'])}")
    return "Available voices:\n" + "\n".join(lines)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def diagnostics() -> str:
    """Return engine state and last generation metrics for debugging."""
    engines = {}
    for name, cfg in MODELS.items():
        engines[name] = {
            "loaded": name in _models,
            "model_id": cfg["id"],
            "sample_rate": _models[name].sample_rate if name in _models else None,
            "voices": len(cfg["voices"]),
        }
    info = {
        "engines": engines,
        "last_metrics": _last_metrics,
    }
    return json.dumps(info, indent=2)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def set_output_device(device: str = "") -> str:
    """List audio output devices, or set the active one.

    Args:
        device: Device index (e.g. "3") or name substring (e.g. "AirPods").
                If empty, lists available devices without changing anything.
    """
    import sounddevice as sd
    global _output_device

    outputs = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            is_default = i == sd.default.device[1]
            is_active = (
                (_output_device is None and is_default)
                or _output_device == i
                or (isinstance(_output_device, str) and _output_device in d["name"])
            )
            outputs.append({"index": i, "name": d["name"], "active": is_active})

    if not device:
        lines = [f"  [{'*' if d['active'] else ' '}] {d['index']}: {d['name']}" for d in outputs]
        return "Audio output devices (* = active):\n" + "\n".join(lines)

    # Set device
    if device.isdigit():
        _output_device = int(device)
    else:
        _output_device = device

    return json.dumps({"status": "ok", "device": _output_device})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()

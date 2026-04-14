"""
Mod³ TTS Server — gives Claude a voice via multiple TTS engines on Apple Silicon.

Multi-model support: Voxtral, Kokoro, Chatterbox, Spark.
Voice presets are resolved to the correct engine automatically.

Interfaces:
  MCP (default):  stdio-based MCP tools for Claude Code
  HTTP (--http):  REST API for OpenClaw and external consumers
  Both (--all):   MCP on stdio + HTTP on a port, shared model cache

Tools (MCP):
  speak(text, voice, speed, emotion) — non-blocking speech, returns job ID
  speech_status(job_id)              — check job or get latest metrics
  stop()                             — interrupt current speech
  list_voices()                      — list available voice presets
  set_output_device(device)          — list/switch audio output
  diagnostics()                      — engine state + last metrics
"""

import json
import logging
import os
import threading
import time
import uuid
import wave
from collections import OrderedDict
from typing import Any

import anyio
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

from bus import ModalityBus
from modality import ModalityType, ModuleStatus
from modules.voice import PlaceholderDecoder, VoiceModule
from pipeline_state import InterruptInfo, PipelineState

logger = logging.getLogger("mod3.server")

_MODEL_REGISTRY = {
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


def _create_bus() -> ModalityBus:
    bus = ModalityBus()
    bus.register(VoiceModule(decoder=PlaceholderDecoder()))
    return bus


_bus = _create_bus()
_bus_vad_lock = threading.Lock()


def _get_voice_module() -> VoiceModule | None:
    module = getattr(_bus, "_modules", {}).get(ModalityType.VOICE)
    return module if isinstance(module, VoiceModule) else None


def _engine_module():
    import engine

    return engine


def _try_engine_module():
    try:
        return _engine_module(), None
    except Exception as exc:
        return None, exc


def _model_registry() -> dict[str, dict[str, Any]]:
    engine_module, _ = _try_engine_module()
    return engine_module.MODELS if engine_module is not None else _MODEL_REGISTRY


def _adaptive_player_class():
    from adaptive_player import AdaptivePlayer

    return AdaptivePlayer


def _resolve_voice_via_bus(voice: str) -> tuple[str, str]:
    voice_module = _get_voice_module()
    if voice_module is None or voice_module.encoder is None:
        raise ValueError("Voice module is not registered on the ModalityBus.")

    for engine_name, cfg in _model_registry().items():
        if voice in cfg["voices"]:
            return engine_name, voice

    raise ValueError(f"Unknown voice '{voice}'. Use list_voices() to see options.")


def _read_wav_as_mono_float32(file_path: str) -> tuple[bytes, int]:
    with wave.open(file_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.float32)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio.astype(np.float32).tobytes(), sample_rate


def _set_bus_voice_state(
    *,
    status: ModuleStatus,
    active_job: str | None = None,
    current_text: str = "",
    progress: float | None = None,
    last_output_text: str | None = None,
    error: str | None = None,
) -> None:
    voice_module = _get_voice_module()
    if voice_module is None:
        return

    state = voice_module.state
    state.status = status
    state.active_job = active_job
    state.current_text = current_text
    state.last_activity = time.time()
    state.error = error
    if progress is not None:
        state.progress = progress
    if last_output_text is not None:
        state.last_output_text = last_output_text


mcp = FastMCP(
    "mod3",
    instructions=(
        "Mod³ voice channel with multi-model TTS (Voxtral, Kokoro, Chatterbox, Spark) "
        "running locally on Apple Silicon. "
        'Voice messages arrive as <channel source="mod3" speaker="..." confidence="...">. '
        "Use the speak tool to respond via voice. speak() is non-blocking. "
        "Use speech_status to check completion. Use stop to interrupt. "
        "Keep spoken text conversational and concise — this is voice, not a document. "
        "For permission prompts, reply verbally with 'yes [code]' or 'no [code]'."
    ),
)

# ---------------------------------------------------------------------------
# Claude Code channel capabilities
# ---------------------------------------------------------------------------

_CHANNEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "claude/channel": {},
    "claude/channel/permission": {},
}

# Store the write stream for emitting channel notifications outside request context
_write_stream = None
_write_stream_lock = threading.Lock()


async def emit_channel_event(content: str, meta: dict[str, str] | None = None):
    """Emit a channel notification to Claude Code.

    Sends a ``notifications/claude/channel`` JSON-RPC notification over the
    active MCP session.  Can be called from tool handlers or background tasks
    while the server is running.

    Args:
        content: The textual content to relay (e.g. transcribed speech).
        meta: Optional string-keyed metadata dict (speaker, confidence, etc.).
    """
    with _write_stream_lock:
        ws = _write_stream
    if ws is None:
        raise RuntimeError("MCP session not active — cannot emit channel event")

    notification = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel",
        params={"content": content, "meta": meta or {}},
    )
    await ws.send(SessionMessage(message=JSONRPCMessage(notification)))


async def emit_permission_verdict(request_id: str, behavior: str):
    """Emit a permission verdict notification to Claude Code.

    Sends a ``notifications/claude/channel/permission`` JSON-RPC notification
    with a verdict (allow/deny) for a previously received permission request.

    Args:
        request_id: The 5-letter request ID from the original permission request.
        behavior: ``"allow"`` or ``"deny"``.
    """
    with _write_stream_lock:
        ws = _write_stream
    if ws is None:
        raise RuntimeError("MCP session not active — cannot emit permission verdict")

    notification = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel/permission",
        params={"request_id": request_id, "behavior": behavior},
    )
    await ws.send(SessionMessage(message=JSONRPCMessage(notification)))
    logger.info("permission verdict sent: request_id=%s behavior=%s", request_id, behavior)


def _handle_permission_request(params: dict[str, Any]) -> None:
    """Handle an incoming permission request by speaking it aloud via TTS.

    Called when the read-stream interceptor detects a
    ``notifications/claude/channel/permission_request`` notification from
    Claude Code.  Formats a spoken prompt and plays it through the default
    voice so the user can respond verbally.

    Args:
        params: The notification params containing ``request_id``,
                ``tool_name``, ``description``, and ``input_preview``.
    """
    request_id = params.get("request_id", "unknown")
    tool_name = params.get("tool_name", "a tool")
    description = params.get("description", "")

    # Build a concise spoken prompt
    prompt_parts = [f"Claude wants to run {tool_name}"]
    if description:
        prompt_parts.append(f": {description}")
    prompt_parts.append(f". Say yes {request_id} or no {request_id}.")
    prompt_text = "".join(prompt_parts)

    logger.info("permission request received: id=%s tool=%s", request_id, tool_name)
    _start_speech(prompt_text, voice="bm_lewis", speed=1.25)


# Patch run_stdio_async to inject experimental capabilities and capture the
# write stream so emit_channel_event can send notifications at any time.
_original_run_stdio = mcp.run_stdio_async

_PERMISSION_REQUEST_METHOD = "notifications/claude/channel/permission_request"


async def _patched_run_stdio():
    global _write_stream
    async with stdio_server() as (read_stream, write_stream):
        with _write_stream_lock:
            _write_stream = write_stream

        # Wrap the read stream to intercept permission_request notifications.
        # These use a custom method that the MCP session cannot parse into a
        # typed ClientNotification, so we handle them here and forward
        # everything else to the MCP server unchanged.
        send_inner, receive_inner = anyio.create_memory_object_stream[SessionMessage | Exception](0)

        async def _filter_read_stream():
            async with read_stream, send_inner:
                async for message in read_stream:
                    if isinstance(message, Exception):
                        await send_inner.send(message)
                        continue

                    root = message.message.root
                    if isinstance(root, JSONRPCNotification) and root.method == _PERMISSION_REQUEST_METHOD:
                        # Handle permission request — speak it via TTS
                        _handle_permission_request(root.params or {})
                        continue

                    # All other messages pass through to the MCP server
                    await send_inner.send(message)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_filter_read_stream)
                await mcp._mcp_server.run(
                    receive_inner,
                    write_stream,
                    mcp._mcp_server.create_initialization_options(
                        experimental_capabilities=_CHANNEL_CAPABILITIES,
                    ),
                )
        finally:
            with _write_stream_lock:
                _write_stream = None


mcp.run_stdio_async = _patched_run_stdio

# ---------------------------------------------------------------------------
# Reflex arc — shared pipeline state
# ---------------------------------------------------------------------------

pipeline_state = PipelineState()


# ---------------------------------------------------------------------------
# Barge-in file watcher — monitors /tmp/mod3-barge-in.json for pause signals
# ---------------------------------------------------------------------------

_BARGEIN_SIGNAL = "/tmp/mod3-barge-in.json"
_bargein_last_mtime: float = 0.0


def _bargein_watcher():
    """Background thread that watches for barge-in signal file changes."""
    global _bargein_last_mtime
    import json as _json

    while True:
        try:
            import os

            if os.path.exists(_BARGEIN_SIGNAL):
                mtime = os.path.getmtime(_BARGEIN_SIGNAL)
                if mtime > _bargein_last_mtime:
                    _bargein_last_mtime = mtime
                    with open(_BARGEIN_SIGNAL) as f:
                        signal = _json.load(f)
                    if signal.get("event") == "user_speaking_start":
                        if pipeline_state.is_speaking:
                            info = pipeline_state.interrupt(reason="barge_in")
                            if info:
                                # Write interrupt context back to signal file
                                signal["interrupted"] = {
                                    "spoken_pct": info.spoken_pct,
                                    "delivered_text": info.delivered_text,
                                    "full_text": info.full_text,
                                }
                                with open(_BARGEIN_SIGNAL, "w") as f:
                                    _json.dump(signal, f, indent=2)
                            logging.info(
                                "Barge-in: paused playback (%.0f%% delivered)", info.spoken_pct * 100 if info else 0
                            )
        except Exception as e:
            logging.debug("Barge-in watcher error: %s", e)
        time.sleep(0.1)  # 100ms poll


_bargein_thread = threading.Thread(target=_bargein_watcher, daemon=True)
_bargein_thread.start()


async def _emit_interruption(info: InterruptInfo):
    """Emit a channel notification when playback is interrupted.

    Called by the inbound pipeline (VAD reflex) after pipeline_state.interrupt()
    returns an InterruptInfo.  Notifies Claude Code that speech was cut short.
    """
    await emit_channel_event(
        content=f"[interrupted — speech halted at '{info.delivered_text}']",
        meta={
            "source": "mod3-voice",
            "type": "interruption",
            "spoken_pct": str(round(info.spoken_pct, 2)),
            "reason": info.reason,
        },
    )


# ---------------------------------------------------------------------------
# Job tracking (MCP only — local speaker playback)
# ---------------------------------------------------------------------------

MAX_JOBS = 20
_last_metrics: dict | None = None
_output_device: int | str | None = None
_jobs: OrderedDict[str, dict] = OrderedDict()
_current_player: Any | None = None
_current_player_lock = threading.Lock()


def _prune_jobs():
    """Keep only the last MAX_JOBS entries."""
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)


# ---------------------------------------------------------------------------
# Speech queue — serial playback with enriched status
# ---------------------------------------------------------------------------


class SpeechQueue:
    """Thread-safe queue for serial speech playback.

    When speak() is called while audio is playing, the new request is
    queued and will play automatically when the current item finishes.
    All queue operations are protected by a single lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._queue: list[dict] = []  # pending jobs (not yet playing)
        self._active_job_id: str | None = None  # job_id currently playing
        self._draining = False  # True while the drain thread is running

    def enqueue(self, job_id: str, params: dict) -> int:
        """Add a job to the queue. Returns the queue position (0 = will play next).

        If nothing is currently playing and the queue is empty, triggers
        drain immediately so the job starts without delay.
        """
        with self._lock:
            self._queue.append({"job_id": job_id, **params})
            position = len(self._queue) - 1
            if not self._draining:
                self._draining = True
                threading.Thread(target=self._drain, daemon=True).start()
            return position

    def cancel(self, job_id: str) -> bool:
        """Remove a queued (not yet playing) job. Returns True if found and removed."""
        with self._lock:
            for i, entry in enumerate(self._queue):
                if entry["job_id"] == job_id:
                    self._queue.pop(i)
                    return True
        return False

    def cancel_all_queued(self) -> int:
        """Remove all queued (not yet playing) jobs. Returns count removed."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count

    def get_queue_snapshot(self) -> list[dict]:
        """Return a snapshot of queued jobs (does not include the active job)."""
        with self._lock:
            return list(self._queue)

    @property
    def active_job_id(self) -> str | None:
        with self._lock:
            return self._active_job_id

    @property
    def depth(self) -> int:
        """Number of jobs waiting (not including the active one)."""
        with self._lock:
            return len(self._queue)

    def _drain(self):
        """Process queued jobs one at a time until the queue is empty."""
        while True:
            with self._lock:
                if not self._queue:
                    self._draining = False
                    self._active_job_id = None
                    return
                entry = self._queue.pop(0)
                self._active_job_id = entry["job_id"]

            # Run the speech job (blocking — one at a time)
            _run_speech_job(entry)


_speech_queue = SpeechQueue()


# ---------------------------------------------------------------------------
# Adaptive playback (MCP speaker output)
# ---------------------------------------------------------------------------


def _estimate_duration_sec(text: str, speed: float) -> float:
    """Rough estimate of speech duration from text length and speed.

    Heuristic: ~150 words per minute at speed 1.0, average word ~5 chars.
    """
    words = len(text.split())
    if words == 0:
        words = max(1, len(text) / 5)
    return (words / 150.0) * 60.0 / speed


def _run_speech_job(entry: dict) -> None:
    """Execute a single speech job (blocking). Called from the drain thread."""
    global _last_metrics, _current_player

    job_id = entry["job_id"]
    text = entry["text"]
    voice = entry["voice"]
    stream = entry.get("stream", True)
    streaming_interval = entry.get("streaming_interval", 1.0)
    speed = entry.get("speed", 1.0)
    emotion = entry.get("emotion", 0.5)

    try:
        engine_module = _engine_module()
        AdaptivePlayer = _adaptive_player_class()
        engine, resolved_voice = _resolve_voice_via_bus(voice)
        model = engine_module.get_model(engine)
        player = AdaptivePlayer(sample_rate=model.sample_rate, device=_output_device)
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        _set_bus_voice_state(
            status=ModuleStatus.ERROR,
            active_job=None,
            current_text="",
            error=str(e),
        )
        with _current_player_lock:
            if _current_player is not None:
                pass  # leave existing player alone on setup error
        return

    with _current_player_lock:
        _current_player = player

    _jobs[job_id]["status"] = "speaking"
    _jobs[job_id]["start_time"] = time.time()
    _jobs[job_id]["engine"] = engine
    _jobs[job_id]["voice"] = resolved_voice
    _jobs[job_id]["player"] = player
    _set_bus_voice_state(
        status=ModuleStatus.ENCODING,
        active_job=job_id,
        current_text=text[:100],
        progress=0.0,
        error=None,
    )

    # Register with the reflex arc so inbound VAD can interrupt us
    pipeline_state.start_speaking(text, player)
    try:
        for chunk in engine_module.generate_audio(
            text,
            voice=resolved_voice,
            stream=stream,
            streaming_interval=streaming_interval,
            speed=speed,
            emotion=emotion,
        ):
            player.queue_audio(chunk.samples, chunk_meta=chunk.metadata if chunk.metadata else None)
            _set_bus_voice_state(
                status=ModuleStatus.ENCODING,
                active_job=job_id,
                current_text=text[:100],
            )
            # Update position after each chunk so PipelineState tracks progress
            pipeline_state.update_position(*player.get_progress())
    except Exception as e:
        _jobs[job_id]["error"] = str(e)
    finally:
        player.mark_done()

    metrics = player.wait(timeout=120.0)
    # Final position update and clear speaking state
    pipeline_state.update_position(*player.get_progress())
    pipeline_state.stop_speaking()

    result = metrics.to_dict()
    result["engine"] = engine
    result["voice"] = resolved_voice
    _jobs[job_id]["metrics"] = result
    _jobs[job_id]["status"] = "error" if _jobs[job_id]["error"] else "done"
    _last_metrics = result
    _set_bus_voice_state(
        status=ModuleStatus.ERROR if _jobs[job_id]["error"] else ModuleStatus.IDLE,
        active_job=None,
        current_text="",
        progress=1.0 if not _jobs[job_id]["error"] else 0.0,
        last_output_text=text[:100],
        error=_jobs[job_id]["error"],
    )

    with _current_player_lock:
        if _current_player is player:
            _current_player = None


def _start_speech(
    text: str,
    voice: str,
    stream: bool = True,
    streaming_interval: float = 1.0,
    speed: float = 1.0,
    emotion: float = 0.5,
) -> tuple[str, int]:
    """Submit speech to the queue. Returns (job_id, queue_position).

    queue_position is 0 if playing immediately, >0 if queued behind others.
    """
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "status": "queued",
        "engine": None,
        "voice": voice,
        "text": text[:100],
        "full_text": text,
        "submitted_time": time.time(),
        "start_time": None,
        "metrics": None,
        "error": None,
        "player": None,
        "speed": speed,
        "estimated_duration_sec": round(_estimate_duration_sec(text, speed), 1),
    }
    _prune_jobs()

    position = _speech_queue.enqueue(
        job_id,
        {
            "text": text,
            "voice": voice,
            "stream": stream,
            "streaming_interval": streaming_interval,
            "speed": speed,
            "emotion": emotion,
        },
    )
    return job_id, position


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


def _get_currently_playing_info() -> dict | None:
    """Return info about the currently playing job, or None if idle."""
    with _current_player_lock:
        if _current_player is None:
            return None

    active_id = _speech_queue.active_job_id
    if active_id is None:
        return None

    job = _jobs.get(active_id)
    if job is None or job["status"] != "speaking":
        return None

    start_time = job.get("start_time")
    elapsed = round(time.time() - start_time, 1) if start_time else 0.0
    estimated = job.get("estimated_duration_sec", 0.0)
    remaining = max(0.0, round(estimated - elapsed, 1))

    return {
        "job_id": active_id,
        "text_preview": job.get("text", "")[:50],
        "elapsed_sec": elapsed,
        "remaining_sec": remaining,
    }


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
    emotion: float = 0.5,
) -> str:
    """Synthesize text to speech and play it through the user's speakers.

    Non-blocking: returns immediately with a job ID while audio plays or is
    queued. If nothing is playing, starts immediately. If audio is already
    playing, the new request is queued and will play automatically when the
    current item finishes.

    The response always includes the current queue state so the agent knows
    exactly what's happening on the output channel without a separate status call.

    Args:
        text: The text to speak aloud. Keep it conversational.
        voice: Voice preset. Use list_voices() to see options.
               Defaults to "bm_lewis" (Kokoro).
        stream: If True, plays audio chunks as they generate (lower latency).
                If False, generates all audio first then plays (better prosody).
        speed: Speed multiplier (engines with speed support). Default 1.25.
        emotion: Emotion/exaggeration intensity 0.0-1.0 (Chatterbox only). Default 0.5.
    """
    if not text.strip():
        return json.dumps({"status": "error", "error": "Nothing to say"})

    # Check if user is currently speaking (barge-in signal file)
    user_state = "idle"
    try:
        if os.path.exists(_BARGEIN_SIGNAL):
            with open(_BARGEIN_SIGNAL) as _bf:
                _bsig = json.load(_bf)
            if _bsig.get("event") == "user_speaking_start":
                user_state = "recording"
    except Exception:
        pass  # signal file missing or corrupt — assume idle

    # If user is currently recording, don't play — just inform the agent.
    # The agent is responsible for re-calling speak() after the user finishes.
    # We intentionally do NOT enqueue the job or create a _jobs entry, because
    # a "held" job in the queue becomes a zombie: the drain thread tries to play
    # it immediately (ignoring the hold), and if anything goes wrong the job
    # can't be cleared by stop().
    if user_state == "recording":
        est_duration = _estimate_duration_sec(text, speed)
        return json.dumps(
            {
                "status": "held",
                "reason": "User is currently speaking — re-send this speak() call after user finishes.",
                "user_state": "recording",
                "estimated_duration_sec": round(est_duration, 1),
            }
        )

    try:
        job_id, position = _start_speech(text, voice, stream=stream, speed=speed, emotion=emotion)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

    # If position is 0 and nothing else was playing, it starts immediately
    currently_playing = _get_currently_playing_info()

    if currently_playing is None or currently_playing["job_id"] == job_id:
        # Playing immediately (no queue ahead)
        result = {"status": "speaking", "job_id": job_id}
        return json.dumps(result)

    # Something is already playing — return enriched queue status
    queue_snapshot = _speech_queue.get_queue_snapshot()
    queue_ahead = []
    for entry in queue_snapshot:
        qid = entry["job_id"]
        if qid == job_id:
            break  # don't include self or anything after self
        qjob = _jobs.get(qid)
        est = qjob.get("estimated_duration_sec", 0.0) if qjob else 0.0
        preview = qjob.get("text", "")[:50] if qjob else entry.get("text", "")[:50]
        queue_ahead.append(
            {
                "job_id": qid,
                "text_preview": preview,
                "estimated_sec": est,
            }
        )

    # Compute estimated wait: remaining on current + all queued ahead
    wait = currently_playing.get("remaining_sec", 0.0)
    for item in queue_ahead:
        wait += item.get("estimated_sec", 0.0)
    wait = round(wait, 1)

    # The queue_position as seen by the user: 1-indexed position in the
    # overall playback order (1 = next after currently playing)
    queue_position = len(queue_ahead) + 1

    result = {
        "status": "queued",
        "job_id": job_id,
        "queue_position": queue_position,
        "currently_playing": currently_playing,
        "queue_ahead": queue_ahead,
        "estimated_wait_sec": wait,
        "actions": (
            f"To cancel this queued item, call stop(job_id='{job_id}'). "
            "To cancel all and speak immediately, call stop() then speak()."
        ),
    }
    if user_state != "idle":
        result["user_state"] = user_state
    return json.dumps(result)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def speech_status(job_id: str = "", verbose: bool = False) -> str:
    """Check status of a speech job, or get the most recent result.

    Always includes queue state so the agent has full output channel awareness.

    Args:
        job_id: The job ID returned by speak(). If empty, returns the latest job.
        verbose: If True, include per-chunk metrics. Default False (summary only).
    """
    if not job_id:
        if not _jobs:
            return json.dumps({"status": "idle", "message": "No speech jobs", "queue_depth": 0})
        job_id = next(reversed(_jobs))

    job = _jobs.get(job_id)
    if not job:
        return json.dumps({"status": "error", "error": f"Unknown job '{job_id}'"})

    result = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "speaking":
        start = job.get("start_time")
        if start:
            result["elapsed_sec"] = round(time.time() - start, 1)
    elif job["status"] == "queued":
        # Find this job's position in the queue
        queue_snapshot = _speech_queue.get_queue_snapshot()
        for i, entry in enumerate(queue_snapshot):
            if entry["job_id"] == job_id:
                result["queue_position"] = i + 1
                break
    if job.get("metrics"):
        metrics = job["metrics"]
        if not verbose and "chunks" in metrics:
            chunks = metrics["chunks"]["per_chunk"]
            rtfs = [c["rtf"] for c in chunks if c.get("rtf")]
            metrics = {
                **metrics,
                "chunks": {
                    "count": metrics["chunks"]["count"],
                    "avg_rtf": round(sum(rtfs) / len(rtfs), 2) if rtfs else 0,
                    "min_rtf": round(min(rtfs), 2) if rtfs else 0,
                },
            }
        result["metrics"] = metrics
    if job.get("error"):
        result["error"] = job["error"]

    # Always include queue state
    currently_playing = _get_currently_playing_info()
    queue_depth = _speech_queue.depth
    result["queue"] = {
        "depth": queue_depth,
        "currently_playing": currently_playing,
    }

    return json.dumps(result)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def stop(job_id: str = "") -> str:
    """Stop current speech or cancel a specific queued item.

    Args:
        job_id: If provided, cancels that specific queued job (not yet playing).
                If the job_id is the currently playing job, interrupts playback.
                If empty, interrupts current playback AND clears the entire queue.
    """
    if job_id:
        # Try to cancel a specific queued (not yet playing) job
        if _speech_queue.cancel(job_id):
            if job_id in _jobs:
                _jobs[job_id]["status"] = "cancelled"
            return json.dumps(
                {
                    "status": "ok",
                    "message": f"Cancelled queued job '{job_id}'",
                    "queue_depth": _speech_queue.depth,
                }
            )

        # Check if it's the currently playing job
        active = _speech_queue.active_job_id
        if active == job_id:
            with _current_player_lock:
                player = _current_player
            if player is not None:
                player.flush()
            return json.dumps(
                {
                    "status": "ok",
                    "message": f"Interrupted playing job '{job_id}'",
                    "queue_depth": _speech_queue.depth,
                }
            )

        # Job exists but already done
        if job_id in _jobs:
            return json.dumps(
                {
                    "status": "ok",
                    "message": f"Job '{job_id}' already finished (status: {_jobs[job_id]['status']})",
                }
            )

        return json.dumps({"status": "error", "error": f"Unknown job '{job_id}'"})

    # No job_id: stop everything — interrupt current + clear queue
    cleared = _speech_queue.cancel_all_queued()
    # Mark all cleared queued and held jobs as cancelled
    for jid, jdata in _jobs.items():
        if jdata["status"] in ("queued", "held"):
            jdata["status"] = "cancelled"

    with _current_player_lock:
        player = _current_player
    if player is None and cleared == 0:
        return json.dumps({"status": "ok", "message": "Nothing playing"})

    if player is not None:
        player.flush()

    parts = []
    if player is not None:
        parts.append("interrupted current playback")
    if cleared > 0:
        parts.append(f"cancelled {cleared} queued item{'s' if cleared != 1 else ''}")

    return json.dumps({"status": "ok", "message": "; ".join(parts).capitalize()})


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def vad_check(file_path: str, threshold: float = 0.5) -> str:
    """Check if an audio file contains speech using Silero VAD.

    Use this before transcription to avoid Whisper hallucinations on
    silence or ambient noise.

    Args:
        file_path: Path to a WAV audio file.
        threshold: Speech probability threshold 0-1 (default 0.5). Higher = stricter.
    """
    try:
        voice_module = _get_voice_module()
        if voice_module is None or voice_module.gate is None:
            from vad import detect_speech_file

            result = detect_speech_file(file_path, threshold=threshold)
            return json.dumps(
                {
                    "has_speech": result.has_speech,
                    "confidence": result.confidence,
                    "speech_ratio": result.speech_ratio,
                    "num_segments": result.num_segments,
                    "total_speech_sec": result.total_speech_sec,
                    "total_audio_sec": result.total_audio_sec,
                }
            )

        raw_audio, sample_rate = _read_wav_as_mono_float32(file_path)
        with _bus_vad_lock:
            previous_threshold = getattr(voice_module.gate, "threshold", threshold)
            voice_module.gate.threshold = threshold
            gate_result = voice_module.gate.check(raw_audio, sample_rate=sample_rate, sample_width=4)
            _bus.perceive(
                raw_audio,
                modality=ModalityType.VOICE,
                channel="mcp:vad_check",
                sample_rate=sample_rate,
                sample_width=4,
                transcript="speech detected",
            )
            voice_module.gate.threshold = previous_threshold

        return json.dumps(
            {
                "has_speech": gate_result.passed,
                "confidence": gate_result.confidence,
                "speech_ratio": gate_result.metadata.get("speech_ratio", 0.0),
                "num_segments": gate_result.metadata.get("num_segments", 0),
                "total_speech_sec": gate_result.metadata.get("total_speech_sec", 0.0),
                "total_audio_sec": gate_result.metadata.get("total_audio_sec", 0.0),
            }
        )
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
def list_voices() -> str:
    """List all available voice presets grouped by engine."""
    if _get_voice_module() is None:
        logger.warning("list_voices called without a registered bus voice module")

    models = _model_registry()
    lines = []
    for engine, cfg in models.items():
        extras = []
        if cfg.get("supports_speed"):
            extras.append("speed")
        if cfg.get("supports_exaggeration"):
            extras.append("emotion")
        if cfg.get("supports_pitch"):
            extras.append("pitch")
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
    engine_module, engine_error = _try_engine_module()
    models = engine_module.MODELS if engine_module is not None else _MODEL_REGISTRY
    loaded_engines = engine_module.get_loaded_engines() if engine_module is not None else []
    engines = {}
    for name, cfg in models.items():
        engines[name] = {
            "loaded": name in loaded_engines,
            "model_id": cfg["id"],
            "voices": len(cfg["voices"]),
        }
    info = {
        "engines": engines,
        "bus": {
            "health": _bus.health(),
            "hud": _bus.hud(),
        },
        "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "speaking"),
        "queued_jobs": sum(1 for j in _jobs.values() if j["status"] == "queued"),
        "total_jobs": len(_jobs),
        "queue_depth": _speech_queue.depth,
        "output_device": _output_device,
        "last_metrics": _last_metrics,
    }
    if engine_error is not None:
        info["engine_import_error"] = str(engine_error)
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

    if device.isdigit():
        _output_device = int(device)
    else:
        _output_device = device

    return json.dumps({"status": "ok", "device": _output_device})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_http(host: str = "0.0.0.0", port: int = 7860):
    """Start the HTTP API server."""
    import uvicorn

    from http_api import app

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mod³ TTS Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP API only")
    parser.add_argument("--all", action="store_true", help="Run both MCP (stdio) and HTTP")
    parser.add_argument("--channel", action="store_true", help="Run as channel server with voice input")
    parser.add_argument("--dashboard", action="store_true", help="Run HTTP API with voice/text dashboard (no MCP)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address")
    args = parser.parse_args()

    if args.http:
        _run_http(host=args.host, port=args.port)
    elif args.all:
        # HTTP in background thread, MCP on stdio
        http_thread = threading.Thread(
            target=_run_http,
            kwargs={"host": args.host, "port": args.port},
            daemon=True,
        )
        http_thread.start()
        mcp.run()
    elif args.dashboard:
        # Dashboard mode: HTTP server with WebSocket voice/text chat
        # Swap PlaceholderDecoder → WhisperDecoder for real STT
        from modules.text import TextModule
        from modules.voice import VoiceModule, WhisperDecoder

        _bus._modules.clear()
        _bus.register(VoiceModule(decoder=WhisperDecoder()))
        _bus.register(TextModule())
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting dashboard mode (WhisperDecoder enabled)")
        _run_http(host=args.host, port=args.port)
    elif args.channel:
        # Channel mode: MCP on stdio + inbound voice pipeline
        from bus import ModalityBus
        from inbound import InboundPipeline
        from modules.voice import VoiceModule

        bus = ModalityBus()
        bus.register(VoiceModule())
        inbound = InboundPipeline(bus=bus, pipeline_state=pipeline_state)
        inbound.start()
        try:
            mcp.run()  # MCP on stdio with channel capabilities
        finally:
            inbound.stop()
    else:
        mcp.run()

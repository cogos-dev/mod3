"""Mod³ HTTP API — REST interface for TTS synthesis, VAD, and dashboard.

Endpoints:
  POST /v1/synthesize  — text → audio bytes (WAV/PCM/OGG-Opus) + structured metrics
  POST /v1/audio/speech — OpenAI-compatible TTS endpoint
  POST /v1/vad         — audio file → speech detection result
  POST /v1/transcribe  — audio file → transcript (Whisper STT)
  POST /v1/filter      — text → hallucination check
  GET  /v1/voices      — list available engines and voices
  GET  /v1/jobs        — list recent generation jobs with full metrics
  GET  /v1/jobs/{id}   — get a specific job's metrics
  GET  /health         — server health check
  POST /shutdown       — graceful server shutdown (kernel lifecycle)
  GET  /capabilities   — machine-readable capability manifest
  WS   /ws/chat        — dashboard voice/text chat
  GET  /dashboard      — dashboard UI
  POST /v1/sessions/{id}/seats             — register a channel-client seat
  DELETE /v1/sessions/{id}/seats/{seat_id} — revoke a seat
  GET  /v1/sessions/{id}/seats/{seat_id}/events — SSE event stream for a seat
  GET  /v1/sessions/{id}/seats             — list seats in a session
  POST /v1/sessions/{id}/messages          — fan dashboard text to all seats in session
  GET  /v1/sessions/{id}/messages          — recent chat history for hydration
  POST /v1/sessions/broadcast-message     — fan dashboard text to ALL seats (all sessions)
  POST /v1/dashboard-chat                  — REST dashboard-chat (for channel clients)
  GET  /v1/logs/chat-flow                  — recent chat-flow events (JSON)
  GET  /v1/logs/chat-flow/stream           — live SSE stream of chat-flow events
"""

import asyncio
import io
import json
import logging
import os
import signal
import struct
import time
import uuid
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, Request, Response, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from _version import __version__
from audio_subscribers import get_default_audio_subscribers
from bus import ModalityBus
from chat_flow_log import (
    CHAT_FAN_OUT,
    CHAT_MESSAGE_RECEIVED,
    CHAT_MESSAGE_SENT,
    get_chat_flow_log,
)
from engine import MODELS, generate_audio, get_loaded_engines
from modality import EncodedOutput, ModalityType
from modules.text import TextModule
from modules.voice import VoiceModule
from schemas.http import (
    BusActRequest,
    ComposeProfileRequest,
    CompositionCreateRequest,
    CompositionUpdateRequest,
    RegisterProfileRequest,
    SessionRegisterRequest,
    ShutdownRequest,
    SpeakRequest,
    SpeechRequest,
    SynthesizeRequest,
    VadFilterRequest,
)
from session_registry import (
    get_default_registry,
    resolve_output_device,
)
from vad import detect_speech_file, is_hallucination
from vad import is_model_loaded as vad_loaded
from voice_profiles import VoiceProfileRegistry

logger = logging.getLogger("mod3.http")

_server_start_time = time.time()
_shutting_down = False

# Voice profile registry — mkdir only; no IO on import.
_registry = VoiceProfileRegistry()

# Composition (draft) registry — mkdir only; no IO on import.
from compositions import Composition, CompositionRegistry, Segment  # noqa: E402

_compositions = CompositionRegistry()


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Unified FastAPI lifespan — replaces all @app.on_event hooks.

    Startup order (pre-yield):
      1. Kokoro warmup — spawns a daemon thread; non-blocking, never fails startup.
      2. Kernel-bus → dashboard bridge — subscribes to cycle-trace events.
         Non-blocking: the subscriber's backoff loop handles an unreachable kernel.
      3. Auto-create the 'main' session in the voice-TTS session registry.
         channel_client.py targets this session by default (clients/channel_client.py:79).
         Idempotent: re-registration on restart updates metadata without touching voice.

    Shutdown order (post-yield, reverse of startup):
      2. Stop kernel-bus bridge.
      1. STT executor drain.

    Each phase catches and logs its own errors so a failure in one phase does not
    prevent the remaining phases from running.
    """
    import threading

    from bus_bridge_runner import start_bridge, stop_bridge

    # --- startup ---

    # 1. Kokoro warmup (thread spawn; never blocks or fails startup)
    def _do_warmup():
        try:
            from engine import get_model

            get_model("kokoro")
            logger.info("Kokoro TTS engine pre-warmed successfully")
        except Exception as e:
            logger.warning("Kokoro pre-warm failed (will lazy-load on first request): %s", e)

    threading.Thread(target=_do_warmup, daemon=True, name="kokoro-warmup").start()

    # 2. Kernel-bus → dashboard bridge
    try:
        await start_bridge(application.state)
    except Exception as e:  # noqa: BLE001 — never fail startup on bridge wiring
        logger.warning("bus-bridge startup failed (non-fatal): %s", e)

    # 3. Ensure the 'main' session exists.
    #
    #    channel_client.py uses _DEFAULT_SESSION_ID = "main" (clients/channel_client.py:79).
    #    POST /v1/sessions/main/seats registers seats into a *seat* registry keyed by
    #    session_id — that path auto-creates the seat bucket — but the voice-TTS
    #    session_registry (GET /v1/sessions) is a separate registry that requires an
    #    explicit register call before it will list the session.  Without this, the
    #    dashboard sees "No active sessions" and /v1/sessions returns {sessions: []}
    #    even when 3 channel clients are running.
    #
    #    We only guarantee 'main'; other sessions are created on explicit request.
    #    Idempotent: re-registering an existing session_id updates metadata without
    #    touching the assigned voice (see SessionRegistry.register).
    _MAIN_SESSION_ID = "main"
    try:
        _sr = get_default_registry()
        _sr_result = _sr.register(
            session_id=_MAIN_SESSION_ID,
            participant_id="channel-client-pool",
            participant_type="agent",
            preferred_voice=None,
            preferred_output_device="system-default",
        )
        if _sr_result.created:
            logger.info("auto-created session '%s' on startup", _MAIN_SESSION_ID)
        else:
            logger.debug("session '%s' already registered (idempotent restart)", _MAIN_SESSION_ID)
    except Exception as e:  # noqa: BLE001 — never fail startup on session init
        logger.warning("auto-create 'main' session failed (non-fatal): %s", e)

    # 3b. Wire seat-liveness ↔ session-lifecycle so orphaned sessions get reaped.
    #
    #     A seat with no live SSE stream is not a seat (per the
    #     seat-as-coordination-surface model). Two hooks close the loop:
    #       * liveness_check: the reaper asks the seat registry whether a
    #         session still has a live SSE connection before pruning it, so a
    #         live-but-quiet seat is preserved.
    #       * on_session_idle: when a session's last SSE stream closes (client
    #         killed/crashed/closed without the graceful DELETE-seat hook), the
    #         seat registry tells the session registry to deregister it. 'main'
    #         is never reaped — it is the daemon's persistent pool.
    try:
        from seats import get_seat_registry as _get_seat_registry
        from session_registry import MAIN_SESSION_ID as _MAIN_SID
        from session_registry import get_default_registry as _get_session_registry

        _seat_reg = _get_seat_registry()
        _sess_reg = _get_session_registry()
        _sess_reg.set_liveness_check(_seat_reg.has_live_stream)

        def _on_session_idle(session_id: str) -> None:
            if session_id == _MAIN_SID:
                return
            res = _sess_reg.deregister(session_id)
            if res.get("status") == "ok":
                logger.info("deregistered session '%s' on last-SSE-stream close", session_id)

        _seat_reg.set_on_session_idle(_on_session_idle)
    except Exception as e:  # noqa: BLE001 — never fail startup on liveness wiring
        logger.warning("seat-liveness wiring failed (non-fatal): %s", e)

    yield  # application is running

    # --- shutdown (reverse order) ---

    # 2. Stop kernel-bus bridge
    try:
        await stop_bridge(application.state, timeout_s=2.0)
    except Exception as e:  # noqa: BLE001
        logger.debug("bus-bridge shutdown error (non-fatal): %s", e)

    # 1. Drain and shut down the dedicated STT executor (§4 of ARCHITECTURE.md).
    #    Allows any in-flight mlx_whisper.transcribe() to finish before exit.
    try:
        from channels import shutdown_stt_executor

        shutdown_stt_executor(wait=True)
        logger.debug("STT executor shut down")
    except Exception as e:  # noqa: BLE001
        logger.debug("STT executor shutdown error (non-fatal): %s", e)


app = FastAPI(
    title="Mod³",
    description="Local multi-model TTS on Apple Silicon",
    lifespan=_lifespan,
)


try:
    from server import _bus as _shared_bus
except Exception:
    _shared_bus = ModalityBus()

_bus = _shared_bus
_bus_vad_lock = Lock()
_stt_transcribe_lock = Lock()
_stt_decoder: "WhisperDecoder | None" = None


def _get_stt_decoder() -> "WhisperDecoder":
    """Lazy-load WhisperDecoder with large-v3-turbo for /v1/transcribe."""
    global _stt_decoder
    if _stt_decoder is None:
        from modules.voice import WhisperDecoder

        _stt_decoder = WhisperDecoder(
            model="mlx-community/whisper-large-v3-turbo",
            load_base=False,
        )
    return _stt_decoder


def _ensure_bus_modules() -> None:
    modules = getattr(_bus, "_modules", {})
    if ModalityType.TEXT not in modules:
        _bus.register(TextModule())
    if ModalityType.VOICE not in modules:
        _bus.register(VoiceModule())


def _get_voice_module() -> VoiceModule | None:
    module = getattr(_bus, "_modules", {}).get(ModalityType.VOICE)
    return module if isinstance(module, VoiceModule) else None


def _resolve_voice_via_bus(voice: str) -> str:
    voice_module = _get_voice_module()
    if voice_module is None or voice_module.encoder is None:
        raise ValueError("Voice module is not registered on the ModalityBus.")

    for cfg in MODELS.values():
        if voice in cfg["voices"]:
            return voice

    # Voice profile registry surfaces cloned voices as first-class names.
    if _registry.get(voice) is not None:
        return voice

    raise ValueError(f"Unknown voice '{voice}'. Use /v1/voices to see options.")


def _read_wav_as_mono_float32(raw_wav: bytes) -> tuple[bytes, int]:
    import numpy as np

    with wave.open(io.BytesIO(raw_wav), "rb") as wav_file:
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


_ensure_bus_modules()

# ---------------------------------------------------------------------------
# Job ledger — full lifecycle tracking for every generation
# ---------------------------------------------------------------------------

MAX_JOBS = 100
_jobs: OrderedDict[str, dict] = OrderedDict()
_jobs_lock = Lock()


def _record_job(job: dict) -> str:
    job_id = uuid.uuid4().hex[:8]
    job["job_id"] = job_id
    with _jobs_lock:
        _jobs[job_id] = job
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)
    return job_id


def _update_job(job_id: str, updates: dict):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


def encode_wav(samples, sample_rate: int) -> bytes:
    """Encode float32 samples as 16-bit PCM WAV."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    num_samples = len(pcm)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    # WAV header (44 bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", 1))  # mono
    buf.write(struct.pack("<I", sample_rate))  # sample rate
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))  # block align
    buf.write(struct.pack("<H", 16))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


# Opus only accepts these sample rates (RFC 7587).
_OPUS_VALID_RATES = frozenset({8000, 12000, 16000, 24000, 48000})


def encode_ogg(samples, sample_rate: int) -> bytes:
    """Encode float32 samples as OGG/Opus.

    Mirrors encode_wav() — takes the same inputs, returns raw bytes.
    Opus requires one of {8000, 12000, 16000, 24000, 48000} Hz; if the
    engine produces a non-standard rate (e.g. 22050) the samples are
    resampled to 24000 before encoding.

    Target bitrate is controlled by soundfile/libopus defaults (~24 kbps
    for speech at 24 kHz mono).
    """
    import numpy as np
    import soundfile as sf

    samples = np.asarray(samples, dtype=np.float32)
    if sample_rate not in _OPUS_VALID_RATES:
        # Resample to 24000 — nearest Opus-compatible rate to most TTS engines.
        try:
            import math

            from scipy.signal import resample_poly

            target_rate = 24000
            gcd = math.gcd(target_rate, sample_rate)
            up, down = target_rate // gcd, sample_rate // gcd
            samples = resample_poly(samples, up, down).astype(np.float32)
            sample_rate = target_rate
        except ImportError:
            raise RuntimeError(
                f"OGG/Opus encoding requires sample_rate in {sorted(_OPUS_VALID_RATES)}; "
                f"got {sample_rate} Hz and scipy is not available for resampling."
            )

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="OGG", subtype="OPUS")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Request / Response models — imported from schemas.http
# (SynthesizeRequest, SpeechRequest, ShutdownRequest, SessionRegisterRequest,
#  RegisterProfileRequest, BusActRequest, VadFilterRequest)


# ---------------------------------------------------------------------------
# Shutdown middleware — reject new requests once shutdown is initiated
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _reject_during_shutdown(request: Request, call_next):
    """Return 503 for new requests once graceful shutdown has been initiated."""
    if _shutting_down and request.url.path != "/health":
        return JSONResponse(
            status_code=503,
            content={"error": "server is shutting down"},
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# localhost CSRF / DNS-rebinding guard
# ---------------------------------------------------------------------------
#
# Threat model: a malicious web page the user visits can issue cross-origin
# requests to the localhost daemon (localhost CSRF).  With DNS rebinding the
# attacker also controls the Host header.  The browser sends an Origin header
# on cross-origin fetches and same-site form POST; it does NOT send Origin on
# same-origin requests from the dashboard or on programmatic non-browser
# clients (httpx / MCP clients).
#
# Defence — two layers applied only to state-changing methods (POST, PUT,
# PATCH, DELETE):
#
#   1. Host-header allowlist: must be localhost / 127.0.0.1 (with optional
#      port).  Blocks DNS-rebinding attacks where the attacker substitutes a
#      hostname that resolves to 127.0.0.1.
#
#   2. Origin / Referer check (when the header is present): the origin must
#      be in the allowed set.  Same-origin requests from the dashboard omit
#      Origin entirely — those are always permitted.  Non-browser clients
#      (httpx, MCP) also omit Origin — permitted.  Only cross-origin browser
#      requests carry Origin, and those are rejected unless explicitly listed.
#
# Allowed origins are configured via MOD3_ALLOWED_ORIGINS (comma-separated
# scheme+host[:port]).  The default covers all common localhost variants so
# the dashboard and channel client work without configuration.
#
# Read-only GET / HEAD / OPTIONS requests are deliberately NOT gated — they
# are lower-risk and blocking them would break CORS preflight and monitoring.

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})

# Build the allowed origins set once at import time.
# Default: all localhost variants on the canonical port and no port.
_DEFAULT_ALLOWED_ORIGINS: frozenset[str] = frozenset(
    {
        "http://localhost",
        "http://localhost:7860",
        "http://127.0.0.1",
        "http://127.0.0.1:7860",
        "http://[::1]",
        "http://[::1]:7860",
    }
)


def _build_allowed_origins() -> frozenset[str]:
    """Return the effective allowed-origin set from env + defaults."""
    raw = os.environ.get("MOD3_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return _DEFAULT_ALLOWED_ORIGINS
    extras = frozenset(o.strip().rstrip("/") for o in raw.split(",") if o.strip())
    return _DEFAULT_ALLOWED_ORIGINS | extras


_ALLOWED_ORIGINS: frozenset[str] = _build_allowed_origins()


def _is_localhost_host(host_header: str) -> bool:
    """Return True when the Host header is a loopback address (with optional port)."""
    # Strip port: "localhost:7860" → "localhost"
    bare = host_header.split(":")[0].lower().strip("[]")
    return bare in {"localhost", "127.0.0.1", "::1"}


def _origin_is_allowed(origin: str) -> bool:
    """Return True when origin is in the allowed set (exact match after stripping trailing slash)."""
    return origin.rstrip("/") in _ALLOWED_ORIGINS


@app.middleware("http")
async def _localhost_csrf_guard(request: Request, call_next):
    """Reject cross-origin state-changing requests (CSRF / DNS-rebinding guard).

    Skips read-only methods (GET, HEAD, OPTIONS) and the /health probe.
    Applies to all mutating methods (POST, PUT, PATCH, DELETE).

    Rules:
      1. Host header must be a loopback address (stops DNS rebinding).
      2. If Origin is present it must be in the allowed set.
      3. Absent Origin is always permitted (non-browser or same-origin).

    On violation: 403 with a JSON body describing which check failed.
    """
    method = request.method.upper()
    path = request.url.path

    # Read-only methods and the health probe are never gated.
    if method in _SAFE_METHODS or path == "/health":
        return await call_next(request)

    # --- 1. Host header check (DNS-rebinding) ---
    host = request.headers.get("host", "")
    if host and not _is_localhost_host(host):
        logger.warning(
            "CSRF guard: rejected %s %s — disallowed Host header %r",
            method,
            path,
            host,
        )
        return JSONResponse(
            status_code=403,
            content={
                "error": "forbidden",
                "detail": (
                    f"Host header {host!r} is not a localhost address. "
                    "mod3 only accepts requests addressed to localhost."
                ),
            },
        )

    # --- 2. Origin check ---
    origin = request.headers.get("origin", "")
    if origin and not _origin_is_allowed(origin):
        logger.warning(
            "CSRF guard: rejected %s %s — disallowed Origin %r",
            method,
            path,
            origin,
        )
        return JSONResponse(
            status_code=403,
            content={
                "error": "forbidden",
                "detail": (f"Origin {origin!r} is not allowed. Set MOD3_ALLOWED_ORIGINS to add origins."),
            },
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/synthesize")
def synthesize(req: SynthesizeRequest):
    """Synthesize text to audio. Returns raw audio bytes + full metrics in headers and job ledger."""
    import numpy as np

    t_request = time.perf_counter()

    # ADR-082 Phase 1: session routing. If the request names a session, we
    # honor the session's assigned voice (unless the caller explicitly
    # picked a non-default voice) and account the job against the session's
    # queue + serializer so multi-session callers can see round-robin.
    session_id = req.session_id
    session_payload: dict | None = None
    if session_id:
        registry = get_default_registry()
        session = registry.get(session_id)
        if session is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"session '{session_id}' is not registered — POST /v1/sessions/register first",
                },
            )
        if req.voice == "bm_lewis" and session.assigned_voice != "bm_lewis":
            req.voice = session.assigned_voice
        # Register the submission with the serializer for accounting only.
        # The synthesize endpoint is non-blocking on the audio side (we
        # return bytes synchronously), so we do not run the registry's
        # dispatcher here — we just record the submission.
        try:
            registry.submit(session_id, {"type": "synthesize", "text": req.text[:200]})
        except Exception as exc:  # noqa: BLE001
            logger.debug("session submit accounting failed: %s", exc)
        session_payload = {
            "session_id": session.session_id,
            "assigned_voice": session.assigned_voice,
            "preferred_output_device": session.preferred_output_device,
        }

    job_id = _record_job(
        {
            "type": "synthesize",
            "status": "generating",
            "requested_at": time.time(),
            "text": req.text[:200],
            "voice": req.voice,
            "speed": req.speed,
            "emotion": req.emotion,
            "format": req.format,
            "engine": None,
            "session_id": session_id,
            "timeline": [{"event": "request_received", "t": 0.0}],
        }
    )

    try:
        req.voice = _resolve_voice_via_bus(req.voice)
    except ValueError as e:
        _update_job(job_id, {"status": "error", "error": str(e)})
        return JSONResponse(status_code=400, content={"error": str(e), "job_id": job_id})

    t_gen_start = time.perf_counter()
    _update_job(job_id, {"timeline_append": True})
    _append_timeline(job_id, "generation_start", t_gen_start - t_request)

    chunks = list(
        generate_audio(
            req.text,
            voice=req.voice,
            speed=req.speed,
            emotion=req.emotion,
            stream=False,
            ref_audio=req.ref_audio,
        )
    )
    t_gen_end = time.perf_counter()

    if not chunks:
        _update_job(job_id, {"status": "error", "error": "No audio generated"})
        return JSONResponse(status_code=400, content={"error": "No audio generated", "job_id": job_id})

    sample_rate = chunks[0].sample_rate
    all_samples = np.concatenate([c.samples for c in chunks])
    duration = len(all_samples) / sample_rate
    gen_time = t_gen_end - t_gen_start

    # Per-chunk metrics
    chunk_metrics = []
    for c in chunks:
        if c.metadata:
            chunk_metrics.append(c.metadata)

    t_encode_start = time.perf_counter()
    if req.format == "pcm":
        pcm = (np.clip(all_samples, -1.0, 1.0) * 32767).astype(np.int16)
        audio_bytes = pcm.tobytes()
        media_type = "audio/pcm"
        wav_for_ws = encode_wav(all_samples, sample_rate)  # dashboard always gets WAV
    elif req.format == "ogg":
        audio_bytes = encode_ogg(all_samples, sample_rate)
        media_type = "audio/ogg; codecs=opus"
        wav_for_ws = encode_wav(all_samples, sample_rate)  # dashboard always gets WAV
    else:
        audio_bytes = encode_wav(all_samples, sample_rate)
        media_type = "audio/wav"
        wav_for_ws = audio_bytes
    t_encode_end = time.perf_counter()

    total_time = t_encode_end - t_request
    engine = chunks[0].metadata.get("engine", "") if chunks[0].metadata else ""

    # Finalize job record
    _append_timeline(job_id, "generation_complete", t_gen_end - t_request)
    _append_timeline(job_id, "encoding_complete", t_encode_end - t_request)

    # Wave 4.3 — route to any dashboard WebSocket subscribers for this
    # session before returning the HTTP response. Mod3 emits the WAV over
    # the /ws/audio/{session_id} channel; the MCP shim and the kernel both
    # consult /v1/sessions/{id}/subscribers to skip local playback when
    # this path fired, so there's no double-play. Pure HTTP callers without
    # a session (or without a subscriber) still get their bytes in the
    # response body exactly as before.
    ws_delivered = 0
    if session_id:
        subs = get_default_audio_subscribers()
        try:
            ws_delivered = subs.emit_wav(
                session_id,
                wav_for_ws,
                job_id=job_id,
                duration_sec=round(duration, 3),
                sample_rate=sample_rate,
            )
        except Exception as exc:  # noqa: BLE001 — never fail synthesize on a WS push
            logger.debug("ws audio emit failed: %s", exc)

    _update_job(
        job_id,
        {
            "status": "complete",
            "engine": engine,
            "metrics": {
                "audio_duration_sec": round(duration, 3),
                "total_samples": len(all_samples),
                "sample_rate": sample_rate,
                "generation_time_sec": round(gen_time, 3),
                "encoding_time_sec": round(t_encode_end - t_encode_start, 4),
                "total_time_sec": round(total_time, 3),
                "rtf": round(duration / gen_time, 2) if gen_time > 0 else 0,
                "chunks": len(chunk_metrics),
                "per_chunk": chunk_metrics,
                "output_bytes": len(audio_bytes),
                "output_format": req.format,
                "ws_subscribers_delivered": ws_delivered,
            },
        },
    )

    headers = {
        "X-Mod3-Job-Id": job_id,
        "X-Mod3-Engine": engine,
        "X-Mod3-Voice": req.voice,
        "X-Mod3-Duration-Sec": f"{duration:.3f}",
        "X-Mod3-Sample-Rate": str(sample_rate),
        "X-Mod3-Gen-Time-Sec": f"{gen_time:.3f}",
        "X-Mod3-Total-Time-Sec": f"{total_time:.3f}",
        "X-Mod3-RTF": f"{duration / gen_time:.2f}" if gen_time > 0 else "0",
        "X-Mod3-Chunks": str(len(chunk_metrics)),
        "X-Mod3-WS-Subscribers": str(ws_delivered),
    }
    if session_payload is not None:
        headers["X-Mod3-Session-Id"] = session_payload["session_id"]

    # Update last_used_at for registered voice profiles (fire-and-forget; never
    # blocks or fails the response).
    try:
        _registry.update_last_used_at(req.voice)
    except Exception:  # noqa: BLE001
        pass

    return Response(content=audio_bytes, media_type=media_type, headers=headers)


@app.post("/v1/audio/speech")
def audio_speech(req: SpeechRequest):
    """OpenAI-compatible TTS endpoint. Accepts OpenAI format, returns WAV audio."""
    import numpy as np

    t_request = time.perf_counter()

    # ADR-082 Phase 1: optional session routing. Same semantics as
    # /v1/synthesize — the session's assigned voice overrides ``voice`` when
    # the caller passed the default, and the submission is accounted against
    # the session's queue.
    session_id = req.session_id
    if session_id:
        registry = get_default_registry()
        session = registry.get(session_id)
        if session is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"session '{session_id}' is not registered — POST /v1/sessions/register first",
                },
            )
        # OpenAI default is af_heart; if the caller left it at the default,
        # prefer the session's voice.
        if req.voice == "af_heart" and session.assigned_voice != "af_heart":
            req.voice = session.assigned_voice
        try:
            registry.submit(session_id, {"type": "audio_speech", "text": req.input[:200]})
        except Exception as exc:  # noqa: BLE001
            logger.debug("session submit accounting failed: %s", exc)

    voice = req.voice
    try:
        voice = _resolve_voice_via_bus(voice)
    except ValueError:
        voice = "af_heart"

    job_id = _record_job(
        {
            "type": "audio_speech",
            "status": "generating",
            "requested_at": time.time(),
            "text": req.input[:200],
            "voice": voice,
            "speed": req.speed,
            "session_id": session_id,
            "timeline": [{"event": "request_received", "t": 0.0}],
        }
    )

    chunks = list(
        generate_audio(
            req.input,
            voice=voice,
            speed=req.speed,
            stream=False,
        )
    )
    t_gen_end = time.perf_counter()

    if not chunks:
        _update_job(job_id, {"status": "error", "error": "No audio generated"})
        return JSONResponse(status_code=500, content={"error": "No audio generated", "job_id": job_id})

    sample_rate = chunks[0].sample_rate
    all_samples = np.concatenate([c.samples for c in chunks])
    duration = len(all_samples) / sample_rate
    gen_time = t_gen_end - t_request

    audio_bytes = encode_wav(all_samples, sample_rate)
    total_time = time.perf_counter() - t_request
    engine = chunks[0].metadata.get("engine", "") if chunks[0].metadata else ""

    _update_job(
        job_id,
        {
            "status": "complete",
            "engine": engine,
            "metrics": {
                "audio_duration_sec": round(duration, 3),
                "generation_time_sec": round(gen_time, 3),
                "total_time_sec": round(total_time, 3),
                "rtf": round(duration / gen_time, 2) if gen_time > 0 else 0,
            },
        },
    )

    headers = {
        "X-Mod3-Job-Id": job_id,
        "X-Mod3-Engine": engine,
        "X-Mod3-Voice": voice,
        "X-Mod3-Duration-Sec": f"{duration:.3f}",
        "X-Mod3-Sample-Rate": str(sample_rate),
        "X-Mod3-Gen-Time-Sec": f"{gen_time:.3f}",
        "X-Mod3-Total-Time-Sec": f"{total_time:.3f}",
    }
    if session_id:
        headers["X-Mod3-Session-Id"] = session_id

    # Update last_used_at for registered voice profiles.
    try:
        _registry.update_last_used_at(voice)
    except Exception:  # noqa: BLE001
        pass

    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


@app.post("/v1/vad")
async def vad_check(file: UploadFile):
    """Check if an audio file contains speech. Returns VAD result with timing."""
    import tempfile

    t_start = time.perf_counter()

    job_id = _record_job(
        {
            "type": "vad",
            "status": "processing",
            "requested_at": time.time(),
            "timeline": [{"event": "request_received", "t": 0.0}],
        }
    )

    content = await file.read()
    t_load = time.perf_counter()

    voice_module = _get_voice_module()
    if voice_module is not None and voice_module.gate is not None:
        raw_audio, sample_rate = _read_wav_as_mono_float32(content)
        with _bus_vad_lock:
            gate_result = voice_module.gate.check(raw_audio, sample_rate=sample_rate, sample_width=4)
            _bus.perceive(
                raw_audio,
                modality=ModalityType.VOICE,
                channel="http:v1/vad",
                sample_rate=sample_rate,
                sample_width=4,
                transcript="speech detected",
            )

        class _Result:
            has_speech = gate_result.passed
            confidence = gate_result.confidence
            speech_ratio = gate_result.metadata.get("speech_ratio", 0.0)
            num_segments = gate_result.metadata.get("num_segments", 0)
            total_speech_sec = gate_result.metadata.get("total_speech_sec", 0.0)
            total_audio_sec = gate_result.metadata.get("total_audio_sec", 0.0)

        result = _Result()
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            result = detect_speech_file(tmp.name)

    t_end = time.perf_counter()
    processing_time = t_end - t_start

    _update_job(
        job_id,
        {
            "status": "complete",
            "metrics": {
                "has_speech": result.has_speech,
                "confidence": result.confidence,
                "speech_ratio": result.speech_ratio,
                "num_segments": result.num_segments,
                "total_speech_sec": result.total_speech_sec,
                "total_audio_sec": result.total_audio_sec,
                "processing_time_sec": round(processing_time, 4),
                "file_load_time_sec": round(t_load - t_start, 4),
                "vad_time_sec": round(t_end - t_load, 4),
            },
        },
    )

    return {
        "job_id": job_id,
        "has_speech": result.has_speech,
        "confidence": result.confidence,
        "speech_ratio": result.speech_ratio,
        "num_segments": result.num_segments,
        "total_speech_sec": result.total_speech_sec,
        "total_audio_sec": result.total_audio_sec,
        "processing_time_sec": round(processing_time, 4),
    }


@app.post("/v1/filter")
async def filter_transcription(req: VadFilterRequest):
    """Check if a transcription is a known Whisper hallucination.

    Body: {"text": "thank you"}
    Returns: {"is_hallucination": true, "text": "thank you"}
    """
    return {
        "is_hallucination": is_hallucination(req.text),
        "text": req.text,
    }


@app.post("/v1/transcribe")
async def transcribe_audio(file: UploadFile):
    """Transcribe an audio file to text using Whisper.

    Accepts WAV, OGG, MP3, M4A audio files.
    Returns transcript with language detection and timing metrics.
    """
    import subprocess
    import tempfile

    import numpy as np

    t_start = time.perf_counter()

    content = await file.read()
    if not content:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty audio file"},
        )

    # Determine format from content-type or filename extension
    filename = file.filename or ""
    content_type = file.content_type or ""

    is_wav = (
        filename.lower().endswith(".wav")
        or content_type == "audio/wav"
        or content_type == "audio/x-wav"
        or content[:4] == b"RIFF"
    )

    try:
        if is_wav:
            raw_audio, sample_rate = _read_wav_as_mono_float32(content)
            audio = np.frombuffer(raw_audio, dtype=np.float32)
        else:
            # Convert non-WAV to WAV using ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_in:
                tmp_in.write(content)
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path + ".wav"
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        tmp_in_path,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        tmp_out_path,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                with open(tmp_out_path, "rb") as f:
                    wav_content = f.read()
                raw_audio, sample_rate = _read_wav_as_mono_float32(wav_content)
                audio = np.frombuffer(raw_audio, dtype=np.float32)
            finally:
                if os.path.exists(tmp_in_path):
                    os.unlink(tmp_in_path)
                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)
    except Exception as e:
        logger.warning("Failed to parse audio file: %s", e)
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to parse audio: {e}"},
        )

    duration_sec = len(audio) / sample_rate if sample_rate > 0 else 0.0

    # Resample to 16kHz if needed (Whisper expects 16kHz)
    if sample_rate != 16000 and sample_rate > 0:
        ratio = 16000 / sample_rate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    decoder = _get_stt_decoder()

    t_stt_start = time.perf_counter()
    with _stt_transcribe_lock:
        event = decoder.decode(b"", audio=audio)
    t_stt_end = time.perf_counter()

    stt_ms = (t_stt_end - t_stt_start) * 1000

    transcript = event.content
    language = event.metadata.get("language", "en")

    # Hallucination filter already applied in decoder.decode(), but check metadata
    if event.metadata.get("filtered"):
        transcript = ""

    return {
        "transcript": transcript,
        "language": language,
        "duration_sec": round(duration_sec, 3),
        "stt_ms": round(stt_ms, 1),
    }


# ---------------------------------------------------------------------------
# Job introspection
# ---------------------------------------------------------------------------


@app.get("/v1/jobs")
def list_jobs(limit: int = 20, type: str = ""):
    """List recent generation jobs with metrics. Optionally filter by type."""
    with _jobs_lock:
        jobs = list(reversed(_jobs.values()))
    if type:
        jobs = [j for j in jobs if j.get("type") == type]
    return {"jobs": jobs[:limit], "total": len(jobs)}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    """Get full details for a specific job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": f"Job '{job_id}' not found"})
    return job


# ---------------------------------------------------------------------------
# Voices and health
# ---------------------------------------------------------------------------


@app.get("/v1/voices")
def voices():
    """List available engines and their voices, including registered profiles."""
    engines = {}
    for name, cfg in MODELS.items():
        supports = []
        if cfg.get("supports_speed"):
            supports.append("speed")
        if cfg.get("supports_exaggeration"):
            supports.append("emotion")
        if cfg.get("supports_pitch"):
            supports.append("pitch")
        engines[name] = {
            "model_id": cfg["id"],
            "voices": list(cfg["voices"]),
            "default_voice": cfg["default_voice"],
            "supports": supports,
        }

    # Merge registered voice profiles into their parent engine's voice list.
    for profile in _registry.list():
        if profile.engine not in engines:
            continue
        engines[profile.engine]["voices"].append(profile.name)
        engines[profile.engine].setdefault("custom_voices", []).append(profile.name)

    return {"engines": engines}


# ---------------------------------------------------------------------------
# Voice profiles — registration, listing, deletion
# ---------------------------------------------------------------------------


@app.post("/v1/voices/profiles")
def register_profile(req: RegisterProfileRequest):
    """Register a new voice profile from a reference audio path.

    The reference audio is loaded and processed through the engine's
    prepare_conditionals step; the resulting Conditionals are stored on disk
    alongside a metadata sidecar. This may be slow on first call if the
    chatterbox model is not yet loaded.

    Returns the registered profile as JSON (200), or:
      409 if a profile with that name already exists,
      404 if ref_audio_path does not exist,
      400 for validation errors (invalid name, engine not supported, etc.).
    """
    from fastapi import HTTPException

    from engine import get_model

    # Resolve model_id from engine registry.
    engine_cfg = MODELS.get(req.engine)
    if engine_cfg is None or not engine_cfg.get("supports_cloning"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"engine {req.engine!r} does not support voice cloning; "
                f"supported engines: {[k for k, v in MODELS.items() if v.get('supports_cloning')]}"
            ),
        )

    import pathlib

    if not pathlib.Path(req.ref_audio_path).exists():
        raise HTTPException(status_code=404, detail=f"ref_audio_path not found: {req.ref_audio_path}")

    # Check for duplicate before loading the model (cheaper).
    existing = _registry.get(req.name)
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"profile {req.name!r} already exists")

    try:
        model = get_model(req.engine)
        # Chatterbox requires ref_sr (positional int) and RETURNS the
        # Conditionals. Turbo uses sample_rate (autodetect when None) and
        # STORES to self._conds rather than returning. Handle both shapes.
        if req.engine == "chatterbox-turbo":
            model.prepare_conditionals(req.ref_audio_path, exaggeration=req.exaggeration)
            conds = model._conds
        else:
            conds = model.prepare_conditionals(req.ref_audio_path, ref_sr=24000, exaggeration=req.exaggeration)
        if conds is None:
            raise RuntimeError(f"prepare_conditionals returned None and model._conds is None for engine {req.engine}")
        profile = _registry.register(
            name=req.name,
            engine=req.engine,
            ref_audio_path=req.ref_audio_path,
            conds=conds,
            ref_text=req.ref_text,
            exaggeration=req.exaggeration,
            model_id=engine_cfg["id"],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        msg = str(exc)
        if "already exists" in msg:
            raise HTTPException(status_code=409, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc

    return profile.to_json()


def _load_wav_24k_mono(path: str):
    """Read a 24 kHz, 16-bit PCM WAV (mono or stereo) into an int16 numpy array.

    Stereo inputs are downmixed to mono. Any other sample rate, sample width,
    or channel count raises ValueError with a human-readable message.
    """
    import wave

    import numpy as np

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        if sr != 24000:
            raise ValueError(f"{path}: sample rate must be 24000 Hz, got {sr} Hz")
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError(f"{path}: sample width must be 16-bit, got {sw * 8}-bit")
        nch = wf.getnchannels()
        if nch not in (1, 2):
            raise ValueError(f"{path}: must be mono or stereo, got {nch} channels")
        frames = wf.readframes(wf.getnframes())

    arr = np.frombuffer(frames, dtype=np.int16)
    if nch == 2:
        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return arr


def _write_wav_24k_mono(path: str, samples) -> None:
    """Write a numpy int16 array as a 24 kHz mono 16-bit PCM WAV."""
    import wave

    import numpy as np

    samples = np.ascontiguousarray(samples, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(samples.tobytes())


@app.post("/v1/voices/profiles/compose")
def compose_profile(req: ComposeProfileRequest):
    """Concatenate multiple reference clips into one and register as a profile.

    Each segment must be a 16-bit PCM WAV at 24 kHz. Stereo files are
    downmixed; other sample rates or bit depths are rejected with 400.
    Segments are joined in the given order with `gap_sec` seconds of silence
    between consecutive entries. The combined WAV is written to
    ``~/.mod3/voices/sources/<name>.wav`` and then registered through the
    same path used by POST /v1/voices/profiles.

    Returns the registered profile as JSON (200), or:
      400 — empty segment_paths, engine doesn't support cloning, or any
            segment fails WAV validation
      404 — any segment path does not exist
      409 — a profile with that name already exists
    """
    import pathlib

    import numpy as np
    from fastapi import HTTPException

    if not req.segment_paths:
        raise HTTPException(status_code=400, detail="segment_paths must include at least one entry")

    engine_cfg = MODELS.get(req.engine)
    if engine_cfg is None or not engine_cfg.get("supports_cloning"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"engine {req.engine!r} does not support voice cloning; "
                f"supported engines: {[k for k, v in MODELS.items() if v.get('supports_cloning')]}"
            ),
        )

    if _registry.get(req.name) is not None:
        raise HTTPException(status_code=409, detail=f"profile {req.name!r} already exists")

    paths = [pathlib.Path(p).expanduser() for p in req.segment_paths]
    for p in paths:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"segment not found: {p}")

    try:
        chunks = [_load_wav_24k_mono(str(p)) for p in paths]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    gap = np.zeros(int(req.gap_sec * 24000), dtype=np.int16)
    parts: list = []
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(chunk)
    combined = np.concatenate(parts)

    sources_dir = pathlib.Path.home() / ".mod3" / "voices" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    out_path = sources_dir / f"{req.name}.wav"
    _write_wav_24k_mono(str(out_path), combined)

    return register_profile(
        RegisterProfileRequest(
            name=req.name,
            engine=req.engine,
            ref_audio_path=str(out_path),
            ref_text=req.ref_text,
            exaggeration=req.exaggeration,
        )
    )


@app.get("/v1/voices/profiles")
def list_profiles(
    request: Request,
    tag: list[str] | None = None,
    favorite: bool | None = None,
    engine: str | None = None,
    sort: str | None = None,
):
    """List registered voice profiles with optional filtering and sorting.

    Query parameters:
      ?tag=<tag>        Filter to profiles having this tag (repeatable; OR semantics).
      ?favorite=true    Filter to favorited profiles only.
      ?engine=<engine>  Filter to profiles registered with this engine.
      ?sort=name        Sort by name ascending (default).
      ?sort=last_used   Sort by last_used_at descending (nulls last).
      ?sort=rating      Sort by rating descending (nulls last).

    Multiple filter types compose with AND; within multi-value ?tag= the
    semantics are OR (profile matches if it has *any* of the listed tags).
    """
    # FastAPI doesn't automatically collect repeated query params for
    # list[str] | None in older versions — read directly from the request.
    tag_values = request.query_params.getlist("tag") if tag is None else tag

    profiles = _registry.list()

    # --- filter ---
    if tag_values:
        tag_set = set(tag_values)
        profiles = [p for p in profiles if tag_set.intersection(p.tags)]

    if favorite is not None:
        profiles = [p for p in profiles if p.favorite == favorite]

    if engine is not None:
        profiles = [p for p in profiles if p.engine == engine]

    # --- sort ---
    if sort == "last_used":
        # Most-recently-used first; profiles without last_used_at sort to the end.
        # key tuple: (has_value, timestamp_str) — both reversed so highest wins first.
        # has_value=1 for set profiles, 0 for null; ISO timestamps sort
        # lexicographically so most-recent string is largest.
        profiles = sorted(
            profiles,
            key=lambda p: (1, p.last_used_at) if p.last_used_at else (0, ""),
            reverse=True,
        )
    elif sort == "rating":
        profiles = sorted(
            profiles,
            key=lambda p: (0 if p.rating is not None else 1, -(p.rating or 0)),
        )
    else:
        # default: name ascending (registry.list() already sorts this way, but
        # re-sort after filtering to preserve order)
        profiles = sorted(profiles, key=lambda p: p.name)

    return {"profiles": [p.to_json() for p in profiles]}


@app.get("/v1/voices/profiles/{name}")
def get_profile(name: str):
    """Return a single voice profile by name. 404 if not found."""
    from fastapi import HTTPException

    profile = _registry.get(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"profile {name!r} not found")
    return profile.to_json()


@app.patch("/v1/voices/profiles/{name}")
async def patch_profile(name: str, request: Request):
    """Update curation metadata fields on an existing profile.

    Accepts a JSON body with any subset of:
      favorite: bool
      notes: str
      tags: list[str]
      last_used_at: str | null  (ISO 8601 timestamp)
      rating: int | null        (1-5 when set)

    Returns the updated profile (200), 404 if not found, 400 on invalid values.
    """
    from fastapi import HTTPException

    body_bytes = await request.body()
    try:
        updates = json.loads(body_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")

    # Validate field types before touching disk.
    allowed = {"favorite", "notes", "tags", "last_used_at", "rating"}
    unknown = set(updates) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown fields: {sorted(unknown)}")

    if "favorite" in updates and not isinstance(updates["favorite"], bool):
        raise HTTPException(status_code=400, detail="'favorite' must be a boolean")
    if "notes" in updates and not isinstance(updates["notes"], str):
        raise HTTPException(status_code=400, detail="'notes' must be a string")
    if "tags" in updates:
        if not isinstance(updates["tags"], list) or not all(isinstance(t, str) for t in updates["tags"]):
            raise HTTPException(status_code=400, detail="'tags' must be a list of strings")

    try:
        profile = _registry.patch_metadata(name, updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if profile is None:
        raise HTTPException(status_code=404, detail=f"profile {name!r} not found")

    return profile.to_json()


@app.delete("/v1/voices/profiles/{name}")
def delete_profile(name: str):
    """Remove a voice profile by name."""
    removed = _registry.delete(name)
    if not removed:
        return JSONResponse(
            status_code=404,
            content={"deleted": False, "error": "not found"},
        )
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Compositions — voice-lab iteration unit (draft of a profile)
# ---------------------------------------------------------------------------


def _composition_to_dict(c: Composition) -> dict:
    return c.to_json()


def _segments_from_request(specs) -> list[Segment]:
    return [Segment(path=s.path, label=s.label, duration_sec=s.duration_sec) for s in specs]


@app.get("/v1/voices/compositions")
def list_compositions():
    """List all saved composition drafts."""
    return {"compositions": [_composition_to_dict(c) for c in _compositions.list()]}


@app.post("/v1/voices/compositions")
def create_composition(req: CompositionCreateRequest):
    """Create a new composition draft. Returns 409 if `name` is taken."""
    from fastapi import HTTPException

    try:
        comp = Composition(
            name=req.name,
            segments=_segments_from_request(req.segments),
            engine=req.engine,
            exaggeration=req.exaggeration,
            gap_sec=req.gap_sec,
            notes=req.notes,
        )
        created = _compositions.create(comp)
    except ValueError as exc:
        msg = str(exc)
        status = 409 if "already exists" in msg else 400
        raise HTTPException(status_code=status, detail=msg) from exc
    return _composition_to_dict(created)


@app.get("/v1/voices/compositions/{name}")
def get_composition(name: str):
    from fastapi import HTTPException

    comp = _compositions.get(name)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"composition {name!r} not found")
    return _composition_to_dict(comp)


@app.patch("/v1/voices/compositions/{name}")
def update_composition(name: str, req: CompositionUpdateRequest):
    from fastapi import HTTPException

    patch = req.model_dump(exclude_none=True)
    if "segments" in patch:
        patch["segments"] = [
            {"path": s["path"], "label": s.get("label", ""), "duration_sec": s.get("duration_sec")}
            for s in patch["segments"]
        ]
    try:
        updated = _compositions.update(name, patch)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"composition {name!r} not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _composition_to_dict(updated)


@app.delete("/v1/voices/compositions/{name}")
def delete_composition(name: str):
    if not _compositions.delete(name):
        return JSONResponse(status_code=404, content={"deleted": False, "error": "not found"})
    return {"deleted": True}


@app.post("/v1/voices/compositions/{name}/register")
def register_composition(name: str, profile_name: str | None = None):
    """Synthesize a profile from a saved composition.

    Routes to POST /v1/voices/profiles/compose with the composition's segments
    + settings. `profile_name` query param overrides the composition's `name`
    for the resulting profile (useful for A/B-ing the same draft).
    """
    from fastapi import HTTPException

    comp = _compositions.get(name)
    if comp is None:
        raise HTTPException(status_code=404, detail=f"composition {name!r} not found")
    return compose_profile(
        ComposeProfileRequest(
            name=profile_name or comp.name,
            engine=comp.engine,
            segment_paths=[s.path for s in comp.segments],
            gap_sec=comp.gap_sec,
            exaggeration=comp.exaggeration,
        )
    )


@app.get("/v1/voices/segments")
def get_segment_audio(path: str):
    """Serve a segment WAV for inline preview in the voice lab.

    Path must point to an existing 16-bit PCM WAV under one of:
      - ~/.mod3/
      - ~/.claude/jobs/
      - /tmp/voice_lab/
      - $TMPDIR / /tmp segment subtrees
    Anything else is rejected with 403.
    """
    import pathlib

    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    target = pathlib.Path(path).expanduser().resolve()
    allowed_roots = [
        (pathlib.Path.home() / ".mod3").resolve(),
        (pathlib.Path.home() / ".claude" / "jobs").resolve(),
        pathlib.Path("/tmp/voice_lab").resolve(),
    ]
    if not any(target.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="path is not within an allowed segment root")
    if not target.exists() or target.suffix.lower() != ".wav":
        raise HTTPException(status_code=404, detail="segment WAV not found")
    return FileResponse(str(target), media_type="audio/wav")


@app.post("/v1/stop")
def stop_speech(job_id: str = ""):
    """Stop current speech and/or cancel queued items.

    If job_id is provided, cancels that specific job.
    If empty, interrupts current playback and clears the queue.
    Returns interruption context for barge-in support.
    """
    try:
        from server import _speech_queue, pipeline_state

        if job_id:
            cancelled = _speech_queue.cancel(job_id)
            return {"status": "ok", "message": f"Cancelled {job_id}" if cancelled else f"Job {job_id} not found"}
        else:
            # Get interrupt info before stopping
            interrupt_info = None
            if pipeline_state.is_speaking:
                info = pipeline_state.interrupt(reason="http_barge_in")
                if info:
                    interrupt_info = {
                        "spoken_pct": info.spoken_pct,
                        "delivered_text": info.delivered_text,
                        "full_text": info.full_text,
                        "reason": info.reason,
                    }
            cancelled_count = _speech_queue.cancel_all_queued()
            return {
                "status": "ok",
                "message": f"Interrupted playback; cancelled {cancelled_count} queued items",
                "interrupted": interrupt_info,
            }
    except ImportError:
        return JSONResponse(status_code=503, content={"error": "Speech queue not available in HTTP-only mode"})


@app.post("/v1/speak")
def speak_enqueue(req: SpeakRequest):
    """Queue-aware speak endpoint. Wraps _start_speech; returns immediately.

    Unlike /v1/synthesize (which blocks, returns WAV bytes, and requires the
    caller to manage playback), this endpoint enqueues the request in mod3's
    speech queue and returns a job token. The drain thread owns all audio
    playback — callers never spawn afplay or aplay.

    Returns:
        {
            "job_id": str,          # correlation handle; poll /v1/speech_status
            "queue_position": int,  # 0 = playing immediately, N = queued
            "status": str           # "speaking" | "queued"
        }

    Poll GET /v1/jobs/{job_id} for completion status.
    Stop via POST /v1/stop?job_id={job_id}.
    """
    if not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "text required"})
    try:
        from server import _start_speech

        job_id, position = _start_speech(
            req.text,
            req.voice,
            stream=req.stream,
            speed=req.speed,
            emotion=req.emotion,
            session_id=req.session_id or None,
            ref_audio=req.ref_audio or None,
        )
        return {
            "job_id": job_id,
            "queue_position": position,
            "status": "queued" if position > 0 else "speaking",
        }
    except ImportError:
        return JSONResponse(status_code=503, content={"error": "Speech queue not available in HTTP-only mode"})


# ---------------------------------------------------------------------------
# Sessions — ADR-082 Phase 1
# ---------------------------------------------------------------------------


@app.post("/v1/sessions/register")
def session_register(req: SessionRegisterRequest):
    """Register a session with the Mod3 communication bus (ADR-082).

    Body:
      {
        "session_id": "...",
        "participant_id": "cog" | "sandy" | "alice" | ...,
        "participant_type": "agent" | "user",
        "preferred_voice": "bm_lewis" | ... | null,
        "preferred_output_device": "system-default" | "<device-name>"
      }

    Returns the SessionChannel with a live-resolved output_device.
    """
    registry = get_default_registry()
    try:
        result = registry.register(
            session_id=req.session_id,
            participant_id=req.participant_id,
            participant_type=req.participant_type,
            preferred_voice=req.preferred_voice,
            preferred_output_device=req.preferred_output_device or "system-default",
            priority=req.priority,
        )
    except Exception as exc:  # noqa: BLE001 — surface the error verbatim
        return JSONResponse(status_code=400, content={"error": str(exc)})

    payload = result.session.to_dict(device_resolver=resolve_output_device)
    payload["created"] = result.created
    # Top-level live device snapshot so the caller does not have to
    # round-trip; the nested one is available for debugging.
    payload["output_device"] = registry.resolve_device(result.session.session_id).to_dict()
    return payload


@app.post("/v1/sessions/{session_id}/deregister")
def session_deregister(session_id: str):
    """Drop a session — drains/cancels pending jobs, returns voice to pool."""
    registry = get_default_registry()
    result = registry.deregister(session_id)
    if result.get("status") == "not_found":
        return JSONResponse(status_code=404, content=result)
    return result


@app.get("/v1/sessions")
def session_list():
    """List all registered sessions plus a serializer snapshot."""
    registry = get_default_registry()
    return {
        "sessions": registry.list_serialized(),
        "serializer": registry.serializer.snapshot(),
        "voice_pool": registry.voice_pool(),
        "voice_holders": registry.voice_holder_snapshot(),
    }


@app.get("/v1/sessions/{session_id}")
def session_get(session_id: str):
    """Get a single session's current state (with live device resolution)."""
    registry = get_default_registry()
    session = registry.get(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"error": f"session '{session_id}' not found"})
    payload = session.to_dict(device_resolver=resolve_output_device)
    payload["output_device"] = registry.resolve_device(session_id).to_dict()
    return payload


@app.get("/v1/sessions/{session_id}/subscribers")
def session_subscribers(session_id: str):
    """Wave 4.3 — does this session have any live audio WebSocket subscribers?

    The kernel queries this before spawning afplay: if any dashboard or
    native client has attached to ``/ws/audio/{session_id}``, the bytes are
    routed over the WebSocket and the server-side fallback player is
    skipped. Unknown sessions return ``{"subscribed": false, "count": 0}``
    instead of 404 so the kernel's check stays a single well-defined
    predicate regardless of registration state.
    """
    subs = get_default_audio_subscribers()
    count = subs.count(session_id)
    return {
        "session_id": session_id,
        "subscribed": count > 0,
        "count": count,
    }


# ---------------------------------------------------------------------------
# Channel seats — register / revoke / SSE stream / dashboard fan-out
# (supports channel_client.py seat-based session attachment)
# ---------------------------------------------------------------------------


@app.post("/v1/sessions/{session_id}/seats")
async def seat_register(session_id: str, request: Request):
    """Register a channel-client seat in *session_id*.

    Body (JSON):
      {
        "client_type": "claude-code-channel" | "generic",
        "device_uuid": "<persistent uuid>"
      }

    Returns:
      {
        "seat_id": "...",
        "session_id": "...",
        "auth_token": "<bearer token echo for confirmation>"
      }

    Auto-creates the session bucket if it does not exist.
    Access policy is enforced via access.py when that module is available.
    """
    from seats import get_seat_registry

    try:
        body = await request.json()
    except Exception:
        body = {}

    client_type = body.get("client_type", "generic")
    device_uuid = body.get("device_uuid", "")
    # Identity claims — optional; absent = unattributed seat
    # Wave 6b: iss/sub as user identity (backward compat field names preserved)
    user_iss: str | None = body.get("user_iss") or body.get("iss") or None
    user_sub: str | None = body.get("user_sub") or body.get("sub") or None
    # Wave 6c / Primitive 2: agent identity for agentic harnesses (Claude Code, Cursor)
    assistant_iss: str | None = body.get("assistant_iss") or None
    assistant_sub: str | None = body.get("assistant_sub") or None
    # Primitive 4: channel pipeline mode — optional; absent = intentional (default)
    channel_mode: str = body.get("channel_mode") or "intentional"

    # Optional: enforce access policy from access.py if available
    try:
        import access as _access

        peer = request.client.host if request.client else "127.0.0.1"
        identifier = device_uuid or peer
        allowed = _access.is_allowed(identifier, peer)
        if not allowed:
            # Emit pairing request event to any existing seats in this session
            registry = get_seat_registry()
            code = _access.add_pending(identifier)
            registry.fan_out(
                session_id,
                {
                    "type": "pairing_request",
                    "identifier": identifier,
                    "code": code,
                },
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "access denied",
                    "pairing_code": code,
                    "message": f"Run `/mod3:access pair {code}` to approve this device",
                },
            )
    except ImportError:
        pass  # access.py not yet ported — allow all (localhost-only in dev)

    registry = get_seat_registry()
    seat = registry.register(
        session_id=session_id,
        client_type=client_type,
        device_uuid=device_uuid,
        user_iss=user_iss,
        user_sub=user_sub,
        assistant_iss=assistant_iss,
        assistant_sub=assistant_sub,
        channel_mode=channel_mode,
    )
    logger.info("Seat registered: %s in session %s (mode=%s)", seat.seat_id, session_id, channel_mode)

    # Mirror the seat-bearing session into the SessionRegistry so GET /v1/sessions
    # enumerates it. Without this, only sessions registered explicitly via
    # POST /v1/sessions/register (today: just "main", seeded at startup) show up
    # in the dashboard sidebar — Claude Code channel clients that bind their own
    # session UUID per PR #103 are invisible to /v1/sessions and the dashboard
    # reports "No active sessions". Idempotent: SessionRegistry.register preserves
    # the existing voice allocation on re-register.
    try:
        from session_registry import get_default_registry as _get_session_registry

        _participant_id = f"channel-client::{client_type}"
        _get_session_registry().register(
            session_id=session_id,
            participant_id=_participant_id,
            participant_type="agent",
            preferred_voice=None,
            preferred_output_device="system-default",
        )
    except Exception as e:  # noqa: BLE001 — never fail seat registration on mirror
        logger.warning("session-registry mirror failed (non-fatal): %s", e)

    # Fix A — session-registration-unification (SYNC half):
    # After the local seat/session registers, fire a best-effort callback to
    # the kernel so the kernel's ChannelSessionRegistry stays authoritative
    # (fixes the kernel:1 / mod3:99 skew reported in
    # cog://mem/semantic/insights/cross-session-tapestry-2026-05-26).
    #
    # Payload shape matches POST /v1/channel-sessions/register.  We pass the
    # EXISTING session_id — the kernel mints only when session_id is absent,
    # so this is idempotent and never re-mints.
    #
    # iss / sub: the mod3 Seat object uses the user_iss/user_sub split
    # (Wave 6c / Primitive 2). We forward user_iss as "iss" and user_sub as
    # "sub" per the ChannelSessionRecord field names on the kernel side.
    # The feat/mod3-session-identity-claims merge fully wires these; on the
    # current main branch user_iss / user_sub are available from this
    # handler's local vars (see binding at lines ~1446-1450 above).
    #
    # DEPENDENCY NOTE: iss/sub population is conditional on the identity
    # claims being present on the seat (Wave 6b+). Pre-Wave-6b callers that
    # don't supply iss/sub register with empty strings; the kernel record is
    # still valid and the callback is still idempotent.
    try:
        import httpx as _httpx

        _kernel_payload: dict = {
            "session_id": session_id,  # pass EXISTING id — never re-mint
            "participant_id": f"channel-client::{client_type}",
            "participant_type": "agent",
            "kind": "channel-client",
            # Forward identity claims — user_iss/user_sub are bound above from
            # the seat request body (see Wave 6c handling ~line 1446).
            # Pass None-as-absent: omitempty on the Go side ignores empty strings.
            "iss": user_iss or None,
            "sub": user_sub or None,
        }
        # Drop None values so the JSON payload stays clean.
        _kernel_payload = {k: v for k, v in _kernel_payload.items() if v is not None}
        _httpx.post(
            f"{_cogos_kernel_url}/v1/channel-sessions/register",
            json=_kernel_payload,
            timeout=1.5,
        )
        logger.debug(
            "kernel session-register callback OK: session=%s participant=%s",
            session_id,
            _kernel_payload["participant_id"],
        )
    except Exception as _kce:  # noqa: BLE001 — never block seat registration on kernel callback
        logger.warning("kernel session-register callback failed (non-fatal): %s", _kce)

    # Emit presence.started when any identity claim is present.
    #
    # Shape (Wave 6c / Primitive 2):
    #   { type, user_iss, user_sub, assistant_iss, assistant_sub,
    #     session_id, seat_id, harness_session_id }
    #
    # harness_session_id: the Claude Code session ID carried as device_uuid by
    # convention (the channel client sets it to its own Claude Code session ID).
    #
    # Fan-out to ALL sessions so the dashboard and any listening reconciler
    # can see the event. Kernel reconciler integration is Wave 6c / ChannelProvider.
    #
    # Trust model (v1): claims are accepted at face value. The launch context
    # (stdio child of kernel process inheriting node identity) is the implicit
    # attestation. Crypto enforcement deferred to v2.
    if user_iss or user_sub or assistant_iss or assistant_sub:
        presence_event: dict = {
            "type": "presence.started",
            "user_iss": user_iss,
            "user_sub": user_sub,
            "assistant_iss": assistant_iss,
            "assistant_sub": assistant_sub,
            "session_id": session_id,
            "seat_id": seat.seat_id,
            "harness_session_id": device_uuid or None,
        }
        registry.fan_out_all(presence_event)
        logger.info(
            "presence.started emitted: user=%r agent=%r seat=%s session=%s",
            user_sub,
            assistant_sub,
            seat.seat_id,
            session_id,
        )

    identity_resp: dict = {}
    if user_iss or user_sub:
        identity_resp["user_iss"] = user_iss
        identity_resp["user_sub"] = user_sub
    if assistant_iss or assistant_sub:
        identity_resp["assistant_iss"] = assistant_iss
        identity_resp["assistant_sub"] = assistant_sub

    return {
        "seat_id": seat.seat_id,
        "session_id": seat.session_id,
        "client_type": seat.client_type,
        "channel_mode": seat.channel_mode,
        **identity_resp,
    }


@app.delete("/v1/sessions/{session_id}/seats/{seat_id}")
def seat_revoke(session_id: str, seat_id: str):
    """Revoke a seat — closes its SSE stream and removes it from the registry."""
    from seats import get_seat_registry

    registry = get_seat_registry()
    removed = registry.revoke(session_id, seat_id)
    if not removed:
        return JSONResponse(
            status_code=404,
            content={"error": f"seat '{seat_id}' not found in session '{session_id}'"},
        )

    # Fix A — session-registration-unification (SYNC half, deregister cascade):
    # After the local seat is revoked, notify the kernel so it can drop its
    # ChannelSessionRecord. Best-effort / non-fatal — the seat is already gone
    # locally; a kernel-side orphan will be cleaned up by Fix B's reaper.
    try:
        import httpx as _httpx

        _httpx.post(
            f"{_cogos_kernel_url}/v1/channel-sessions/{session_id}/deregister",
            timeout=1.5,
        )
        logger.debug(
            "kernel session-deregister callback OK: session=%s seat=%s",
            session_id,
            seat_id,
        )
    except Exception as _kde:  # noqa: BLE001 — never block seat revoke on kernel callback
        logger.warning("kernel session-deregister callback failed (non-fatal): %s", _kde)

    return {"status": "revoked", "seat_id": seat_id, "session_id": session_id}


@app.get("/v1/sessions/{session_id}/seats/{seat_id}/events")
async def seat_events(session_id: str, seat_id: str):
    """SSE stream of events for a channel-client seat.

    Streams ``text/event-stream`` until the seat is revoked or the client
    disconnects.  Events follow the shape:

      event: user_message
      data: {"type":"user_message","content":"<text>","input_type":"text"}

      event: pairing_request
      data: {"type":"pairing_request","identifier":"<uuid>","code":"abcde"}
    """
    from fastapi.responses import StreamingResponse

    from seats import get_seat_registry, sse_stream

    registry = get_seat_registry()
    seat = registry.get(session_id, seat_id)
    if seat is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"seat '{seat_id}' not found in session '{session_id}'"},
        )

    async def _generate():
        async for chunk in sse_stream(seat):
            yield chunk

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/sessions/{session_id}/seats")
def seat_list(session_id: str):
    """List all seats currently attached to *session_id*."""
    from seats import get_seat_registry

    registry = get_seat_registry()
    return {
        "session_id": session_id,
        "seats": registry.list_session_seats(session_id),
    }


@app.post("/v1/sessions/{session_id}/messages")
async def session_message(session_id: str, request: Request):
    """Fan a dashboard text message to all seats in *session_id*.

    Body (JSON):
      {
        "content": "<text>",
        "input_type": "text" | "voice",
        "role": "user",
        "seat_id": "<originating-seat-id>"   (optional — skipped in fan-out to prevent echo)
      }

    Returns the number of seats that received the event.
    """
    from seats import get_seat_registry

    try:
        body = await request.json()
    except Exception:
        body = {}

    content = body.get("content", "")
    input_type = body.get("input_type", "text")
    role = body.get("role", "user")
    originating_seat = body.get("seat_id") or None

    msg_id = str(uuid.uuid4())[:8]
    try:
        get_chat_flow_log().emit(
            CHAT_MESSAGE_RECEIVED,
            session_id,
            msg_id,
            "http",
            [],
            content,
            "inbound",
        )
    except Exception:  # noqa: BLE001
        pass

    registry = get_seat_registry()
    count = registry.fan_out(
        session_id,
        {
            "type": "user_message",
            "content": content,
            "input_type": input_type,
            "role": role,
        },
        exclude_seat=originating_seat,
    )

    # Mark the session active so the lifecycle reaper does not prune a quiet
    # but still-coordinating session on the next sweep.
    try:
        from session_registry import get_default_registry as _get_session_registry

        _get_session_registry().touch(session_id)
    except Exception:  # noqa: BLE001 — touch is best-effort
        pass

    try:
        from message_store import get_default_store as _get_msg_store

        _get_msg_store().append(
            session_id=session_id,
            role=role if role in ("user", "assistant") else "user",
            content=content,
            input_type=input_type,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("message-store append (session_message) failed: %s", e)

    try:
        seat_ids = [s["seat_id"] for s in registry.list_session_seats(session_id)]
        get_chat_flow_log().emit(
            CHAT_FAN_OUT,
            session_id,
            msg_id,
            "http",
            seat_ids,
            content,
            "inbound",
        )
    except Exception:  # noqa: BLE001
        pass

    return {"status": "ok", "session_id": session_id, "seats_notified": count}


@app.get("/v1/sessions/{session_id}/messages")
async def session_messages_history(session_id: str, limit: int = 100):
    """Recent chat history for *session_id*.

    Used by the dashboard to hydrate the chat pane on refresh and when the
    operator clicks a session row in the sidebar. RAM-only ring buffer — lost
    on mod3 restart. ``limit`` caps the response size; the buffer itself is
    bounded per session (see ``message_store.DEFAULT_MAX_PER_SESSION``).
    """
    from message_store import get_default_store as _get_msg_store

    if limit <= 0 or limit > 1000:
        limit = 100
    messages = _get_msg_store().get(session_id, limit=limit)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@app.post("/v1/sessions/broadcast-message")
async def broadcast_message(request: Request):
    """Fan a dashboard text message to ALL channel-client seats across all sessions.

    This is the dashboard→MCP-harness receive path.  When the operator types
    in the dashboard chat input, the message needs to reach every attached
    Claude Code channel-client seat so each seat can forward it as a
    ``notifications/claude/channel`` MCP notification to its host process.

    Body (JSON):
      {
        "content": "<text>",
        "input_type": "text" | "voice",
        "role": "user"
      }

    Optional @-mention addressing:
      {
        "content": "@<session_id> hello",
        "target_session_id": "<session_id>"   -- explicit target, skips fan_out_all
      }

    Returns:
      {"status": "ok", "seats_notified": <int>, "target": "all" | "<session_id>"}
    """
    from seats import get_seat_registry

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    content = body.get("content", "")
    input_type = body.get("input_type", "text")
    role = body.get("role", "user")
    target_session_id = body.get("target_session_id") or None

    if not content:
        return JSONResponse(status_code=400, content={"error": "content is required"})

    msg_id = str(uuid.uuid4())[:8]
    registry = get_seat_registry()

    event = {
        "type": "user_message",
        "content": content,
        "input_type": input_type,
        "role": role,
    }

    if target_session_id:
        # Addressed message — fan only to the target session's seats.
        count = registry.fan_out(target_session_id, event)
        target_label = target_session_id
        try:
            get_chat_flow_log().emit(
                CHAT_MESSAGE_RECEIVED,
                target_session_id,
                msg_id,
                "http-broadcast",
                [],
                content,
                "inbound",
            )
        except Exception:  # noqa: BLE001
            pass
    else:
        # Broadcast to all sessions.
        count = registry.fan_out_all(event)
        target_label = "all"
        try:
            get_chat_flow_log().emit(
                CHAT_MESSAGE_RECEIVED,
                "broadcast",
                msg_id,
                "http-broadcast",
                [],
                content,
                "inbound",
            )
        except Exception:  # noqa: BLE001
            pass

    # Persist into chat history under the resolved session id so the dashboard
    # can hydrate that session's pane on refresh. For broadcast (no target),
    # we attribute to "main" — that's where the dashboard send button routes
    # when no @-mention or sidebar selection is active.
    try:
        from message_store import get_default_store as _get_msg_store

        _get_msg_store().append(
            session_id=target_session_id or "main",
            role=role if role in ("user", "assistant") else "user",
            content=content,
            input_type=input_type,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("message-store append (broadcast) failed: %s", e)

    logger.debug(
        "broadcast-message: target=%s seats_notified=%d content=%r",
        target_label,
        count,
        content[:80],
    )
    return {"status": "ok", "seats_notified": count, "target": target_label}


@app.post("/v1/dashboard-chat")
async def dashboard_chat_post(request: Request):
    """REST alternative to the WebSocket dashboard-chat path.

    Used by channel_client.py's mod3_dashboard_post tool.  Broadcasts
    the message to all connected WebSocket dashboard-chat subscribers AND
    fans it out to any channel seats in the same session.

    Body (JSON):
      {
        "text": "<message>",
        "role": "assistant" | "user",
        "session_id": "<id>",
        "seat_id": "<id>"
      }
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid JSON"})

    text = body.get("text", "")
    role = body.get("role", "assistant")
    session_id = body.get("session_id")
    # seat_id identifies the channel-client that sent this message; exclude it
    # from the fan-out so the originator does not receive its own broadcast back
    # and trigger an echo loop.
    originating_seat = body.get("seat_id") or None

    # Fan to WebSocket dashboard-chat subscribers (existing server.py mechanism)
    try:
        from server import _dashboard_chat_broadcast

        _dashboard_chat_broadcast({"type": "chat", "role": role, "text": text, "session_id": session_id})
    except (ImportError, AttributeError):
        logger.debug("_dashboard_chat_broadcast not available (server.py not loaded or renamed)")

    # Also fan to any seat SSE streams in the session, skipping the sender.
    if session_id:
        from seats import get_seat_registry

        registry = get_seat_registry()
        registry.fan_out(
            session_id,
            {
                "type": "assistant_message",
                "content": text,
                "role": role,
                "session_id": session_id,
            },
            exclude_seat=originating_seat,
        )

    # Persist into the per-session history so the dashboard can render this
    # assistant turn on refresh / sidebar-switch. No session_id means we drop
    # the message from history (still broadcast for live dashboards) — this
    # path is reached when channel_client could not resolve a session id.
    if session_id:
        try:
            from message_store import get_default_store as _get_msg_store

            _get_msg_store().append(
                session_id=session_id,
                role=role if role in ("user", "assistant") else "assistant",
                content=text,
                input_type="text",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("message-store append (dashboard-chat) failed: %s", e)

    try:
        get_chat_flow_log().emit(
            CHAT_MESSAGE_SENT,
            session_id or "",
            str(uuid.uuid4())[:8],
            "http",
            [],
            text,
            "outbound",
        )
    except Exception:  # noqa: BLE001
        pass

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Chat-flow log endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/logs/chat-flow")
def chat_flow_log_query(
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100,
):
    """Return recent structured chat-flow events from the in-memory ring buffer.

    Query parameters:
      session_id  — filter to a single session (optional)
      event_type  — comma-separated event type filter, e.g. chat.message_received,chat.fan_out
      since       — ISO 8601 timestamp; only events at or after this time are returned
      limit       — max events to return (default 100, max 1000)

    Returns:
      {"events": [...], "count": N}

    Verification:
      curl 'http://localhost:7860/v1/logs/chat-flow?limit=20'
    """
    limit = max(1, min(limit, 1000))
    events = get_chat_flow_log().query(
        session_id=session_id,
        event_type=event_type,
        since=since,
        limit=limit,
    )
    return {"events": events, "count": len(events)}


@app.get("/v1/logs/chat-flow/stream")
async def chat_flow_log_stream(
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100,
):
    """SSE live-tail of structured chat-flow events.

    Opens a ``text/event-stream`` that pushes new events as they arrive.
    Supports the same filter params as GET /v1/logs/chat-flow.
    Each event is emitted as::

      event: chat_flow
      data: {"ts":"...","event_type":"chat.message_received",...}

    Keep-alive comments are sent every 15 s.

    Close by disconnecting (the server cleans up the subscription on disconnect).
    """
    from fastapi.responses import StreamingResponse

    event_types: set[str] | None = None
    if event_type:
        event_types = {t.strip() for t in event_type.split(",") if t.strip()}

    limit = max(1, min(limit, 1000))
    log = get_chat_flow_log()

    async def _generate():
        q = log.subscribe()
        try:
            KEEPALIVE = 15.0
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=KEEPALIVE)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                # Apply filters
                if session_id and event.get("session_id") != session_id:
                    continue
                if event_types and event.get("event_type") not in event_types:
                    continue
                data = json.dumps(event, separators=(",", ":"))
                yield f"event: chat_flow\ndata: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            log.unsubscribe(q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    """Health check — standardized CogOS service format."""
    try:
        loaded = get_loaded_engines()

        # Engine status: loaded/unloaded for each registered engine
        engines = {}
        for engine_name in MODELS:
            engines[engine_name] = "loaded" if engine_name in loaded else "unloaded"

        # Modality availability
        modalities = {
            "tts": len(loaded) > 0,
            "stt": False,  # STT not yet implemented as a server modality
            "vad": vad_loaded(),
        }

        # Queue state from job ledger
        with _jobs_lock:
            total = len(_jobs)
            active = sum(1 for j in _jobs.values() if j.get("status") in ("generating", "processing"))

        # Overall status: ok if at least one TTS engine loaded, degraded if none
        status = "ok" if loaded else "degraded"

        return {
            "status": status,
            "service": "mod3",
            "version": __version__,
            "uptime_sec": round(time.time() - _server_start_time, 1),
            "engines": engines,
            "modalities": modalities,
            "queue": {
                "depth": total,
                "active_jobs": active,
            },
            "routing": "channel-client",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "service": "mod3",
                "version": __version__,
                "error": str(e),
            },
        )


@app.post("/shutdown")
async def shutdown(req: Optional[ShutdownRequest] = None):
    """Initiate graceful server shutdown.

    Called by the CogOS kernel for lifecycle management. Returns immediately
    with confirmation, then drains active jobs and exits.

    Body (optional): {"timeout_sec": 5, "reason": "kernel-restart"}
    """
    global _shutting_down

    if _shutting_down:
        return JSONResponse(
            status_code=409,
            content={"status": "already_shutting_down"},
        )

    if req is None:
        req = ShutdownRequest()

    timeout_sec = req.timeout_sec
    reason = req.reason

    _shutting_down = True
    logger.info("Shutdown requested: reason=%s timeout=%.1fs", reason, timeout_sec)

    async def _graceful_exit():
        """Wait for active jobs to drain, then signal the process to stop."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with _jobs_lock:
                active = sum(1 for j in _jobs.values() if j.get("status") in ("generating", "processing"))
            if active == 0:
                break
            await asyncio.sleep(0.25)

        with _jobs_lock:
            remaining = sum(1 for j in _jobs.values() if j.get("status") in ("generating", "processing"))

        if remaining:
            logger.warning("Shutdown timeout reached with %d active jobs — forcing exit", remaining)
        else:
            logger.info("All jobs drained — exiting cleanly")

        # Send SIGINT to our own process, which uvicorn handles gracefully
        os.kill(os.getpid(), signal.SIGINT)

    # Fire-and-forget: schedule the shutdown coroutine on the running loop
    asyncio.ensure_future(_graceful_exit())

    return {
        "status": "shutting_down",
        "reason": reason,
        "timeout_sec": timeout_sec,
    }


@app.get("/capabilities")
def capabilities():
    """Machine-readable capability manifest for service discovery."""
    voices = {name: cfg["voices"] for name, cfg in MODELS.items()}
    return {
        "service": "mod3",
        "version": __version__,
        "description": "Model Modality Modulator — local TTS, STT, and VAD on Apple Silicon",
        "modalities": ["voice"],
        "capabilities": {
            "tts": {
                "engines": list(MODELS.keys()),
                "default_voice": "bm_lewis",
                "default_speed": 1.25,
                "endpoint": "/v1/synthesize",
            },
            "stt": {
                "engine": "mlx_whisper",
                "model": "mlx-community/whisper-large-v3-turbo",
                "languages": ["en"],
                "endpoint": None,
            },
            "vad": {
                "engine": "silero_v5",
                "endpoint": "/v1/vad",
            },
        },
        "voices": voices,
        "endpoints": {
            "synthesize": "POST /v1/synthesize",
            "speech": "POST /v1/audio/speech",
            "vad": "POST /v1/vad",
            "voices": "GET /v1/voices",
            "health": "GET /health",
            "shutdown": "POST /shutdown",
            "capabilities": "GET /capabilities",
        },
        "protocols": {
            "mcp": True,
            "http": True,
            "websocket": True,
        },
    }


@app.get("/diagnostics")
def diagnostics():
    """Diagnostics snapshot with bus state."""
    with _jobs_lock:
        total = len(_jobs)
        active = sum(1 for j in _jobs.values() if j.get("status") in ("generating", "processing"))
    return {
        "engines_loaded": get_loaded_engines(),
        "vad_loaded": vad_loaded(),
        "jobs": {
            "total": total,
            "active": active,
        },
        "bus": {
            "health": _bus.health(),
            "hud": _bus.hud(),
        },
    }


# ---------------------------------------------------------------------------
# Modality Bus endpoints
# ---------------------------------------------------------------------------


@app.get("/v1/bus/hud")
def bus_hud():
    """Agent HUD — live state of all modalities, channels, and queues."""
    return _bus.hud()


@app.get("/v1/bus/health")
def bus_health():
    """Full modality bus health report."""
    return _bus.health()


@app.post("/v1/bus/perceive")
async def bus_perceive(file: UploadFile, modality: str = "voice", channel: str = ""):
    """Run raw input through the modality bus: gate → decode → cognitive event."""
    raw = await file.read()
    event = _bus.perceive(raw, modality=modality, channel=channel)
    if event is None:
        return {"status": "filtered", "modality": modality, "channel": channel}
    return {
        "status": "ok",
        "event": {
            "modality": event.modality.value,
            "content": event.content,
            "confidence": event.confidence,
            "source_channel": event.source_channel,
            "timestamp": event.timestamp,
            "metadata": event.metadata,
        },
    }


@app.post("/v1/bus/act")
def bus_act(req: BusActRequest):
    """Route a cognitive intent through the bus: resolve modality → encode → queue.

    Body: {"content": "hello world", "modality": "voice", "channel": "discord-voice",
           "voice": "bm_lewis", "speed": 1.25}
    """
    from modality import CognitiveIntent, ModalityType

    content = req.content
    modality = req.modality
    channel = req.channel
    metadata = {}
    for k in ("voice", "speed", "emotion"):
        v = getattr(req, k, None)
        if v is not None:
            metadata[k] = v

    intent = CognitiveIntent(
        modality=ModalityType(modality) if modality else None,
        content=content,
        target_channel=channel,
        metadata=metadata,
    )

    output = _bus.act(intent, channel=channel, blocking=True)
    assert isinstance(output, EncodedOutput), "Expected blocking act() to return EncodedOutput"

    return {
        "status": "ok",
        "modality": output.modality.value,
        "format": output.format,
        "duration_sec": output.duration_sec,
        "bytes": len(output.data),
        "metadata": output.metadata,
    }


def get_bus() -> ModalityBus:
    """Get the global bus instance (for server.py integration)."""
    return _bus


# ---------------------------------------------------------------------------
# Dashboard — voice/text chat via WebSocket
# ---------------------------------------------------------------------------

_logger = logging.getLogger("mod3.dashboard")

_dashboard_dir = Path(__file__).parent / "dashboard"


@app.get("/dashboard")
@app.get("/dashboard/")
async def dashboard_page():
    """Serve the dashboard UI (handles both /dashboard and /dashboard/)."""
    index = _dashboard_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "dashboard not found"}, status_code=404)


@app.get("/dashboard/{filename:path}")
async def dashboard_static(filename: str):
    """Serve any file inside the dashboard/ directory.

    This covers the Sessions browser (sessions.html), Voice Lab
    (voice-lab.html), Console (console.html), and any supporting JS files
    that live alongside the main index.html.
    """
    # Prevent path traversal.
    safe = Path(filename).parts
    if ".." in safe:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    target = _dashboard_dir / filename
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    return JSONResponse({"error": f"{filename} not found"}, status_code=404)


# ---------------------------------------------------------------------------
# Claude Code ACP-client proxy
# ---------------------------------------------------------------------------
# The sessions.html dashboard page calls /v1/claude-code/spawn on the mod3
# origin (localhost:7860) so it avoids browser CORS restrictions when the
# kernel is on a different port. This endpoint proxies the request to the
# CogOS kernel and returns the kernel's response verbatim.

_cogos_kernel_url = os.environ.get("COGOS_KERNEL_URL", "http://localhost:6931")


@app.get("/v1/providers/available")
async def get_providers_available():
    """Return the list of inference providers available on the CogOS kernel.

    Proxies GET /v1/providers from the kernel and returns the same JSON.
    Falls back to a static default list if the kernel is unreachable, so the
    dashboard's backend selector always has options to show.

    Response shape::

        {
          "providers": [
            {"name": "lmstudio-eclipse", "type": "lmstudio-eclipse", "available": true},
            ...
          ]
        }
    """
    import httpx

    target = f"{_cogos_kernel_url}/v1/providers"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(target)
        if resp.status_code == 200:
            return Response(
                content=resp.content,
                status_code=200,
                media_type="application/json",
            )
    except Exception:
        pass

    # Fallback: static list derived from known provider names so the UI is
    # never empty when the kernel is unreachable.
    return JSONResponse(
        content={
            "providers": [
                {"name": "lmstudio-eclipse", "type": "lmstudio-eclipse", "available": True},
                {"name": "ollama", "type": "ollama", "available": True},
                {"name": "lmstudio-darkstar", "type": "lmstudio-darkstar", "available": False},
                {"name": "claude-code", "type": "claude-code", "available": True},
                {"name": "codex", "type": "codex", "available": False},
                {"name": "mlx-lm", "type": "mlx-lm", "available": False},
            ],
            "source": "fallback",
        }
    )


@app.post("/v1/claude-code/spawn")
async def proxy_claude_code_spawn(request: Request):
    """Proxy POST /v1/claude-code/spawn to the CogOS kernel.

    Accepts the same body as the kernel endpoint:
      { project?, session_id?, dangerously_load_development_channels? }

    Returns the kernel's response (201 on success, 4xx/5xx on error).
    This proxy exists so the dashboard UI can call a same-origin endpoint
    rather than cross-origin to localhost:6931.
    """
    import httpx

    try:
        body = await request.body()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "cannot read request body"})

    target = f"{_cogos_kernel_url}/v1/claude-code/spawn"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                target,
                content=body,
                headers={"Content-Type": "application/json"},
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )
    except httpx.ConnectError:
        logger.warning("claude-code spawn proxy: kernel unreachable at %s", target)
        return JSONResponse(
            status_code=503,
            content={
                "error": {"type": "kernel_unavailable", "message": f"CogOS kernel unreachable at {_cogos_kernel_url}"}
            },
        )
    except Exception as exc:
        logger.exception("claude-code spawn proxy: unexpected error")
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "proxy_error", "message": str(exc)}},
        )


@app.websocket("/ws/audio/{session_id}")
async def ws_audio(websocket: WebSocket, session_id: str):
    """Wave 4.3 — per-session playback channel for the dashboard.

    The dashboard (or any client) opens ``ws://host:7860/ws/audio/<sid>``
    to receive audio frames that would otherwise play through afplay /
    sounddevice. The wire contract per send from the server:

      1. JSON text frame: ``{"type": "audio_header", "session_id": ...,
         "job_id": ..., "duration_sec": ..., "sample_rate": ..., "bytes": N,
         "format": "wav", "seq": N}``
      2. Binary frame: the raw WAV bytes.

    The browser decodes via ``AudioContext.decodeAudioData`` — browsers
    accept a whole-WAV in one blob so we don't need chunking for the v1
    implementation. A future iteration can stream PCM frames for lower
    latency; the header envelope is the forward-compatibility seam.

    On disconnect the subscriber is removed and the session falls back to
    ``afplay`` (kernel) / ``sd.play`` (MCP shim) automatically — the
    subscribers registry tracks liveness, so the very next
    ``/v1/sessions/<sid>/subscribers`` probe returns ``subscribed: false``.

    Client → server frames are ignored for v1 (binary-only path). RTVI 1.3.0
    clients send a JSON ``client-ready`` handshake on connect; the server
    replies with ``bot-ready`` before entering the drain loop. Binary-only
    clients that skip the handshake are tolerated silently.

    RTVI 1.3.0 wire shapes (B+ workstream, see docs/rtvi/G2-decision-record.md):
      Inbound:  {"label":"rtvi-ai","type":"client-ready","id":"<uuid>",
                 "data":{"version":"1.3.0","about":{...}}}
      Outbound: {"label":"rtvi-ai","type":"bot-ready","id":"<client-ready-id>",
                 "data":{"version":"1.3.0","about":{"server":"mod3","version":"0.5.0"}}}

    After handshake, clients may send JSON-framed inbound audio (``raw-audio``)
    and VAD signals (``user-started-speaking`` / ``user-stopped-speaking``).
    These are routed to the existing VAD/STT pipeline via ``_bus.perceive()``.
    Clients may also send ``disconnect-bot`` to request a graceful close.

    Note: a simultaneously active mic path (InboundPipeline) will process audio
    in parallel. WS-source audio and mic audio do not conflict at the bus level —
    both enter ``perceive()`` independently. Use the ``MOD3_DISABLE_MIC`` env
    var to suppress the mic path when a WS client is the sole source.

    RTVI 1.3.0 inbound types handled in the drain loop (B+ workstream):
      raw-audio:        T3 — base64 int16 PCM 16kHz routed to VAD/STT pipeline
      raw-audio-batch:  T3 — batch variant of raw-audio
      user-started-speaking: T3 — client-side VAD onset signal
      user-stopped-speaking: T3 — client-side VAD offset signal
      disconnect-bot:   T5 — client-initiated graceful close
    """
    _RTVI_PROTOCOL_VERSION = "1.3.0"
    _RTVI_HANDSHAKE_TIMEOUT = 5.0  # seconds to wait for client-ready before giving up

    await websocket.accept()
    subs = get_default_audio_subscribers()
    loop = asyncio.get_running_loop()
    subscriber = subs.register(session_id, websocket, loop)
    try:
        # --- RTVI 1.3.0 handshake (T2) ----------------------------------------
        # Attempt to receive client-ready. On timeout or non-JSON, continue without
        # handshake (legacy binary-only clients are tolerated).
        try:
            first_msg = await asyncio.wait_for(
                websocket.receive(),
                timeout=_RTVI_HANDSHAKE_TIMEOUT,
            )
            text = first_msg.get("text") if first_msg.get("type") != "websocket.disconnect" else None
            if text:
                try:
                    parsed = json.loads(text)
                    msg_type = parsed.get("type")
                    msg_id = parsed.get("id", str(uuid.uuid4()))
                    if msg_type == "client-ready":
                        data = parsed.get("data") or {}
                        version_str = data.get("version", "")
                        major = int(version_str.split(".")[0]) if version_str else 0
                        if major != 1:
                            # Major version mismatch — reject with RTVI error frame.
                            error_frame = json.dumps(
                                {
                                    "label": "rtvi-ai",
                                    "type": "error",
                                    "id": msg_id,
                                    "data": {
                                        "error": (
                                            f"RTVI major version mismatch: "
                                            f"client={version_str!r} server={_RTVI_PROTOCOL_VERSION!r}"
                                        ),
                                        "fatal": True,
                                    },
                                }
                            )
                            await websocket.send_text(error_frame)
                            await websocket.close()
                            return
                        # Version OK — send bot-ready.
                        bot_ready = json.dumps(
                            {
                                "label": "rtvi-ai",
                                "type": "bot-ready",
                                "id": msg_id,
                                "data": {
                                    "version": _RTVI_PROTOCOL_VERSION,
                                    "about": {"server": "mod3", "version": "0.5.0"},
                                },
                            }
                        )
                        await websocket.send_text(bot_ready)
                        logger.debug(
                            "/ws/audio/%s: RTVI handshake complete (client version=%s)",
                            session_id,
                            version_str,
                        )
                    elif msg_type == "websocket.disconnect":
                        # Client disconnected before handshake — fall through to finally.
                        return
                    # Any other RTVI type before handshake is silently ignored;
                    # we proceed to the drain loop.
                except (json.JSONDecodeError, ValueError, AttributeError, IndexError):
                    pass  # non-JSON or malformed: tolerate as legacy binary-only client
            elif first_msg.get("type") == "websocket.disconnect":
                return  # immediate disconnect
        except asyncio.TimeoutError:
            pass  # no handshake frame received — tolerate as legacy client
        # --- drain loop -------------------------------------------------------
        # Keep the connection open; drain client frames so the socket close
        # handshake fires promptly. T5 adds disconnect-bot handling here.
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            # RTVI 1.3.0: dispatch inbound JSON frames (T3 raw-audio/VAD + T5 disconnect-bot).
            text = msg.get("text")
            if text:
                try:
                    parsed = json.loads(text)
                    rtvi_type = parsed.get("type")
                    if rtvi_type == "disconnect-bot":
                        logger.debug("/ws/audio/%s: disconnect-bot received, closing", session_id)
                        break
                    elif rtvi_type in ("raw-audio", "raw-audio-batch"):
                        # Decode base64 int16 PCM and route to VAD/STT pipeline.
                        data = parsed.get("data") or {}
                        audio_b64 = data.get("audio", "")
                        if audio_b64:
                            import base64 as _base64  # noqa: I001,PLC0415
                            import numpy as _np  # noqa: PLC0415

                            pcm_bytes = _base64.b64decode(audio_b64)
                            # Convert int16 LE bytes → float32 for ModalityBus.perceive()
                            pcm_int16 = _np.frombuffer(pcm_bytes, dtype=_np.int16)
                            pcm_float32 = pcm_int16.astype(_np.float32) / 32768.0
                            raw_bytes = pcm_float32.tobytes()
                            try:
                                _ensure_bus_modules()
                                _bus.perceive(
                                    raw_bytes,
                                    modality="voice",
                                    channel=f"rtvi-ws:{session_id}",
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.debug(
                                    "/ws/audio/%s raw-audio perceive error: %s",
                                    session_id,
                                    exc,
                                )
                    elif rtvi_type == "user-started-speaking":
                        # Client-side VAD onset — log for now; future: bypass Silero VAD
                        logger.debug("/ws/audio/%s: user-started-speaking (client VAD)", session_id)
                    elif rtvi_type == "user-stopped-speaking":
                        # Client-side utterance boundary — log for now; future: trigger endpointing
                        logger.debug("/ws/audio/%s: user-stopped-speaking (client VAD)", session_id)
                except (json.JSONDecodeError, AttributeError, ValueError):
                    pass  # ignore non-JSON or malformed frames
    except Exception as exc:  # noqa: BLE001 — disconnect is the normal exit
        logger.debug("/ws/audio/%s disconnect: %s", session_id, exc)
    finally:
        subs.unregister(session_id, subscriber)


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """Dashboard voice/text chat — one session per connection."""
    await websocket.accept()

    loop = asyncio.get_running_loop()

    from agent_loop import AgentLoop
    from channels import BrowserChannel
    from pipeline_state import PipelineState
    from providers import auto_detect_provider

    provider = auto_detect_provider()
    ps = PipelineState()

    agent = AgentLoop(
        bus=_bus,
        provider=provider,
        pipeline_state=ps,
    )

    channel = BrowserChannel(
        ws=websocket,
        bus=_bus,
        pipeline_state=ps,
        loop=loop,
        on_event=agent.handle_event,
    )

    agent.channel_id = channel.channel_id
    agent._channel_ref = channel

    _logger.info("Dashboard session started: %s (provider: %s)", channel.channel_id, provider.name)

    try:
        await channel.run()
    finally:
        _logger.info("Dashboard session ended: %s", channel.channel_id)


# ---------------------------------------------------------------------------
# ACP (Agent Client Protocol) WebSocket endpoint
# ---------------------------------------------------------------------------

# ACP JSON-RPC error codes
_ACP_METHOD_NOT_FOUND = -32601
_ACP_INVALID_PARAMS = -32602
_ACP_INTERNAL_ERROR = -32603
_ACP_PROTOCOL_VERSION = 1


@app.websocket("/ws/acp")
async def ws_acp(websocket: WebSocket):
    """ACP (Agent Client Protocol) endpoint — JSON-RPC 2.0 over WebSocket.

    Implements a minimal ACP server that routes prompts through mod3's
    existing AgentLoop. The wire format matches Zed's Agent Client Protocol
    spec (https://github.com/zed-industries/agent-client-protocol).

    Supported methods:
      initialize      -- capability negotiation
      session/new     -- create a session
      session/prompt  -- submit a user prompt (streams via session/update notifications)
      session/cancel  -- cancel in-flight prompt (notification, no response)
      session/list    -- list registered TTS sessions (optional)
      session/load    -- retrieve state of a specific session (optional)
      session/resume  -- bind this ACP connection to a named session (optional)
      authenticate    -- auth handshake; no-op when authMethods is empty

    Prompts are fanned to attached channel-client seats via the seat registry.
    Responses flow through the channel client (speak / mod3_dashboard_post),
    not back through this WebSocket.

    /ws/chat is not deprecated; both endpoints are live in parallel.
    """
    import json
    from uuid import uuid4

    from schemas.acp import (
        AgentCapabilities,
        AuthenticateResult,
        InitializeResult,
        JsonRpcResponse,
        PromptCapabilities,
        SessionCapabilities,
        SessionListItem,
        SessionListResult,
        SessionLoadResult,
        SessionNewResult,
        SessionPromptResult,
        SessionResumeResult,
    )

    await websocket.accept()
    _logger.info("ACP session opened")

    # Per-connection state
    _sessions: dict[str, dict] = {}  # sessionId -> {"cancel": asyncio.Event}
    _initialized = False

    async def _send(obj: dict) -> None:
        try:
            await websocket.send_text(json.dumps(obj, separators=(",", ":")))
        except Exception:  # noqa: BLE001
            pass

    async def _send_response(request_id: int | str, result: object) -> None:
        resp = JsonRpcResponse.ok(
            request_id=request_id,
            result=result if isinstance(result, dict) else result.model_dump(),  # pyright: ignore[reportAttributeAccessIssue]
        )
        await _send(resp.model_dump(exclude_none=True))

    async def _send_error(request_id: int | str | None, code: int, message: str) -> None:
        resp = JsonRpcResponse.err(request_id=request_id, code=code, message=message)
        await _send(resp.model_dump(exclude_none=True))

    async def _send_notification(notif: dict) -> None:
        await _send(notif)

    async def _stream_prompt(session_id: str, text: str, request_id: int | str) -> None:
        """Fan the prompt to all channel-client seats in this session.

        The kernel bus bridge (cogos_agent_bridge) has been removed in favour of
        the channel-client architecture (PR #40). /ws/acp now fans the prompt
        through ``POST /v1/sessions/{session_id}/messages`` so Claude Code
        channel clients receive it as ``notifications/claude/channel``.

        Responses flow back through the channel client's mod3_dashboard_post
        tool or speak tool — not through this WebSocket. The ACP caller
        therefore receives a single resolution frame immediately after fan-out.
        """
        from seats import get_seat_registry

        registry = get_seat_registry()
        seats_count = registry.fan_out(
            session_id,
            {
                "type": "user_message",
                "content": text,
                "input_type": "text",
                "role": "user",
            },
        )
        if seats_count == 0:
            await _send_error(
                request_id,
                -32000,
                "No channel-client seats attached to this session. "
                "Start a session with 'claude --dangerously-load-development-channels server:mod3' "
                "so a channel client is present to handle the prompt.",
            )
            return

        # Resolve immediately — responses flow through the channel client,
        # not back through this WebSocket connection.
        result = SessionPromptResult(stopReason="end_turn")
        await _send_response(request_id, result)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(None, -32700, "Parse error")
                continue

            jsonrpc = msg.get("jsonrpc")
            if jsonrpc != "2.0":
                await _send_error(None, -32600, "Invalid request: jsonrpc must be '2.0'")
                continue

            method = msg.get("method", "")
            msg_id = msg.get("id")  # None for notifications
            params = msg.get("params") or {}

            # Notifications have no id and expect no response.
            is_notification = "id" not in msg

            # ---- initialize ----
            if method == "initialize":
                _initialized = True
                result = InitializeResult(
                    agentCapabilities=AgentCapabilities(
                        promptCapabilities=PromptCapabilities(audio=False, image=False, embeddedContext=False),
                        sessionCapabilities=SessionCapabilities(list=True, resume=True),
                        loadSession=True,
                    )
                )
                await _send_response(msg_id, result)

            # ---- session/new ----
            elif method == "session/new":
                session_id = f"mod3-acp-{uuid4().hex[:12]}"
                _sessions[session_id] = {"cancel": asyncio.Event()}
                result = SessionNewResult(sessionId=session_id)
                await _send_response(msg_id, result)

            # ---- session/prompt ----
            elif method == "session/prompt":
                session_id = params.get("sessionId", "")
                prompt_blocks = params.get("prompt", [])
                # Extract text from content blocks.
                text_parts = [
                    b.get("text", "") for b in prompt_blocks if isinstance(b, dict) and b.get("type") == "text"
                ]
                user_text = " ".join(text_parts).strip()
                if not user_text:
                    await _send_error(msg_id, _ACP_INVALID_PARAMS, "prompt must contain at least one text block")
                    continue
                # Ensure session exists (create on-demand for resilience).
                if session_id not in _sessions:
                    _sessions[session_id] = {"cancel": asyncio.Event()}
                _acp_msg_id = str(uuid4())[:8]
                try:
                    get_chat_flow_log().emit(
                        CHAT_MESSAGE_RECEIVED,
                        session_id,
                        _acp_msg_id,
                        "acp",
                        [],
                        user_text,
                        "inbound",
                    )
                except Exception:  # noqa: BLE001
                    pass
                await _stream_prompt(session_id, user_text, msg_id)
                try:
                    from seats import get_seat_registry as _get_sr

                    _seat_ids = [s["seat_id"] for s in _get_sr().list_session_seats(session_id)]
                    get_chat_flow_log().emit(
                        CHAT_FAN_OUT,
                        session_id,
                        _acp_msg_id,
                        "acp",
                        _seat_ids,
                        user_text,
                        "inbound",
                    )
                except Exception:  # noqa: BLE001
                    pass

            # ---- session/cancel (notification) ----
            elif method == "session/cancel":
                session_id = params.get("sessionId", "")
                session_info = _sessions.get(session_id)
                if session_info:
                    session_info["cancel"].set()
                # No response for notifications.

            # ---- session/list ----
            elif method == "session/list":
                from session_registry import get_default_registry

                registry = get_default_registry()
                items = [
                    SessionListItem(
                        sessionId=s["session_id"],
                        state=s.get("state", "idle"),
                        participantId=s.get("participant_id", ""),
                        participantType=s.get("participant_type", ""),
                    )
                    for s in registry.list_serialized()
                ]
                result = SessionListResult(sessions=items)
                await _send_response(msg_id, result)

            # ---- session/load ----
            elif method == "session/load":
                from session_registry import get_default_registry

                target_id = params.get("sessionId", "")
                if not target_id:
                    await _send_error(msg_id, _ACP_INVALID_PARAMS, "sessionId is required")
                    continue
                registry = get_default_registry()
                session_obj = registry.get(target_id)
                if session_obj is None:
                    await _send_error(
                        msg_id,
                        _ACP_INTERNAL_ERROR,
                        f"session '{target_id}' not found",
                    )
                    continue
                state_dict = session_obj.to_dict()
                result = SessionLoadResult(sessionId=target_id, state=state_dict)
                await _send_response(msg_id, result)

            # ---- session/resume ----
            elif method == "session/resume":
                target_id = params.get("sessionId", "")
                if not target_id:
                    await _send_error(msg_id, _ACP_INVALID_PARAMS, "sessionId is required")
                    continue
                # Bind the ACP connection to the named session so subsequent
                # session/prompt calls fan to seats in that session.
                if target_id not in _sessions:
                    _sessions[target_id] = {"cancel": asyncio.Event()}
                result = SessionResumeResult(sessionId=target_id)
                await _send_response(msg_id, result)

            # ---- authenticate ----
            elif method == "authenticate":
                # authMethods is [] — no auth is required. Return success
                # unconditionally. If authMethods were non-empty, methodId
                # handling would go here.
                result = AuthenticateResult(success=True)
                await _send_response(msg_id, result)

            # ---- unknown ----
            else:
                if not is_notification:
                    await _send_error(msg_id, _ACP_METHOD_NOT_FOUND, f"Method not found: {method}")

    except Exception as exc:  # noqa: BLE001 — disconnect is the normal exit
        _logger.debug("/ws/acp disconnect: %s", exc)
    finally:
        _logger.info("ACP session closed")


# ---------------------------------------------------------------------------
# Dashboard chat WebSocket — symmetric outbound channel (Path B)
# ---------------------------------------------------------------------------


@app.websocket("/ws/dashboard-chat")
async def ws_dashboard_chat(websocket: WebSocket):
    """Symmetric dashboard chat channel — broadcasts mod3_dashboard_post messages.

    Any number of dashboard tabs can connect here. When Claude Code calls the
    mod3_dashboard_post MCP tool, the message is fanned out to every connected
    subscriber as a JSON text frame:

      {"type": "chat", "role": "assistant", "text": "...", "session_id": "..."}

    User-role messages (role="user") are also supported for future bidirectional
    wiring; the chat panel renders both sides with distinct styling.

    Client → server frames are not processed in v0. The connection stays open
    until the client disconnects; no heartbeat is required.
    """
    from server import (
        _dashboard_chat_register,
        _dashboard_chat_unregister,
    )

    await websocket.accept()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _dashboard_chat_register(q, loop)
    _logger.info("dashboard-chat subscriber connected")

    import json as _json

    async def _drain_client():
        """Drain incoming client frames; set disconnect flag on close."""
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    # Signal the send loop to exit by enqueuing a sentinel.
                    await q.put(None)
                    return
        except Exception:  # noqa: BLE001
            await q.put(None)

    drain_task = asyncio.ensure_future(_drain_client())
    try:
        while True:
            message = await q.get()
            if message is None:
                # Sentinel — client disconnected.
                break
            await websocket.send_text(_json.dumps(message, separators=(",", ":")))
    except Exception as exc:  # noqa: BLE001 — disconnect is the normal exit
        _logger.debug("/ws/dashboard-chat send error: %s", exc)
    finally:
        drain_task.cancel()
        try:
            await drain_task
        except (asyncio.CancelledError, Exception):
            pass
        _dashboard_chat_unregister(q)
        _logger.info("dashboard-chat subscriber disconnected")


# Mount dashboard static files (after explicit routes so they don't shadow /v1/*)
if _dashboard_dir.exists():
    # VAD assets need their own mount (ONNX workers request from this path)
    _vad_dir = _dashboard_dir / "vad"
    if _vad_dir.exists():
        app.mount("/dashboard/vad", StaticFiles(directory=str(_vad_dir)), name="dashboard_vad")
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir)), name="dashboard_static")


# ONNX Runtime WASM workers request .wasm and .onnx files at the root path.
# These catch-all routes serve them from dashboard/vad/.
@app.get("/{filename:path}.wasm")
async def serve_wasm(filename: str):
    wasm_path = _dashboard_dir / "vad" / f"{filename}.wasm"
    if wasm_path.exists():
        return FileResponse(str(wasm_path), media_type="application/wasm")
    return JSONResponse({"detail": "Not Found"}, status_code=404)


@app.get("/{filename:path}.onnx")
async def serve_onnx(filename: str):
    onnx_path = _dashboard_dir / "vad" / f"{filename}.onnx"
    if onnx_path.exists():
        return FileResponse(str(onnx_path), media_type="application/octet-stream")
    return JSONResponse({"detail": "Not Found"}, status_code=404)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_timeline(job_id: str, event: str, t: float):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job and "timeline" in job:
            job["timeline"].append({"event": event, "t": round(t, 4)})

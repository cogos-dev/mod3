#!/usr/bin/env python3
"""Mod³ MCP shim — thin stdio proxy to a running Mod³ HTTP service.

Instead of spawning a full server.py (which loads TTS models, ~4GB VRAM),
this shim implements the MCP stdio protocol and forwards tool calls to
the Mod³ HTTP API at localhost:7860.

Tools that are purely local (set_output_device, await_voice_input) are
handled in-process without touching the HTTP service.

For `speak`, the shim posts to /v1/synthesize for audio generation, then
plays the returned WAV bytes locally via sounddevice.

Usage:
    python mcp_shim.py              # normal MCP stdio mode
    python mcp_shim.py --test       # connectivity check, then exit
"""

import io
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import OrderedDict
from typing import Any

logger = logging.getLogger("mod3.shim")

MOD3_BASE = os.environ.get("MOD3_URL", "http://localhost:7860")

# ---------------------------------------------------------------------------
# Lightweight audio playback (only needs sounddevice, not full TTS stack)
# ---------------------------------------------------------------------------

_output_device: Any = None
_current_player_lock = threading.Lock()
_current_sd_stream = None
_playback_interrupt = threading.Event()

# Job tracking (lightweight — just for speak/stop/status)
_jobs: OrderedDict = OrderedDict()
_jobs_lock = threading.Lock()
_MAX_JOBS = 50

# Barge-in signal file — must match server.py (_BARGEIN_SIGNAL there).
# Previously this was ``~/.mod3_bargein_signal.json`` but that was never
# written by anyone; the canonical path is the one the producer and server
# already use: /tmp/mod3-barge-in.json.
_BARGEIN_SIGNAL = os.environ.get("BARGEIN_SIGNAL", "/tmp/mod3-barge-in.json")


def _http_request(method: str, path: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict | bytes]:
    """Make an HTTP request to the Mod3 service. Returns (status_code, parsed_json_or_bytes)."""
    url = f"{MOD3_BASE}{path}"
    headers = {"Content-Type": "application/json"} if body is not None else {}
    data = json.dumps(body).encode() if body is not None else None

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            if "application/json" in content_type:
                return resp.status, json.loads(raw)
            elif "audio/" in content_type:
                return resp.status, raw
            else:
                try:
                    return resp.status, json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    return resp.status, raw
    except urllib.error.HTTPError as e:
        try:
            body_bytes = e.read()
            return e.code, json.loads(body_bytes)
        except Exception:
            return e.code, {"error": str(e)}
    except urllib.error.URLError as e:
        return 0, {"error": f"Mod3 service unreachable: {e.reason}"}
    except Exception as e:
        return 0, {"error": f"Request failed: {e}"}


def _play_wav_bytes(wav_bytes: bytes, job_id: str):
    """Play WAV audio bytes through speakers via sounddevice."""
    global _current_sd_stream
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError:
        logger.error("sounddevice/numpy not available — cannot play audio")
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = "sounddevice not installed"
        return

    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sw == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        elif sw == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
        else:
            audio = np.frombuffer(frames, dtype=np.float32)

        if ch > 1:
            audio = audio.reshape(-1, ch)[:, 0]  # mono mixdown

        duration = len(audio) / sr
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "speaking"
                _jobs[job_id]["start_time"] = time.time()
                _jobs[job_id]["duration_sec"] = round(duration, 2)

        _playback_interrupt.clear()
        device = _output_device
        with _current_player_lock:
            _current_sd_stream = job_id

        sd.play(audio, samplerate=sr, device=device, blocking=True)

        with _current_player_lock:
            _current_sd_stream = None

        if not _playback_interrupt.is_set():
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["status"] = "done"
                    _jobs[job_id]["metrics"] = {
                        "audio_duration_sec": round(duration, 2),
                        "sample_rate": sr,
                    }
    except Exception as e:
        logger.error("Playback error: %s", e)
        with _current_player_lock:
            _current_sd_stream = None
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(e)


def _estimate_duration(text: str, speed: float) -> float:
    words = len(text.split())
    base_wpm = 160 * speed
    return (words / base_wpm) * 60


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_speak(
    text: str, voice: str = "bm_lewis", stream: bool = True, speed: float = 1.25, emotion: float = 0.5
) -> str:
    """Synthesize via HTTP, play locally."""
    if not text.strip():
        return json.dumps({"status": "error", "error": "Nothing to say"})

    # Check barge-in
    try:
        if os.path.exists(_BARGEIN_SIGNAL):
            with open(_BARGEIN_SIGNAL) as f:
                sig = json.load(f)
            if sig.get("event") == "user_speaking_start":
                return json.dumps(
                    {
                        "status": "held",
                        "reason": "User is currently speaking — re-send after user finishes.",
                        "user_state": "recording",
                        "estimated_duration_sec": round(_estimate_duration(text, speed), 1),
                    }
                )
    except Exception:
        pass

    # Request synthesis from HTTP service
    status, resp = _http_request(
        "POST",
        "/v1/synthesize",
        {
            "text": text,
            "voice": voice,
            "speed": speed,
            "emotion": emotion,
            "format": "wav",
        },
        timeout=60.0,
    )

    if status == 0:
        return json.dumps({"status": "error", "error": resp.get("error", "Service unreachable")})
    if status != 200:
        err = resp.get("error", f"HTTP {status}") if isinstance(resp, dict) else f"HTTP {status}"
        return json.dumps({"status": "error", "error": err})
    if not isinstance(resp, bytes):
        return json.dumps({"status": "error", "error": "Expected audio bytes from synthesize"})

    # Create job and play in background
    job_id = f"shim-{int(time.time() * 1000)}"
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "generating",
            "text": text[:100],
            "voice": voice,
            "created": time.time(),
        }
        while len(_jobs) > _MAX_JOBS:
            _jobs.popitem(last=False)

    t = threading.Thread(target=_play_wav_bytes, args=(resp, job_id), daemon=True)
    t.start()

    return json.dumps({"status": "speaking", "job_id": job_id})


def tool_stop(job_id: str = "") -> str:
    """Stop playback."""
    try:
        import sounddevice as sd
    except ImportError:
        pass

    if job_id:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            return json.dumps({"status": "error", "error": f"Unknown job '{job_id}'"})
        if job["status"] == "speaking":
            _playback_interrupt.set()
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass
            with _jobs_lock:
                _jobs[job_id]["status"] = "interrupted"
            return json.dumps({"status": "ok", "message": f"Interrupted '{job_id}'"})
        return json.dumps({"status": "ok", "message": f"Job '{job_id}' status: {job['status']}"})

    # Stop all
    _playback_interrupt.set()
    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass
    with _jobs_lock:
        for j in _jobs.values():
            if j["status"] in ("speaking", "generating"):
                j["status"] = "interrupted"
    return json.dumps({"status": "ok", "message": "Stopped all playback"})


def tool_speech_status(job_id: str = "", verbose: bool = False) -> str:
    """Check job status."""
    with _jobs_lock:
        if not job_id:
            if not _jobs:
                return json.dumps({"status": "idle", "message": "No speech jobs", "queue_depth": 0})
            job_id = next(reversed(_jobs))
        job = _jobs.get(job_id)

    if not job:
        return json.dumps({"status": "error", "error": f"Unknown job '{job_id}'"})

    result = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "speaking" and "start_time" in job:
        result["elapsed_sec"] = round(time.time() - job["start_time"], 1)
    if job.get("metrics"):
        result["metrics"] = job["metrics"]
    if job.get("error"):
        result["error"] = job["error"]

    # Queue state
    with _jobs_lock:
        speaking = sum(1 for j in _jobs.values() if j["status"] == "speaking")
    result["queue"] = {"depth": speaking, "currently_playing": None}

    return json.dumps(result)


def tool_list_voices() -> str:
    """List voices via HTTP."""
    status, resp = _http_request("GET", "/v1/voices")
    if status != 200:
        return json.dumps({"status": "error", "error": "Could not reach Mod3 service"})

    engines = resp.get("engines", {})
    lines = []
    for engine, cfg in engines.items():
        supports = cfg.get("supports", [])
        tag = f" ({', '.join(supports)})" if supports else ""
        voices = cfg.get("voices", [])
        lines.append(f"  {engine}{tag}: {', '.join(voices)}")
    return "Available voices:\n" + "\n".join(lines)


def tool_diagnostics() -> str:
    """Diagnostics via HTTP."""
    status, resp = _http_request("GET", "/diagnostics")
    if status != 200:
        return json.dumps({"status": "error", "error": "Could not reach Mod3 service"})
    return json.dumps(resp, indent=2)


def tool_set_output_device(device: str = "") -> str:
    """List or set audio output device (local only)."""
    global _output_device
    try:
        import sounddevice as sd
    except ImportError:
        return json.dumps({"status": "error", "error": "sounddevice not installed"})

    outputs = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_output_channels"] > 0:
            is_default = i == sd.default.device[1]
            is_active = (
                (_output_device is None and is_default)
                or _output_device == i
                or (isinstance(_output_device, str) and _output_device in d["name"])
            )
            outputs.append({"index": i, "name": d["name"], "active": is_active, "default": is_default})

    if not device:
        return json.dumps({"devices": outputs})

    if device == "default":
        _output_device = None
        return json.dumps({"status": "ok", "message": "Tracking system default"})

    # Try numeric index
    try:
        idx = int(device)
        for d in outputs:
            if d["index"] == idx:
                _output_device = idx
                return json.dumps({"status": "ok", "device": d["name"], "index": idx})
        return json.dumps({"status": "error", "error": f"No output device at index {idx}"})
    except ValueError:
        pass

    # Try name substring
    for d in outputs:
        if device.lower() in d["name"].lower():
            _output_device = d["index"]
            return json.dumps({"status": "ok", "device": d["name"], "index": d["index"]})

    return json.dumps({"status": "error", "error": f"No device matching '{device}'"})


def tool_await_voice_input(timeout_sec: float = 180.0) -> str:
    """Block until SuperWhisper recording finishes (local only)."""
    _rec_dir = os.path.expanduser("~/Documents/superwhisper/recordings")

    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            if os.path.exists(_BARGEIN_SIGNAL):
                with open(_BARGEIN_SIGNAL) as f:
                    signal = json.load(f)
                if signal.get("event") == "user_speaking_end":
                    break
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    else:
        return json.dumps({"status": "timeout", "error": f"No recording completed within {timeout_sec}s"})

    # Find latest transcript
    try:
        folders = sorted(
            [d for d in os.listdir(_rec_dir) if d.isdigit()],
            key=int,
            reverse=True,
        )
        if folders:
            meta_path = os.path.join(_rec_dir, folders[0], "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                raw = meta.get("rawResult", "").strip()
                result = meta.get("result", raw).strip()
                duration_ms = meta.get("duration", 0)
                return json.dumps(
                    {
                        "status": "ok",
                        "transcript": result if result else raw,
                        "raw_transcript": raw,
                        "duration_sec": round(duration_ms / 1000, 1),
                        "folder": folders[0],
                        "source": "superwhisper",
                    }
                )
    except Exception as e:
        logger.warning("await_voice_input error: %s", e)

    return json.dumps({"status": "error", "error": "Could not retrieve transcript"})


def tool_vad_check(file_path: str, threshold: float = 0.5) -> str:
    """VAD check via HTTP."""
    if not os.path.exists(file_path):
        return json.dumps({"status": "error", "error": f"File not found: {file_path}"})

    # Read WAV and send to HTTP endpoint
    try:
        with open(file_path, "rb") as f:
            wav_data = f.read()
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})

    # The HTTP API expects multipart file upload, use urllib
    boundary = "----Mod3ShimBoundary"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(file_path)}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + wav_data
        + f"\r\n--{boundary}--\r\n".encode()
    )

    url = f"{MOD3_BASE}/v1/vad"
    if threshold != 0.5:
        url += f"?threshold={threshold}"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.dumps(json.loads(resp.read()))
    except Exception as e:
        return json.dumps({"status": "error", "error": f"VAD request failed: {e}"})


# ---------------------------------------------------------------------------
# Tool registry (matches server.py exactly)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "speak",
        "description": (
            "Synthesize text to speech and play it through the user's speakers.\n\n"
            "Non-blocking: returns immediately with a job ID while audio plays or is\n"
            "queued. If nothing is playing, starts immediately. If audio is already\n"
            "playing, the new request is queued and will play automatically when the\n"
            "current item finishes.\n\n"
            "The response always includes the current queue state so the agent knows\n"
            "exactly what's happening on the output channel without a separate status call.\n\n"
            "Args:\n"
            "    text: The text to speak aloud. Keep it conversational.\n"
            "    voice: Voice preset. Use list_voices() to see options.\n"
            '           Defaults to "bm_lewis" (Kokoro).\n'
            "    stream: If True, plays audio chunks as they generate (lower latency).\n"
            "            If False, generates all audio first then plays (better prosody).\n"
            "    speed: Speed multiplier (engines with speed support). Default 1.25.\n"
            "    emotion: Emotion/exaggeration intensity 0.0-1.0 (Chatterbox only). Default 0.5."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to speak aloud. Keep it conversational."},
                "voice": {
                    "type": "string",
                    "default": "bm_lewis",
                    "description": 'Voice preset. Use list_voices() to see options. Defaults to "bm_lewis" (Kokoro).',
                },
                "stream": {
                    "type": "boolean",
                    "default": True,
                    "description": "If True, plays audio chunks as they generate (lower latency).",
                },
                "speed": {
                    "type": "number",
                    "default": 1.25,
                    "description": "Speed multiplier (engines with speed support). Default 1.25.",
                },
                "emotion": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Emotion/exaggeration intensity 0.0-1.0 (Chatterbox only). Default 0.5.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "speech_status",
        "description": (
            "Check status of a speech job, or get the most recent result.\n\n"
            "Always includes queue state so the agent has full output channel awareness.\n\n"
            "Args:\n"
            "    job_id: The job ID returned by speak(). If empty, returns the latest job.\n"
            "    verbose: If True, include per-chunk metrics. Default False (summary only)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "default": "",
                    "description": "The job ID returned by speak(). If empty, returns the latest job.",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, include per-chunk metrics. Default False (summary only).",
                },
            },
        },
    },
    {
        "name": "stop",
        "description": (
            "Stop current speech or cancel a specific queued item.\n\n"
            "Args:\n"
            "    job_id: If provided, cancels that specific queued job (not yet playing).\n"
            "            If the job_id is the currently playing job, interrupts playback.\n"
            "            If empty, interrupts current playback AND clears the entire queue."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "default": "",
                    "description": "If provided, cancels that specific job. If empty, stops everything.",
                },
            },
        },
    },
    {
        "name": "list_voices",
        "description": "List all available voice presets grouped by engine.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "await_voice_input",
        "description": (
            "Block until the user finishes a SuperWhisper recording, then return the transcript.\n\n"
            "This closes the voice input loop: instead of waiting for the user to paste\n"
            "their transcribed text, you can directly receive what they said. Use this\n"
            'when speak() returns "held" (user is recording) or when you want to listen\n'
            "for the next voice input.\n\n"
            "Polls the barge-in signal file for user_speaking_end, then reads the\n"
            "transcript from SuperWhisper's recordings directory.\n\n"
            "Args:\n"
            "    timeout_sec: Maximum seconds to wait for recording to finish. Default 180 (3 minutes)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "timeout_sec": {
                    "type": "number",
                    "default": 180,
                    "description": "Maximum seconds to wait for recording to finish. Default 180 (3 minutes).",
                },
            },
        },
    },
    {
        "name": "diagnostics",
        "description": "Return engine state and last generation metrics for debugging.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_output_device",
        "description": (
            "List audio output devices, or set the active one.\n\n"
            "Args:\n"
            '    device: Device index (e.g. "3"), name substring (e.g. "AirPods"),\n'
            '            or "default" to track the system default automatically.\n'
            "            If empty, lists available devices without changing anything."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "default": "",
                    "description": "Device index, name substring, or 'default'. If empty, lists devices.",
                },
            },
        },
    },
    {
        "name": "vad_check",
        "description": (
            "Check if an audio file contains speech using Silero VAD.\n\n"
            "Use this before transcription to avoid Whisper hallucinations on\n"
            "silence or ambient noise.\n\n"
            "Args:\n"
            "    file_path: Path to a WAV audio file.\n"
            "    threshold: Speech probability threshold 0-1 (default 0.5). Higher = stricter."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to a WAV audio file."},
                "threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Speech probability threshold 0-1 (default 0.5). Higher = stricter.",
                },
            },
            "required": ["file_path"],
        },
    },
]

TOOL_DISPATCH = {
    "speak": lambda args: tool_speak(
        args["text"],
        voice=args.get("voice", "bm_lewis"),
        stream=args.get("stream", True),
        speed=args.get("speed", 1.25),
        emotion=args.get("emotion", 0.5),
    ),
    "speech_status": lambda args: tool_speech_status(
        job_id=args.get("job_id", ""),
        verbose=args.get("verbose", False),
    ),
    "stop": lambda args: tool_stop(job_id=args.get("job_id", "")),
    "list_voices": lambda args: tool_list_voices(),
    "await_voice_input": lambda args: tool_await_voice_input(
        timeout_sec=args.get("timeout_sec", 180.0),
    ),
    "diagnostics": lambda args: tool_diagnostics(),
    "set_output_device": lambda args: tool_set_output_device(
        device=args.get("device", ""),
    ),
    "vad_check": lambda args: tool_vad_check(
        file_path=args["file_path"],
        threshold=args.get("threshold", 0.5),
    ),
}


# ---------------------------------------------------------------------------
# MCP stdio protocol
# ---------------------------------------------------------------------------

SERVER_INFO = {
    "name": "mod3",
    "version": "0.3.0-shim",
}

CAPABILITIES = {
    "tools": {},
}


def _read_message() -> dict | None:
    """Read a JSON-RPC message from stdin (newline-delimited)."""
    try:
        line = sys.stdin.readline()
        if not line:
            return None
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _write_message(msg: dict):
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _jsonrpc_response(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _jsonrpc_error(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def handle_initialize(msg: dict) -> dict:
    return _jsonrpc_response(
        msg["id"],
        {
            "protocolVersion": "2024-11-05",
            "serverInfo": SERVER_INFO,
            "capabilities": CAPABILITIES,
        },
    )


def handle_tools_list(msg: dict) -> dict:
    return _jsonrpc_response(msg["id"], {"tools": TOOLS})


def handle_tools_call(msg: dict) -> dict:
    params = msg.get("params", {})
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    handler = TOOL_DISPATCH.get(tool_name)
    if not handler:
        return _jsonrpc_error(msg["id"], -32602, f"Unknown tool: {tool_name}")

    try:
        result_text = handler(arguments)
    except Exception as e:
        result_text = json.dumps({"status": "error", "error": str(e)})

    return _jsonrpc_response(
        msg["id"],
        {
            "content": [{"type": "text", "text": result_text}],
        },
    )


def handle_notifications_initialized(msg: dict):
    """Client sends this after initialize — no response needed."""
    pass


METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "tools/list": handle_tools_list,
    "tools/call": handle_tools_call,
    "notifications/initialized": handle_notifications_initialized,
    "ping": lambda msg: _jsonrpc_response(msg["id"], {}),
}


def run_stdio():
    """Main MCP stdio loop."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    while True:
        msg = _read_message()
        if msg is None:
            break  # EOF

        method = msg.get("method", "")
        handler = METHOD_HANDLERS.get(method)

        if handler is None:
            # Unknown method — if it has an id, return error; if notification, ignore
            if "id" in msg:
                _write_message(_jsonrpc_error(msg["id"], -32601, f"Method not found: {method}"))
            continue

        result = handler(msg)
        if result is not None:
            _write_message(result)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test():
    """Quick connectivity check."""
    print(f"Mod3 shim — testing connection to {MOD3_BASE}")

    status, resp = _http_request("GET", "/health")
    if status == 200:
        engines = resp.get("engines", {})
        loaded = [k for k, v in engines.items() if v == "loaded"]
        print(f"  OK: Mod3 service healthy — {len(loaded)} engine(s) loaded: {', '.join(loaded) or 'none'}")
    elif status == 0:
        print(f"  WARN: Mod3 service not reachable at {MOD3_BASE}")
        print("        Tools will return errors until the service starts.")
    else:
        print(f"  WARN: Unexpected status {status} from /health")

    # Check sounddevice
    try:
        import sounddevice as sd

        default_out = sd.query_devices(sd.default.device[1])
        print(f"  OK: sounddevice available — default output: {default_out['name']}")
    except ImportError:
        print("  WARN: sounddevice not installed — speak/stop will fail")
    except Exception as e:
        print(f"  WARN: sounddevice error: {e}")

    print("  Shim ready.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        run_stdio()

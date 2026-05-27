# Mod³ — Model Modality Modulator

[![CI](https://github.com/myrgic/mod3/actions/workflows/ci.yml/badge.svg)](https://github.com/myrgic/mod3/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Give your AI agent a voice.

Mod³ is a Python MCP server that provides text-to-speech for Claude Code, Cursor, and other MCP-compatible AI tools. It runs four TTS engines locally on Apple Silicon, generates speech faster than realtime, and returns immediately so the agent keeps working while audio plays.

## What it does

- **Non-blocking speech** -- `speak()` returns immediately with a job ID. Audio plays in the background. The agent writes code while it talks.
- **Queue-aware output** -- Every `speak()` return includes queue position, estimated wait time, and active job state. The agent knows what's playing without making a separate status call.
- **Barge-in detection** -- VAD (voice activity detection) monitors the microphone. If the user starts talking, playback stops and the agent is notified. No talking over people.
- **Turn-taking** -- Bidirectional awareness of who's speaking. The agent can check user state before deciding to speak or wait.
- **Multi-model routing** -- Four TTS engines behind one interface. Voice name determines which engine handles the request.
- **Voice profile registry** -- Cloned voices are stored as named profiles under `~/.mod3/voices/` and addressable as first-class voice IDs alongside built-in engine presets.
- **Continuous open-mic** -- Always-on VAD with auto-start barge-in and tunable endpointing; Whisper STT uses multi-strategy deduplication (Z-function, sentence-level, N-way) to eliminate phrase doubling.
- **Adaptive buffering** -- EMA-based arrival rate tracking with dynamic startup threshold. Gapless playback under normal load, graceful degradation under GPU contention.
- **Structured metrics** -- Every call returns TTFA, RTF, per-chunk timing, buffer health, underrun counts, and memory usage. The agent can diagnose its own audio quality.
- **Observability** -- Per-phase wall-time instrumentation and W3C traceparent propagation through `CogOSProvider`; trace IDs flow from inbound request to every pipeline phase.

## Engines

| Engine | Model | Size | TTFA | Control Surfaces |
|--------|-------|------|------|-----------------|
| **Kokoro** | Kokoro-82M-bf16 | 82M | ~60ms | Speed, emphasis (ALL CAPS), pacing (punctuation) |
| **Voxtral** | Voxtral-4B-TTS-mlx-4bit | 4B | ~500ms | 20 voice presets, multi-language |
| **Chatterbox** | chatterbox-4bit | ~1B | ~60ms | Emotion/exaggeration (0-1), voice cloning |
| **Spark** | Spark-TTS-0.5B-bf16 | 0.5B | ~1s | Pitch (5-level), speed, gender |

Models are downloaded on first use via HuggingFace Hub.

## Quick Start

```bash
git clone https://github.com/myrgic/mod3.git
cd mod3
./setup.sh
```

### HTTP-MCP (recommended)

Start mod3 as a persistent daemon and connect via HTTP-MCP. This is the canonical transport going forward. The daemon stays alive between agent sessions so TTS engines stay warm and multiple clients can share one instance.

```bash
# Start the server (or configure as a launchd service)
python server.py --http
```

Then point your MCP client at the HTTP-MCP endpoint:

```json
{
  "mcpServers": {
    "mod3": {
      "type": "http",
      "url": "http://localhost:7860/mcp"
    }
  }
}
```

### stdio MCP (deprecated)

> **Deprecated.** stdio MCP is still functional but is being phased out. Each client session spawns a new mod3 process, which means TTS engines cold-start on every connection (~60s for Kokoro) and state is not shared across sessions. Prefer HTTP-MCP above. A `DeprecationWarning` is printed to stderr at boot when this path is active. Removal is tracked in [issue #11](https://github.com/myrgic/mod3/issues/11).

For users who have not migrated yet, the stdio path remains available. Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "mod3": {
      "command": "/path/to/mod3/.venv/bin/python",
      "args": ["/path/to/mod3/server.py"]
    }
  }
}
```

## MCP Tools

### `speak(text, voice?, stream?, speed?, emotion?)`

Synthesize text and play through speakers. Returns immediately with a job ID, queue state, and estimated wait time.

```
speak("Hello world")                                        → default voice (eng_uk_m_davids @ 1.25x)
speak("Hello world", voice="casual_male")                   → Voxtral
speak("Hello world", voice="chatterbox", emotion=0.8)       → Chatterbox with high emotion
speak("Hello world", voice="am_michael", speed=1.4)         → Kokoro fast
```

### `output(text, mode?, stream?)`

Unified output tool. `mode` selects the channel: `"audio"` (TTS only), `"text"` (dashboard chat bubble only), or `"both"` (simultaneous TTS + chat bubble). Defaults to `"audio"`. Replaces separate speak-and-notify patterns with a single call.

### `speech_status(job_id?, verbose?)`

Check if speech is still playing, or get metrics from the last completed job. Pass `verbose=True` for per-chunk detail.

### `stop()`

Interrupt current speech immediately.

### `vad_check()`

Check microphone for voice activity. Returns whether the user is currently speaking, enabling the agent to wait for a natural pause before responding.

### `list_voices()`

List all available voices grouped by engine, with control surface tags. Includes cloned voices from the voice profile registry (`~/.mod3/voices/`).

### `set_output_device(device?)`

List audio output devices, or switch the active one mid-session.

### `diagnostics()`

Show loaded engines, active jobs, output device, and last generation metrics.

## Architecture

Key modules:

- **`server.py`** -- MCP tool definitions, multi-model registry, sentence chunking, non-blocking job management, queue-aware returns
- **`http_api.py`** -- FastAPI HTTP server; mounts the HTTP-MCP transport at `/mcp`, the ACP WebSocket endpoint at `/ws/acp`, and per-session audio at `/ws/audio/{session_id}`; implements ACP `session/list`, `session/load`, `session/resume`, and `authenticate`
- **`channels.py`** -- `ChannelMode` enum (passthrough / transcribe / agent) and composable directed-acyclic stage graph; pipeline stages are wired at startup from registered `@register_stage` classes
- **`inbound.py`** -- Intentional pipeline stages (VAD, STT, intent classification) as `@register_stage`-decorated classes; consumed by the channel stage graph
- **`bus.py`** -- Session-aware event bus; sessions are first-class, per-session routing replaces broadcast fan-out (ADR-082)
- **`bus_bridge.py`** -- SSE bridge that forwards CogOS kernel events (identity projection, voice config) to connected dashboard and channel clients
- **`seats.py`** -- Seat registration and identity claim management; `register_session` emits `presence.started` with `iss`/`sub` pairs for both user and agent identities
- **`identity_projection_handler.py`** -- Handles incoming CogOS identity-projection events; updates active seat voice config from `IdentityVoiceProfile`
- **`adaptive_player.py`** -- Callback-based audio playback with EMA arrival rate tracking, adaptive startup threshold, and structured metrics collection
- **`voice_profiles.py`** / **`voice_profile_io.py`** / **`voice_profile_schema.py`** -- Voice profile registry and schema; cloned voices stored under `~/.mod3/voices/` addressable as first-class voice IDs; `IdentityVoiceProfile` schema mirrors the CogOS identity CRD for voice config received via projection events
- **`chat_flow_log.py`** -- Structured turn lifecycle log with per-phase wall-time instrumentation and W3C traceparent propagation
- **`dashboard/`** -- Three-column browser dashboard: sessions sidebar, main chat panel, and Settings / Traces / Debug side panel with hierarchical span tree

The adaptive player is model-agnostic. Any TTS engine that produces audio chunks feeds the same pipeline.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- espeak-ng (`brew install espeak-ng`) -- required for Kokoro's phonemizer

## Using Voice as a Modality

See [`skills/voice/SKILL.md`](skills/voice/SKILL.md) for the full guide on dual-modal communication -- when to speak vs write, non-blocking patterns, reading metrics, and anti-patterns.

Voice carries the ephemeral (context, intent, tone). Text carries the persistent (code, data, decisions). Both channels active simultaneously.

## Ecosystem

Mod³ is the voice layer in the [CogOS](https://github.com/myrgic/cogos) ecosystem. It integrates as a modality channel -- the kernel routes intents to Mod³ when voice output is appropriate. Works standalone without CogOS.

| Repo | Purpose |
|------|---------|
| [cogos](https://github.com/myrgic/cogos) | The daemon |
| **mod3** | **Voice -- this repo** |
| [constellation](https://github.com/myrgic/constellation) | Distributed identity and trust |
| [plugins](https://github.com/myrgic/plugins) | Agent skill library |
| [charts](https://github.com/myrgic/charts) | Helm charts for deployment |

## License

MIT

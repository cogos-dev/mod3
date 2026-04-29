# Mod³ — Model Modality Modulator

[![CI](https://github.com/cogos-dev/mod3/actions/workflows/ci.yml/badge.svg)](https://github.com/cogos-dev/mod3/actions/workflows/ci.yml)
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
- **Adaptive buffering** -- EMA-based arrival rate tracking with dynamic startup threshold. Gapless playback under normal load, graceful degradation under GPU contention.
- **Structured metrics** -- Every call returns TTFA, RTF, per-chunk timing, buffer health, underrun counts, and memory usage. The agent can diagnose its own audio quality.

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
git clone https://github.com/cogos-dev/mod3.git
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

> **Deprecated.** stdio MCP is still functional but is being phased out. Each client session spawns a new mod3 process, which means TTS engines cold-start on every connection (~60s for Kokoro) and state is not shared across sessions. Prefer HTTP-MCP above. A `DeprecationWarning` is printed to stderr at boot when this path is active. Removal is tracked in [issue #11](https://github.com/cogos-dev/mod3/issues/11).

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
speak("Hello world")                                        → default voice (bm_lewis @ 1.25x)
speak("Hello world", voice="casual_male")                   → Voxtral
speak("Hello world", voice="chatterbox", emotion=0.8)       → Chatterbox with high emotion
speak("Hello world", voice="am_michael", speed=1.4)         → Kokoro fast
```

### `speech_status(job_id?, verbose?)`

Check if speech is still playing, or get metrics from the last completed job. Pass `verbose=True` for per-chunk detail.

### `stop()`

Interrupt current speech immediately.

### `vad_check()`

Check microphone for voice activity. Returns whether the user is currently speaking, enabling the agent to wait for a natural pause before responding.

### `list_voices()`

List all available voices grouped by engine, with control surface tags.

### `set_output_device(device?)`

List audio output devices, or switch the active one mid-session.

### `diagnostics()`

Show loaded engines, active jobs, output device, and last generation metrics.

## Architecture

Two files:

- **`server.py`** -- MCP tool definitions, multi-model registry, sentence chunking, non-blocking job management, queue-aware returns
- **`adaptive_player.py`** -- Callback-based audio playback with EMA arrival rate tracking, adaptive startup threshold, and structured metrics collection

The adaptive player is model-agnostic. Any TTS engine that produces audio chunks feeds the same pipeline.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- espeak-ng (`brew install espeak-ng`) -- required for Kokoro's phonemizer

## Using Voice as a Modality

See [`skills/voice/SKILL.md`](skills/voice/SKILL.md) for the full guide on dual-modal communication -- when to speak vs write, non-blocking patterns, reading metrics, and anti-patterns.

Voice carries the ephemeral (context, intent, tone). Text carries the persistent (code, data, decisions). Both channels active simultaneously.

## Ecosystem

Mod³ is the voice layer in the [CogOS](https://github.com/cogos-dev/cogos) ecosystem. It integrates as a modality channel -- the kernel routes intents to Mod³ when voice output is appropriate. Works standalone without CogOS.

| Repo | Purpose |
|------|---------|
| [cogos](https://github.com/cogos-dev/cogos) | The daemon |
| **mod3** | **Voice -- this repo** |
| [constellation](https://github.com/cogos-dev/constellation) | Distributed identity and trust |
| [skills](https://github.com/cogos-dev/skills) | Agent skill library |
| [charts](https://github.com/cogos-dev/charts) | Helm charts for deployment |
| [desktop](https://github.com/cogos-dev/desktop) | macOS dashboard app |

## License

MIT

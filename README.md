# Mod³ — Model Modality Modulator

Give your AI agent a voice. Mod³ is an MCP server that runs local TTS models on Apple Silicon, with adaptive buffering, multi-model routing, non-blocking speech, and structured metrics.

Built for [Claude Code](https://claude.ai/claude-code), works with any MCP-compatible client.

## What it does

- **Non-blocking speech** — `speak()` returns immediately. Audio plays in the background while the agent keeps working. Two output channels: voice for the ephemeral, text for the persistent.
- **Multi-model routing** — Four TTS engines, one interface. Voice name auto-routes to the right model.
- **Adaptive buffering** — EMA-based arrival rate tracking with dynamic startup threshold. Gapless playback under normal conditions, graceful degradation under GPU contention.
- **Structured metrics** — Every call returns TTFA, RTF, per-chunk timing, buffer health, underrun counts, memory usage. The agent can diagnose its own audio quality.
- **Sentence chunking** — Text is split at sentence boundaries for natural prosody. Feathered edges (fade-out + breath gap) between sentences.

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
git clone https://github.com/slowbro/mod3.git
cd mod3
./setup.sh
```

Then add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "tts": {
      "command": "/path/to/mod3/.venv/bin/python",
      "args": ["/path/to/mod3/server.py"]
    }
  }
}
```

## MCP Tools

### `speak(text, voice?, stream?, speed?)`

Synthesize text and play through speakers. Returns immediately with a job ID.

```
speak("Hello world")                          → default voice (bm_lewis @ 1.25x)
speak("Hello world", voice="casual_male")     → Voxtral
speak("Hello world", voice="chatterbox", speed=0.8)  → Chatterbox with emotion
```

### `speech_status(job_id?)`

Check if speech is still playing, or get metrics from the last completed job.

### `list_voices()`

List all available voices grouped by engine.

### `diagnostics()`

Show loaded models, memory usage, and last generation metrics.

## Architecture

Two files:

- **`server.py`** — MCP tool definitions, multi-model registry, sentence chunking, non-blocking job management
- **`adaptive_player.py`** — Callback-based audio playback with EMA arrival rate tracking, adaptive startup threshold, and structured metrics collection

The adaptive player is model-agnostic. Any TTS engine that produces audio chunks feeds the same pipeline.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- espeak-ng (`brew install espeak-ng`) — required for Kokoro's phonemizer

## Using Voice as a Modality

See [`skills/voice/SKILL.md`](skills/voice/SKILL.md) for the full guide on dual-modal communication — when to speak vs write, non-blocking patterns, reading metrics, and anti-patterns.

The short version: voice carries the ephemeral (context, intent, tone). Text carries the persistent (code, data, decisions). Both channels active simultaneously. That's the point.

## License

MIT

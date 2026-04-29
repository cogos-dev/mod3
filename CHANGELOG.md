# Changelog

## Unreleased

### Changed
- **FastAPI lifespan migration** (`http_api.py`) — replaced all `@app.on_event("startup")` / `@app.on_event("shutdown")` decorators with a single `@asynccontextmanager` lifespan passed to `FastAPI(lifespan=...)`. Startup order: Kokoro warmup thread, kernel-bus bridge, CogOS agent bridge. Shutdown order: reverse. Eliminates the DeprecationWarning emitted on every boot.

## [0.4.0] - 2026-04-19

### Added — Voice pipeline
- **Bidirectional voice pipeline** — full duplex audio (capture → STT → agent_loop → TTS → playback) with WebRTC echo cancellation
- **MCP shim** — bridges mod3 tools through cogos kernel as MCP tool surface
- **Bus-mediated dashboard chat** — dashboard chat goes through cogos kernel buses instead of in-process loop, so external observers see the same conversation events

### Added — Bargein provider registry
- **Pluggable `BargeinProvider` interface** (`bargein/providers/base.py`) — was a hardcoded SuperWhisper file watcher; now extensible
- **`SuperWhisperProvider`** (`bargein/providers/superwhisper.py`) — first provider, opt-in via `MOD3_BARGEIN_PROVIDERS=superwhisper`. Absorbs the SuperWhisper SQLite + filesystem detection logic that was previously drifting in a sibling repo
- **`BargeinRegistry`** (`bargein/__init__.py`) — registry + shared `handle_bargein_start()` helper, used by both legacy file watcher and provider dispatch
- **`BargeinRegistry.wait_for_event()`** — synchronous wait primitive used by `await_voice_input()` to block on in-process registry events
- New `"superwhisper"` value in `BargeinSource` literal

### Added — From earlier work, never released
- Queue-aware `speak()` returns with enriched metadata (PR #4)
- `SpeechQueue` for serial playback (thread-safe)
- User-state detection (held status when user is recording)
- `/v1/stop` HTTP endpoint for playback control
- `vad_check` MCP tool

### Changed
- Default `MOD3_BARGEIN_PROVIDERS=` (empty) preserves current behavior — no providers auto-start
- `await_voice_input()` now waits on both `BargeinRegistry` events AND legacy `/tmp/mod3-barge-in.json` for backward compat

### Fixed
- **Speaking lock ownership** — `(pid, job_id)`-aware with idempotent re-acquire. Two overlapping mod3 processes can no longer falsely interrupt each other.
- **Bus subscriber endpoint** — `KernelBusSubscriber` honors `COGOS_ENDPOINT` at call time (previously hardcoded `localhost:6931`)
- **Session-scoped reply routing** — kernel replies with `session_id` get routed to the matching browser channel; older payloads fall back to broadcast
- **Signal path unification** — `mcp_shim.py` reads from `/tmp/mod3-barge-in.json` (was orphan `~/.mod3_bargein_signal.json` that nobody wrote to)
- Held job zombie drain bug
- Pyright type errors in Gate abstract class
- Various ruff lint issues

### Reviewed by
- claude-opus-4-7 (interactive)
- gpt-5.4 (peer review, 3 passes)

### Notes on versioning
This release jumps from `v0.1.0` to `v0.4.0`. An earlier `v0.2.0` tag exists from before the cogos-dev org transition (no GitHub release was created). `v0.3.0` was bumped in `pyproject.toml` and added to the CHANGELOG but never tagged. `v0.4.0` captures everything since the last released version (`v0.1.0`).

## [0.3.0] - 2026-04-04

### Added
- **HTTP API** — FastAPI server alongside MCP, shared model cache
  - `POST /v1/synthesize` — text → WAV/PCM audio bytes with full generation metrics
  - `POST /v1/audio/speech` — OpenAI-compatible TTS endpoint
  - `POST /v1/vad` — Silero VAD speech detection on audio files
  - `POST /v1/filter` — Whisper hallucination check (Bag of Hallucinations)
  - `GET /v1/voices` — list engines and voice presets
  - `GET /v1/jobs` — job ledger with lifecycle tracking and per-chunk metrics
  - `GET /v1/jobs/{id}` — specific job details
  - `GET /health` — server health with engine/VAD status
- **Silero VAD** — voice activity detection input gate, prevents Whisper hallucinations on silence/noise
- **Bag of Hallucinations (BoH)** — post-filter for known Whisper phantom phrases ("thank you", "subscribe", etc.)
- **`vad_check` MCP tool** — run VAD on a local audio file from Claude Code
- **Job ledger** — every HTTP request (synthesize, VAD, filter) gets a job ID with full lifecycle timeline
- **Server startup modes** — `--http` (HTTP only), `--all` (MCP + HTTP), default MCP only
- **OpenClaw speech provider plugin** (`integrations/openclaw/`) — drop-in local TTS for Discord voice channels

### Changed
- **Engine extraction** — inference core moved to `engine.py`, shared by MCP and HTTP interfaces
- **Server refactored** — `server.py` imports from `engine.py` instead of defining models inline

## [0.2.0] - 2026-04-04

### Added
- **Non-blocking speech** — `speak()` returns immediately with job ID, audio plays in background
- **Multi-model routing** — Voxtral, Kokoro, Chatterbox, Spark engines, voice name auto-routes
- **Sentence chunking** — pysbd splits text at sentence boundaries for natural prosody
- **Feathered edges** — fade-out + adaptive gap between sentences
- **Adaptive sentence gaps** — 50-200ms scaled by sentence length
- **`stop()` tool** — interrupt current speech immediately
- **`speech_status()` tool** — check job status, verbose flag for per-chunk detail
- **`set_output_device()` tool** — list/switch audio outputs mid-session
- **`diagnostics()` tool** — engine state, active jobs, memory usage
- **Separate `emotion` param** — Chatterbox exaggeration decoupled from speed
- **Job cleanup** — OrderedDict capped at 20 entries
- **200ms silence leader** — prevents audio device clipping first word

### Fixed
- Race condition in `wait()` returning before generation started
- Audio cut-off at end of speech (now uses `finished_callback`)
- Sample rate now model-aware (fixes Spark 16kHz calculations)
- README clone URL and tool signatures

## [0.1.0] - 2026-04-03

### Added
- Initial release
- Adaptive playback engine with EMA arrival rate tracking
- Voxtral 4B TTS support
- Structured per-call metrics (TTFA, RTF, buffer health, underruns, memory)
- Voice modality skill doc (`skills/voice/SKILL.md`)

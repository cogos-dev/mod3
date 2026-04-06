# Changelog

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

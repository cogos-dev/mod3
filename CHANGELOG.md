# Changelog

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

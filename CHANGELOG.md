# Changelog

## [0.8.0] - 2026-08-02

### Fixed — CI: unpinned `mcp` install broke test collection

- **`Import & Unit Tests` was failing on every PR.** Both the CI workflow's inline `pip install` and `requirements.txt` declared `mcp` with no upper bound, so a fresh install picked up `mcp==2.0.0`, which moved the high-level server helper out of `mcp.server.fastmcp` (imported at module level by `jobs_registry.py` and `clients/channel_client.py`). Collecting `tests/test_speaking_lock.py` then failed with `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`, interrupting the whole run. `mcp` is now capped at `>=1.20.0,<2.0.0` in both `.github/workflows/ci.yml` (the `test-import` and `test-api` jobs) and `requirements.txt`. Verified clean in a fresh venv: 925 passed, 19 skipped, 0 collection errors.

### Added — ledger sink: the seat's spoken and dashboard turns become durable

- **The channel client's two mouths now write to a durable conversation ledger.** `mod3_speak` (with `post_to_chat=True`) and `mod3_dashboard_post` (as `assistant`) also commit the utterance as a receipt into the ledger repo's `conversations/inbox/` (`clients/ledger_sink.py`), riding the existing conversations-ingest workflow into `ledger.json`. mod3's message store is RAM-only, so before this every spoken word was lost on daemon restart; now the spoken half of a conversation survives. Turns declare `origin=seat` and a `seat-*` `from` — the wake-line watcher's self-echo suppression keys, enforced by the sink itself, not caller convention. Fire-and-forget (`asyncio.to_thread`): speech never blocks on a git push; failures log and swallow; kill switch `MOD3_LEDGER_SINK=0`. An interprocess flock per repo serializes the git sequence across concurrent channel-client processes; receipt ids carry an entropy suffix so same-millisecond writers can't collide; the client's shutdown path drains in-flight sinks (bounded 15s) so a session's final utterance isn't dropped. `mod3_speak` gains an optional `ledger_thread` arg so speech during a threaded conversation lands on that thread; defaults to `$MOD3_LEDGER_THREAD` or `"voice"`. The daemon is untouched — this is entirely client-side. (The sink module was originally named `theseus_sink` after an internal codename; renamed to `ledger_sink` before this shipped in a release, with `MOD3_THESEUS_*` env vars and the old module name kept as backward-compatible aliases.)

### Fixed — P0: server.py's __main__/import duality double-executed the whole speech runtime

- **`jobs_registry.py` is now the single source of truth for job state, the speech queue, the modality bus, pipeline (barge-in) state, and the dashboard-chat relay — shared by both server.py (the MCP process, run as `__main__`) and http_api.py.** Production runs `python server.py --http`, which makes server.py execute as `__main__`. http_api.py previously read shared state via `from server import _bus` (and five other lazy cross-imports) — since a module run as `__main__` is never the same module object as one later imported by its own name, that silently re-executed server.py's entire top level a second time under the module name `"server"`: a second job registry, a second speech queue, a second `ModalityBus`, and a second `_bargein_watcher` thread with its own `pipeline_state`. Confirmed empirically (`tests/test_server_topology.py` reproduces the real subprocess shape): the watcher bound to `__main__` never saw the real job's `pipeline_state` as speaking, so it misread the other instance's live utterance as a foreign cross-process speaker and forcibly cleared the shared speaking-lock file out from under it. `http_api.py` and `server.py` now both import everything from `jobs_registry.py` directly — no more lazy `from server import ...` anywhere. `GET /diagnostics` gained a `topology.server_reimported` sentinel (`"server" in sys.modules`) so this class of regression is directly observable going forward, not just inferable from symptoms.
- **A barge-in-interrupted job now reaches a terminal state instead of wedging at "speaking".** `pipeline_state.interrupt()` calls `player.flush()` directly, silencing audio immediately, but never told the generation loop to stop calling the TTS engine for more chunks — the loop's only prior stop signal was the file-based speaking-lock check, which an in-process interrupt never touches. A barged-in job stayed at status "speaking" / HUD "encoding" until the engine finished synthesizing the rest of the (unheard) text, which could take minutes. The generation loop now also checks the same `InterruptInfo` the interrupt callback already receives; either that or a lock-loss finalizes the job as `"interrupted"` (not `"done"`) with the partial `spoken_pct`/`delivered_text`/`reason` and the buffer/starvation fields from the adaptive-buffer feature. The HTTP `/v1/stop` and MCP `stop()` manual-interrupt paths now route through `pipeline_state.interrupt()` too (falling back to a direct `player.flush()` only if nothing was speaking per pipeline_state), so a manual stop finalizes the job record the same way.

### Added — Adaptive pre-playback buffer + job retention

- **Job retention closes an observability hole.** `GET /v1/jobs/{id}` for a job launched via `POST /v1/speak` was always `"not found"` — `/v1/speak` enqueues through `server.py`'s own queue-based job registry, which is entirely separate from `http_api.py`'s `_jobs` (used by `/v1/synthesize`, `/v1/audio/speech`, `/v1/vad`). `GET /v1/jobs` and `/v1/jobs/{id}` now merge both registries. `server.py`'s `_prune_jobs()` is now time-based (`JOB_RETENTION_SECONDS = 600`) instead of a hard `MAX_JOBS=20` count cap, so a finished job's full metrics stay queryable for at least 10 minutes; `MAX_JOBS` (now 500) is kept as a safety net against unbounded growth. The `job_id` format is unchanged.
- **Two-loop adaptive pre-playback buffer.** Under GPU contention (measured: 1.26x wall-clock synthesis time with one concurrent local-LLM call sharing Metal with TTS) the drain thread's fixed startup buffer could empty mid-utterance. `adaptive_player.py` now sizes `initial_buffer_ms` per job (feedforward) from mod3's own recent chunk-deficit telemetry (EMA of synth_time/audio_duration), optionally corroborated by a ≤100ms-timeboxed probe of the LMS lane (`GET http://localhost:1234/v1/models`; a timeout is read as "assume quiet"). Defaults: quiet 150ms, elevated 500ms, heavy capped at 2000ms. During playback, a true starvation event (buffer empty while more chunks are still expected) grows the target buffer and holds output silent until it rebuilds — the target never shrinks mid-utterance. `initial_buffer_ms`, `final_buffer_ms`, and `starvation_count` are recorded in the job's metrics. The policy math (`compute_initial_buffer_ms`, `grow_target_buffer_ms`, etc.) is pure and unit-tested independently of the audio runtime. Env var `MOD3_ADAPTIVE_BUFFER_PROBE=0` disables the LMS probe (falls back to the primary telemetry signal only). Requires a daemon restart to take effect.

### Fixed — Barge-in signal now expires instead of holding the gate forever

- **A stuck `user_speaking_start` signal no longer blocks `speak()`/`/v1/speak` indefinitely.** `_bargein_user_recording()` (server.py) and the legacy file watcher's interrupt branch both read `/tmp/mod3-barge-in.json` with no staleness check: whatever `event` the file last recorded was treated as still-true, forever. On 2026-07-07 the mic/VAD writer in `inbound.py` died mid-utterance without ever writing `user_speaking_end`, leaving the file pinned at `user_speaking_start` for roughly three days — every `speak()` request in that window was silently held with "User is currently speaking," and the operator heard no voice output until noticing and deleting the file by hand. Both readers now treat a `user_speaking_start` event older than `MOD3_BARGEIN_SIGNAL_TTL_SEC` (default 120s) as idle, parsing the signal's own `timestamp` field (falling back to file mtime if it's missing or unparseable). 120s rather than a value closer to `inbound.py`'s ~2s VAD-refresh cadence because not every producer refreshes continuously — the SuperWhisper-family producers write the start event once per utterance and lean on their own internal 150s staleness check, and real dictation there legitimately runs 60s+. Missing/corrupt-file-is-idle behavior is unchanged.

### Fixed — Pinned ML-critical dependencies to prevent fresh-install drift

- **`requirements.txt` now pins `transformers`, `mlx-lm`, and `mlx-audio` to exact known-working versions, and declares `torchaudio` explicitly.** A deploy-worktree cutover attempt found that a fresh install from the previously-unpinned `requirements.txt` resolved `transformers` to a version that breaks `mlx_lm`'s module-level `AutoTokenizer.register("NewlineTokenizer", ...)` call (`AttributeError: 'str' object has no attribute '__module__'`), which took down the chatterbox-turbo model-load path entirely (HTTP 500 on `/v1/speak` and `/v1/synthesize` for that voice). Separately, `torchaudio` — required by `vad.py`'s silero-vad path — was never declared in `requirements.txt` at all, so a fresh install silently reported `modalities.vad: false` instead of failing loudly. Both are now pinned to the versions verified working in the long-running dev venv (`transformers==5.9.0`, `mlx-lm==0.31.3`, `mlx-audio==0.4.3`, `torchaudio==2.10.0`). Added `tests/test_dependency_sanity.py` as a cheap, network-free regression guard (bare-import checks) so a future unpinned-floor drift fails CI instead of surfacing as a live-service HTTP 500.

### Added: per-packet VAD confidence endpoint

- **`POST /v1/vad/confidence` adds a lightweight per-packet VAD check for barge-in loops (Discord voice, etc.) that don't need full utterance-level detection.** Accepts a raw 512-sample (16kHz) or 256-sample (8kHz) int16 PCM frame and returns `{confidence, available, latency_ms}` in under 5ms via the vendored ONNX Silero path (no torch dependency), versus `/v1/vad`'s full WAV-upload torch pipeline. `available=false` when onnxruntime isn't installed; listed in `GET /capabilities` as `vad_confidence`. A follow-up fix in the same window corrected a Silero output-shape bug (a `(1,1)` array where a scalar was expected) that had silently zeroed every confidence reading despite `available=true`. (#128)

### Fixed — Honest CI: unmasked test suite, pre-existing failures fixed or skipped

- **The test suite no longer runs masked.** Several pre-existing failures were being hidden from CI signal; they're now fixed or explicitly skipped with a reason, so a green run means the suite actually passed. Covers: CSRF-guarded `TestClient` calls now override the `Host` header so the CSRF check sees the expected origin; the STT pool-isolation assertion uses a local executor instead of depending on ambient global state; the `mlx_lm` tokenizer regression guard is skipped when `mlx` is unavailable in the environment; a TTS-model-dependent test is skipped in the Import & Unit Tests job where the model isn't loaded; the disconnect-bot websocket path sends a close frame both before and after the RTVI handshake so shutdown doesn't hang mid-test.

### Fixed: VAD mic idle-release and barge-in gate retry

- **The mic no longer stays open indefinitely, and a single stale-signal check no longer holds voice output forever.** `InboundPipeline` previously acquired the capture device once at boot and never released it; a server observed at 3.9 days uptime had held the mic open the whole time. It now releases the device after `MOD3_MIC_IDLE_RELEASE_SECONDS` of no detected speech (default 300s; 0 restores the old always-open behavior) and re-acquires on demand. Separately, `speak()` and `/v1/speak` checked the barge-in signal file exactly once and returned a terminal "held" response with no retry, which held all voice output for an entire 12-hour flight on one VAD false positive. `check_bargein_gate()` now retries up to `MOD3_GATE_RETRY_ATTEMPTS` times (default 3, roughly a 10s budget) and surfaces an explicit escalation message once a session has been held twice in a row, without ever forcing playback over a real speaker. (#134)

### Added: TTS model idle-unload

- **`MOD3_TTS_IDLE_UNLOAD_SECONDS` opts a background watcher into evicting loaded TTS models from memory after a configurable idle period,** freeing RAM and Metal GPU cache. Disabled by default (env unset or `0`); the next synthesis call transparently reloads from disk through the existing lazy-singleton path. (#133)

### Added: RTVI audio streaming from the speak/drain path

- **`mod3_speak` and `/v1/speak` now deliver audio to `/ws/audio` subscribers as RTVI 1.3.0 frames (`bot-tts-started`, `bot-tts-audio`, `bot-tts-stopped`), not just to the local speaker.** The drain path emits per-chunk over the websocket alongside local `AdaptivePlayer` playback; the closing frame is guaranteed exactly once whether the job finishes normally, is barged in on, or throws mid-synthesis. No subscriber means no change in behavior. (#131)

### Fixed: drain threads hardened against BaseException

- **`SpeechQueue` and `ChannelQueue` drain loops previously caught only `Exception`, so a `SystemExit`, `KeyboardInterrupt`, or `MemoryError` could kill the drain thread while leaving its running/draining flags stuck true.** New jobs then queued forever with `active_jobs=0`, the traced cause of a stale `queue_depth` counter seen in production. Both loops now reset their state in a `finally` block on every exit path, normal or abnormal. (#129)

### Added: seat fan-out events for streamed audio and barge-in

- **Two new seat event types fan out during speech synthesis: `tts_chunk` (OGG/Opus-encoded, monotonically indexed, `is_final` on the last chunk) and `bargein` (fired whenever `pipeline_state.interrupt()` runs), letting session seats cancel pending deliveries.** Both are no-ops when no session_id is present or fan-out fails. (#127)

### Added: speech-to-text endpoint

- **`POST /v1/transcribe` accepts an audio upload (WAV directly, or OGG/MP3/M4A via ffmpeg) and returns a transcription from `mlx-community/whisper-large-v3-turbo`, lazy-loaded on first request and filtered through the existing hallucination check.** This is what lets Hermes route its STT pipeline through mod3 instead of running faster-whisper directly. (#126)

### Added: OGG/Opus output format

- **`/v1/synthesize` and the `mod3_speak` MCP tool (via `skip_playback=True`) can now return OGG/Opus-encoded audio instead of WAV/PCM,** RFC 7587/7845 compliant (resampled to 24kHz, `codecs=opus` on the content type so Telegram renders it as a voice bubble). `/v1/speak` stays WAV-only; the playback queue never honored a format parameter. (#125)

### Added: kernel channel-session sync on seat register and revoke

- **Seat registration and revoke now fire best-effort callbacks to the kernel's `/v1/channel-sessions/*` routes, carrying the existing session_id and iss/sub identity claims,** so the kernel's own session bookkeeping stays in sync with mod3's. An unreachable kernel never blocks the seat operation itself. (#118)

### Fixed: inbound voice pipeline restored

- **The mic-to-VAD-to-STT pipeline had silently stopped starting on every default deploy since an earlier PR removed the symbols it imported from `server.py`,** and the broad `except` around pipeline startup swallowed the resulting `ImportError` without logging (`/health` just read `vad: false`). Rewired onto `SeatRegistry.fan_out_all()`, the delivery path already used elsewhere, so voice input reaches Claude Code hosts again. (#123)

### Fixed: seat and session lifecycle hardening

- **SSE stream teardown now revokes the seat, not just the channel, closing a leak where zombie seats kept accumulating queued messages after their connection dropped** (the per-seat queue is also now bounded, drop-oldest past 1024). A double-failure path in the Ollama local-fallback provider (kernel error plus a non-2xx Ollama response) previously bubbled a raw HTTP error into a voice turn instead of a spoken one; it's now caught and returns gracefully. (#122)
- **`SessionRegistry` gained a periodic reaper that prunes sessions idle past a TTL (default 600s) with no live SSE stream,** tied to actual connection liveness rather than the graceful-DELETE hook a crashed or killed client never runs. Dashboards had been accumulating 100+ stale sessions over days; `"main"` is never reaped. (#121)

### Fixed: voice inference stays on the local floor when Eclipse is down

- Hardened the local-fallback provider path so a downed remote node degrades voice inference to the local floor instead of failing the turn. (#120)

### Changed — `mod3_speak` mirrors to dashboard chat by default

- **Speech now appears in the chat panel + per-session history.** The MCP `mod3_speak` tool previously only hit `POST /v1/speak`: audio played, but the dashboard chat pane stayed empty and nothing landed in the per-session ring buffer. Operators going back to reread a conversation later saw their own prompts but no agent responses for any turn delivered as speech. The tool now POSTs the spoken text to `/v1/dashboard-chat` (as an `assistant` message under the seat's `session_id`) *before* invoking `/v1/speak`, so the transcript lands at roughly the same time audio begins. The chat mirror is best-effort — a failed POST logs and continues, so audio playback is never blocked. New `post_to_chat` parameter (default `True`) opts out for non-conversational audio (system sounds, UI cues).

### Fixed — Channel client survives mod3 restarts

- **Reconnect the SSE stream + re-register the seat after disconnect.** `ChannelClient.run_sse_subscription` was a one-shot: on any stream disconnect (mod3 restart, network blip, server-side close), the coroutine returned silently and the seat was lost until the MCP child was respawned. Symptom: every mod3 restart made attached Claude Code sessions invisible to the dashboard sidebar even though the channel client process was still alive. The subscription now runs in a reconnect loop: on disconnect it re-POSTs to `/v1/sessions/<id>/seats` to get a fresh seat_id (mod3 wiped its registry on restart, so the old one is invalid) and resumes streaming. Backoff is exponential (1s → 2s → 4s, capped at 30s) and resets to 1s after a healthy stream survives ≥5s; the inner register loop retries until mod3 responds rather than thrashing the stream while mod3 is still down. `asyncio.CancelledError` still exits cleanly so shutdown paths are unaffected.

### Fixed — Dashboard version pills

- **Render the mod3 version alongside the kernel version, and stop double-prefixing the kernel `v`.** The kernel `/health` endpoint emits `"version": "v0.10.0"` (with a leading `v`); mod3 `/health` emits `"version": "0.7.0"` (no prefix). The previous pill renderer hardcoded `' v' + d.version`, so the kernel pill displayed `kernel vv0.10.0`, and the mod3 pill hardcoded the bare label `mod3` with no version at all. A new `formatVersion()` normalizer strips a leading `v` before re-prefixing; both pills funnel through it, so the header now shows `kernel v0.10.0` and `mod3 v0.7.0` consistently.

### Added — Dashboard session switching + chat persistence across refresh

- **Per-session chat history.** New `message_store.py` keeps a per-session ring buffer (default 500 messages) of `{id, session_id, role, content, input_type, ts}` records. `POST /v1/sessions/{id}/messages`, `POST /v1/sessions/broadcast-message`, and `POST /v1/dashboard-chat` all append into the store under the resolved session id; broadcasts without a target attribute fall back to the `"main"` session. New endpoint `GET /v1/sessions/{id}/messages?limit=N` (default 100, max 1000) returns the recent slice for hydration. RAM-only; restart wipes the buffer (parity with seats / SessionRegistry / chat-flow log).
- **Sidebar click selects the active chat session.** The sidebar click handler in `dashboard/index.html` is no longer a stub — clicking a session row sets `window.__activeChatSessionId`, persists it to `localStorage.mod3.activeChatSession`, calls `acpTransport.sessionResume(sid)` when ACP transport is available, fetches `GET /v1/sessions/<sid>/messages`, and re-renders the chat pane. Outbound sends from the chat input now route to the active session by default (explicit `@<sid>` mentions still win); when the active session is the dashboard's own, the local AgentLoop continues to handle the reply.
- **Refresh hydration.** On page load, after the dashboard's own session registers, the active chat session is restored from `localStorage.mod3.activeChatSession` (falling back to the dashboard's own session_id when nothing is saved). The chat pane re-renders from `GET /v1/sessions/<sid>/messages` so a page refresh no longer drops the conversation. The dashboard's own participant identity (`localStorage.mod3.sessionId`) is unchanged by the active-session selection — switching sessions only changes what this tab is *sending to*, not who it *is*.

### Fixed — Dashboard sidebar enumerates seat-bearing sessions

- **Mirror seat-bearing sessions into SessionRegistry.** `POST /v1/sessions/{id}/seats` now idempotently registers the session in the voice-TTS `SessionRegistry` after the seat lands. Before this fix only the startup-seeded `"main"` session showed up in `GET /v1/sessions`; Claude Code channel clients bound to their own session UUID per PR #103 were attached at `/v1/sessions/{id}/seats` but invisible to the sidebar, which reported "No active sessions" with a live channel client. `SessionRegistry.register` preserves existing voice allocation on repeat calls, so multiple seats under the same session_id don't reshuffle the voice. Mirror failures log a warning and never break seat registration.

### Added — Dashboard ↔ Claude Code channel binding

- **Bind mod3 seats to the real Claude Code session_id.** Each Claude Code session now registers a distinct channel-client seat instead of all sessions collapsing into the legacy `"main"` sentinel. `clients/channel_client.py` reads `~/.claude/sessions/<parent-pid>.json` at startup to discover the harness session_id; the kernel's `/v1/claude-code/spawn` flow continues to pass `--session <id>` directly via a temp `.mcp.json`. Dashboard `sessions.html` posts `mod3:claude-code-spawned` (same-origin) to its opener after spawning; `index.html` listens, polls `/v1/sessions/<id>/seats` until the seat appears, then calls `acpTransport.sessionResume(session_id)` to bind its ACP connection. New `AcpTransport.sessionResume(sessionId)` method wraps the existing server-side `/ws/acp` `session/resume` handler. (#103)
- **Hotfix: state-file resolution.** `${CLAUDE_CODE_SESSION_ID:-main}` env-substitution in `mcp.channel.json` doesn't fire because that variable isn't in the parent claude process's env at MCP-spawn time. Replaced with `_resolve_claude_session_id()` walking up the parent PID chain to read Claude Code's own `~/.claude/sessions/<PID>.json` state file. (#105)
- **Hotfix: resolver polls for state file (startup race).** Claude Code writes its `~/.claude/sessions/<PID>.json` AFTER spawning MCP children — the resolver's one-shot check lost the race and fell back to `"main"`. Now polls with a 10s deadline + 100ms interval; PPID chain is snapshotted once. Live-fire test simulates a 200ms-late state file to catch regressions. (#106)
- **Hotfix: longer poll timeout + live-PID fallback.** Observed gap on this machine was ~35 seconds, not <10 — bumped `poll_timeout_s` default from 10s to 60s. Added a fallback that picks the most-recently-modified `~/.claude/sessions/*.json` whose PID is still alive (kill -0 check) when the parent-chain ancestry doesn't trace back to claude (bg-spare re-parenting / wrapper-script interposition). Liveness, not mtime, is the freshness signal — Claude Code only updates the state file on status changes. (#107)
- **Diagnostic: startup log at `~/.mod3/channel-client-startup.log`.** channel_client now writes a one-line entry on every launch with timestamp, pid, ppid, resolved session_id, and source (env / parent-chain / fallback / default). Auto-trims to last 200 lines. Lets operators post-mortem what the resolver picked — Claude Code captures the MCP child's stderr internally and it's not easily retrievable. Best-effort write; never raises. (#108)
- **Hotfix: parse `--resume <id>` from parent argv (placeholder rewrite bug).** Claude Code writes a *placeholder* session_id to `~/.claude/sessions/<PID>.json` at startup, then rewrites the file with the actual `--resume` target ~28 seconds later. The state-file resolver read the placeholder and registered the seat at the wrong session_id. Fix: walk the parent-chain looking for a claude process and parse `--resume <id>` from its argv first — argv is set at exec() and never changes, immune to the rewrite. Falls through to the state-file resolver for non-resume launches. (#109)

### Security

- **`_localhost_csrf_guard` middleware now rejects state-changing requests (POST/PUT/PATCH/DELETE) whose Host header isn't loopback, and checks Origin against an allowlist when the browser sends one,** closing a DNS-rebinding path to the localhost daemon. Default bind address moved from `0.0.0.0` to `127.0.0.1` (opt into LAN exposure explicitly with `--host 0.0.0.0`). Read-only methods and `/health` are deliberately not gated; configurable via `MOD3_ALLOWED_ORIGINS`. (#117)
- **Pre-existing auth posture surfaced.** During review of #103, the security review flagged that `/v1/sessions/{id}/seats`, `/v1/sessions/broadcast-message`, and `/v1/claude-code/spawn` have no auth or CSRF protection. The findings predate this work (the localhost-only design assumed no untrusted browser context) but the dashboard wiring now exercises these endpoints from same-origin scripts. Phased mitigation tracked in #104.

## [0.7.0] - 2026-05-19

### Added — Wave-6b session identity claims

- **`iss`/`sub` on seat registration** -- `register_session` now emits `presence.started` with issuer and subject fields set from the CogOS identity context. (#89)
- **Multi-identity harness binding** -- a single harness seat can now carry multiple identity claims (user + agent simultaneously); `seats.py` updated with `user_iss`/`user_sub` and `agent_iss`/`agent_sub` pairs. (#91)

### Added — Voice subsystem

- **`VoiceProfile` schema adoption** -- `voice_profile_schema.py` is the canonical schema layer; mod3 now reads voice config from CogOS identity projection events via `IdentityVoiceProfile`. `cog://voices/*` URIs are resolved to the local registry under `~/.mod3/voices/`. (#90)
- **URI resolver docstring fix** -- corrected stale comment on `resolve_voices_uri` that referenced the old field names. (#97)

### Added — Channel pipeline composability

- **`ChannelMode` + composable stage graph** -- `channels.py` introduces `ChannelMode` (passthrough / transcribe / agent) and a directed acyclic stage graph; pipeline stages are composed at startup rather than hard-wired. (#92)
- **`@register_stage` intentional stages** -- `inbound.py` extracts the intentional pipeline stages (VAD, STT, intent classification) into `@register_stage`-decorated classes so the stage graph can enumerate and wire them automatically. (#98)

### Added — ACP transport

- **`session/list`, `session/load`, `session/resume`, `authenticate`** -- the four missing ACP methods are now wired in `http_api.py`; mod3 is a conforming ACP server for session lifecycle. (#100)
- **Auto-create main session + `session/update` wire-shape fix** -- a `main` session is created at startup so clients can connect immediately; the `session/update` request shape now matches the ACP spec. (#101)

### Added — SSE bridge for identity-projection events

- **`/v1/events/identity-projection` SSE endpoint** -- `bus_bridge.py` wires a Server-Sent Events handler for CogOS identity-projection events so the dashboard and channel clients receive voice and identity updates in real time. (#99)

## [0.6.0] - 2026-05-16

### Added — RTVI 1.3.0 audio-plane compatibility

- **`rtvi-client` seat type** — `VALID_CLIENT_TYPES` extended to accept RTVI 1.3.0 clients. (#75)
- **`/ws/audio/{session_id}` client-ready/bot-ready handshake** — RTVI protocol negotiation on WebSocket connect. (#77)
- **Raw-audio inbound routing** — `client-audio` frames routed to VAD/STT pipeline. (#80)
- **RTVI transcript and speaking-lifecycle emission** — `bot-speaking`, `bot-stopped-speaking`, and `transcript` server events emitted on the audio plane. (#78)
- **`disconnect-bot` graceful close** — server-side teardown on client disconnect-bot message. (#76)
- **Full-session RTVI 1.3.0 integration test** — T1–T6 coverage for the audio WebSocket path. (#81)
- **G2 executed decision record** — RTVI 1.3.0 B+ selected over LIVEKIT/Pipecat-native alternatives. (#79)

### Added — Smart Turn v3 end-of-utterance detection

- **Smart Turn ONNX v3.2-cpu vendor** — replaces v1 model; CoreML execution provider wired for both ONNX sessions on Apple Silicon. (#74)
- **Smart Turn v3 integration** — end-of-utterance detector replaces legacy silence-threshold heuristic. (#73)
- **`voice_confidence` wrapper** — shared confidence accessor for VAD and RTVI; decision doc + weight fetch (F2, F4). (#72)
- **Rung-1 Silero VAD + Smart Turn vendor scaffold + RTVI scoping doc** (F1, F3, G1). (#71)

## [0.5.0] - 2026-05-15

### Default voice

The CogOS-driven speech default is now `eng_uk_m_davids` (Chatterbox-Turbo cloned British male, "David S"). This replaces the prior default `bm_lewis`. Prosody is more natural under the Chatterbox-Turbo stack; the voice ID is stable and registered in the voice profile registry.

### Added — Channel architecture (ADR-082)

- **Session-aware communication bus** — sessions are first-class citizens on the event bus; per-session routing replaces broadcast fan-out. (#acad6f1)
- **ACP transport endpoint** — `/ws/acp` accepts connections from ACP-compatible clients and routes prompts through the kernel cycle via `cogos_agent_bridge`. (#31, #34)
- **Claude Code channel via separated channel-client** — dedicated channel-client module (supersedes the in-process bridge approach from #39). (#40)
- **Single-path channel routing** — removed superseded bridge, fallback, and mcp_shim layers; one canonical routing path through the bus. (#42)
- **ACP client e2e flow tests** and ACP-client pattern documentation. (#44)
- **Sessions browser** — dashboard UI panel showing ACP-client projects and sessions. (#43)
- **Echo suppression** — originating seat excluded from fan-out to prevent self-echo. (#45)

### Added — Dashboard surface

- **Three-column shell** — skeleton layout with sessions sidebar, main panel, and Settings / Traces / Debug side panel. (#d0ed93b)
- **Settings panel** — transport, voice, and audio controls in the settings tab. (#36)
- **Three-tab page nav** — Dashboard / Console / Voice Lab. (#33)
- **Real-time trace panel** — phase timeline with kernel sub-spans. (#56)
- **Hierarchical span tree** — agent-prism-inspired nested span display replacing the flat Gantt. (#a80d72a)
- **Debug tab bus event stream** — live bus events in the Debug tab. (#66)
- **Providers/available endpoint** — dynamic backend selector populated from `/providers/available`. (#ade87e8)
- **Accessibility and keyboard shortcuts** (Wave 3H+I). (#67)
- **Participant panel** and auto-register on page load. (#cff22f3)

### Added — Voice and TTS

- **Queue-aware `POST /v1/speak` endpoint** — HTTP counterpart to the `speak()` MCP tool; returns queue position, estimated wait, and active job state. (#54)
- **Voice profile registry** (Phases 1–3) — cloned voices as first-class voice IDs stored under `~/.mod3/voices/`; voice profile I/O, schema, and profile management. (#21)
- **Unified `output()` MCP tool** — single tool with `mode` ∈ `{audio, text, both}` dispatching to TTS, dashboard chat, or both simultaneously. (#55)
- **`bargein.event` with position tracking** — emitted on TTS interrupt with byte-level position so consumers know how much was spoken. (#57)
- **RTVI 1.3.0 audio envelope** for `/ws/audio/{session_id}` sidecar. (#29)
- **`/ws/audio/{session_id}` WebSocket** for per-session playback routing. (#69dd70d)

### Added — STT and open-mic

- **Continuous voice** — auto-start VAD, barge-in integration, and tunable endpointing for always-on mic capture. (#38)
- **Multi-strategy Whisper dedup** — Z-function, sentence-level, and N-way deduplication strategies to eliminate phrase doubling. (#53)
- **Dedicated STT thread executor** — isolates Whisper inference onto its own `ThreadPoolExecutor` to prevent blocking the async event loop. (#25)

### Added — Observability

- **Structured chat-flow log** — `chat_flow_log.py` captures turn lifecycle; `/v1/logs/chat-flow` endpoints expose the log over HTTP. (#46)
- **Per-phase wall-time instrumentation** — every pipeline phase records wall-clock durations for turn observability. (#51)
- **W3C traceparent injection** — `CogOSProvider` injects a W3C-compliant `traceparent` header; `trace_id` propagated through `chat_flow_log`. (#52)
- **`trace_id` propagation** — trace IDs flow through all phase events: `stt_capture`, `stt_transcribe`, `tts_synthesize`, `tts_playback_start`. (#58, #60)

### Added — CogOS modality node (RFC-0001)

- **Cog-native modality node scaffolding** — `modality.py`, Pipecat integration, and RFC-0001 design doc. (#27)
- **Typed API surfaces** — `schemas.http`, `schemas.ws_chat`, `schemas.ws_audio`. (#28)

### Added — MCP transport

- **HTTP-MCP mount at `/mcp`** — `install_mcp_route()` mounts FastAPI-native streamable-HTTP MCP transport; guarded against double-install. (#c05ed89, #9922d58)
- **`.mcp.json` switched to HTTP transport** — project-level MCP config updated to use the canonical HTTP path. (#f1cc22b)

### Changed

- **FastAPI lifespan migration** (`http_api.py`) — replaced all `@app.on_event("startup")` / `@app.on_event("shutdown")` decorators with a single `@asynccontextmanager` lifespan. Startup order: Kokoro warmup thread, kernel-bus bridge, CogOS agent bridge. Shutdown order: reverse. Eliminates the DeprecationWarning emitted on every boot. (#ba5e8e9)
- **Dashboard inference routed through CogOS kernel** — provider requests go to `/v1/chat/completions` on the kernel instead of in-process MLX. (#49)
- **Voice dropdown populated dynamically** from `/v1/voices`. (#48)
- **Version read from `pyproject.toml`** via `importlib.metadata` instead of a hardcoded constant. (#eebc588)
- **Generic example identifiers** in MCP tool schemas (scrubbed participant-specific examples). (#8, #9)

### Deprecated

- **stdio MCP transport (Phase 1 soft deprecation)** — `python server.py` (no args), `--all`, and `--channel` now emit a `DeprecationWarning` to stderr at boot. The stdio path remains fully functional; no behavior has changed. CLI `--help` text for `--all` and `--channel` notes the deprecation. HTTP-MCP (`python server.py --http`, connect via `/mcp`) is the canonical transport. Tracked in [#11](https://github.com/myrgic/mod3/issues/11); Phases 2–4 (flip default, retire `mcp_shim.py`, remove stdio) are separate future PRs. (#26, #f180d9fb)

### Fixed

- **Queue deadlock + Spark speed `KeyError`** — resolved a deadlock in queue stability and a missing-key error in Spark speed routing. (#20)
- **Kernel health endpoint URL** — corrected the URL used by the dashboard to check kernel health. (#68)
- **Channel-client 404** — `mod3_speak` in channel-client was calling the wrong endpoint; switched to `/v1/speak`. (#69)
- **Trace panel**: `trace_id` grouping, `turn_total` Gantt exclusion, kernel sub-span extraction. (#62)
- **Trace panel**: wall-clock Gantt, expand-state preservation, turn dedup. (#59)
- **Tracing**: propagate `trace_id` to all phase events; trace panel SSE and render. (#58)
- **Output**: `mode='audio'` now also emits text bubble to dashboard. (#61)
- **STT**: suppressed Whisper phrase-doubling via conditioning params and dedup backstop. (#50)
- **Bridge**: subscribes to per-bus SSE endpoint for agent responses. (#35)
- **Dashboard**: cycle trace removal, chat default to `/ws/chat`, ACP spec compliance. (#32)
- **Dashboard**: persist output device, close sink-timing race. (#08d679f)
- **Dashboard**: route WebSocket audio to the selected output device. (#7e441ee)
- **Dashboard**: audio WebSocket buffer must be `ArrayBuffer`, not `Uint8Array`. (#4da2533)
- **Dashboard**: serve `index.html` for `GET /dashboard/` (trailing slash). (#47)
- **Channels**: clean teardown on WebSocket disconnect. (#24)
- **MCP**: start MCP session manager during FastAPI lifespan (was missed on `--http` start). (#180d9fb)
- Various lint and formatting fixes (ruff).

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
This release jumps from `v0.1.0` to `v0.4.0`. An earlier `v0.2.0` tag exists from before the org rename (no GitHub release was created). `v0.3.0` was bumped in `pyproject.toml` and added to the CHANGELOG but never tagged. `v0.4.0` captures everything since the last released version (`v0.1.0`).

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

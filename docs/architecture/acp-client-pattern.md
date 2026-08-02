# ACP-Client Pattern: mod3 as Claude Code Driver

## What this is

The ACP-client pattern makes mod3 the **front-door** for Claude Code rather
than a peripheral surface attached to an already-running session. mod3 lists
projects and sessions, spawns Claude Code subprocesses on demand, and the
spawned process registers back as a seat on the mod3 dashboard.

This is distinct from the **channel pattern** (PR #40), where Claude Code
starts first and mod3 is attached as a channel client.

| Pattern | Driver | Backend | Entry point |
|---------|--------|---------|-------------|
| Channel | Claude Code session | mod3 = voice/text surface | `claude --channels server:mod3` |
| ACP-client | mod3 dashboard | Claude Code = inference backend | `http://localhost:7860/dashboard/sessions.html` |

Both patterns converge at the same seat-registration mechanism
(`POST /v1/sessions/{id}/seats`). The difference is who initiates.

---

## Kernel endpoints (CogOS)

The kernel (port 6931) exposes three endpoints that the ACP-client pattern depends on:

```
GET  /v1/claude-code/projects
GET  /v1/claude-code/projects/{project}/sessions
POST /v1/claude-code/spawn
```

See the cogos `internal/engine/serve_claude_code.go` handler for implementation
details and the `CLAUDE_PROJECTS_DIR` env-var override.

---

## mod3 surface

### Dashboard page

`http://localhost:7860/dashboard/sessions.html` — served via
`GET /dashboard/{filename}` in `http_api.py`. Shows:

- Project grid: all project directories under `~/.claude/projects/`, sorted by
  last activity. Click to drill into sessions.
- Session list: per-project `.jsonl` files with `first_prompt_summary`,
  turn count, message count, and `last_modified` timestamp.
- **Resume** button: calls `POST /v1/claude-code/spawn` with the selected
  `session_id`.
- **Mount New Session** / **New Session in Project** buttons: call
  `POST /v1/claude-code/spawn` without a `session_id`.

### Spawn proxy

`POST /v1/claude-code/spawn` in mod3 proxies to the kernel endpoint.
This avoids browser CORS issues (dashboard at port 7860 calling kernel at 6931).
The `COGOS_KERNEL_URL` env-var overrides the default `http://localhost:6931`.

---

## Spawn lifecycle

```
1.  Operator clicks "Resume" on sessions.html
2.  Browser POST /v1/claude-code/spawn → mod3 (port 7860)
3.  mod3 proxies → kernel POST /v1/claude-code/spawn (port 6931)
4.  Kernel writes temp .mcp.json (points channel_client.py at mod3,
    passing --session <claude_session_id> directly via args)
5.  Kernel calls ClaudeCodeProvider.SpawnBackground(...)
6.  Kernel returns { process_id, session_id, status: "spawned", spawned_at }
7.  mod3 returns the kernel response verbatim (201)
8.  sessions.html posts {type:"mod3:claude-code-spawned", session_id, ...}
    to window.opener (same-origin guarded). Dashboard banner shows
    "Session resumed (process <id>)"
9.  Claude Code subprocess starts, loads .mcp.json
10. channel_client.py resolves its session_id (see "Channel-client
    session resolution" below) and POSTs to /v1/sessions/{id}/seats
11. Seat appears at /v1/sessions/{id}/seats
12. dashboard/index.html's message listener receives the spawn
    postMessage, polls /v1/sessions/{id}/seats until the seat appears
    (up to 15s), then calls acpTransport.sessionResume(session_id)
13. Dashboard's ACP transport is now bound to that session — subsequent
    chat prompts fan to the spawned subprocess's seat via session/prompt
14. Subprocess processes the prompt; replies via mod3_dashboard_post
    MCP tool, which appears in the dashboard chat panel
```

Steps 9-13 happen asynchronously. The spawn response confirms the
subprocess was started; the dashboard postMessage bridge plus seat-poll
ensures the dashboard's ACP connection ends up bound to the spawned
session's seats without operator intervention. The bridge was added in
mod3 PR #103; without it, the operator would have to manually invoke
`acpTransport.sessionResume(...)` from DevTools or use `@<session_id>`
chat-mentions.

### Channel-client session resolution

`clients/channel_client.py` resolves its mod3 session_id in this order:

1. Explicit `--session <id>` arg (set by the kernel spawn path above)
2. `MOD3_SESSION_ID` env var
3. **Parse `--resume <id>` from any claude ancestor's argv** — covers
   the manual `claude --dangerously-load-development-channels server:mod3
   --resume <id>` launch path. argv is exec()-set and never changes,
   immune to the state-file rewrite below.
4. `~/.claude/sessions/<parent-pid>.json` — Claude Code writes this
   state file at startup with the harness session_id. The resolver
   walks the parent-chain and polls up to 60s (the file is written
   ~30s after MCP children spawn).
5. Most-recently-modified `~/.claude/sessions/*.json` whose owning
   PID is still alive (kill -0 check) — handles bg-spare re-parenting
   where the ancestry doesn't trace back to claude.
6. `_DEFAULT_SESSION_ID = "main"` — last-resort fallback for non-Claude
   MCP clients.

The first resolved value wins. A diagnostic log line is appended to
`~/.mod3/channel-client-startup.log` on every launch with the resolved
session_id and source — useful for post-mortem when seat registration
lands on an unexpected session.

---

## Testing

```bash
# Unit + integration tests covering the spawn proxy and dashboard static delivery
cd mod3
PYTHONPATH=. pytest tests/test_acp_client_flow.py -v

# Curl smoke tests (kernel must be running on port 6931)
curl http://localhost:6931/v1/claude-code/projects
curl http://localhost:6931/v1/claude-code/projects/-Users-yourname/sessions

# Spawn a new session (kernel must be running)
curl -X POST http://localhost:7860/v1/claude-code/spawn \
  -H 'Content-Type: application/json' \
  -d '{"project": "-Users-yourname"}'

# Resume a specific session
curl -X POST http://localhost:7860/v1/claude-code/spawn \
  -H 'Content-Type: application/json' \
  -d '{"project": "-Users-yourname", "session_id": "<session_uuid>"}'
```

---

## Operator Resume flow (manual verification)

After the three PRs are merged and both services are running:

1. Open `http://localhost:7860/dashboard/sessions.html`
2. Confirm project list loads (requires kernel on port 6931)
3. Click a project — confirm session list with preview cards
4. Click **Resume** on any session
5. Banner shows "Session resumed (process ...)"
6. Switch to `http://localhost:7860/dashboard`
7. Watch the Participants pill — count should increment as the channel client registers its seat

**Do not click Resume from an automated context** — it starts a real Claude
Code subprocess with real token spend.

---

## Env-var reference

| Var | Default | Effect |
|-----|---------|--------|
| `COGOS_KERNEL_URL` | `http://localhost:6931` | mod3 spawn proxy target |
| `CLAUDE_PROJECTS_DIR` | `~/.claude/projects` | kernel project listing root |

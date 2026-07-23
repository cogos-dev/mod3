# ACP Spec Boundary — mod3 session methods

This file documents the structural mapping between ACP's optional session-management
methods and mod3's substrate invariants. Its purpose is to make boundary conditions
explicit so native replacement of any mod3-internal layer stays possible without
rediscovery.

## ACP wire-level invariants (what the spec requires)

- `session/list` response: `{sessions: [{sessionId, ...metadata}]}`
  — advertised via `agentCapabilities.sessionCapabilities.list: true`
- `session/load` response: `{sessionId, state: {...}}`
  — advertised via `agentCapabilities.loadSession: true`
- `session/resume` response: `{sessionId}`
  — advertised via `agentCapabilities.sessionCapabilities.resume: true`
- `authenticate` request: `{methodId: string}`
  — called by client when `authMethods` is non-empty; no-op when `authMethods: []`
- All four are JSON-RPC 2.0 requests (carry `id`, expect a response).

## mod3 substrate invariants at this boundary

- **Session** in mod3 is a `SessionChannel` in the `SessionRegistry` — a voice-output
  coordination object with `session_id`, `participant_id`, `state`, and device preference.
  This is a TTS/audio session, not a Claude Code conversation session.
- **Seat** is a channel-client slot (e.g., a Claude Code subprocess) attached to a
  `SessionChannel`. Multiple seats can attach to one session.
- **Claude Code session** is a kernel-side entity tracked by the CogOS kernel. mod3's
  `/v1/claude-code/spawn` proxies creation to the kernel at `COGOS_KERNEL_URL`.
  mod3 does not own conversation state — the kernel does.
- **ACP `sessionId`** (issued by `session/new`) is a mod3-local identifier
  (`mod3-acp-<hex>`), distinct from both `SessionChannel.session_id` and the
  kernel's Claude Code session ID. It scopes the per-connection ACP state (`_sessions`
  dict on the WebSocket handler).

## Mapping: ACP method → mod3 implementation

| ACP method | mod3 implementation | Notes |
|---|---|---|
| `session/list` | `SessionRegistry.list_serialized()` | Lists TTS/audio sessions; not Claude Code conversations |
| `session/load` | `SessionRegistry.get(session_id)` → `to_dict()` | Returns mod3 session state |
| `session/resume` | Create/re-use ACP session scoped to the named session_id | Reconnects ACP handle; no kernel spawn |
| `authenticate` | No-op success — `authMethods: []` means no auth required | Returns `{success: true}` |

## What `session/list` does NOT enumerate

`session/list` returns mod3's internal TTS session channels, not Claude Code conversation
history. The sessions.html dashboard's "Resume" button works at a different layer — it
calls `/v1/claude-code/spawn` with a project + session_id to re-attach a channel client.

If a future consumer needs ACP `session/list` to enumerate kernel Claude Code sessions,
that would require a proxy to `COGOS_KERNEL_URL/v1/sessions` — a distinct integration
point not covered here. The current mapping is deliberate and explicitly scoped to
mod3's own session registry.

## Capability advertisement

The `InitializeResult` is updated to advertise these capabilities:

```python
sessionCapabilities = {
    "list": True,
    "resume": True,
}
```

And `loadSession: True` on `AgentCapabilities`.

## Deferred (out of scope)

- File-system callbacks (`fs/read_text_file`, `fs/write_text_file`)
- Terminal methods
- `session/request_permission`
- `session/close`
- `session/set_mode`
- Dashboard UX redesign consuming these methods (SessionConfigOption-driven backend selector)
- Dropping `/ws/chat` transport
- Auto-create of `main` session (separate orchestrator)
- `session/update` wire-shape fix (separate orchestrator)

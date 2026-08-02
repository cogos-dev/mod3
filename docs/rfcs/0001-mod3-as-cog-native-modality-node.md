# RFC-0001: mod3 as a Cog-Native Modality Node

| Field | Value |
|---|---|
| Status | Proposed |
| Author | @chazmaniandinkle |
| Date | 2026-05-13 |
| Commit | `5be113c` (schemas baseline) |
| Relates | ADR-062 Recursive Node Architecture, RFC-0007 named-provider dispatch, RFC-0008 Inference Control Plane, Channel-Provider Interface design |

## Summary

Mod3 is currently an audio inference server tangled with local agent logic. This
RFC reframes it as a Cog-native modality node: a Python subprocess that provides
a standardized, schema-governed interface for whichever modalities it supports.
Its modalities are providers that map to a canonical schema for their particular
modality type. The kernel's `pkg/modality` layer (`types.go`, `wire.go`) is the
Go side of this contract; `mod3/schemas/` (committed at `5be113c`) is the Python
mirror. This RFC proposes a four-layer architecture, specifies the D2 wire
protocol conformance, describes the channel-with-modalities framing, and opens
three decisions for operator ratification.

## 1. Motivation

### The current state

Mod3 was built to serve one use case: voice. It has grown to handle TTS
synthesis, VAD, STT, session management, a WebSocket dashboard, an agent loop,
and a CogOS kernel bridge, all in one Python server process. The `agent_loop.py`
and `cogos_agent_bridge.py` modules implement local agent logic that duplicates
what the CogOS kernel is supposed to do. The result is a system with two planes
tangled: audio inference (CPU/GPU-bound, latency-sensitive, language-independent)
and agent control (stateful, session-scoped, architecture-dependent).

### The reframe

Mod3 is an audio inference provider. The analogy is exact: Anthropic is an
inference provider for Claude; mod3 is an inference provider for audio. It
synthesizes speech (TTS), detects speech (VAD), and transcribes speech (STT).
It does not need to know about sessions, routing, or agent identity. Those are
the kernel's concerns.

Under this reframe, mod3 implements `pkg/modality.Module` for the Voice modality.
The kernel spawns mod3's worker subprocesses, routes cognitive intents to the
encoder, receives cognitive events from the decoder, and manages sessions. Mod3
answers synthesis and transcription requests over the D2 wire protocol. Nothing
more.

### Why this matters

The untangling produces a clean composability story. A modality node that knows
only about audio can be replaced with a better audio stack without touching the
agent. An agent that knows only about cognitive intents and events can be ported
to a different modality node without touching audio code. The kernel routes
between them.

This is exactly the separation ADR-062 describes at the architectural level: a
node advertises its capabilities on its card; the parent kernel routes through
the card. Mod3's card says it can encode Voice intents (TTS) and decode Voice
events (STT with VAD gating). The kernel routes accordingly.

## 2. Design

### 2.1 Four-Layer Architecture

Mod3 organizes along four layers. Each layer has a distinct scope and a distinct
set of consumers.

**Layer 1: Native inference primitives.**
The raw output shapes of the underlying ML models. For audio, these are
`AudioChunk`, `TranscriptResult`, `PartialTranscript`, and `VADResult`, defined
in `schemas/primitives.py`. These types are Python-internal; they do not cross
the subprocess wire directly. They are what the MLX engine, Silero VAD, and
WhisperDecoder yield before any encoding.

**Layer 2: Pipecat-compatible service wrappers.**
Optional integration layer for inference testing and pipeline composition.
`Mod3TTSService` (in `integrations/pipecat/tts_service.py`, committed on branch
`wave/2026-05-13-mod3/pipecat`) subclasses Pipecat's `TTSService` and wraps
`engine.generate_audio()`. This layer does not require Pipecat to be installed;
it is gated behind an optional dependency (`pip install mod3[pipecat]`).

The Pipecat layer serves two purposes: it gives integration engineers a familiar
interface for testing mod3 engines in isolation, and it validates that mod3's
output shape is structurally identical to what Pipecat expects, which is the
strongest available compatibility signal short of formal interface tests.

**Layer 3: Wire surfaces.**
The surfaces through which external clients communicate with mod3. Three
surfaces exist today:

- HTTP REST API (`http_api.py`): `/v1/synthesize`, `/v1/vad`, and related
  endpoints. Clients that cannot speak the D2 wire protocol use these.
- MCP shim (`mcp_shim.py`): thin MCP adapter for Claude Code and similar MCP
  clients. Deprecated in favor of the kernel's native voice path (commit
  `f78fea4`).
- RTVI-shaped WebSocket audio sidecar (`/ws/audio/{session_id}`): the audio
  plane for real-time TTS delivery and microphone input. Described in
  `docs/architecture/audio-sidecar.md`.

The D2 wire protocol (Layer 3 for kernel-native clients) is described in
Section 2.2.

**Layer 4: Node-Card mesh participation.**
Mod3 registers itself as a child node of the local Root node (see Section 2.4).
Its Node Card advertises one capability type not yet in ADR-062's vocabulary:
`modality`. The proposed amendment is in Section 3.1.

### 2.2 D2 Wire Protocol Conformance

The D2 wire protocol is JSON-lines over stdin/stdout pipes. The Python schema
(`schemas/wire.py`, committed at `5be113c`) mirrors the Go struct in
`cogos/pkg/modality/wire.go` field-for-field. Field names and JSON tags are
byte-identical across the two implementations; the wire format is the source of
truth, and both sides converge to it independently.

The kernel spawns each worker subprocess as:

```
python -m mod3.worker {tts,vad,stt}
```

Each subprocess:

1. Emits `{"id":"__startup__","type":"event","event":"ready","status":"ok"}`
   on startup before reading any input.
2. Reads `WireMessage` records from stdin in a loop.
3. Dispatches operations to the matching inference code.
4. Writes `WireMessage` responses and events to stdout.
5. Handles `health` and `shutdown` lifecycle commands.
6. On any operation error, emits a `WireMessage(type="error")` correlated to
   the request `id`.

The `mod3.worker` package (committed on branch
`wave/2026-05-13-mod3/worker-cli`) implements this. The five operations are:

| Module | Op | Request | Response |
|---|---|---|---|
| `tts` | `synthesize` | `TTSSynthesizeRequest` | `TTSSynthesizeResponse` |
| `tts` | `stream` | `TTSStreamRequest` | sequence of `tts.chunk` events |
| `vad` | `detect` | `VADDetectRequest` | `VADDetectResponse` |
| `stt` | `transcribe` | `STTTranscribeRequest` | `STTTranscribeResponse` |
| `stt` | `transcribe_streaming` | `STTStreamingRequest` | `STTStreamingResponse` |

All request and response shapes are defined in `schemas/operations.py`. The Go
consumer (`cogos/pkg/modality/wire.go`, the `SubprocessConn.Request()` method)
reads these shapes from the subprocess stdout. Any field name change in
`schemas/operations.py` is a breaking change that requires a matching change in
the Go consumer.

### 2.3 Channel-with-Multiple-Modalities Framing

A **channel** (`ChannelDescriptor` in `schemas/channel.py`) is a transport-bound
identity that declares which modalities it can receive and deliver. The kernel's
`ChannelRegistry` routes output to every channel that supports a requested
modality. Mod3's dashboard registers itself as a channel with both Voice and Text
modalities on output, and Voice on input (microphone).

The framing:

- A **channel** is a transport-bound receiver/sender (WebSocket, stdio, Discord,
  etc.).
- A **modality** within a channel is an inference provider. It maps raw signals
  to cognitive events (decode path) and cognitive intents to raw signals
  (encode path).
- The modality's interface is its schema. Changing a modality's capabilities
  means changing the canonical schema for that modality type
  (`schemas/primitives.py`, `schemas/operations.py`).

Creating or modifying a modality is always a schema change first. The schema is
the contract. Implementation follows schema.

This framing is grounded in the Channel-Provider Interface design (an internal
design note), which defines channels as nodes in the Constellation, each
advertising a set of modalities through its capability card. Mod3 is the
first `audio`-kind provider in that design.

### 2.4 Sidecar Audio Plane

The real-time audio plane is described in full in
`docs/architecture/audio-sidecar.md` (committed on branch
`wave/2026-05-13-mod3/sidecar-doc`). The summary relevant to this RFC:

- ACP carries structured control messages (agent responses, tool calls, session
  events).
- A parallel RTVI-shaped WebSocket at `/ws/audio/{session_id}` carries raw audio
  (TTS chunks down, mic frames up).
- The two connections share a `session_id`. The audio channel has no independent
  lifecycle.
- `TTSChunkEvent` (in `schemas/operations.py`) is the natural payload shape for
  the server-to-client direction; it maps directly to the RTVI `bot-tts-audio`
  message type with mod3-specific extension fields.
- The current `/ws/chat` endpoint is the migration target: deprecate in favor of
  (a) ACP for control and (b) `/ws/audio/{session_id}` for audio.

## 3. Open Questions for Ratification

### 3.1 ADR-062 Amendment: Add `modality` Capability Type

ADR-062 defines the Node Card capability vocabulary with three types: `tool`,
`shard`, and `bus`. There is no `inference` or `modality` type. Mod3 as a
modality node cannot be described in the current vocabulary.

**Proposed amendment to ADR-062, Section 2 (The Node Card):**

> Add a fourth capability type to the `capabilities` block:
>
> ```yaml
> - id: voice_modality
>   type: modality
>   modality: voice         # one of: text | voice | vision | spatial
>   ops:                    # operations this modality exposes
>     - tts/synthesize
>     - tts/stream
>     - vad/detect
>     - stt/transcribe
>     - stt/transcribe_streaming
>   transport: d2-wire      # transport protocol for this modality's ops
> ```
>
> The `modality` type is preferred over `inference` because it generalizes to
> non-audio modalities (vision, spatial) without implying that the capability is
> limited to language model inference. A Vision modality node does inference too,
> but its schema vocabulary differs entirely from an audio modality node's.
>
> The `modality` type is orthogonal to `tool` (a callable action) and `shard`
> (queryable knowledge). A modality node may also advertise `tool` capabilities
> (e.g., `/v1/synthesize` as an HTTP tool) for callers that cannot speak the D2
> wire protocol, but the canonical interface is the `modality` capability.

**This amendment requires operator ratification before ADR-062 can be marked
Accepted.**

### 3.2 Position in the Node Tree

ADR-062 defines the node tree: Agent -> Workspace -> Root -> Team -> Federation.
Where does mod3 sit?

**Proposed:** Mod3 is a child of the local Root node. It is not a Workspace
(it does not contain tasks, agents, or project-scoped knowledge). It is not a
Team node. It is an inference service, registered as a child of Root with the
Root node's consent (Root controls registration per ADR-062 Section 9).

Concretely: mod3 registers at Root startup via the same registration protocol
ADR-062 Section 9 specifies for workspace-to-root registration, but with a
Node Card that advertises `type: modality` capabilities rather than workspace
capabilities. The Root node routes Voice modality requests to mod3.

An alternative is sibling to Root, but that requires a federation-level parent
to mediate, which is unnecessary complexity for a single-machine deployment.
Child of Root is the minimal correct position.

### 3.3 Sub-Sentence Streaming End-to-End

The schemas added `tts/stream` at `5be113c`. The `mod3.worker tts` subprocess
(branch `wave/2026-05-13-mod3/worker-cli`) implements the Python side. However,
the current Go-side voice encoder in `cogos/pkg/modality/modality_voice.go`
implements `voiceEncoder.Encode()` as a one-shot call: it sends a
`tts/synthesize` request and waits for a single `WireMessage(type="response")`.
It does not consume streamed `tts.chunk` events.

Adopting end-to-end sub-sentence streaming requires a Go-side change in
`modality_voice.go::voiceEncoder.Encode()` to send a `tts/stream` request and
read the resulting event sequence instead of a single response. This is a bounded
change (the `SubprocessConn.Request()` method in `wire.go` already handles the
wire mechanics; only the response-reading loop changes), but it is a breaking
change to the current voice encoding path.

**Decision for ratification:** adopt streaming in `v0.5` after validating the
end-to-end path on the `feat/issue-155-block-peers` branch, or defer to a
separate RFC. The capability is implemented on the Python side; the Go-side
wire-up is the remaining work.

### 3.4 Local AgentLoop Deprecation

Mod3 has `agent_loop.py` and `cogos_agent_bridge.py` as local agent paths.
Under this RFC, the agent lives in the kernel; mod3 provides voice inference.

**Proposed deprecation timeline:**

1. Mark `agent_loop.py` and `cogos_agent_bridge.py` as deprecated now (emit
   `DeprecationWarning` on import; add `[DEPRECATED]` header in docstring).
2. Remove in mod3 `v0.5` after the kernel-agent path is end-to-end working
   (kernel can issue a TTS intent via the D2 wire and receive playback on the
   dashboard without going through mod3's local agent path).
3. The `/ws/chat` endpoint is deprecated on the same schedule (it relies on
   `AgentLoop` directly).

### 3.5 Dashboard Role

The mod3 dashboard (`dashboard/`) serves two purposes: first-class development
interface for tuning voices and testing TTS engines, and eventual end-user UI
for voice-enabled operator workflows. Both roles are in scope. The details of
the dashboard's ACP integration, session lifecycle, and UI evolution are out of
scope for this RFC.

## 4. Non-Goals

- Cross-modality reasoning. The kernel routes between modalities; mod3 does not.
- Routing strategy among multiple TTS engines. That is a separate concern
  (relevant to RFC-0007 named-provider dispatch and RFC-0008's Observatory).
- Session storage or UAS integration. Session state lives in the kernel.
- The Pipecat integration's feature roadmap. `Mod3TTSService` is a thin wrapper;
  its evolution follows Pipecat's `TTSService` contract.

## 5. References

- **ADR-062** Recursive Node Architecture (internal architecture decision record): the node primitive, capability cards, and the one-level visibility rule that constrains mod3's position in the tree.
- **RFC-034** Substrate Kernel Categorical Split (internal RFC): Observer-as-Reconciler vocabulary referenced in RFC-0008; defines the Observer/Observatory distinction this RFC relies on for inference routing context.
- **RFC-0007** Named-Provider Dispatch (`cogos/docs/rfcs/0007-dispatch-provider-override.md`, PR #230, merged): the provider-name override mechanism that mod3 registers with as a Voice inference provider.
- **RFC-0008** Inference Control Plane via Node-State Observatory (`cogos/docs/rfcs/0008-inference-control-plane.md`, PR #233, merged with Observer-as-Reconciler amendment): classification of mod3 as a `cog-native` InferenceChannel with `RuntimeKind: cog-native-mlx`.
- **Channel-Provider Interface design** (internal design note): defines the `audio`-kind adapter contract that mod3 implements; mod3 is the first `audio`-kind provider in that design.
- **`cogos/pkg/modality/`** (in-flight on branch `feat/issue-155-block-peers` in the cogos repo): the Go implementation of the Module interface (`types.go`), wire protocol (`wire.go`), and voice wiring (`modality_voice.go`).
- **`mod3/schemas/`** (commit `5be113c`): the Python mirror of the kernel's modality schemas. Canonical schemas for wire protocol (`wire.py`), operations (`operations.py`), primitives (`primitives.py`), modality types (`modality.py`), and channel descriptor (`channel.py`).
- **`docs/architecture/audio-sidecar.md`** (branch `wave/2026-05-13-mod3/sidecar-doc`): operational mapping of the RTVI-shaped audio plane to the ACP control plane.

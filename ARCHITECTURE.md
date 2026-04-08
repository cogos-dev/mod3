# Mod3 Architecture: The Modality Bus

The modality bus is the sensorimotor boundary between cognitive agents and physical signals. Agents think in cognitive events ("someone spoke", "say this"); the bus translates between those events and raw bytes (audio, text, future: vision, spatial).

```
                        ModalityBus
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
    │  │  Voice   │  │  Text   │  │ Vision* │ ...  │
    │  │ Module   │  │ Module  │  │ Module  │      │
    │  └────┬─────┘  └────┬────┘  └────┬────┘      │
    │       │             │            │            │
    │  ┌────┴─────────────┴────────────┴────┐      │
    │  │         Event Log + Listeners       │      │
    │  └────┬─────────────┬────────────┬────┘      │
    │       │             │            │            │
    │  ┌────┴────┐  ┌─────┴─────┐  ┌──┴───┐       │
    │  │ Channel │  │  Channel  │  │ ...  │       │
    │  │ discord │  │  http-api │  │      │       │
    │  └─────────┘  └───────────┘  └──────┘       │
    └──────────────────────────────────────────────┘

    * Vision/Spatial are defined in ModalityType but not yet implemented.
```

## Core Types (modality.py)

### Cognitive Primitives

The agent never touches raw bytes. It sees these:

```python
@dataclass
class CognitiveEvent:          # Input percept
    modality: ModalityType     # VOICE, TEXT, VISION, SPATIAL
    content: str               # The meaning (transcribed text, caption, etc.)
    source_channel: str        # Which channel it arrived on
    confidence: float          # Decoder certainty (0.0 - 1.0)
    timestamp: float
    metadata: dict[str, Any]

@dataclass
class CognitiveIntent:         # Output intent (not yet encoded)
    modality: ModalityType | None  # None = let the bus decide
    content: str               # What to communicate
    target_channel: str        # Specific channel, or "" for bus routing
    priority: int              # Higher = more urgent
    metadata: dict[str, Any]   # voice, speed, emotion, etc.

@dataclass
class EncodedOutput:           # Raw signal ready for delivery
    modality: ModalityType
    data: bytes                # WAV, PNG, JSON, etc.
    format: str                # "wav", "png", "text", etc.
    duration_sec: float
    metadata: dict[str, Any]
```

### Abstract Base Classes

Every modality module implements three components:

```python
class Gate(ABC):
    def check(self, raw: bytes, **kwargs) -> GateResult: ...

class Decoder(ABC):
    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent: ...

class Encoder(ABC):
    def encode(self, intent: CognitiveIntent) -> EncodedOutput: ...

class ModalityModule(ABC):
    modality_type -> ModalityType   # Which modality this handles
    gate -> Gate | None             # Input filter (None = pass all)
    decoder -> Decoder | None       # raw -> CognitiveEvent
    encoder -> Encoder | None       # CognitiveIntent -> EncodedOutput
    state -> ModuleState            # Live HUD state
    health() -> dict                # Diagnostics
```

`Gate` is optional. Text has no gate (all text passes). Voice uses VAD (Voice Activity Detection) to reject silence.

## The Bus (bus.py)

`ModalityBus` manages module registration, signal routing, and state tracking.

### perceive() -- Input Path

```
raw bytes ──→ Gate.check() ──→ Decoder.decode() ──→ CognitiveEvent
                  │                   │
              (rejected?)        (empty content?)
                  ↓                   ↓
               None               None (filtered)
```

```python
bus.perceive(raw: bytes, modality: str | ModalityType, channel: str = "", **kwargs)
    -> CognitiveEvent | None
```

1. Resolve the modality module from the registry
2. If the module has a gate, run `gate.check(raw)`. Emit a `modality.gate` bus event. Return `None` if rejected.
3. Run `decoder.decode(raw)`. If content is empty (e.g., hallucination filtered), emit `modality.filtered` and return `None`.
4. Stamp `source_channel`, emit `modality.input`, return the event.

### act() -- Output Path

```
CognitiveIntent ──→ resolve modality ──→ Encoder.encode() ──→ EncodedOutput
                                                                    │
                                                          channel.deliver()
```

```python
bus.act(intent: CognitiveIntent, channel: str = "", blocking: bool = False)
    -> QueuedJob | EncodedOutput
```

1. Resolve output modality: explicit on intent, or inferred from channel capabilities (prefers voice over text), or defaults to text.
2. Encode via the module's encoder. Emits `modality.encode_start` and `modality.output` bus events.
3. If the target channel has a `deliver` callback, call it with the encoded output.
4. If `blocking=True`, returns `EncodedOutput` directly. Otherwise queues via `OutputQueueManager` and returns a `QueuedJob`.

### hud() -- Agent Awareness

```python
bus.hud() -> dict
```

Returns a live snapshot of all modules and channels: current status, active jobs, queue depths, recent events. Designed to be injected into the agent's context window so it knows what the body is doing.

### Channels

Channels declare which modalities they support. The bus auto-routes output based on channel capabilities.

```python
bus.register_channel("discord-voice", [ModalityType.VOICE, ModalityType.TEXT],
                     deliver=send_to_discord)
```

### Bus Events

Every boundary crossing is recorded as a `BusEvent` (type, modality, channel, timestamp, data). Listeners can subscribe via `bus.on_event(callback)` for ledger integration. The bus keeps the last 500 events in memory.

## Current Modalities

### Voice (modules/voice.py)

| Component | Class | Implementation |
|-----------|-------|----------------|
| Gate | `VoiceGate` | Silero VAD via `vad.detect_speech()`. Threshold-configurable (default 0.5). Rejects audio with no detected speech. |
| Decoder | `WhisperDecoder` | `mlx_whisper` STT on Apple Silicon. Lazy-loads `mlx-community/whisper-turbo`. Applies `vad.is_hallucination()` filter to reject phantom transcripts. |
| Decoder (legacy) | `PlaceholderDecoder` | Accepts pre-transcribed text. Used by the MCP server for the `speak` tool path where text is already known. |
| Encoder | `VoiceEncoder` | Wraps `engine.synthesize()` (Kokoro, Voxtral, Chatterbox, Spark). Default voice: `bm_lewis` at 1.25x speed. Returns WAV bytes. |

### Text (modules/text.py)

| Component | Class | Implementation |
|-----------|-------|----------------|
| Gate | None | All text passes through. |
| Decoder | `TextDecoder` | Identity transform: `bytes.decode("utf-8")` -> `CognitiveEvent`. |
| Encoder | `TextEncoder` | Identity transform: `intent.content.encode("utf-8")` -> `EncodedOutput`. |

Text exists so it is a first-class modality on the bus, not a special case.

## Integration Points

### MCP Server (server.py)

The MCP server creates the bus singleton at module level:

```python
_bus = _create_bus()  # ModalityBus with VoiceModule(decoder=PlaceholderDecoder())
```

MCP tools (`speak`, `diagnostics`, `vad_check`) use `_bus` for voice state tracking, health reports, and VAD. The `speak` tool resolves voices through the bus's voice module, sets encoder state, and uses the engine directly for synthesis (the adaptive player handles local playback).

The `diagnostics` tool returns `_bus.health()` and `_bus.hud()`.

### HTTP API (http_api.py)

The HTTP API imports the bus singleton from the MCP server:

```python
from server import _bus as _shared_bus  # Shared instance when co-hosted
_bus = _shared_bus                       # Falls back to fresh ModalityBus if import fails
```

It ensures both Text and Voice modules are registered, then exposes the bus directly:

| Endpoint | Bus Method |
|----------|------------|
| `GET /v1/bus/hud` | `_bus.hud()` |
| `GET /v1/bus/health` | `_bus.health()` |
| `POST /v1/bus/perceive` | `_bus.perceive(raw, modality, channel)` |
| `POST /v1/bus/act` | `_bus.act(intent, channel, blocking=True)` |
| `GET /health` | includes `_bus.health()` and `_bus.hud()` |

When running with `--all`, both MCP and HTTP share the same bus instance and model cache.

## Adding a New Modality

1. **Create `modules/your_modality.py`** -- implement `Gate`, `Decoder`, `Encoder` (all optional), and a `ModalityModule` subclass that wires them together. See `modules/text.py` for the minimal case or `modules/voice.py` for the full pattern.

2. **Add the modality type** to `ModalityType` in `modality.py` if needed. `VISION` and `SPATIAL` are already defined.

3. **Register with the bus** where it is created (`server.py` and/or `http_api.py`):
   ```python
   bus.register(VisionModule())
   bus.register_channel("webcam-feed", [ModalityType.VISION])
   ```

4. **No routing changes needed.** The bus auto-routes `act()` based on channel capabilities. The HTTP API's `/v1/bus/perceive` and `/v1/bus/act` already accept any registered modality via the `modality` parameter.

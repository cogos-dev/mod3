"""Modality Bus — the sensorimotor boundary of the cognitive agent.

Manages modality modules, routes inputs and outputs, tracks live state.
The agent interacts with the bus, never with raw signals directly.

    bus = ModalityBus()
    bus.register(VoiceModule())
    bus.register(TextModule())

    # Input: raw audio → gate → decode → cognitive event
    event = bus.perceive(raw_audio, modality="voice", channel="discord-voice")

    # Output: cognitive intent → route → encode → channel delivery
    output = bus.act(CognitiveIntent(content="hello"), channel="discord-voice")

    # HUD: what's happening right now across all modalities
    hud = bus.hud()
"""

from __future__ import annotations

import logging
import time
from queue import OutputQueueManager, QueuedJob
from typing import Any, Callable

from modality import (
    CognitiveEvent,
    CognitiveIntent,
    EncodedOutput,
    ModalityModule,
    ModalityType,
)

logger = logging.getLogger("mod3.bus")


# ---------------------------------------------------------------------------
# Bus event — everything that crosses the boundary gets recorded
# ---------------------------------------------------------------------------


class BusEvent:
    """Record of a boundary crossing. Feeds the ledger."""

    __slots__ = ("type", "modality", "channel", "timestamp", "data")

    def __init__(self, type: str, modality: str, channel: str, data: dict[str, Any] | None = None):
        self.type = type
        self.modality = modality
        self.channel = channel
        self.timestamp = time.time()
        self.data = data or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "modality": self.modality,
            "channel": self.channel,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Channel descriptor
# ---------------------------------------------------------------------------


class ChannelDescriptor:
    """Declares what modalities a channel supports."""

    def __init__(self, channel_id: str, modalities: list[ModalityType], deliver: Callable | None = None):
        self.channel_id = channel_id
        self.modalities = set(modalities)
        self.deliver = deliver  # Optional: callback to deliver encoded output
        self.active = True


# ---------------------------------------------------------------------------
# Modality Bus
# ---------------------------------------------------------------------------


class ModalityBus:
    """The sensorimotor boundary. Manages modules, routes signals, tracks state."""

    def __init__(self):
        self._modules: dict[ModalityType, ModalityModule] = {}
        self._channels: dict[str, ChannelDescriptor] = {}
        self._queue_manager = OutputQueueManager()
        self._event_log: list[BusEvent] = []
        self._max_events = 500
        self._listeners: list[Callable[[BusEvent], None]] = []

    # -- Registration --

    def register(self, module: ModalityModule) -> None:
        """Register a modality module."""
        self._modules[module.modality_type] = module
        logger.info(f"registered modality: {module.modality_type.value}")

    def register_channel(
        self, channel_id: str, modalities: list[ModalityType], deliver: Callable | None = None
    ) -> None:
        """Register a channel with its supported modalities."""
        self._channels[channel_id] = ChannelDescriptor(channel_id, modalities, deliver)
        logger.info(f"registered channel: {channel_id} ({[m.value for m in modalities]})")

    def on_event(self, listener: Callable[[BusEvent], None]) -> None:
        """Subscribe to bus events (for ledger integration)."""
        self._listeners.append(listener)

    # -- Perception (input) --

    def perceive(
        self,
        raw: bytes,
        modality: str | ModalityType,
        channel: str = "",
        **kwargs,
    ) -> CognitiveEvent | None:
        """Process raw input through gate → decoder → cognitive event.

        Returns None if the gate rejected the input.
        """
        mod_type = ModalityType(modality) if isinstance(modality, str) else modality
        module = self._modules.get(mod_type)
        if not module:
            raise ValueError(f"No module registered for modality: {mod_type}")

        # Gate check
        if module.gate is not None:
            gate_result = module.gate.check(raw, **kwargs)
            self._emit(
                BusEvent(
                    "modality.gate",
                    mod_type.value,
                    channel,
                    {"passed": gate_result.passed, "confidence": gate_result.confidence, "reason": gate_result.reason},
                )
            )
            if not gate_result.passed:
                return None

        # Decode
        if module.decoder is None:
            raise ValueError(f"Module {mod_type} has no decoder")

        event = module.decoder.decode(raw, channel=channel, **kwargs)

        # Empty content after decoding (e.g., hallucination filtered)
        if not event.content:
            self._emit(
                BusEvent(
                    "modality.filtered",
                    mod_type.value,
                    channel,
                    event.metadata,
                )
            )
            return None

        event.source_channel = channel
        self._emit(
            BusEvent(
                "modality.input",
                mod_type.value,
                channel,
                {"content": event.content[:200], "confidence": event.confidence},
            )
        )
        return event

    # -- Action (output) --

    def act(
        self,
        intent: CognitiveIntent,
        channel: str = "",
        blocking: bool = False,
    ) -> QueuedJob | EncodedOutput:
        """Encode a cognitive intent and deliver it.

        If blocking=True, waits for encoding and returns EncodedOutput.
        If blocking=False (default), queues the job and returns QueuedJob.
        """
        # Resolve modality
        mod_type = self._resolve_output_modality(intent, channel)
        module = self._modules.get(mod_type)
        if not module or module.encoder is None:
            raise ValueError(f"No encoder for modality: {mod_type}")

        intent.modality = mod_type
        target = channel or intent.target_channel or "default"

        def _do_encode():
            self._emit(
                BusEvent(
                    "modality.encode_start",
                    mod_type.value,
                    target,
                    {"content": intent.content[:200]},
                )
            )
            output = module.encoder.encode(intent)
            self._emit(
                BusEvent(
                    "modality.output",
                    mod_type.value,
                    target,
                    {
                        "format": output.format,
                        "duration_sec": output.duration_sec,
                        "bytes": len(output.data),
                    },
                )
            )
            # Deliver if channel has a delivery callback
            ch = self._channels.get(target)
            if ch and ch.deliver:
                ch.deliver(output)
            return output

        if blocking:
            return _do_encode()

        return self._queue_manager.submit(
            target,
            _do_encode,
            content_preview=intent.content[:100],
            modality=mod_type.value,
        )

    def _resolve_output_modality(self, intent: CognitiveIntent, channel: str) -> ModalityType:
        """Decide which modality to use for output."""
        # Explicit modality requested
        if intent.modality is not None:
            return intent.modality

        # Check what the target channel supports
        ch = self._channels.get(channel)
        if ch:
            # Prefer voice if available, fall back to text
            if ModalityType.VOICE in ch.modalities:
                return ModalityType.VOICE
            return ModalityType.TEXT

        # Default to text
        return ModalityType.TEXT

    # -- HUD (agent awareness) --

    def hud(self) -> dict[str, Any]:
        """Live state snapshot for the agent's context window.

        Returns the current state of all modules and channels —
        what's being spoken, what was just heard, queue depths.
        """
        modules = {}
        for mod_type, module in self._modules.items():
            state = module.state
            modules[mod_type.value] = {
                "status": state.status.value,
                "active_job": state.active_job,
                "queue_depth": self._queue_manager.get_queue(mod_type.value).depth
                if mod_type.value in self._queue_manager._queues
                else 0,
                "current_text": state.current_text,
                "progress": state.progress,
                "last_output": state.last_output_text,
                "last_activity": state.last_activity,
                "error": state.error,
            }

        channels = {}
        for cid, ch in self._channels.items():
            channels[cid] = {
                "modalities": [m.value for m in ch.modalities],
                "active": ch.active,
                "queue_depth": self._queue_manager.get_queue(cid).depth if cid in self._queue_manager._queues else 0,
            }

        return {
            "timestamp": time.time(),
            "modules": modules,
            "channels": channels,
            "queues": self._queue_manager.status(),
            "recent_events": [e.to_dict() for e in self._event_log[-10:]],
        }

    # -- Diagnostics --

    def health(self) -> dict[str, Any]:
        """Full health report."""
        return {
            "modules": {mt.value: m.health() for mt, m in self._modules.items()},
            "channels": {
                cid: {"modalities": [m.value for m in ch.modalities], "active": ch.active}
                for cid, ch in self._channels.items()
            },
            "queues": self._queue_manager.status(),
            "event_count": len(self._event_log),
        }

    # -- Internal --

    def _emit(self, event: BusEvent):
        """Record and broadcast a bus event."""
        self._event_log.append(event)
        if len(self._event_log) > self._max_events:
            self._event_log = self._event_log[-self._max_events :]
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

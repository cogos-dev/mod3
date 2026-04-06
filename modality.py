"""Modality module base classes — the sensorimotor boundary.

The agent thinks in cognitive events ("someone spoke", "say this").
Modality modules translate between cognitive events and raw signals.

Each module has three components:
  Gate    — should this input pass? (VAD for voice, change detection for vision)
  Decoder — raw signal → cognitive event (STT, OCR, sensor decode)
  Encoder — cognitive intent → raw signal (TTS, image gen, motor encode)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModalityType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    SPATIAL = "spatial"


class ModuleStatus(str, Enum):
    IDLE = "idle"
    ENCODING = "encoding"
    DECODING = "decoding"
    LOADING = "loading"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Cognitive primitives — what the agent sees
# ---------------------------------------------------------------------------


@dataclass
class CognitiveEvent:
    """An input percept that crossed the sensorimotor boundary."""

    modality: ModalityType
    content: str  # The meaning (transcribed text, caption, etc.)
    source_channel: str = ""  # Which channel it arrived on
    confidence: float = 1.0  # How sure the decoder is
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveIntent:
    """An output intent from the agent, not yet encoded."""

    modality: ModalityType | None  # None = let the bus decide
    content: str  # What to communicate
    target_channel: str = ""  # Specific channel, or "" for bus routing
    priority: int = 0  # Higher = more urgent
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncodedOutput:
    """Raw signal ready for channel delivery."""

    modality: ModalityType
    data: bytes  # The raw output (audio bytes, image bytes, etc.)
    format: str = ""  # "wav", "png", "json", etc.
    duration_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gate result
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Result of an input gate check."""

    passed: bool
    confidence: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module state — what the agent can see in its HUD
# ---------------------------------------------------------------------------


@dataclass
class ModuleState:
    """Live operational state of a modality module."""

    status: ModuleStatus = ModuleStatus.IDLE
    active_job: str | None = None  # Current job ID if encoding/decoding
    queue_depth: int = 0  # How many items waiting
    current_text: str = ""  # What's being spoken/generated right now
    progress: float = 0.0  # 0.0 - 1.0 for current job
    last_input: CognitiveEvent | None = None
    last_output_text: str = ""
    last_activity: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class Gate(ABC):
    """Input gate — decides if raw input contains signal worth decoding."""

    @abstractmethod
    def check(self, raw: bytes, **kwargs) -> GateResult:
        """Check if this input should pass through to the decoder."""
        ...


class Decoder(ABC):
    """Decoder — transforms raw signal into a cognitive event."""

    @abstractmethod
    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent:
        """Transform raw input into a cognitive event."""
        ...


class Encoder(ABC):
    """Encoder — transforms cognitive intent into raw signal."""

    @abstractmethod
    def encode(self, intent: CognitiveIntent) -> EncodedOutput:
        """Transform a cognitive intent into raw output."""
        ...


class ModalityModule(ABC):
    """A modality module — the full sensorimotor interface for one modality."""

    @property
    @abstractmethod
    def modality_type(self) -> ModalityType: ...

    @property
    @abstractmethod
    def gate(self) -> Gate | None:
        """Input gate, or None if no gating needed (e.g., text)."""
        ...

    @property
    @abstractmethod
    def decoder(self) -> Decoder | None:
        """Input decoder, or None if input not supported."""
        ...

    @property
    @abstractmethod
    def encoder(self) -> Encoder | None:
        """Output encoder, or None if output not supported."""
        ...

    @property
    def state(self) -> ModuleState:
        """Live state for the agent's HUD. Override for custom tracking."""
        return ModuleState()

    def health(self) -> dict[str, Any]:
        """Health check for diagnostics."""
        return {
            "modality": self.modality_type.value,
            "has_gate": self.gate is not None,
            "has_decoder": self.decoder is not None,
            "has_encoder": self.encoder is not None,
            "status": self.state.status.value,
        }

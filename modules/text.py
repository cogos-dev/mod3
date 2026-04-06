"""Text modality module — the trivial case.

No gate (all text passes), no transform needed.
Text in → cognitive event. Intent → text out.
This exists so text is a first-class modality on the bus,
not a special case.
"""

from modality import (
    CognitiveEvent,
    CognitiveIntent,
    Decoder,
    EncodedOutput,
    Encoder,
    Gate,
    ModalityModule,
    ModalityType,
)


class TextDecoder(Decoder):
    """Text → CognitiveEvent. Identity transform."""

    def decode(self, raw: bytes, **kwargs) -> CognitiveEvent:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return CognitiveEvent(
            modality=ModalityType.TEXT,
            content=text,
            source_channel=kwargs.get("channel", ""),
            confidence=1.0,
        )


class TextEncoder(Encoder):
    """CognitiveIntent → text bytes. Identity transform."""

    def encode(self, intent: CognitiveIntent) -> EncodedOutput:
        return EncodedOutput(
            modality=ModalityType.TEXT,
            data=intent.content.encode("utf-8"),
            format="text",
        )


class TextModule(ModalityModule):
    """Text modality — passthrough, no gate, no transform."""

    def __init__(self):
        self._decoder = TextDecoder()
        self._encoder = TextEncoder()

    @property
    def modality_type(self) -> ModalityType:
        return ModalityType.TEXT

    @property
    def gate(self) -> Gate | None:
        return None

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

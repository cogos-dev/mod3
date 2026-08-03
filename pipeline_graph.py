"""Channel pipeline-as-composable-graph — stage registry and mode defaults.

This module implements Primitive 4 of the substrate channel primitive design.
The pipeline is a declared composable stage graph with a ChannelMode field
that switches between two compositions:

  INTENTIONAL (default):
    mic → denoise → vad → stt → emit
    Session-scoped, explicit participation. The user turns the mic on
    deliberately. This is the current default for all Claude Code / MCP
    channels.

  AMBIENT:
    mic → denoise → vad → diarize → ecapa_match → stt → attribute →
    mention_detect → emit
    Always-on mic, VAD-gated attention, continuous diarization. Cog listens
    without responding unless mentioned or "appropriate silence" is detected
    (OQ-5, per an operator transcript). This is the multi-human-attendee
    mode (e.g. Speaker A + Speaker B with a hot mic).

Stage registration:

  Use the ``@register_stage(name)`` decorator on any class that implements a
  pipeline stage. The InboundPipeline will instantiate registered stages by
  name when composing a pipeline.

  Stages listed in DEFAULT_PIPELINES that are NOT registered are silently
  skipped with a warning log. This means ambient mode is safe to declare
  today — the unregistered stages (diarize, ecapa_match, attribute,
  mention_detect) produce substrate-visible TODO warnings until their
  implementations land in follow-on PRs.

No side effects on import.
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger("mod3.pipeline_graph")


# ---------------------------------------------------------------------------
# ChannelMode
# ---------------------------------------------------------------------------


class ChannelMode(str, Enum):
    """Pipeline composition mode for a channel.

    INTENTIONAL: session-scoped, explicit participation (default).
    AMBIENT: always-on, VAD-gated attention, continuous diarization.
    """

    INTENTIONAL = "intentional"
    AMBIENT = "ambient"


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

#: Global name → stage factory mapping.
#: Populated via ``@register_stage(name)`` decorators.
STAGE_REGISTRY: dict[str, type] = {}


def register_stage(name: str):
    """Decorator that registers a pipeline stage class under *name*.

    Usage::

        @register_stage("vad")
        class VADStage:
            def process(self, audio): ...

    The stage class is stored in STAGE_REGISTRY keyed by *name*. The
    InboundPipeline looks up names in this registry when composing a
    pipeline from a mode's DEFAULT_PIPELINES entry.
    """

    def decorator(cls):
        if name in STAGE_REGISTRY:
            logger.warning(
                "register_stage: overwriting existing registration for %r (was %r, now %r)",
                name,
                STAGE_REGISTRY[name],
                cls,
            )
        STAGE_REGISTRY[name] = cls
        logger.debug("register_stage: registered %r → %r", name, cls)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Default pipeline compositions by mode
# ---------------------------------------------------------------------------

#: Ordered stage names for each ChannelMode.
#:
#: INTENTIONAL matches today's InboundPipeline: denoise → vad → stt → emit.
#: AMBIENT adds the multi-human layers that are not yet implemented; those
#: stages will be skipped with a warning until their PRs land.
DEFAULT_PIPELINES: dict[ChannelMode, list[str]] = {
    ChannelMode.INTENTIONAL: ["denoise", "vad", "stt", "emit"],
    ChannelMode.AMBIENT: [
        "denoise",
        "vad",
        "diarize",  # follow-on PR: pyannote-based speaker diarization
        "ecapa_match",  # follow-on PR: ECAPA-TDNN identity matching
        "stt",
        "attribute",  # follow-on PR: attach speaker identity to transcript
        "mention_detect",  # follow-on PR: detect Cog mentions / appropriate silence
        "emit",
    ],
}


def resolve_pipeline(
    mode: ChannelMode | str,
    pipeline_stages: list[str] | None = None,
) -> list[str]:
    """Return the ordered stage name list for *mode*, with optional override.

    Args:
        mode: The ChannelMode (or its string value) to look up in
            DEFAULT_PIPELINES. Unknown strings default to INTENTIONAL.
        pipeline_stages: When provided, returned as-is (caller override).

    Returns:
        Ordered list of stage names.
    """
    if pipeline_stages is not None:
        return list(pipeline_stages)

    if isinstance(mode, str):
        try:
            mode = ChannelMode(mode)
        except ValueError:
            logger.warning(
                "resolve_pipeline: unknown mode %r; defaulting to INTENTIONAL",
                mode,
            )
            mode = ChannelMode.INTENTIONAL

    return list(DEFAULT_PIPELINES.get(mode, DEFAULT_PIPELINES[ChannelMode.INTENTIONAL]))


def compose_stages(stage_names: list[str]) -> list[object]:
    """Instantiate registered stages from *stage_names*, skipping unknowns.

    For each name in *stage_names*:
      - If the name is in STAGE_REGISTRY, instantiate the class (no-arg
        constructor) and include it in the returned list.
      - If the name is NOT in STAGE_REGISTRY, log a warning and skip. This
        makes ambient mode safe to declare before all stages are implemented.

    Returns:
        List of instantiated stage objects in pipeline order.
    """
    stages = []
    for name in stage_names:
        if name in STAGE_REGISTRY:
            stages.append(STAGE_REGISTRY[name]())
        else:
            logger.warning(
                "compose_stages: stage %r is not registered — skipping (implement and @register_stage(%r) to enable)",
                name,
                name,
            )
    return stages


__all__ = [
    "ChannelMode",
    "DEFAULT_PIPELINES",
    "STAGE_REGISTRY",
    "compose_stages",
    "register_stage",
    "resolve_pipeline",
]

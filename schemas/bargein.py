"""Barge-in context schema for injecting interrupt state into the next agent turn.

Sibling to `pipeline_state.InterruptInfo` — that type is the raw record captured
at the moment TTS playback is halted (timestamp, spoken_pct, delivered_text,
full_text, reason). `BargeinContext` is the agent-facing view: it adds the
precomputed `unspoken` remainder, the interrupting user's transcript (when
known), and a classified `source`, plus a `format_for_prompt()` renderer for
system-prompt injection. A2/A3 construct one of these from an `InterruptInfo`
(and, on the STT path, the resulting transcript) and hand it to agent_loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pipeline_state import InterruptInfo

BargeinSource = Literal["browser_vad", "mcp_signal", "manual", "superwhisper"]


@dataclass
class BargeinContext:
    """Agent-facing snapshot of a TTS interrupt, ready for prompt injection."""

    spoken: str
    unspoken: str
    full_text: str
    spoken_pct: float
    user_said: str | None
    interrupted_at: datetime
    source: BargeinSource

    @classmethod
    def from_interrupt_info(
        cls,
        info: InterruptInfo,
        source: BargeinSource,
        user_said: str | None = None,
    ) -> BargeinContext:
        """Build a BargeinContext from a pipeline_state.InterruptInfo record."""
        full_text = info.full_text or ""
        spoken = info.delivered_text or ""
        if full_text.startswith(spoken):
            unspoken = full_text[len(spoken) :].strip()
        else:
            unspoken = full_text[len(spoken) :].strip()
        return cls(
            spoken=spoken,
            unspoken=unspoken,
            full_text=full_text,
            spoken_pct=info.spoken_pct,
            user_said=user_said,
            interrupted_at=datetime.fromtimestamp(info.timestamp),
            source=source,
        )

    def format_for_prompt(self) -> str:
        """Render a terse system-prompt-friendly string (3-6 lines)."""
        lines: list[str] = ["[Your previous reply was interrupted.]"]
        if self.spoken:
            lines.append(f'Spoken: "{self.spoken}"')
        if self.unspoken:
            lines.append(f'Unspoken: "{self.unspoken}"')
        if self.user_said:
            lines.append(f'User said: "{self.user_said}"')
        else:
            lines.append(f"User interrupted at {self.spoken_pct * 100:.0f}% (via {self.source}).")
        return "\n".join(lines)

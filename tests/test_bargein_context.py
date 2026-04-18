"""Tests for the barge-in context injection path through AgentLoop.

Verifies the A2/A3 flow end-to-end:
  pipeline_state.last_interrupt (InterruptInfo)
    -> AgentLoop._prepare_bargein_context()
    -> AgentLoop._pending_bargein (BargeinContext)
    -> AgentLoop._inject_pending_bargein(system_prompt)
    -> provider.chat(system=...)

Run: python -m pytest tests/test_bargein_context.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock

import pytest

# Ensure the project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_loop import AgentLoop  # noqa: E402
from modality import CognitiveEvent, ModalityType  # noqa: E402
from pipeline_state import InterruptInfo, PipelineState  # noqa: E402
from providers import ProviderResponse  # noqa: E402
from schemas.bargein import BargeinContext  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeProvider:
    """Minimal async provider: records chat() kwargs, returns empty response."""

    name = "fake"

    def __init__(self, response: ProviderResponse | None = None):
        self.calls: list[dict] = []
        self._response = response or ProviderResponse(tool_calls=[], text="")

    async def chat(self, messages, tools=None, system: str = ""):
        self.calls.append({"messages": list(messages), "tools": tools, "system": system})
        return self._response


def _make_loop(provider: FakeProvider | None = None) -> AgentLoop:
    """Build an AgentLoop with a MagicMock bus and a fresh PipelineState."""
    bus = MagicMock()
    prov = provider or FakeProvider()
    state = PipelineState()
    return AgentLoop(bus=bus, provider=prov, pipeline_state=state, channel_id="test")


def _seed_interrupt(
    state: PipelineState,
    *,
    full_text: str = "Hello there how are you today friend",
    delivered_text: str = "Hello there how",
    spoken_pct: float = 0.45,
    reason: str = "vad_reflex",
    timestamp: float | None = None,
) -> InterruptInfo:
    """Directly seed pipeline_state._last_interrupt (matches the prod consume pattern)."""
    info = InterruptInfo(
        timestamp=timestamp if timestamp is not None else time.time(),
        spoken_pct=spoken_pct,
        delivered_text=delivered_text,
        full_text=full_text,
        reason=reason,
    )
    with state._lock:
        state._last_interrupt = info
    return info


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_bargein_from_interrupt():
    """_prepare_bargein_context consumes last_interrupt and builds a BargeinContext."""
    loop = _make_loop()
    info = _seed_interrupt(loop.pipeline_state)

    loop._prepare_bargein_context(user_text="wait go back")

    assert isinstance(loop._pending_bargein, BargeinContext)
    assert loop._pending_bargein.source == "browser_vad"
    assert loop._pending_bargein.user_said == "wait go back"
    assert loop._pending_bargein.full_text == info.full_text
    assert loop._pending_bargein.spoken == info.delivered_text
    assert abs(loop._pending_bargein.spoken_pct - info.spoken_pct) < 1e-9
    # Unspoken = full minus delivered prefix, stripped
    assert loop._pending_bargein.unspoken == "are you today friend"
    # Interrupt must be cleared so subsequent turns don't re-consume it
    assert loop.pipeline_state.last_interrupt is None


def test_prepare_bargein_none_when_no_interrupt():
    """When pipeline_state has no interrupt, _pending_bargein is cleared to None."""
    loop = _make_loop()
    loop._pending_bargein = MagicMock()  # something non-None
    loop._prepare_bargein_context(user_text="hi")
    assert loop._pending_bargein is None


def test_inject_pending_bargein_into_prompt():
    """_inject_pending_bargein appends the rendered context and clears pending."""
    loop = _make_loop()
    loop._pending_bargein = BargeinContext(
        spoken="Hello there",
        unspoken="how are you today",
        full_text="Hello there how are you today",
        spoken_pct=0.4,
        user_said="wait go back",
        interrupted_at=__import__("datetime").datetime.now(),
        source="browser_vad",
    )

    result = loop._inject_pending_bargein("BASE PROMPT")

    assert "BASE PROMPT" in result
    assert "[Your previous reply was interrupted.]" in result
    assert 'User said: "wait go back"' in result
    # Consumed — must not leak into subsequent turns
    assert loop._pending_bargein is None


def test_inject_noop_when_no_pending():
    """_inject_pending_bargein returns the prompt unchanged when nothing is pending."""
    loop = _make_loop()
    loop._pending_bargein = None
    assert loop._inject_pending_bargein("BASE PROMPT") == "BASE PROMPT"


def test_stale_interrupt_guarded():
    """Interrupts older than the 30s freshness window are dropped, not injected."""
    loop = _make_loop()
    # Timestamp far in the past (well beyond the 30s guard)
    _seed_interrupt(loop.pipeline_state, timestamp=time.time() - 120.0)

    loop._prepare_bargein_context(user_text=None)

    assert loop._pending_bargein is None
    # Stale record must also be cleared so it can't rot in state
    assert loop.pipeline_state.last_interrupt is None


def test_full_flow_through_process_turn(monkeypatch):
    """End-to-end: _process builds a prompt containing the bargein render.

    Monkey-patches _fetch_kernel_context to avoid any HTTP call, and uses a
    FakeProvider to capture the `system` kwarg passed into chat().
    """
    # Neutralize the kernel-context HTTP fetch: return "" so the prompt is deterministic.
    import agent_loop as agent_loop_module

    monkeypatch.setattr(agent_loop_module, "_fetch_kernel_context", lambda: "")
    # Prevent exchange-logging HTTP call
    monkeypatch.setattr(agent_loop_module, "_log_exchange_to_bus", lambda *a, **kw: None)

    provider = FakeProvider(response=ProviderResponse(tool_calls=[], text=""))
    loop = _make_loop(provider=provider)

    _seed_interrupt(
        loop.pipeline_state,
        full_text="The capital of France is Paris and also other cities",
        delivered_text="The capital of France",
        spoken_pct=0.35,
    )

    event = CognitiveEvent(
        modality=ModalityType.VOICE,
        content="wait actually ask about Germany",
        source_channel="test",
    )

    asyncio.run(loop._process(event))

    assert len(provider.calls) == 1, "provider.chat should have been called exactly once"
    system_prompt = provider.calls[0]["system"]
    # Barge-in banner is present
    assert "[Your previous reply was interrupted.]" in system_prompt
    # User's new utterance was threaded into the BargeinContext
    assert 'User said: "wait actually ask about Germany"' in system_prompt
    # Spoken prefix surfaced in the render
    assert 'Spoken: "The capital of France"' in system_prompt
    # Base prompt still present
    assert "You are Cog" in system_prompt
    # Interrupt consumed from pipeline_state
    assert loop.pipeline_state.last_interrupt is None
    # Pending bargein cleared after injection
    assert loop._pending_bargein is None

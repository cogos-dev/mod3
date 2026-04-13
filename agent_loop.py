"""Agent loop — receives percepts, calls LLM with tools, dispatches actions.

The agent loop is the bridge between the ModalityBus (perception/action)
and the InferenceProvider (thinking). It maintains conversation history
and routes tool calls through the bus.
"""

from __future__ import annotations

import json as _json
import logging
import os
import time
from typing import TYPE_CHECKING

import httpx

from bus import ModalityBus
from modality import CognitiveEvent, CognitiveIntent, ModalityType
from pipeline_state import PipelineState
from providers import AGENT_TOOLS, InferenceProvider

if TYPE_CHECKING:
    from channels import BrowserChannel

logger = logging.getLogger("mod3.agent_loop")

# Base system prompt — kernel context is appended dynamically
_BASE_SYSTEM_PROMPT = (
    "You are Cog, a voice assistant running on Mod³ (Apple Silicon, fully local). "
    "You respond using tool calls. Use speak() for conversational voice responses — "
    "keep them concise, 1-3 sentences. Use send_text() only when the content is "
    "better read than heard (code, lists, links, structured data). "
    "No markdown in speak() text. Speak naturally. "
    "If the user asks something you can't do, say so briefly."
)

# CogOS kernel endpoint for context enrichment
_COGOS_ENDPOINT = os.environ.get("COGOS_ENDPOINT", "http://localhost:6931")

# Bus endpoint for logging exchanges (observation channel)
_COGOS_BUS_ENDPOINT = f"{_COGOS_ENDPOINT}/v1/bus"


def _fetch_kernel_context() -> str:
    """Pull active context from CogOS kernel to enrich the system prompt.

    Returns a context block string, or empty string if kernel unavailable.
    This is the afferent path: kernel → local model.
    """
    try:
        resp = httpx.get(f"{_COGOS_ENDPOINT}/health", timeout=2.0)
        if resp.status_code != 200:
            return ""
        health = resp.json()

        parts = []
        identity = health.get("identity", "cog")
        state = health.get("state", "unknown")
        parts.append(f"Kernel identity: {identity}, state: {state}")

        # Try to get active session context
        try:
            ctx_resp = httpx.get(f"{_COGOS_ENDPOINT}/v1/context", timeout=2.0)
            if ctx_resp.status_code == 200:
                ctx = ctx_resp.json()
                nucleus = ctx.get("nucleus", "")
                if nucleus:
                    parts.append(f"Active nucleus: {nucleus}")
                process_state = ctx.get("state", "")
                if process_state:
                    parts.append(f"Process state: {process_state}")
        except Exception:
            pass

        # Check for barge-in context (what was Claude saying when interrupted?)
        signal_file = os.environ.get("BARGEIN_SIGNAL", "/tmp/mod3-barge-in.json")
        try:
            if os.path.exists(signal_file):
                with open(signal_file) as f:
                    signal = _json.load(f)
                interrupted = signal.get("interrupted")
                if interrupted:
                    delivered = interrupted.get("delivered_text", "")
                    pct = interrupted.get("spoken_pct", 0)
                    parts.append(
                        f"[barge-in] Claude's speech was interrupted at {pct * 100:.0f}%. "
                        f'Delivered: "{delivered}". '
                        f"The user interrupted to say something — acknowledge and respond to them."
                    )
        except Exception:
            pass

        if parts:
            return "\n\nKernel context:\n" + "\n".join(f"- {p}" for p in parts)
        return ""
    except Exception:
        return ""


def _log_exchange_to_bus(user_text: str, assistant_text: str, provider_name: str):
    """Log the local model exchange to the CogOS bus (observation channel).

    This is the efferent path: local model → kernel → Claude can observe.
    """
    try:
        payload = {
            "type": "modality.voice.exchange",
            "from": f"mod3-reflex:{provider_name}",
            "payload": {
                "user": user_text,
                "assistant": assistant_text,
                "provider": provider_name,
                "timestamp": time.time(),
            },
        }
        httpx.post(
            _COGOS_BUS_ENDPOINT,
            json=payload,
            timeout=2.0,
        )
    except Exception as e:
        logger.debug("Failed to log exchange to bus: %s", e)


MAX_HISTORY = 50


class AgentLoop:
    """Conversational agent that receives percepts and acts through the bus."""

    def __init__(
        self,
        bus: ModalityBus,
        provider: InferenceProvider,
        pipeline_state: PipelineState,
        channel_id: str = "",
    ):
        self.bus = bus
        self.provider = provider
        self.pipeline_state = pipeline_state
        self.channel_id = channel_id
        self.conversation: list[dict[str, str]] = []
        self._channel_ref: BrowserChannel | None = None
        self._processing = False

    async def handle_event(self, event: CognitiveEvent) -> None:
        """Called when a CognitiveEvent arrives from the channel."""
        if not event.content.strip():
            return

        if self._processing:
            logger.warning("agent busy, dropping: %s", event.content[:50])
            return

        self._processing = True
        try:
            await self._process(event)
        except Exception as e:
            logger.error("agent_loop error: %s", e, exc_info=True)
            try:
                if self._channel_ref:
                    await self._channel_ref.send_response_text(f"[error: {e}]")
                    await self._channel_ref.send_response_complete()
            except Exception:
                pass  # channel may be dead, don't block finally
        finally:
            self._processing = False

    async def _process(self, event: CognitiveEvent) -> None:
        """Core: event → provider → tool dispatch."""
        self.conversation.append({"role": "user", "content": event.content})
        self._trim_history()

        t_start = time.perf_counter()

        # Assemble system prompt with kernel context (afferent path)
        kernel_ctx = _fetch_kernel_context()
        system_prompt = _BASE_SYSTEM_PROMPT + kernel_ctx

        response = await self.provider.chat(
            messages=self.conversation,
            tools=AGENT_TOOLS,
            system=system_prompt,
        )

        t_llm = (time.perf_counter() - t_start) * 1000

        # Dispatch tool calls
        assistant_parts: list[str] = []

        for tc in response.tool_calls:
            if tc.name == "speak":
                text = tc.arguments.get("text", "")
                if text:
                    assistant_parts.append(text)
                    # Show text in chat panel
                    if self._channel_ref:
                        await self._channel_ref.send_response_text(text)
                    # Route through bus → VoiceEncoder → TTS → channel.deliver
                    intent = CognitiveIntent(
                        modality=ModalityType.VOICE,
                        content=text,
                        target_channel=self.channel_id,
                        metadata={
                            "voice": self._channel_ref.config.get("voice", "bm_lewis")
                            if self._channel_ref
                            else "bm_lewis",
                            "speed": self._channel_ref.config.get("speed", 1.25) if self._channel_ref else 1.25,
                        },
                    )
                    # Fire-and-forget: bus.act(blocking=False) returns QueuedJob immediately,
                    # OutputQueue drain thread handles TTS encoding + delivery.
                    self.bus.act(intent, channel=self.channel_id)

            elif tc.name == "send_text":
                text = tc.arguments.get("text", "")
                if text:
                    assistant_parts.append(text)
                    if self._channel_ref:
                        await self._channel_ref.send_response_text(text)

        # Fallback: if provider returned text but no tool calls, auto-speak
        if not response.tool_calls and response.text:
            text = response.text
            assistant_parts.append(text)
            if self._channel_ref:
                await self._channel_ref.send_response_text(text)
            intent = CognitiveIntent(
                modality=ModalityType.VOICE,
                content=text,
                target_channel=self.channel_id,
                metadata={
                    "voice": self._channel_ref.config.get("voice", "bm_lewis") if self._channel_ref else "bm_lewis",
                    "speed": self._channel_ref.config.get("speed", 1.25) if self._channel_ref else 1.25,
                },
            )
            self.bus.act(intent, channel=self.channel_id)

        # Update conversation history
        if assistant_parts:
            assistant_text = " ".join(assistant_parts)
            self.conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )

            # Log exchange to CogOS bus (observation channel — Claude can see this)
            _log_exchange_to_bus(event.content, assistant_text, self.provider.name)

        # Signal completion
        if self._channel_ref:
            await self._channel_ref.send_response_complete(
                metrics={"llm_ms": round(t_llm, 1), "provider": self.provider.name}
            )

    def _trim_history(self) -> None:
        """Keep conversation within MAX_HISTORY messages."""
        if len(self.conversation) > MAX_HISTORY:
            self.conversation = self.conversation[-MAX_HISTORY:]

"""CogOS kernel agent bridge (MOD3_USE_COGOS_AGENT=1).

When the env flag is set, Mod³'s agent loop forwards user turns to the
cogos kernel's metabolic cycle instead of the local inference provider:

  browser → WS turn → post_user_message()  ─POST /v1/bus/send─►  kernel
                                                                     │
                                                                     ▼
                                                         bus_dashboard_chat
                                                                     │
                                                                     ▼
                                                   kernel cycle → `respond` tool
                                                                     │
                                                                     ▼
                                                         bus_dashboard_response
                                                                     │
                                                     SSE /v1/events/stream
                                                                     │
                                                                     ▼
                                               KernelBusSubscriber.stream()
                                                                     │
                                                                     ▼
                                                    run_response_bridge()
                                                                     │
                                                                     ▼
                                          BrowserChannel.broadcast_response_text()

The subscriber does its own reconnect with exponential backoff (see
`bus_bridge.py`). Disable the whole fork by leaving `MOD3_USE_COGOS_AGENT`
unset (default).

Note: the kernel's `POST /v1/bus/send` takes a flat `{bus_id, from, to,
message, type}` body — the inner JSON event is serialised into `message`
(matches the pattern used by other cogos producers).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

from bus_bridge import KERNEL_BUS_STREAM_URL, KernelBusSubscriber
from channels import BrowserChannel

logger = logging.getLogger("mod3.cogos_agent")

# Bus names — contract with the kernel side (see ADR / c-agent subagent).
CHAT_BUS_ID = "bus_dashboard_chat"
RESPONSE_BUS_ID = "bus_dashboard_response"

# Kernel endpoints.
_DEFAULT_KERNEL_BASE = os.environ.get("COGOS_ENDPOINT", "http://localhost:6931")
BUS_SEND_URL = f"{_DEFAULT_KERNEL_BASE}/v1/bus/send"

# Env gate.
ENABLE_ENV = "MOD3_USE_COGOS_AGENT"

_POST_TIMEOUT_S = 5.0


def is_enabled() -> bool:
    """True when MOD3_USE_COGOS_AGENT is set to a truthy value."""
    v = os.environ.get(ENABLE_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def post_user_message(text: str, session_id: str) -> bool:
    """POST a user turn to the kernel's `bus_dashboard_chat` bus.

    Returns True if the send succeeded (kernel replied 2xx), False otherwise.
    Logs at warning-level on failure but never raises — callers use graceful
    degradation (e.g. show an error response frame to the dashboard).

    The kernel's handleBusSend (see apps/cogos/bus_api.go) accepts
    `{bus_id, from, to, message, type}` — we JSON-encode the full event dict
    into `message` so the kernel's cycle receives the structured payload.
    """
    event = {
        "type": "user_message",
        "text": text,
        "session_id": session_id,
        "ts": _now_rfc3339(),
    }
    body = {
        "bus_id": CHAT_BUS_ID,
        "from": "mod3-dashboard",
        "type": "user_message",
        "message": json.dumps(event, separators=(",", ":")),
    }
    try:
        async with httpx.AsyncClient(timeout=_POST_TIMEOUT_S) as client:
            resp = await client.post(BUS_SEND_URL, json=body)
    except httpx.HTTPError as exc:
        logger.warning("cogos-agent: post to %s failed: %s", BUS_SEND_URL, exc)
        return False
    if resp.status_code // 100 != 2:
        logger.warning(
            "cogos-agent: post non-2xx: %d body=%r",
            resp.status_code, resp.text[:200],
        )
        return False
    logger.info(
        "cogos-agent: forwarded user turn to kernel bus (session=%s)",
        session_id,
    )
    return True


def _extract_response_text(payload: dict) -> Optional[str]:
    """Dig the assistant reply out of the bus event payload.

    Kernel's `handleBusSend` wraps the sent `message` string inside a
    `{"content": "<message>"}` map. On SSE delivery, the envelope's `data`
    field is that map. We look first for structured keys (`text`, direct
    agent_response shape), then fall through to parsing `content` as JSON.
    """
    if not isinstance(payload, dict):
        return None
    # Direct shape (if an upstream producer wrote the event dict at the top level).
    for key in ("text", "reply", "response"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    # Standard bus envelope: payload = {"content": "<json-encoded event>"}
    content = payload.get("content")
    if isinstance(content, str) and content:
        try:
            inner = json.loads(content)
        except (TypeError, ValueError):
            # Free-form string — treat the whole thing as the reply.
            return content
        if isinstance(inner, dict):
            for key in ("text", "reply", "response"):
                val = inner.get(key)
                if isinstance(val, str) and val:
                    return val
        elif isinstance(inner, str) and inner:
            return inner
    return None


async def run_response_bridge(subscriber: KernelBusSubscriber) -> None:
    """Consume `subscriber` and broadcast agent replies to dashboard clients.

    `BrowserChannel.broadcast_response_text()` is thread-safe via
    `run_coroutine_threadsafe`, matching the existing trace-event pattern.
    Malformed events (no recoverable text) are logged at debug and skipped.
    """
    first_event_logged = False
    forwarded = 0
    async for env in subscriber.stream():
        if env.kind == "connected":
            continue
        text = _extract_response_text(env.payload)
        if not text:
            logger.debug(
                "cogos-agent: skip event with no text kind=%s id=%s",
                env.kind, env.event_id,
            )
            continue
        if not first_event_logged:
            logger.info(
                "cogos-agent: first response forwarded kind=%s event_id=%s",
                env.kind, env.event_id,
            )
            first_event_logged = True
        try:
            BrowserChannel.broadcast_response_text(text)
            forwarded += 1
            logger.debug(
                "cogos-agent: forwarded response event_id=%s (total=%d)",
                env.event_id, forwarded,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort fan-out
            logger.debug("cogos-agent: broadcast failed: %s", exc)


async def start_response_bridge(
    app_state: object,
    *,
    url: str = KERNEL_BUS_STREAM_URL,
) -> None:
    """Construct the response subscriber + bridge task and store on `app_state`.

    No-op (logs once) when `MOD3_USE_COGOS_AGENT` is unset.
    """
    if not is_enabled():
        logger.debug("cogos-agent: response bridge disabled (%s unset)", ENABLE_ENV)
        setattr(app_state, "cogos_agent_subscriber", None)
        setattr(app_state, "cogos_agent_task", None)
        return

    subscriber = KernelBusSubscriber(
        url=url,
        bus_filter=RESPONSE_BUS_ID,
        consumer_id="mod3-dashboard-agent",
    )
    task = asyncio.create_task(
        run_response_bridge(subscriber),
        name="mod3-cogos-agent-bridge",
    )
    setattr(app_state, "cogos_agent_subscriber", subscriber)
    setattr(app_state, "cogos_agent_task", task)
    logger.info(
        "cogos-agent: response bridge started, target=%s bus_id=%s",
        url, RESPONSE_BUS_ID,
    )


async def stop_response_bridge(app_state: object, *, timeout_s: float = 2.0) -> None:
    """Gracefully stop the response bridge: close subscriber, await task, cancel on timeout."""
    subscriber: Optional[KernelBusSubscriber] = getattr(app_state, "cogos_agent_subscriber", None)
    task: Optional[asyncio.Task] = getattr(app_state, "cogos_agent_task", None)
    if subscriber is None and task is None:
        return
    if subscriber is not None:
        try:
            await subscriber.close()
        except Exception:  # pragma: no cover - best-effort
            pass
    if task is not None:
        try:
            await asyncio.wait_for(task, timeout=timeout_s)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pragma: no cover
                pass
    logger.info("cogos-agent: response bridge stopped")

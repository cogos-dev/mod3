"""Kernel-bus → dashboard bridge runner.

Consumes `KernelBusSubscriber.stream()` (see `bus_bridge.py`) and fans the
ADR-083 cycle-trace events out to every connected dashboard WebSocket via
`BrowserChannel.broadcast_trace_event()` (see `channels.py`).

Wiring:

  kernel (bus_cycle_trace)
     └─► SSE /v1/events/stream?bus_id=bus_cycle_trace
            └─► KernelBusSubscriber.stream()       [C1]
                   └─► run_bridge() filter + forward
                          └─► BrowserChannel.broadcast_trace_event()  [C2]

The subscriber does its own reconnect with exponential backoff, so a kernel
that is temporarily unreachable does not affect server startup. Disable the
bridge entirely at process boot by setting env `MOD3_BUS_BRIDGE_DISABLED=1`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from bus_bridge import KERNEL_BUS_STREAM_URL, BusEnvelope, KernelBusSubscriber
from channels import BrowserChannel

logger = logging.getLogger("mod3.bus_bridge")

# ADR-083 kinds the dashboard trace panel cares about. Kept as a module-level
# constant so tests and the lifespan wiring share one definition.
ADR083_KINDS: frozenset[str] = frozenset({"state_transition", "tool_dispatch", "assessment"})

# Kernel-side bus name (see apps/cogos/trace_emit.go:const traceBusID).
TRACE_BUS_ID = "bus_cycle_trace"

# Env flag consulted at startup.
DISABLE_ENV = "MOD3_BUS_BRIDGE_DISABLED"


def is_disabled() -> bool:
    """True when MOD3_BUS_BRIDGE_DISABLED is set to a truthy value."""
    v = os.environ.get(DISABLE_ENV, "").strip().lower()
    return v in ("1", "true", "yes", "on")


async def run_bridge(
    subscriber: KernelBusSubscriber,
    *,
    filter_kinds: Optional[set[str]] = None,
) -> None:
    """Consume `subscriber` and broadcast cycle-trace events to dashboard clients.

    `filter_kinds`:
      - `None`: forward everything (dev mode — useful when inspecting the raw
        stream through a dashboard).
      - set of kind strings: only forward envelopes whose `BusEnvelope.kind`
        is in the set. Unknown kinds are tolerated per ADR-083 — they simply
        won't pass this filter.

    `BrowserChannel.broadcast_trace_event()` is thread-safe and non-blocking:
    it dispatches each WS send via `run_coroutine_threadsafe`. We call it
    directly (no await).
    """
    first_event_logged = False
    forwarded = 0
    async for env in subscriber.stream():
        if filter_kinds is not None and env.kind not in filter_kinds:
            continue
        # The "connected" bootstrap frame has an empty payload; skip silently.
        if env.kind == "connected":
            continue
        if not first_event_logged:
            logger.info(
                "bridge: first event forwarded kind=%s event_id=%s",
                env.kind, env.event_id,
            )
            first_event_logged = True
        try:
            BrowserChannel.broadcast_trace_event(env.payload)
            forwarded += 1
            logger.debug(
                "bridge: forwarded kind=%s event_id=%s (total=%d)",
                env.kind, env.event_id, forwarded,
            )
        except Exception as exc:  # noqa: BLE001 — broadcaster is best-effort
            logger.debug("bridge: broadcast failed: %s", exc)


async def start_bridge(
    app_state: object,
    *,
    url: str = KERNEL_BUS_STREAM_URL,
    bus_filter: str = TRACE_BUS_ID,
    filter_kinds: Optional[set[str]] = frozenset(ADR083_KINDS),
) -> None:
    """Construct the subscriber + bridge task and store them on `app_state`.

    Startup is non-blocking: we don't await the task or probe the kernel.
    The subscriber's own backoff loop handles reconnects. Logs a disabled
    notice and returns cleanly when `MOD3_BUS_BRIDGE_DISABLED` is set.
    """
    if is_disabled():
        logger.info("bridge: disabled via %s=1", DISABLE_ENV)
        setattr(app_state, "bus_bridge_subscriber", None)
        setattr(app_state, "bus_bridge_task", None)
        return

    subscriber = KernelBusSubscriber(url=url, bus_filter=bus_filter, consumer_id="mod3-dashboard")
    task = asyncio.create_task(
        run_bridge(subscriber, filter_kinds=set(filter_kinds) if filter_kinds else None),
        name="mod3-bus-bridge",
    )
    setattr(app_state, "bus_bridge_subscriber", subscriber)
    setattr(app_state, "bus_bridge_task", task)
    logger.info(
        "bridge: started, target=%s bus_id=%s filter=%s",
        url, bus_filter, sorted(filter_kinds) if filter_kinds else "*",
    )


async def stop_bridge(app_state: object, *, timeout_s: float = 2.0) -> None:
    """Gracefully stop the bridge: close subscriber, await task, cancel on timeout."""
    subscriber: Optional[KernelBusSubscriber] = getattr(app_state, "bus_bridge_subscriber", None)
    task: Optional[asyncio.Task] = getattr(app_state, "bus_bridge_task", None)
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
    logger.info("bridge: stopped")

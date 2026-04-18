"""Kernel-bus SSE subscriber.

Consumes http://localhost:6931/v1/events/stream and yields parsed bus events.
Reconnects on disconnect with exponential backoff. Tolerates unknown event kinds
per ADR-083 (cycle-trace event contract).

C3 will consume this to broadcast CycleEvents to dashboard WebSocket clients.

The kernel (see apps/cogos/bus_stream.go) emits SSE frames of the form:

    data: {"id":"live_*_42","type":"bus.event","timestamp":"...","data":{<CogBlock>}}\\n\\n

Heartbeats arrive as SSE comment lines:

    : keep-alive\\n\\n

An initial frame of {"type":"connected","bus_id":"*","timestamp":"..."} is
sent on subscribe — we surface that as a BusEnvelope with kind="connected".
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx

logger = logging.getLogger("mod3.bus_bridge")

KERNEL_BUS_STREAM_URL = "http://localhost:6931/v1/events/stream"


@dataclass
class BusEnvelope:
    """Raw bus-envelope record as received from the kernel SSE stream.

    `raw` is the full outer JSON (the bus.event envelope). `payload` is the
    inner CogBlock dict (envelope["data"]) — may be {} for non-bus.event
    frames (e.g. the initial "connected" frame). `kind` is the best-effort
    event-kind string: preferring payload["kind"] (ADR-083 CycleEvent), then
    payload["type"], then envelope["type"]. Consumers MUST tolerate unknown
    kinds.
    """

    raw: dict
    kind: str
    payload: dict = field(default_factory=dict)
    ts: Optional[str] = None
    event_id: Optional[str] = None


def _extract_kind(envelope: dict, payload: dict) -> str:
    for src in (payload, envelope):
        for key in ("kind", "type"):
            val = src.get(key) if isinstance(src, dict) else None
            if isinstance(val, str) and val:
                return val
    return "unknown"


class KernelBusSubscriber:
    """Async SSE subscriber for the cogos kernel bus stream.

    Usage::

        sub = KernelBusSubscriber()
        async for env in sub.stream():
            handle(env)

    `stream()` yields indefinitely; on any transport error it reconnects
    with exponential backoff clamped to [reconnect_min_s, reconnect_max_s].
    Call `close()` (or cancel the consuming task) to stop.
    """

    def __init__(
        self,
        url: str = KERNEL_BUS_STREAM_URL,
        *,
        bus_filter: str = "*",
        consumer_id: Optional[str] = None,
        reconnect_min_s: float = 1.0,
        reconnect_max_s: float = 30.0,
        request_timeout_s: float = 10.0,
    ) -> None:
        self._url = url
        self._bus_filter = bus_filter
        self._consumer_id = consumer_id
        self._min_backoff = reconnect_min_s
        self._max_backoff = reconnect_max_s
        self._request_timeout = request_timeout_s
        self._last_event_id: Optional[str] = None
        self._closed = asyncio.Event()
        self._client: Optional[httpx.AsyncClient] = None

    async def close(self) -> None:
        self._closed.set()
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - best-effort
                pass
            self._client = None

    def _build_params(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self._bus_filter and self._bus_filter != "*":
            params["bus_id"] = self._bus_filter
        if self._consumer_id:
            params["consumer"] = self._consumer_id
        return params

    def _build_headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        if self._last_event_id:
            # Harmless if the kernel doesn't honor it today; future protocol
            # bump may use it for resume.
            headers["Last-Event-ID"] = self._last_event_id
        return headers

    async def stream(self) -> AsyncIterator[BusEnvelope]:
        backoff = self._min_backoff
        # Generous read timeout — SSE is long-lived with 30s heartbeats.
        timeout = httpx.Timeout(self._request_timeout, read=None)
        while not self._closed.is_set():
            self._client = httpx.AsyncClient(timeout=timeout)
            try:
                async with self._client.stream(
                    "GET",
                    self._url,
                    params=self._build_params(),
                    headers=self._build_headers(),
                ) as resp:
                    if resp.status_code != 200:
                        logger.info(
                            "bus-bridge: non-200 from %s: %s — backing off %.1fs",
                            self._url, resp.status_code, backoff,
                        )
                        await self._sleep_or_close(backoff)
                        backoff = min(self._max_backoff, max(self._min_backoff, backoff * 2))
                        continue

                    logger.info("bus-bridge: connected to %s", self._url)
                    backoff = self._min_backoff  # reset on successful connect

                    async for envelope in self._iter_sse(resp):
                        yield envelope
            except (httpx.HTTPError, asyncio.TimeoutError, ConnectionError) as e:
                logger.info(
                    "bus-bridge: transport error (%s); reconnecting in %.1fs",
                    e.__class__.__name__, backoff,
                )
                await self._sleep_or_close(backoff)
                backoff = min(self._max_backoff, max(self._min_backoff, backoff * 2))
            except asyncio.CancelledError:
                await self.close()
                raise
            finally:
                if self._client is not None:
                    try:
                        await self._client.aclose()
                    except Exception:  # pragma: no cover
                        pass
                    self._client = None

    async def _sleep_or_close(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._closed.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    async def _iter_sse(self, resp: httpx.Response) -> AsyncIterator[BusEnvelope]:
        """Parse the SSE byte stream into BusEnvelope records.

        Minimal SSE parser: we accumulate field lines into the current event,
        dispatch on blank-line boundaries, silently skip comment lines
        (`: heartbeat`), and honor `data:`, `event:`, `id:` fields.
        """
        event_name: Optional[str] = None
        data_lines: list[str] = []
        event_id: Optional[str] = None

        async for raw_line in resp.aiter_lines():
            if self._closed.is_set():
                return
            # httpx strips the trailing \n but preserves empty lines.
            if raw_line == "":
                # Dispatch boundary.
                if data_lines:
                    env = self._parse_event(event_name, "\n".join(data_lines), event_id)
                    if env is not None:
                        yield env
                event_name = None
                data_lines = []
                event_id = None
                continue
            if raw_line.startswith(":"):
                # Comment line / heartbeat.
                continue
            field, _, value = raw_line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data_lines.append(value)
            elif field == "event":
                event_name = value
            elif field == "id":
                event_id = value
                self._last_event_id = value
            # retry / unknown fields: ignore

    def _parse_event(
        self, event_name: Optional[str], data: str, event_id: Optional[str]
    ) -> Optional[BusEnvelope]:
        try:
            envelope: Any = json.loads(data)
        except json.JSONDecodeError:
            logger.debug("bus-bridge: non-JSON data frame dropped: %r", data[:200])
            return None
        if not isinstance(envelope, dict):
            logger.debug("bus-bridge: non-object data frame dropped: %r", envelope)
            return None

        inner = envelope.get("data")
        payload: dict = inner if isinstance(inner, dict) else {}
        kind = _extract_kind(envelope, payload)
        ts = envelope.get("timestamp") or payload.get("ts") or payload.get("timestamp")
        eid = event_id or envelope.get("id")
        if eid and not self._last_event_id:
            self._last_event_id = eid

        if kind not in ("state_transition", "tool_dispatch", "assessment", "bus.event", "connected"):
            # Tolerate unknowns — just log and forward.
            logger.debug("bus-bridge: forwarding unknown event kind=%r", kind)

        return BusEnvelope(
            raw=envelope,
            kind=kind,
            payload=payload,
            ts=ts if isinstance(ts, str) else None,
            event_id=eid if isinstance(eid, str) else None,
        )


# ---------------------------------------------------------------------------
# Manual validation entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sub = KernelBusSubscriber()
    print(f"bus-bridge: subscribing to {sub._url} (Ctrl-C to stop)")
    try:
        async for env in sub.stream():
            print(
                json.dumps(
                    {
                        "kind": env.kind,
                        "ts": env.ts,
                        "id": env.event_id,
                        "payload_keys": sorted(env.payload.keys())[:12],
                    }
                )
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await sub.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass

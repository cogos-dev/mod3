#!/usr/bin/env python3
"""End-to-end smoke test for the Audio Loop + Trace Wiring plan.

Verifies the non-audio half of the loop:
  1. Kernel health endpoint reachable.
  2. Cycle-trace bus (bus_cycle_trace) emits ADR-083-shaped envelopes.
  3. The existing Mod3 KernelBusSubscriber can consume them.

Exit 0 if >=1 event of any recognized kind is validated within 60s.
Exit 1 with a clear error otherwise.

Usage:
    python3 demo/e2e_audio_trace_demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

# Allow running from repo root or from demo/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bus_bridge import KernelBusSubscriber  # noqa: E402

HEALTH_URL = "http://localhost:6931/health"
BUS = "bus_cycle_trace"
KNOWN_KINDS = {"state_transition", "tool_dispatch", "assessment"}
REQUIRED_FIELDS = ("id", "ts", "source", "cycle_id", "kind", "payload")
MAX_WAIT_S = 60
TARGET_EVENTS = 3


async def check_health() -> None:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(HEALTH_URL)
    if r.status_code != 200:
        raise SystemExit(f"health: HTTP {r.status_code}")
    try:
        body = r.json()
    except Exception as e:
        raise SystemExit(f"health: non-JSON body: {e}")
    print(f"[ok] kernel health: status={body.get('status')!r}")


def validate_envelope(env) -> tuple[bool, list[str], dict]:
    """ADR-083 envelope shape check.

    The kernel SSE frame is {id, type, timestamp, data: <CogBlock>}.
    The CogBlock is {bus_id, from, seq, hash, prev, prev_hash, ts, type, v, payload}.
    The ADR-083 CycleEvent (id/ts/source/cycle_id/kind/payload) is the CogBlock's
    payload field — that's what consumers validate.
    """
    cogblock = env.raw.get("data", {}) if isinstance(env.raw, dict) else {}
    inner = cogblock.get("payload") if isinstance(cogblock, dict) else {}
    if not isinstance(inner, dict):
        inner = {}
    missing = [f for f in REQUIRED_FIELDS if f not in inner]
    return (not missing), missing, inner


async def collect() -> int:
    sub = KernelBusSubscriber(bus_filter=BUS, consumer_id="e2e-audio-trace-demo")
    captured: list = []
    kinds = defaultdict(int)
    cycle_ids: set[str] = set()
    t0 = time.time()
    unknown_kinds: set[str] = set()
    first_ts: str | None = None
    last_ts: str | None = None

    print(f"[..] subscribing to {sub._url} bus={BUS} (up to {MAX_WAIT_S}s or {TARGET_EVENTS} events)")
    try:
        async def _run():
            nonlocal first_ts, last_ts
            async for env in sub.stream():
                if env.kind == "connected":
                    continue
                ok, missing, inner = validate_envelope(env)
                kind = inner.get("kind", env.kind)
                if not ok:
                    print(f"[warn] envelope missing fields {missing}: keys={sorted(inner.keys())}")
                    continue
                if kind not in KNOWN_KINDS:
                    unknown_kinds.add(kind)
                    print(f"[warn] unknown kind={kind!r} (tolerated per ADR-083)")
                kinds[kind] += 1
                cid = inner.get("cycle_id")
                if isinstance(cid, str):
                    cycle_ids.add(cid)
                ts = inner.get("ts")
                if isinstance(ts, str):
                    first_ts = first_ts or ts
                    last_ts = ts
                captured.append(inner)
                print(f"[evt] kind={kind} cycle_id={cid} ts={ts} payload_keys={sorted((inner.get('payload') or {}).keys())[:6]}")
                if len(captured) >= TARGET_EVENTS:
                    return

        await asyncio.wait_for(_run(), timeout=MAX_WAIT_S)
    except asyncio.TimeoutError:
        print(f"[..] timeout reached after {MAX_WAIT_S}s; collected {len(captured)} event(s)")
    finally:
        await sub.close()

    elapsed = time.time() - t0
    print("\n=== SUMMARY ===")
    print(f"elapsed: {elapsed:.1f}s  events: {len(captured)}  distinct cycle_ids: {len(cycle_ids)}")
    print(f"time window: {first_ts} .. {last_ts}")
    print(f"count per kind: {dict(kinds)}")
    if unknown_kinds:
        print(f"unknown kinds (tolerated): {sorted(unknown_kinds)}")
    for k in sorted(kinds):
        sample = next((e for e in captured if e.get("kind") == k), None)
        if sample:
            print(f"sample[{k}]: {json.dumps(sample, default=str)[:220]}")

    if len(captured) == 0:
        print("[FAIL] no events captured from bus_cycle_trace within window.", file=sys.stderr)
        return 1
    if not any(k in KNOWN_KINDS for k in kinds):
        print("[FAIL] captured events but none matched known ADR-083 kinds.", file=sys.stderr)
        return 1
    print("[PASS] >=1 ADR-083-shaped event validated.")
    return 0


async def main() -> int:
    await check_health()
    return await collect()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Tests for bus_bridge_runner.run_bridge — filter + fan-out behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the mod3 package root importable (tests live in tests/ subfolder).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bus_bridge import BusEnvelope  # noqa: E402
from bus_bridge_runner import ADR083_KINDS, run_bridge  # noqa: E402


class _FakeSubscriber:
    """Minimal stand-in for KernelBusSubscriber: yields canned envelopes then stops."""

    def __init__(self, envelopes: list[BusEnvelope]) -> None:
        self._envelopes = envelopes

    async def stream(self):
        for env in self._envelopes:
            yield env


def _env(kind: str, payload: dict | None = None, event_id: str = "e1") -> BusEnvelope:
    return BusEnvelope(
        raw={"type": "bus.event", "data": payload or {"kind": kind}},
        kind=kind,
        payload=payload or {"kind": kind, "cycle_id": "c1"},
        ts="2026-04-17T00:00:00Z",
        event_id=event_id,
    )


def test_run_bridge_forwards_only_filtered_kinds():
    envelopes = [
        _env("state_transition", {"kind": "state_transition", "cycle_id": "c1"}, "e1"),
        _env("unknown_future_kind", {"kind": "unknown_future_kind"}, "e2"),
        _env("tool_dispatch", {"kind": "tool_dispatch", "cycle_id": "c1"}, "e3"),
        _env("connected", {}, "e4"),
    ]
    sub = _FakeSubscriber(envelopes)

    with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event") as mock_bcast:
        asyncio.run(run_bridge(sub, filter_kinds=set(ADR083_KINDS)))

    # Only the two ADR-083 envelopes should have been forwarded.
    assert mock_bcast.call_count == 2
    kinds_forwarded = [c.args[0]["kind"] for c in mock_bcast.call_args_list]
    assert kinds_forwarded == ["state_transition", "tool_dispatch"]


def test_run_bridge_no_filter_forwards_all_nonconnected():
    envelopes = [
        _env("state_transition", {"kind": "state_transition"}, "e1"),
        _env("assessment", {"kind": "assessment"}, "e2"),
        _env("weird.ns.kind", {"kind": "weird.ns.kind"}, "e3"),
        _env("connected", {}, "e4"),  # bootstrap frame — always skipped
    ]
    sub = _FakeSubscriber(envelopes)

    with patch("bus_bridge_runner.BrowserChannel.broadcast_trace_event") as mock_bcast:
        asyncio.run(run_bridge(sub, filter_kinds=None))

    assert mock_bcast.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

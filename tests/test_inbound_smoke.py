"""Non-swallowing smoke test for the inbound voice pipeline.

Guards the regression where ``inbound.py`` imported now-removed symbols
(``emit_channel_event`` / ``emit_permission_verdict``) from ``server``,
which made the default-on server-side mic→VAD→STT pipeline silently fail to
start (the start path in ``server._start_inbound_pipeline_if_enabled`` swallows
the resulting ImportError, leaving ``/health`` reporting ``vad:false``).

These tests deliberately do NOT swallow import / start errors, so the
regression cannot silently recur:

  * ``test_inbound_imports_cleanly`` — import the module with no stubbing.
  * ``test_inbound_does_not_import_dead_server_symbols`` — assert the dead
    ``from server import ...`` line is gone.
  * ``test_start_inbound_pipeline_actually_starts`` — with MOD3_INBOUND_ENABLED=1
    the start function returns a running pipeline (not None).
  * ``test_emit_notification_fans_out_transcript`` — a transcript reaches a
    registered seat via the seat registry (the modern delivery path).

No real microphone, audio device, or live daemon is touched: AudioCapture is
mocked and the seat registry is the in-process singleton.

Run with:
  PYTHONPATH=. python -m pytest tests/test_inbound_smoke.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_inbound_imports_cleanly():
    """inbound must import with no 'server' stubbing — the dead import is gone.

    A bare ``import inbound`` is the exact operation that fails on the broken
    tree (ImportError: cannot import name 'emit_channel_event' from 'server').
    No try/except here: a regression surfaces as a hard test failure.
    """
    if "inbound" in sys.modules:
        del sys.modules["inbound"]
    import inbound  # noqa: F401

    assert hasattr(inbound, "InboundPipeline")


def test_inbound_does_not_import_dead_server_symbols():
    """inbound must not import the removed symbols from 'server'.

    Checks the actual import statements, not prose: the regression signature
    is a top-level ``from server import emit_channel_event ...`` (or any import
    of those names), which raises ImportError at module load. The symbol names
    may legitimately appear in docstrings explaining the history.
    """
    import ast

    src = (Path(__file__).resolve().parents[1] / "inbound.py").read_text()
    tree = ast.parse(src)
    dead = {"emit_channel_event", "emit_permission_verdict"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported = {alias.name for alias in node.names}
            assert not (dead & imported), f"inbound imports removed symbols {dead & imported} from {node.module!r}"


def test_start_inbound_pipeline_actually_starts(monkeypatch):
    """server._start_inbound_pipeline_if_enabled returns a running pipeline.

    This exercises the real default-on start path. The path swallows
    exceptions internally (by design — a missing mic must not abort the
    daemon), so here we assert on the *observable result* instead: with
    MOD3_INBOUND_ENABLED=1 the function must return a non-None, running
    pipeline. If the import/start regresses, the swallow turns it into a
    silent ``return None`` and this assertion fails loudly.
    """
    monkeypatch.setenv("MOD3_INBOUND_ENABLED", "1")

    # Mock AudioCapture so no real microphone is opened.
    mock_capture = MagicMock()
    mock_capture.is_active.return_value = False

    for mod in ("inbound", "server"):
        sys.modules.pop(mod, None)

    with patch("capture.AudioCapture", return_value=mock_capture):
        import server

        pipeline = server._start_inbound_pipeline_if_enabled()
        try:
            assert pipeline is not None, "inbound pipeline failed to start (import/start error was swallowed)"
            assert pipeline.is_running, "pipeline returned but not running"
        finally:
            if pipeline is not None:
                pipeline.stop()


def test_start_inbound_pipeline_respects_disable(monkeypatch):
    """MOD3_INBOUND_ENABLED=0 must return None without starting anything."""
    monkeypatch.setenv("MOD3_INBOUND_ENABLED", "0")
    sys.modules.pop("server", None)
    import server

    assert server._start_inbound_pipeline_if_enabled() is None


def test_emit_notification_fans_out_transcript():
    """A transcript reaches a registered seat via the seat registry.

    Verifies the rewired delivery path: _emit_notification → fan_out_all →
    the seat's SSE queue carries a ``user_message`` event with input_type=voice.
    """
    if "inbound" in sys.modules:
        del sys.modules["inbound"]

    from seats import get_seat_registry
    from vad import VADResult

    mock_capture = MagicMock()
    mock_capture.is_active.return_value = False

    with patch("capture.AudioCapture", return_value=mock_capture):
        from inbound import InboundPipeline

        pipeline = InboundPipeline(bus=MagicMock(), pipeline_state=MagicMock())

    registry = get_seat_registry()
    seat = registry.register(
        session_id="smoke-session",
        client_type="claude-code-channel",
        device_uuid="smoke-device",
    )

    event = MagicMock()
    event.content = "hello from the mic"
    event.confidence = 0.95
    vad = VADResult(
        has_speech=True,
        confidence=0.9,
        speech_ratio=0.8,
        num_segments=1,
        total_speech_sec=1.0,
        total_audio_sec=2.0,
    )

    try:
        pipeline._emit_notification(event, vad)

        delivered = seat.queue.get_nowait()
        assert delivered["type"] == "user_message"
        assert delivered["content"] == "hello from the mic"
        assert delivered["input_type"] == "voice"
        assert delivered["role"] == "user"
    finally:
        registry.revoke("smoke-session", seat.seat_id)

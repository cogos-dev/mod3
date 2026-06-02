"""Tests for tts_chunk and bargein seat SSE fan-out events.

Covers:
  * tts_chunk event schema (field names + types) after synthesis
  * fan_out is called per synthesized chunk with the right shape
  * is_final sentinel emitted when engine never marks is_final=True
  * bargein event schema when pipeline_state.interrupt() fires
  * bargein fan_out is called with session_id / job_id / reason
  * pipeline_state.interrupt() callbacks: add/remove, multi-callback, fire-once

Run with:
  PYTHONPATH=. .venv/bin/python -m pytest tests/test_tts_chunk_bargein_seat_events.py -v
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# PipelineState interrupt-callback unit tests
# ---------------------------------------------------------------------------


class TestPipelineStateInterruptCallbacks:
    def _make_state(self):
        from pipeline_state import PipelineState

        return PipelineState()

    def test_add_and_remove_callback(self):
        state = self._make_state()
        cb = MagicMock()
        state.add_interrupt_callback(cb)
        assert cb in state._interrupt_callbacks
        state.remove_interrupt_callback(cb)
        assert cb not in state._interrupt_callbacks

    def test_remove_nonexistent_is_noop(self):
        state = self._make_state()
        cb = MagicMock()
        state.remove_interrupt_callback(cb)  # must not raise

    def test_callback_not_fired_when_not_speaking(self):
        state = self._make_state()
        cb = MagicMock()
        state.add_interrupt_callback(cb)
        result = state.interrupt(reason="test")
        assert result is None
        cb.assert_not_called()

    def test_callback_fired_on_interrupt(self):
        from pipeline_state import PipelineState

        state = PipelineState()
        cb = MagicMock()
        state.add_interrupt_callback(cb)

        player = MagicMock()
        player.flush = MagicMock()
        state.start_speaking("hello world", player)
        info = state.interrupt(reason="vad_reflex")

        assert info is not None
        cb.assert_called_once_with(info)
        assert cb.call_args[0][0].reason == "vad_reflex"

    def test_multiple_callbacks_all_fired(self):
        from pipeline_state import PipelineState

        state = PipelineState()
        cb1, cb2 = MagicMock(), MagicMock()
        state.add_interrupt_callback(cb1)
        state.add_interrupt_callback(cb2)

        player = MagicMock()
        state.start_speaking("hi", player)
        state.interrupt(reason="manual")

        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_callback_removed_before_interrupt_not_fired(self):
        from pipeline_state import PipelineState

        state = PipelineState()
        cb = MagicMock()
        state.add_interrupt_callback(cb)
        state.remove_interrupt_callback(cb)

        player = MagicMock()
        state.start_speaking("hi", player)
        state.interrupt(reason="test")

        cb.assert_not_called()

    def test_failing_callback_does_not_crash_interrupt(self):
        from pipeline_state import PipelineState

        state = PipelineState()

        def bad_cb(info):
            raise RuntimeError("callback exploded")

        state.add_interrupt_callback(bad_cb)
        player = MagicMock()
        state.start_speaking("hi", player)
        # Must not raise even though callback raises
        info = state.interrupt(reason="test")
        assert info is not None


# ---------------------------------------------------------------------------
# Helpers for _run_speech_job tests
# ---------------------------------------------------------------------------


def _make_fake_chunk(samples, sample_rate=24000, is_final=False):
    chunk = MagicMock()
    import numpy as np

    chunk.samples = np.zeros(samples, dtype=np.float32)
    chunk.sample_rate = sample_rate
    chunk.metadata = {"is_final": is_final, "engine": "mock", "gen_time_sec": 0.01, "rtf": 0.1}
    return chunk


def _drain_seat(seat) -> list[dict]:
    items = []
    while True:
        try:
            items.append(seat.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


# ---------------------------------------------------------------------------
# tts_chunk schema tests — shape of each emitted event
# ---------------------------------------------------------------------------

TTS_CHUNK_REQUIRED_FIELDS = {
    "type": str,
    "job_id": str,
    "chunk_index": int,
    "text": str,
    "audio_base64": str,
    "format": str,
    "is_final": bool,
    "session_id": str,
}


class TestTtsChunkSchema:
    """Verify the tts_chunk event shape matches the contract WI-3 parses."""

    def _build_event(self, chunk_index=0, is_final=False, audio_b64="dGVzdA=="):
        return {
            "type": "tts_chunk",
            "job_id": "abc123",
            "chunk_index": chunk_index,
            "text": "Hello world",
            "audio_base64": audio_b64,
            "format": "ogg",
            "is_final": is_final,
            "session_id": "sess-1",
        }

    def test_all_required_fields_present(self):
        event = self._build_event()
        for field in TTS_CHUNK_REQUIRED_FIELDS:
            assert field in event, f"missing field: {field}"

    def test_field_types_correct(self):
        event = self._build_event()
        for field, expected_type in TTS_CHUNK_REQUIRED_FIELDS.items():
            assert isinstance(event[field], expected_type), (
                f"field {field!r}: expected {expected_type.__name__}, got {type(event[field]).__name__}"
            )

    def test_format_is_ogg(self):
        event = self._build_event()
        assert event["format"] == "ogg"

    def test_type_is_tts_chunk(self):
        event = self._build_event()
        assert event["type"] == "tts_chunk"

    def test_is_final_true_on_last_chunk(self):
        event = self._build_event(is_final=True)
        assert event["is_final"] is True

    def test_chunk_index_monotonic(self):
        events = [self._build_event(chunk_index=i) for i in range(3)]
        indices = [e["chunk_index"] for e in events]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# bargein schema tests
# ---------------------------------------------------------------------------

BARGEIN_REQUIRED_FIELDS = {
    "type": str,
    "session_id": str,
    "reason": str,
}


class TestBargeinSchema:
    """Verify the bargein event shape matches the contract WI-3 parses."""

    def _build_event(self, job_id="abc123"):
        return {
            "type": "bargein",
            "session_id": "sess-1",
            "job_id": job_id,
            "reason": "vad_reflex",
        }

    def test_all_required_fields_present(self):
        event = self._build_event()
        for field in BARGEIN_REQUIRED_FIELDS:
            assert field in event, f"missing field: {field}"

    def test_job_id_present_and_str_or_none(self):
        event = self._build_event()
        assert event["job_id"] is None or isinstance(event["job_id"], str)

    def test_type_is_bargein(self):
        event = self._build_event()
        assert event["type"] == "bargein"

    def test_reason_is_string(self):
        event = self._build_event()
        assert isinstance(event["reason"], str)


# ---------------------------------------------------------------------------
# Integration: _run_speech_job fan-out behaviour (mocked synth + seats)
# ---------------------------------------------------------------------------


class TestRunSpeechJobFanOut:
    """Assert fan_out is called with the right event shapes in _run_speech_job.

    Synthesis is mocked (MOD3_WORKER_MOCK-style) so no TTS engine is needed.
    The seat registry fan_out is patched to capture calls.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        # Patch engine module
        mock_engine = MagicMock()

        def _gen_two_chunks(text, **kwargs):
            yield _make_fake_chunk(2400, is_final=False)
            yield _make_fake_chunk(2400, is_final=True)

        mock_engine.generate_audio.side_effect = _gen_two_chunks
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        # Patch AdaptivePlayer
        mock_player = MagicMock()
        mock_player.get_progress.return_value = (0, 0)
        mock_player.wait.return_value = MagicMock(to_dict=lambda: {})

        # Patch _encode_chunk_ogg to avoid soundfile dependency in tests
        ogg_bytes = b"fakeOGGdata"
        import base64

        fake_b64 = base64.b64encode(ogg_bytes).decode("ascii")
        monkeypatch.setattr("server._encode_chunk_ogg", lambda s, r: ogg_bytes)

        # Patch seat registry fan_out
        mock_registry = MagicMock()
        mock_registry.fan_out = MagicMock(return_value=1)

        self.mock_engine = mock_engine
        self.mock_player = mock_player
        self.mock_registry = mock_registry
        self.fake_b64 = fake_b64

    def _run(self, session_id="sess-test"):
        import server

        entry = {
            "job_id": "job001",
            "text": "Hello seat",
            "voice": "bm_lewis",
            "stream": False,
            "session_id": session_id,
        }

        with (
            patch("server._engine_module", return_value=self.mock_engine),
            patch("server._adaptive_player_class", return_value=lambda **kw: self.mock_player),
            patch("server._resolve_voice_via_bus", return_value=("kokoro", "bm_lewis")),
            patch("server._resolve_device_for_entry", return_value=(None, None)),
            patch("server._acquire_speaking_lock", return_value=False),
            patch("server._release_speaking_lock"),
            patch("server._set_bus_voice_state"),
            patch("server._jobs", {entry["job_id"]: {"status": "speaking"}}),
            patch("seats.get_seat_registry", return_value=self.mock_registry),
        ):
            server._run_speech_job(entry)

        return self.mock_registry.fan_out.call_args_list

    def test_fan_out_called_per_chunk(self):
        calls = self._run()
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        # 2 real chunks (both have samples > 0)
        assert len(tts_calls) == 2, f"expected 2 tts_chunk fan_out calls, got {len(tts_calls)}"

    def test_tts_chunk_session_id_matches(self):
        calls = self._run(session_id="my-session")
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        for c in tts_calls:
            assert c.args[0] == "my-session"
            assert c.args[1]["session_id"] == "my-session"

    def test_tts_chunk_job_id_matches(self):
        calls = self._run()
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        for c in tts_calls:
            assert c.args[1]["job_id"] == "job001"

    def test_tts_chunk_monotonic_index(self):
        calls = self._run()
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        indices = [c.args[1]["chunk_index"] for c in tts_calls]
        assert indices == list(range(len(indices)))

    def test_tts_chunk_format_ogg(self):
        calls = self._run()
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        for c in tts_calls:
            assert c.args[1]["format"] == "ogg"

    def test_tts_chunk_last_is_final(self):
        calls = self._run()
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        assert tts_calls[-1].args[1]["is_final"] is True

    def test_tts_chunk_has_audio_base64(self):
        calls = self._run()
        tts_calls = [c for c in calls if c.args[1].get("type") == "tts_chunk"]
        for c in tts_calls:
            b64 = c.args[1]["audio_base64"]
            assert isinstance(b64, str) and len(b64) > 0

    def test_no_fan_out_without_session_id(self):
        """No seat events when session_id is absent."""
        import server

        entry = {
            "job_id": "job002",
            "text": "No session",
            "voice": "bm_lewis",
            "stream": False,
            # no session_id
        }
        with (
            patch("server._engine_module", return_value=self.mock_engine),
            patch("server._adaptive_player_class", return_value=lambda **kw: self.mock_player),
            patch("server._resolve_voice_via_bus", return_value=("kokoro", "bm_lewis")),
            patch("server._resolve_device_for_entry", return_value=(None, None)),
            patch("server._acquire_speaking_lock", return_value=False),
            patch("server._release_speaking_lock"),
            patch("server._set_bus_voice_state"),
            patch("server._jobs", {entry["job_id"]: {"status": "speaking"}}),
            patch("seats.get_seat_registry", return_value=self.mock_registry),
        ):
            server._run_speech_job(entry)

        tts_calls = [c for c in self.mock_registry.fan_out.call_args_list if c.args[1].get("type") == "tts_chunk"]
        assert len(tts_calls) == 0


class TestRunSpeechJobBargeinFanOut:
    """Assert bargein fan_out is called when pipeline_state.interrupt() fires."""

    def test_bargein_fan_out_on_interrupt(self, monkeypatch):
        import server
        from pipeline_state import PipelineState

        # Use a real PipelineState so callbacks actually fire
        real_state = PipelineState()
        monkeypatch.setattr(server, "pipeline_state", real_state)

        mock_engine = MagicMock()

        def _gen_then_interrupt(text, **kwargs):
            # Yield one chunk, then simulate barge-in during synthesis
            yield _make_fake_chunk(2400, is_final=False)
            # Interrupt fires as if VAD detected speech
            real_state.interrupt(reason="vad_reflex")

        mock_engine.generate_audio.side_effect = _gen_then_interrupt
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        mock_player = MagicMock()
        mock_player.get_progress.return_value = (0, 0)
        mock_player.wait.return_value = MagicMock(to_dict=lambda: {})

        mock_registry = MagicMock()
        mock_registry.fan_out = MagicMock(return_value=1)

        monkeypatch.setattr("server._encode_chunk_ogg", lambda s, r: b"fakeOGG")

        entry = {
            "job_id": "job-bargein",
            "text": "Interrupt me",
            "voice": "bm_lewis",
            "stream": False,
            "session_id": "sess-bargein",
        }

        with (
            patch("server._engine_module", return_value=mock_engine),
            patch("server._adaptive_player_class", return_value=lambda **kw: mock_player),
            patch("server._resolve_voice_via_bus", return_value=("kokoro", "bm_lewis")),
            patch("server._resolve_device_for_entry", return_value=(None, None)),
            patch("server._acquire_speaking_lock", return_value=False),
            patch("server._release_speaking_lock"),
            patch("server._set_bus_voice_state"),
            patch("server._jobs", {entry["job_id"]: {"status": "speaking"}}),
            patch("seats.get_seat_registry", return_value=mock_registry),
        ):
            # start_speaking must be called so interrupt() actually fires
            real_state.start_speaking("Interrupt me", mock_player)
            server._run_speech_job(entry)

        bargein_calls = [c for c in mock_registry.fan_out.call_args_list if c.args[1].get("type") == "bargein"]
        assert len(bargein_calls) >= 1, "expected at least one bargein fan_out call"

        ev = bargein_calls[0].args[1]
        assert ev["type"] == "bargein"
        assert ev["session_id"] == "sess-bargein"
        assert ev["job_id"] == "job-bargein"
        assert isinstance(ev["reason"], str)

    def test_bargein_callback_removed_after_job(self, monkeypatch):
        """Callback must be removed from pipeline_state after _run_speech_job completes."""
        import server
        from pipeline_state import PipelineState

        real_state = PipelineState()
        monkeypatch.setattr(server, "pipeline_state", real_state)

        mock_engine = MagicMock()
        mock_engine.generate_audio.return_value = iter([_make_fake_chunk(2400, is_final=True)])
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        mock_player = MagicMock()
        mock_player.get_progress.return_value = (0, 0)
        mock_player.wait.return_value = MagicMock(to_dict=lambda: {})

        monkeypatch.setattr("server._encode_chunk_ogg", lambda s, r: b"fakeOGG")

        entry = {
            "job_id": "job-cleanup",
            "text": "Clean up",
            "voice": "bm_lewis",
            "stream": False,
            "session_id": "sess-cleanup",
        }
        with (
            patch("server._engine_module", return_value=mock_engine),
            patch("server._adaptive_player_class", return_value=lambda **kw: mock_player),
            patch("server._resolve_voice_via_bus", return_value=("kokoro", "bm_lewis")),
            patch("server._resolve_device_for_entry", return_value=(None, None)),
            patch("server._acquire_speaking_lock", return_value=False),
            patch("server._release_speaking_lock"),
            patch("server._set_bus_voice_state"),
            patch("server._jobs", {entry["job_id"]: {"status": "speaking"}}),
            patch("seats.get_seat_registry", return_value=MagicMock()),
        ):
            server._run_speech_job(entry)

        # After job, pipeline_state must have no lingering callbacks
        assert len(real_state._interrupt_callbacks) == 0


# ---------------------------------------------------------------------------
# Sentinel: is_final emitted when engine never marks any chunk final
# ---------------------------------------------------------------------------


class TestTtsChunkFinalSentinel:
    """When no engine chunk has is_final=True, a sentinel tts_chunk is emitted."""

    def test_sentinel_emitted_if_no_chunk_is_final(self, monkeypatch):
        import server

        mock_engine = MagicMock()

        def _gen_no_final(text, **kwargs):
            yield _make_fake_chunk(2400, is_final=False)
            yield _make_fake_chunk(2400, is_final=False)

        mock_engine.generate_audio.side_effect = _gen_no_final
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        mock_player = MagicMock()
        mock_player.get_progress.return_value = (0, 0)
        mock_player.wait.return_value = MagicMock(to_dict=lambda: {})

        mock_registry = MagicMock()
        mock_registry.fan_out = MagicMock(return_value=1)

        monkeypatch.setattr("server._encode_chunk_ogg", lambda s, r: b"OGG")

        entry = {
            "job_id": "job-sentinel",
            "text": "Sentinel test",
            "voice": "bm_lewis",
            "stream": False,
            "session_id": "sess-sentinel",
        }
        with (
            patch("server._engine_module", return_value=mock_engine),
            patch("server._adaptive_player_class", return_value=lambda **kw: mock_player),
            patch("server._resolve_voice_via_bus", return_value=("kokoro", "bm_lewis")),
            patch("server._resolve_device_for_entry", return_value=(None, None)),
            patch("server._acquire_speaking_lock", return_value=False),
            patch("server._release_speaking_lock"),
            patch("server._set_bus_voice_state"),
            patch("server._jobs", {entry["job_id"]: {"status": "speaking"}}),
            patch("seats.get_seat_registry", return_value=mock_registry),
        ):
            server._run_speech_job(entry)

        tts_calls = [c for c in mock_registry.fan_out.call_args_list if c.args[1].get("type") == "tts_chunk"]
        # 2 real + 1 sentinel
        assert len(tts_calls) == 3
        sentinel = tts_calls[-1].args[1]
        assert sentinel["is_final"] is True
        assert sentinel["audio_base64"] == ""

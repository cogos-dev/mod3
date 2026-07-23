"""Tests for job finalization on barge-in (2026-07-23 production incident).

Field evidence: a job barged-in mid-utterance (mic VAD detected real speech,
``pipeline_state.interrupt()`` correctly silenced the player — "paused local
playback (36% delivered)" in the daemon log) stayed at status "speaking" /
HUD "encoding" / progress 0.0 for minutes afterward, never reaching a
terminal state.

Root cause: ``pipeline_state.interrupt()`` calls ``player.flush()`` directly,
which stops audio immediately, but by itself never told
``_run_speech_job``'s generation loop to stop calling the TTS engine for more
chunks — the loop's only prior stop signal was the file-based speaking-lock
check, which an in-process interrupt never touches. Left unfixed, the loop
kept synthesizing the rest of the (unheard) text and the job stayed
"speaking" until that finished naturally.

Fix: ``_on_bargein`` (the interrupt callback already registered per-job) now
also records the ``InterruptInfo`` in ``_interrupt_state``, which the
generation loop checks every iteration alongside the pre-existing lock-loss
check; either path now finalizes the job as "interrupted" with the partial
spoken_pct/delivered_text/reason instead of "done".

Run: python3 -m pytest tests/test_barge_in_finalization.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_fake_chunk(samples: int, sample_rate: int = 24000, is_final: bool = False):
    chunk = MagicMock()
    chunk.samples = np.zeros(samples, dtype=np.float32)
    chunk.sample_rate = sample_rate
    chunk.metadata = {"is_final": is_final, "engine": "mock", "gen_time_sec": 0.01, "rtf": 0.1}
    return chunk


def _make_entry(job_id: str, text: str = "Interrupt me please", session_id: str | None = None) -> dict:
    return {
        "job_id": job_id,
        "text": text,
        "voice": "bm_lewis",
        "stream": False,
        "session_id": session_id,
    }


@pytest.fixture
def mock_player():
    player = MagicMock()
    player.get_progress.return_value = (0, 0)
    player.wait.return_value = MagicMock(
        to_dict=lambda: {"buffer": {"initial_buffer_ms": 150.0, "final_buffer_ms": 150.0, "starvation_count": 0}}
    )
    return player


class TestInProcessBargeinFinalizesJob:
    """pipeline_state.interrupt() firing mid-generation must finalize the job."""

    def test_interrupted_job_reaches_terminal_status_with_partial_metrics(self, mock_player):
        import jobs_registry
        from pipeline_state import PipelineState

        real_state = PipelineState()

        mock_engine = MagicMock()

        def _gen_then_interrupt(text, **kwargs):
            # First chunk plays, then a real barge-in fires mid-utterance —
            # mirrors the mic-VAD path that stopped the daemon's own job.
            yield _make_fake_chunk(2400, is_final=False)
            real_state.interrupt(reason="vad_reflex")
            # The engine keeps yielding more of the (now unheard) text —
            # this is exactly what wedged the job before the fix.
            yield _make_fake_chunk(2400, is_final=False)
            yield _make_fake_chunk(2400, is_final=True)

        mock_engine.generate_audio.side_effect = _gen_then_interrupt
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        job_id = "job-bargein-fin"
        entry = _make_entry(job_id)

        with (
            patch.object(jobs_registry, "pipeline_state", real_state),
            patch.object(jobs_registry, "_jobs", {job_id: {"status": "speaking", "error": None}}),
            patch.object(jobs_registry, "_engine_module", lambda: mock_engine),
            patch.object(jobs_registry, "_adaptive_player_class", lambda: (lambda **kw: mock_player)),
            patch.object(jobs_registry, "_resolve_voice_via_bus", lambda v: ("kokoro", v)),
            patch.object(jobs_registry, "_resolve_device_for_entry", lambda e: (None, None)),
            patch.object(jobs_registry, "_acquire_speaking_lock", lambda jid, txt: True),
            # Never let the lock-loss branch fire — this test isolates the
            # in-process interrupt path specifically.
            patch.object(jobs_registry, "_i_own_speaking_lock", lambda jid: True),
            patch.object(jobs_registry, "_release_speaking_lock", lambda jid: None),
            patch.object(jobs_registry, "_set_bus_voice_state", MagicMock()),
        ):
            jobs_registry._run_speech_job(entry)
            job = jobs_registry._jobs[job_id]

        assert job["status"] == "interrupted", f"expected 'interrupted', got {job['status']!r}"
        assert job["status"] != "done", "an interrupted job must not be mislabeled as a clean completion"

        interrupted = job.get("interrupted")
        assert interrupted is not None, "job record must carry partial-delivery info"
        assert interrupted["reason"] == "vad_reflex"
        assert 0.0 <= interrupted["spoken_pct"] <= 1.0
        assert interrupted["full_text"] == entry["text"]
        # delivered_text is a strict prefix of full_text (word-boundary trimmed)
        assert entry["text"].startswith(interrupted["delivered_text"])

        # Buffer fields from feature 2 (adaptive pre-playback buffer) still
        # land in metrics even for an interrupted job.
        assert "buffer" in job["metrics"]
        assert "starvation_count" in job["metrics"]["buffer"]

        assert job.get("end_time") is not None, "interrupted job must still get an end_time (retention needs it)"

    def test_generation_loop_stops_promptly_after_interrupt(self, mock_player):
        """The engine must not be asked for chunks forever after an interrupt.

        Regression: before the fix, nothing told the loop to stop, so it kept
        consuming the generator until the underlying (mocked, here — real,
        in production) engine ran out of text on its own. This test caps the
        fake generator at a large chunk count and asserts the loop exits long
        before exhausting it.
        """
        import jobs_registry
        from pipeline_state import PipelineState

        real_state = PipelineState()
        chunks_requested = {"count": 0}

        mock_engine = MagicMock()

        def _gen_many_chunks_interrupt_after_first(text, **kwargs):
            for i in range(200):  # a "long" utterance
                chunks_requested["count"] += 1
                yield _make_fake_chunk(2400, is_final=(i == 199))
                if i == 0:
                    real_state.interrupt(reason="vad_reflex")

        mock_engine.generate_audio.side_effect = _gen_many_chunks_interrupt_after_first
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        job_id = "job-bargein-long"
        entry = _make_entry(job_id, text="word " * 400)

        with (
            patch.object(jobs_registry, "pipeline_state", real_state),
            patch.object(jobs_registry, "_jobs", {job_id: {"status": "speaking", "error": None}}),
            patch.object(jobs_registry, "_engine_module", lambda: mock_engine),
            patch.object(jobs_registry, "_adaptive_player_class", lambda: (lambda **kw: mock_player)),
            patch.object(jobs_registry, "_resolve_voice_via_bus", lambda v: ("kokoro", v)),
            patch.object(jobs_registry, "_resolve_device_for_entry", lambda e: (None, None)),
            patch.object(jobs_registry, "_acquire_speaking_lock", lambda jid, txt: True),
            patch.object(jobs_registry, "_i_own_speaking_lock", lambda jid: True),
            patch.object(jobs_registry, "_release_speaking_lock", lambda jid: None),
            patch.object(jobs_registry, "_set_bus_voice_state", MagicMock()),
        ):
            jobs_registry._run_speech_job(entry)
            job = jobs_registry._jobs[job_id]

        assert job["status"] == "interrupted"
        # Stopped at (or right after) the interrupt, nowhere near all 200.
        assert chunks_requested["count"] < 10, (
            f"generation loop kept pulling chunks after interrupt: requested {chunks_requested['count']}"
        )


class TestCrossProcessLockLossAlsoFinalizes:
    """The pre-existing lock-file-loss path must finalize identically."""

    def test_lock_loss_finalizes_as_interrupted(self, mock_player):
        import jobs_registry
        from pipeline_state import PipelineState

        real_state = PipelineState()

        mock_engine = MagicMock()

        def _gen_two_chunks(text, **kwargs):
            yield _make_fake_chunk(2400, is_final=False)
            yield _make_fake_chunk(2400, is_final=True)

        mock_engine.generate_audio.side_effect = _gen_two_chunks
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_engine.get_model.return_value = mock_model

        job_id = "job-lock-lost"
        entry = _make_entry(job_id)

        with (
            patch.object(jobs_registry, "pipeline_state", real_state),
            patch.object(jobs_registry, "_jobs", {job_id: {"status": "speaking", "error": None}}),
            patch.object(jobs_registry, "_engine_module", lambda: mock_engine),
            patch.object(jobs_registry, "_adaptive_player_class", lambda: (lambda **kw: mock_player)),
            patch.object(jobs_registry, "_resolve_voice_via_bus", lambda v: ("kokoro", v)),
            patch.object(jobs_registry, "_resolve_device_for_entry", lambda e: (None, None)),
            patch.object(jobs_registry, "_acquire_speaking_lock", lambda jid, txt: True),
            # We think we hold the lock, but it's already gone -- the
            # cross-process branch (server.py:965 pre-fix) fires.
            patch.object(jobs_registry, "_i_own_speaking_lock", lambda jid: False),
            patch.object(jobs_registry, "_release_speaking_lock", lambda jid: None),
            patch.object(jobs_registry, "_set_bus_voice_state", MagicMock()),
        ):
            jobs_registry._run_speech_job(entry)
            job = jobs_registry._jobs[job_id]

        assert job["status"] == "interrupted"
        assert job["interrupted"]["reason"] == "cross_process_lock_lost"


class TestManualStopRoutesThroughPipelineState:
    """stop() on the active job should also finalize it via pipeline_state,
    not a bare player.flush() the job record never learns about."""

    def test_stop_active_job_uses_pipeline_state_interrupt(self):
        import jobs_registry

        fake_pipeline = MagicMock()
        fake_pipeline.interrupt.return_value = MagicMock(reason="manual_stop")
        mock_player = MagicMock()

        with (
            patch.object(jobs_registry, "pipeline_state", fake_pipeline),
            patch.object(jobs_registry, "_speech_queue") as mock_queue,
            patch.object(jobs_registry, "_current_player", mock_player),
        ):
            mock_queue.cancel.return_value = False
            mock_queue.active_job_id = "active-job"
            mock_queue.depth = 0

            jobs_registry.stop(job_id="active-job")

        fake_pipeline.interrupt.assert_called_once_with(reason="manual_stop")
        # pipeline_state.interrupt() returned non-None, so the direct-flush
        # fallback must NOT also fire (would double-stop / mask the real path).
        mock_player.flush.assert_not_called()

    def test_stop_falls_back_to_flush_when_pipeline_state_had_nothing(self):
        """Edge case: a player exists but pipeline_state didn't think anything
        was speaking. stop() must still silence audio directly."""
        import jobs_registry

        fake_pipeline = MagicMock()
        fake_pipeline.interrupt.return_value = None
        mock_player = MagicMock()

        with (
            patch.object(jobs_registry, "pipeline_state", fake_pipeline),
            patch.object(jobs_registry, "_speech_queue") as mock_queue,
            patch.object(jobs_registry, "_current_player", mock_player),
        ):
            mock_queue.cancel.return_value = False
            mock_queue.active_job_id = "active-job"
            mock_queue.depth = 0

            jobs_registry.stop(job_id="active-job")

        mock_player.flush.assert_called_once()

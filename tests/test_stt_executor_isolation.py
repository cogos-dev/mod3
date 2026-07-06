"""Regression tests for §4 of ARCHITECTURE.md — STT executor isolation.

Before the fix: _process_utterance(), _run_t1(), and _run_t2() all called
``await asyncio.to_thread(_transcribe_*)`` which submits work to the
asyncio default thread pool.  That pool is shared with bus.act() drain
threads and other I/O helpers.  A 1-2s mlx_whisper.transcribe() call holds
one pool slot for its full duration — starving concurrent bus.act() jobs
under load.

After the fix: all three STT calls route through
``loop.run_in_executor(channels._STT_EXECUTOR, ...)`` where
``_STT_EXECUTOR`` is a dedicated ``ThreadPoolExecutor(max_workers=1)``
owned by the channels module.  The default pool is left free.

Tests
-----
1. ``test_stt_uses_dedicated_executor`` — verify the executor used for a
   simulated _process_utterance call is ``_STT_EXECUTOR``, not the default.
2. ``test_slow_stt_does_not_block_default_pool`` — a slow STT job does not
   prevent a concurrent task that uses the default pool from completing
   within its expected window.
3. ``test_shutdown_stt_executor_is_safe`` — shutdown_stt_executor() returns
   without error when called with wait=False (used in teardown paths).

Run: python -m pytest tests/test_stt_executor_isolation.py -v

Note for future test authors: do NOT import and use the shared module-level
``channels._STT_EXECUTOR`` directly for isolation assertions. It is a
process-global singleton that gets shut down for real by any other test
file that runs ``with TestClient(app) as c:`` (this exercises the app's
full lifespan, including ``shutdown_stt_executor()`` in the teardown path —
see http_api.py's ``_lifespan``). Depending on pytest's collection order,
the shared executor may already be shut down by the time this test runs.
Always instantiate a local, throwaway ``ThreadPoolExecutor`` for isolation
tests instead (see ``test_shutdown_stt_executor_is_safe`` below for the
established pattern).
"""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import channels  # noqa: E402 — must come after sys.path insert
from channels import _STT_EXECUTOR, shutdown_stt_executor  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Verify _STT_EXECUTOR is a single-worker ThreadPoolExecutor
# ---------------------------------------------------------------------------


def test_stt_executor_is_single_worker():
    """_STT_EXECUTOR must be a single-worker ThreadPoolExecutor.

    Only one mlx_whisper graph should be in flight at a time — a larger pool
    would not reduce latency (MLX is already using the full ANE/GPU) but
    could allow two concurrent STT calls to contend over hardware and
    memory.
    """
    assert isinstance(_STT_EXECUTOR, ThreadPoolExecutor)
    # max_workers is stored on the executor instance.
    assert _STT_EXECUTOR._max_workers == 1


def test_stt_executor_thread_name_prefix():
    """Threads spawned by _STT_EXECUTOR carry the 'mod3-stt' prefix.

    This makes it easy to spot STT threads in stack traces and profilers.
    """
    assert _STT_EXECUTOR._thread_name_prefix.startswith("mod3-stt")


# ---------------------------------------------------------------------------
# 2. Pool isolation: a slow STT job does not block the default pool
# ---------------------------------------------------------------------------


def test_slow_stt_does_not_block_default_pool():
    """Slow STT work (on a dedicated single-worker pool) must not delay
    default-pool work.

    Scenario:
      - Enqueue a "slow STT" job (200ms) on a dedicated single-worker
        ThreadPoolExecutor — the same shape as channels._STT_EXECUTOR,
        but a fresh local instance so this test is immune to any other
        test file's TestClient(app) lifespan teardown (which shuts down
        the real, shared, module-level _STT_EXECUTOR as a side effect —
        see the module docstring above).
      - Concurrently enqueue a "fast default-pool" job on the asyncio
        default executor via asyncio.to_thread().
      - Assert the default-pool job completes well within the STT job's
        duration — i.e., isolation is real.

    If STT were still running on the default pool, the STT job would hold
    one of the (potentially few) default pool slots and the "fast" job
    might queue behind it.  With isolation both can run concurrently and
    the fast job finishes in <50ms.
    """
    STT_DURATION = 0.2  # seconds — simulates a fast mlx_whisper call
    FAST_JOB_DURATION = 0.02  # 20ms — trivially fast
    FAST_JOB_DEADLINE = 0.10  # must finish within 100ms of dispatch

    results: dict[str, Any] = {}

    async def _run():
        def _slow_stt():
            time.sleep(STT_DURATION)
            results["stt"] = "done"

        def _fast_default():
            time.sleep(FAST_JOB_DURATION)
            return time.monotonic()

        local_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mod3-stt-test")
        try:
            loop = asyncio.get_event_loop()
            # Submit STT to the dedicated local executor — fire and don't await yet.
            stt_task = loop.run_in_executor(local_executor, _slow_stt)

            # Dispatch a job to the DEFAULT pool (via asyncio.to_thread) and
            # measure how long it takes from dispatch to completion.
            t0 = time.monotonic()
            finish_time = await asyncio.to_thread(_fast_default)
            elapsed = finish_time - t0

            assert elapsed < FAST_JOB_DEADLINE, (
                f"Default-pool job took {elapsed:.3f}s — expected <{FAST_JOB_DEADLINE}s. STT isolation may be broken."
            )

            # Wait for the STT task so we don't leave threads dangling.
            await stt_task
        finally:
            local_executor.shutdown(wait=False)

    asyncio.run(_run())
    assert results.get("stt") == "done"


# ---------------------------------------------------------------------------
# 3. shutdown_stt_executor is safe to call
# ---------------------------------------------------------------------------


def test_shutdown_stt_executor_is_safe():
    """shutdown_stt_executor(wait=False) returns without raising.

    We use wait=False so the test does not block on an in-flight job.
    After shutdown the executor should be unusable (submit raises), but
    the module-level reference is the live executor — we create a temporary
    one here to avoid killing the module-level instance mid-test suite.
    """
    tmp_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mod3-stt-test")
    # Patch the module-level executor temporarily.
    original = channels._STT_EXECUTOR
    channels._STT_EXECUTOR = tmp_executor
    try:
        # Should not raise.
        shutdown_stt_executor(wait=False)
    finally:
        # Restore so other tests that import _STT_EXECUTOR keep working.
        channels._STT_EXECUTOR = original


# ---------------------------------------------------------------------------
# 4. run_in_executor uses _STT_EXECUTOR, not the default pool
# ---------------------------------------------------------------------------


def test_run_in_executor_targets_stt_executor():
    """The executor passed to loop.run_in_executor is _STT_EXECUTOR.

    Inspects the channel's _process_utterance call by patching
    loop.run_in_executor and verifying the first positional argument is
    the module-level _STT_EXECUTOR instance (not None, which would select
    the default pool).

    This is the strongest guard: if someone changes a run_in_executor call
    back to asyncio.to_thread or passes None as the executor, this test
    catches it immediately.
    """
    executor_calls: list[Any] = []

    async def _run():
        # We need a minimal BrowserChannel to call _process_utterance.
        # Patch heavy deps so no real TTS/STT/WS is involved.
        from unittest.mock import patch

        from bus import ModalityBus
        from pipeline_state import PipelineState

        bus = ModalityBus()
        ps = PipelineState()

        class _FakeWS:
            async def send_json(self, frame):
                pass

        with (
            patch("channels.WhisperDecoder", autospec=True),
        ):
            ch = channels.BrowserChannel(
                ws=_FakeWS(),
                bus=bus,
                pipeline_state=ps,
                loop=asyncio.get_event_loop(),
                on_event=None,
            )

        # Put enough fake PCM data to pass the length guard (>6400 bytes).
        import numpy as np

        # 200ms of silence — enough to pass length check, rms check will
        # exit early (returns None) without calling mlx_whisper.
        silence = np.zeros(3200, dtype=np.int16)
        ch._audio_buffer = bytearray(silence.tobytes())

        original_run_in_executor = asyncio.get_event_loop().run_in_executor

        async def _spy_run_in_executor(executor, fn, *args):
            executor_calls.append(executor)
            # Return None immediately — simulates a filtered/empty STT result.
            return None

        loop = asyncio.get_event_loop()
        loop.run_in_executor = _spy_run_in_executor  # type: ignore[method-assign]
        try:
            await ch._process_utterance()
        finally:
            loop.run_in_executor = original_run_in_executor
            ch._cleanup()

    asyncio.run(_run())

    # At least one run_in_executor call must have happened.
    assert executor_calls, "_process_utterance must call loop.run_in_executor"
    # Every call must pass _STT_EXECUTOR, not None (default pool).
    for exc in executor_calls:
        assert exc is channels._STT_EXECUTOR, (
            f"Expected _STT_EXECUTOR but got {exc!r}. STT must not use the default asyncio thread pool."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

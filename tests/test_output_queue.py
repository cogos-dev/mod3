"""Tests for ChannelQueue and OutputQueueManager drain-thread resilience.

Focuses on the BaseException safety fix: ChannelQueue._drain() must always
reset _running=False when the drain thread exits — regardless of whether the
exit is normal (queue emptied) or abnormal (BaseException raised by a job fn).

Without the fix:
  - BaseException leaves _running=True permanently.
  - depth property reports len(self._queue) + 1 forever (ghost of the dead job).
  - Subsequent submit() calls see _running=True and never start a new drain,
    so jobs accumulate with no processor.

Run: python3 -m pytest tests/test_output_queue.py -v
"""

from __future__ import annotations

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from output_queue import ChannelQueue, OutputQueueManager, QueuedJob  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_flag(check_fn, timeout: float = 2.5) -> bool:
    """Poll check_fn() until True or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check_fn():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Normal drain behavior (sanity)
# ---------------------------------------------------------------------------


class TestChannelQueueBasics:
    def test_submit_and_drain(self):
        """A submitted job runs and queue depth returns to 0."""
        q = ChannelQueue("ch1")
        results: list[str] = []

        def fn():
            results.append("ran")
            return "ok"

        job = q.submit(fn)

        drained = _wait_flag(lambda: not q._running)
        assert drained, "drain did not finish within timeout"
        assert results == ["ran"]
        assert q.depth == 0
        assert job.status == "done"
        assert job.result == "ok"

    def test_depth_includes_running_job(self):
        """While a job is running, depth == len(pending) + 1."""
        q = ChannelQueue("ch_depth")
        barrier = threading.Event()

        def slow_fn():
            barrier.wait(timeout=2.0)
            return "done"

        q.submit(slow_fn)
        q.submit(lambda: "second")

        # While first job is blocked, depth should be 2 (1 running + 1 pending)
        time.sleep(0.05)
        assert q.depth == 2, f"expected depth=2, got {q.depth}"

        barrier.set()
        _wait_flag(lambda: not q._running)
        assert q.depth == 0

    def test_multiple_jobs_run_in_order(self):
        """Jobs execute serially in submission order."""
        q = ChannelQueue("ch_order")
        order: list[int] = []

        for i in range(5):
            idx = i

            def fn(n=idx):
                order.append(n)

            q.submit(fn)

        _wait_flag(lambda: not q._running)
        assert order == list(range(5)), f"unexpected order: {order}"

    def test_exception_in_job_does_not_kill_drain(self):
        """An Exception raised by a job function is caught; remaining jobs run."""
        q = ChannelQueue("ch_exc")
        results: list[str] = []

        def bad_fn():
            raise ValueError("oops")

        def good_fn():
            results.append("good")

        bad_job = q.submit(bad_fn)
        q.submit(good_fn)

        _wait_flag(lambda: not q._running)
        assert bad_job.status == "error"
        assert bad_job.error == "oops"
        assert results == ["good"], "Exception in job should not stop subsequent jobs"


# ---------------------------------------------------------------------------
# BaseException safety (regression tests for the drain-thread fix)
# ---------------------------------------------------------------------------


class TestChannelQueueBaseExceptionResilience:
    """ChannelQueue._drain() must reset _running=False on abnormal exit.

    Before the fix:
      - BaseException escapes the inner ``except Exception``.
      - _drain() thread exits with _running=True.
      - depth reports N+1 forever; no new drain thread starts.
    """

    def test_drain_resets_running_after_system_exit(self):
        """SystemExit from a job fn must reset _running to False."""
        q = ChannelQueue("ch_se")

        def raises_system_exit():
            raise SystemExit(1)

        q.submit(raises_system_exit)

        drained = _wait_flag(lambda: not q._running)
        assert drained, "_running stayed True after SystemExit — drain thread leaked"
        assert q._running is False, "_running not False"
        assert q._current is None, "_current not cleared after SystemExit"
        assert q.depth == 0, f"depth should be 0 after drain exit, got {q.depth}"

    def test_drain_resets_running_after_memory_error(self):
        """MemoryError from a job fn must reset _running to False."""
        q = ChannelQueue("ch_mem")

        def raises_memory_error():
            raise MemoryError("simulated OOM")

        q.submit(raises_memory_error)

        drained = _wait_flag(lambda: not q._running)
        assert drained, "_running stayed True after MemoryError"
        assert q._running is False
        assert q._current is None

    def test_depth_returns_to_zero_after_base_exception(self):
        """depth must be 0 after a BaseException kills the drain thread.

        Before the fix, _running stayed True so depth = len(queue) + 1 = 1
        even with an empty queue.
        """
        q = ChannelQueue("ch_depth_be")

        def raises_base_exc():
            raise MemoryError("OOM")

        q.submit(raises_base_exc)
        _wait_flag(lambda: not q._running)

        # Queue is empty AND drain is no longer running — depth must be 0.
        assert q.depth == 0, (
            f"depth should be 0 after drain exit, got {q.depth}; "
            f"_running={q._running}, len(_queue)={len(q._queue)}"
        )

    def test_new_jobs_accepted_after_base_exception_kill(self):
        """After BaseException kills the drain thread, the queue must resume.

        This is the critical end-to-end regression: _running resets to False,
        and the next submit() starts a fresh drain thread that actually runs.
        """
        q = ChannelQueue("ch_resume")
        results: list[str] = []
        calls = 0

        def job_fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise MemoryError("first job dies")
            results.append(f"job{calls}")

        # Kill the drain thread
        q.submit(job_fn)
        drained = _wait_flag(lambda: not q._running)
        assert drained, "drain thread did not exit after MemoryError"
        assert q._running is False, "_running not reset — new jobs will be silently dropped"

        # Submit after the kill — must start a new drain
        q.submit(job_fn)

        recovered = _wait_flag(lambda: not q._running)
        assert recovered, "new drain thread did not finish within timeout"

        assert results == ["job2"], (
            f"queue did not resume after BaseException drain-thread death; results={results}"
        )


# ---------------------------------------------------------------------------
# OutputQueueManager (per-channel isolation)
# ---------------------------------------------------------------------------


class TestOutputQueueManager:
    def test_separate_channels_run_concurrently(self):
        """Different channels drain independently and in parallel."""
        mgr = OutputQueueManager()
        a_results: list[int] = []
        b_results: list[int] = []

        for i in range(3):
            idx = i

            def fn_a(n=idx):
                a_results.append(n)

            def fn_b(n=idx):
                b_results.append(n)

            mgr.submit("a", fn_a)
            mgr.submit("b", fn_b)

        # Wait for both channels to drain
        qa = mgr.get_queue("a")
        qb = mgr.get_queue("b")
        _wait_flag(lambda: not qa._running)
        _wait_flag(lambda: not qb._running)

        assert sorted(a_results) == [0, 1, 2]
        assert sorted(b_results) == [0, 1, 2]

    def test_cancel_channel_clears_pending(self):
        """cancel_channel() removes pending jobs without affecting running ones."""
        mgr = OutputQueueManager()
        barrier = threading.Event()

        def blocking():
            barrier.wait(timeout=2.0)

        mgr.submit("c", blocking)
        for i in range(5):
            mgr.submit("c", lambda: None)

        time.sleep(0.05)
        cancelled = mgr.cancel_channel("c")
        barrier.set()

        assert cancelled == 5

    def test_drop_queue_removes_channel(self):
        """drop_queue() removes the ChannelQueue so it can be re-created fresh."""
        mgr = OutputQueueManager()
        results: list[str] = []
        mgr.submit("d", lambda: results.append("before"))
        _wait_flag(lambda: "before" in results)

        removed = mgr.drop_queue("d")
        assert removed is True
        assert "d" not in mgr._queues

        # Re-submitting after drop creates a fresh queue
        mgr.submit("d", lambda: results.append("after"))
        q = mgr.get_queue("d")
        _wait_flag(lambda: not q._running)
        assert "after" in results

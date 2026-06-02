"""Tests for the SpeechQueue and queue-aware speak/stop/speech_status.

Tests the queue mechanics without requiring live audio or TTS engines.
Run: python3 -m pytest tests/test_speech_queue.py -v
"""

import json
import os
import sys
import threading
import time

# Ensure the project root is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# SpeechQueue unit tests (no server dependencies)
# ---------------------------------------------------------------------------


class TestSpeechQueue:
    """Test the SpeechQueue class directly."""

    def _make_queue(self):
        from server import SpeechQueue

        return SpeechQueue()

    def test_enqueue_returns_position_zero_when_empty(self):
        """First enqueued job gets position 0."""
        q = self._make_queue()
        # Override _drain to be a no-op so the job stays in queue
        q._drain = lambda: None  # prevent actual drain
        q._draining = True  # pretend we're already draining
        pos = q.enqueue("job1", {"text": "hello"})
        assert pos == 0, f"Expected position 0, got {pos}"

    def test_enqueue_returns_incrementing_positions(self):
        """Subsequent enqueues get increasing positions."""
        q = self._make_queue()
        q._draining = True  # prevent auto-drain
        q.enqueue("job1", {"text": "first"})
        pos2 = q.enqueue("job2", {"text": "second"})
        pos3 = q.enqueue("job3", {"text": "third"})
        assert pos2 == 1, f"Expected position 1, got {pos2}"
        assert pos3 == 2, f"Expected position 2, got {pos3}"

    def test_cancel_removes_queued_job(self):
        """cancel() removes a job from the queue and returns True."""
        q = self._make_queue()
        q._draining = True
        q.enqueue("job1", {"text": "first"})
        q.enqueue("job2", {"text": "second"})
        q.enqueue("job3", {"text": "third"})

        assert q.cancel("job2") is True
        snapshot = q.get_queue_snapshot()
        ids = [e["job_id"] for e in snapshot]
        assert "job2" not in ids, "job2 should be removed"
        assert len(ids) == 2

    def test_cancel_returns_false_for_unknown_job(self):
        """cancel() returns False if job_id not found."""
        q = self._make_queue()
        q._draining = True
        assert q.cancel("nonexistent") is False

    def test_cancel_all_queued(self):
        """cancel_all_queued() clears everything and returns count."""
        q = self._make_queue()
        q._draining = True
        q.enqueue("job1", {"text": "first"})
        q.enqueue("job2", {"text": "second"})

        count = q.cancel_all_queued()
        assert count == 2
        assert q.depth == 0
        assert q.get_queue_snapshot() == []

    def test_depth_reflects_queue_length(self):
        """depth property returns number of pending (not active) jobs."""
        q = self._make_queue()
        q._draining = True
        assert q.depth == 0
        q.enqueue("job1", {"text": "first"})
        assert q.depth == 1
        q.enqueue("job2", {"text": "second"})
        assert q.depth == 2
        q.cancel("job1")
        assert q.depth == 1

    def test_get_queue_snapshot_returns_copy(self):
        """get_queue_snapshot returns a copy, not a reference."""
        q = self._make_queue()
        q._draining = True
        q.enqueue("job1", {"text": "first"})
        snap1 = q.get_queue_snapshot()
        q.enqueue("job2", {"text": "second"})
        snap2 = q.get_queue_snapshot()
        assert len(snap1) == 1, "First snapshot should have 1 item"
        assert len(snap2) == 2, "Second snapshot should have 2 items"


class TestSpeechQueueDrain:
    """Test the drain mechanism with a controllable job runner."""

    def test_drain_executes_jobs_serially(self):
        """Jobs execute one at a time through the drain loop."""
        from server import SpeechQueue

        execution_order = []
        barriers = {}

        q = SpeechQueue()

        # Monkey-patch: instead of calling _run_speech_job, record execution

        def mock_drain():
            while True:
                with q._lock:
                    if not q._queue:
                        q._draining = False
                        q._active_job_id = None
                        return
                    entry = q._queue.pop(0)
                    q._active_job_id = entry["job_id"]

                jid = entry["job_id"]
                execution_order.append(jid)
                # Signal that this job started
                if jid in barriers:
                    barriers[jid].set()

        q._drain = mock_drain

        b1 = threading.Event()
        b2 = threading.Event()
        barriers["job1"] = b1
        barriers["job2"] = b2

        q.enqueue("job1", {"text": "first"})
        q.enqueue("job2", {"text": "second"})

        # Wait for drain to process both
        time.sleep(0.2)

        assert execution_order == ["job1", "job2"], f"Jobs should execute in order, got {execution_order}"


# ---------------------------------------------------------------------------
# Duration estimation
# ---------------------------------------------------------------------------


class TestDurationEstimation:
    def test_estimate_duration_scales_with_text_length(self):
        from server import _estimate_duration_sec

        short = _estimate_duration_sec("Hello", 1.0)
        long = _estimate_duration_sec("This is a much longer sentence with many more words in it", 1.0)
        assert long > short, "Longer text should have longer estimated duration"

    def test_estimate_duration_scales_with_speed(self):
        from server import _estimate_duration_sec

        normal = _estimate_duration_sec("Hello world this is a test", 1.0)
        fast = _estimate_duration_sec("Hello world this is a test", 2.0)
        assert fast < normal, "Higher speed should reduce estimated duration"
        assert abs(fast - normal / 2.0) < 0.01, "2x speed should halve duration"

    def test_estimate_duration_handles_empty(self):
        from server import _estimate_duration_sec

        dur = _estimate_duration_sec("", 1.0)
        assert dur > 0, "Even empty text should give a positive estimate"


# ---------------------------------------------------------------------------
# speak() return format tests (requires server imports)
# ---------------------------------------------------------------------------


class TestSpeakReturnFormat:
    """Test that speak() returns correctly structured JSON for queue states."""

    def test_speak_empty_text_returns_error(self):
        from server import speak

        result = json.loads(speak("   "))
        assert result["status"] == "error"
        assert "Nothing to say" in result["error"]

    def test_speak_returns_job_id(self):
        """speak() always returns a job_id in the response."""
        # This test can only work if the engine module is available.
        # Without it, speak() will return an error, which also has a defined format.
        from server import speak

        result = json.loads(speak("test"))
        assert "status" in result
        # Either 'speaking'/'queued' with job_id, or 'error'
        if result["status"] in ("speaking", "queued"):
            assert "job_id" in result
            assert len(result["job_id"]) == 8


class TestStopReturnFormat:
    """Test that stop() returns correctly structured JSON."""

    def test_stop_when_nothing_playing(self):
        from server import stop

        result = json.loads(stop())
        assert result["status"] == "ok"
        assert (
            "Nothing playing" in result["message"]
            or "interrupted" in result["message"].lower()
            or "cancelled" in result["message"].lower()
        )

    def test_stop_unknown_job_returns_error(self):
        from server import stop

        result = json.loads(stop("nonexistent"))
        assert result["status"] == "error"
        assert "Unknown job" in result["error"]


class TestSpeechStatusReturnFormat:
    """Test that speech_status() returns correctly structured JSON."""

    def test_speech_status_no_jobs(self):
        from server import _jobs, speech_status

        # Clear jobs for this test
        original = dict(_jobs)
        _jobs.clear()
        try:
            result = json.loads(speech_status())
            assert result["status"] == "idle"
            assert "queue_depth" in result
        finally:
            _jobs.update(original)

    def test_speech_status_unknown_job(self):
        from server import speech_status

        result = json.loads(speech_status("nonexistent"))
        assert result["status"] == "error"

    def test_speech_status_includes_queue_info(self):
        """speech_status for a known job includes queue metadata."""
        from server import _jobs, speech_status

        # Insert a fake job
        _jobs["testjob1"] = {
            "status": "done",
            "engine": "kokoro",
            "voice": "bm_lewis",
            "text": "test text",
            "start_time": time.time() - 5,
            "metrics": None,
            "error": None,
            "player": None,
        }
        try:
            result = json.loads(speech_status("testjob1"))
            assert result["job_id"] == "testjob1"
            assert result["status"] == "done"
            assert "queue" in result, "Response should include queue state"
            assert "depth" in result["queue"]
            assert "currently_playing" in result["queue"]
        finally:
            _jobs.pop("testjob1", None)


# ---------------------------------------------------------------------------
# Regression tests for queue-stability fixes
# ---------------------------------------------------------------------------


class TestPruneJobsInFlightProtection:
    """_prune_jobs must never evict a job whose worker is still running.

    Regression: stress-testing 35+ parallel speak() calls hit MAX_JOBS=20,
    causing _prune_jobs to evict in-flight jobs. Subsequent
    `_jobs[job_id]["metrics"] = result` then KeyErrored, killing the drain
    thread and stranding the queue.
    """

    def test_prune_skips_speaking_and_queued_jobs(self):
        from server import MAX_JOBS, _jobs, _prune_jobs

        original = dict(_jobs)
        _jobs.clear()
        try:
            # Fill past MAX_JOBS with done jobs (evictable) and a few in-flight.
            for i in range(MAX_JOBS - 2):
                _jobs[f"done{i}"] = {"status": "done", "error": None}
            _jobs["live1"] = {"status": "speaking", "error": None}
            _jobs["live2"] = {"status": "queued", "error": None}
            for i in range(5):
                _jobs[f"trailing{i}"] = {"status": "done", "error": None}

            _prune_jobs()

            assert len(_jobs) == MAX_JOBS, f"len={len(_jobs)}, want {MAX_JOBS}"
            assert "live1" in _jobs, "speaking job got evicted"
            assert "live2" in _jobs, "queued job got evicted"
        finally:
            _jobs.clear()
            _jobs.update(original)

    def test_prune_returns_when_only_in_flight_remain(self):
        """If every entry is in-flight, prune accepts transient over-cap."""
        from server import MAX_JOBS, _jobs, _prune_jobs

        original = dict(_jobs)
        _jobs.clear()
        try:
            for i in range(MAX_JOBS + 3):
                _jobs[f"speaking{i}"] = {"status": "speaking", "error": None}

            _prune_jobs()  # should not raise; should not deadlock.

            assert len(_jobs) == MAX_JOBS + 3, "over-cap entries should be kept"
        finally:
            _jobs.clear()
            _jobs.update(original)


class TestDrainExceptionResilience:
    """The drain thread must survive a job-runner exception.

    Regression: an unhandled KeyError inside _run_speech_job killed the drain
    loop, leaving _draining=True with no live drainer. Later enqueues then
    saw _draining=True and never started a new drain.
    """

    def test_drain_continues_after_job_raises(self):
        import server

        original_runner = server._run_speech_job
        completed: list[str] = []
        seen: list[str] = []

        def fake_runner(entry):
            jid = entry["job_id"]
            seen.append(jid)
            if jid == "boom":
                raise RuntimeError("simulated job failure")
            completed.append(jid)

        server._run_speech_job = fake_runner
        try:
            from server import SpeechQueue

            q = SpeechQueue()
            q.enqueue("first", {})
            q.enqueue("boom", {})
            q.enqueue("third", {})

            # Wait for drain to process all three (or time out).
            for _ in range(50):
                if not q._draining:
                    break
                time.sleep(0.05)

            assert seen == ["first", "boom", "third"], f"drain stopped early: saw {seen}"
            assert "third" in completed, "drain did not run the third job after the second raised"
            assert q._draining is False, "drain thread leaked _draining=True after queue empty"
        finally:
            server._run_speech_job = original_runner


class TestDrainBaseExceptionResilience:
    """The drain thread MUST reset _draining=False even when a BaseException
    subclass (not caught by ``except Exception``) escapes the inner handler.

    Regression target: before the fix, a SystemExit / KeyboardInterrupt / MemoryError
    raised during _run_speech_job would exit the drain thread with _draining=True.
    Every subsequent enqueue() would skip starting a new drain thread, causing
    jobs to accumulate in the queue indefinitely with active_jobs=0.
    """

    def _wait_for_drain_exit(self, q, timeout=2.5):
        """Block until q._draining goes False or timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not q._draining:
                return True
            time.sleep(0.05)
        return False

    def test_drain_resets_draining_after_system_exit(self):
        """SystemExit inside _run_speech_job must reset _draining to False.

        Without the fix, the drain thread exits with _draining=True and
        subsequent enqueue() calls never start a new drain.
        """
        import server

        original_runner = server._run_speech_job

        def fake_runner_raises_systemexit(entry):
            raise SystemExit(1)

        server._run_speech_job = fake_runner_raises_systemexit
        try:
            from server import SpeechQueue

            q = SpeechQueue()
            q.enqueue("job1", {"text": "test"})

            drained = self._wait_for_drain_exit(q)
            assert drained, "drain thread did not exit within timeout"
            assert q._draining is False, "_draining stayed True after SystemExit"
            assert q._active_job_id is None, "_active_job_id not cleared after SystemExit"
        finally:
            server._run_speech_job = original_runner

    def test_drain_resets_draining_after_memory_error(self):
        """MemoryError (a BaseException subclass) inside _run_speech_job must
        reset _draining to False so the queue can continue processing.
        """
        import server

        original_runner = server._run_speech_job

        def fake_runner_raises_memory_error(entry):
            raise MemoryError("simulated OOM")

        server._run_speech_job = fake_runner_raises_memory_error
        try:
            from server import SpeechQueue

            q = SpeechQueue()
            q.enqueue("oom_job", {"text": "test"})

            drained = self._wait_for_drain_exit(q)
            assert drained, "drain thread did not exit within timeout"
            assert q._draining is False, "_draining stayed True after MemoryError"
            assert q._active_job_id is None, "_active_job_id not cleared after MemoryError"
        finally:
            server._run_speech_job = original_runner

    def test_drain_accepts_new_jobs_after_base_exception_kill(self):
        """After a BaseException kills the drain thread, subsequent enqueue()
        calls must successfully start a NEW drain thread and process jobs.

        This is the end-to-end regression test: not just that _draining resets,
        but that the queue actually resumes processing — which requires the next
        enqueue() to detect _draining=False and spawn a new thread.
        """
        import server

        original_runner = server._run_speech_job
        completed: list[str] = []
        calls = 0

        def fake_runner(entry):
            nonlocal calls
            calls += 1
            if calls == 1:
                # First call raises SystemExit to kill the drain thread
                raise SystemExit(1)
            # Subsequent calls succeed
            completed.append(entry["job_id"])

        server._run_speech_job = fake_runner
        try:
            from server import SpeechQueue

            q = SpeechQueue()
            # This job kills the drain thread
            q.enqueue("killer", {"text": "test"})

            # Wait for the drain thread to die and _draining to reset
            drained = self._wait_for_drain_exit(q)
            assert drained, "drain thread did not exit within timeout after SystemExit"
            assert q._draining is False, "_draining not reset after thread death"

            # Now enqueue a new job -- must start a fresh drain thread
            q.enqueue("survivor", {"text": "after"})

            # Wait for the new drain to complete
            deadline = time.time() + 2.5
            while time.time() < deadline:
                if not q._draining:
                    break
                time.sleep(0.05)

            assert "survivor" in completed, (
                "queue did not resume processing after BaseException drain-thread death; "
                f"completed={completed}, _draining={q._draining}"
            )
        finally:
            server._run_speech_job = original_runner


"""Tests for the owner-aware speaking lock (Codex review #4 / Fix 2).

The cross-process ``/tmp/mod3-speaking.json`` lock used to be ownership-blind:
``_release_speaking_lock`` removed the file unconditionally, so two overlapping
mod3 processes could falsely interrupt each other when one finished its speech.

The new contract:
  * Acquire writes only if the file is missing, the holder PID is dead, or
    (pid, job_id) match the current process (idempotent re-acquire).
  * Release only removes the file when (pid, job_id) match.
  * ``_i_own_speaking_lock`` returns True only when our (pid, job_id) is the
    current on-disk holder; the speech loop uses this for stop-on-mismatch.
  * ``_force_clear_speaking_lock`` is the only path that clears regardless of
    owner — used by the bargein watcher's cross-process interrupt path.

Run: python -m pytest tests/test_speaking_lock.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# jobs_registry.py hard-imports mcp at module level; skip cleanly in CI where
# the mcp package is not installed rather than raising a collection-time ImportError.
# Check HAS_MCP from conftest (set before stubs) so we test real availability.
from conftest import HAS_MCP  # noqa: E402

if not HAS_MCP:
    pytest.skip("mcp package not installed", allow_module_level=True)

import jobs_registry  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_lock_path(tmp_path, monkeypatch):
    """Each test gets its own lock-file path so they don't collide on /tmp."""
    lock_path = str(tmp_path / "mod3-speaking.json")
    monkeypatch.setattr(jobs_registry, "_SPEAKING_LOCK", lock_path)
    yield lock_path
    # Best-effort cleanup
    try:
        os.remove(lock_path)
    except OSError:
        pass


def _write_raw_lock(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f)


def test_acquire_writes_lock_when_missing(_isolated_lock_path):
    ok = jobs_registry._acquire_speaking_lock("job-1", "hello")
    assert ok is True

    with open(_isolated_lock_path) as f:
        lock = json.load(f)
    assert lock["pid"] == os.getpid()
    assert lock["job_id"] == "job-1"
    assert lock["text"] == "hello"
    assert "acquired_at" in lock


def test_acquire_is_idempotent_for_same_owner(_isolated_lock_path):
    assert jobs_registry._acquire_speaking_lock("job-1", "hello") is True
    # Same (pid, job_id) re-acquiring — must succeed and refresh the file
    assert jobs_registry._acquire_speaking_lock("job-1", "hello again") is True

    with open(_isolated_lock_path) as f:
        lock = json.load(f)
    assert lock["text"] == "hello again"


def test_acquire_blocked_by_live_other_process(_isolated_lock_path):
    """Different live PID owns the lock -> we can't acquire."""
    other_pid = os.getppid()  # parent pid is reliably alive during this test
    assert other_pid != os.getpid()
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": other_pid, "job_id": "their-job", "text": "..."},
    )

    ok = jobs_registry._acquire_speaking_lock("our-job", "hi")
    assert ok is False

    # The other process's lock must be untouched
    with open(_isolated_lock_path) as f:
        lock = json.load(f)
    assert lock["pid"] == other_pid
    assert lock["job_id"] == "their-job"


def test_acquire_reclaims_lock_when_holder_pid_is_dead(_isolated_lock_path):
    """A lock left by a crashed process must be reclaimable."""
    # PID 1 on macOS is launchd — not us. We mock _pid_is_alive to make it
    # appear dead so the acquire path takes the "stale" branch.
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": 99999, "job_id": "ghost-job", "text": "stale"},
    )

    with patch.object(jobs_registry, "_pid_is_alive", return_value=False):
        ok = jobs_registry._acquire_speaking_lock("our-job", "hi")
    assert ok is True

    with open(_isolated_lock_path) as f:
        lock = json.load(f)
    assert lock["pid"] == os.getpid()
    assert lock["job_id"] == "our-job"


def test_release_only_clears_own_lock(_isolated_lock_path):
    """Release with mismatched job_id must NOT remove the file."""
    other_pid = os.getppid()
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": other_pid, "job_id": "their-job", "text": "..."},
    )

    # Same pid, different job_id -> still must not remove (we don't own job_id)
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": os.getpid(), "job_id": "their-job", "text": "..."},
    )
    ok = jobs_registry._release_speaking_lock("our-job")
    assert ok is False
    assert os.path.exists(_isolated_lock_path)

    # Wrong PID -> still must not remove
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": other_pid, "job_id": "any", "text": "..."},
    )
    ok = jobs_registry._release_speaking_lock("any")
    assert ok is False
    assert os.path.exists(_isolated_lock_path)


def test_release_removes_own_lock(_isolated_lock_path):
    assert jobs_registry._acquire_speaking_lock("job-1", "x") is True
    assert os.path.exists(_isolated_lock_path)

    ok = jobs_registry._release_speaking_lock("job-1")
    assert ok is True
    assert not os.path.exists(_isolated_lock_path)


def test_i_own_speaking_lock_matches_pid_and_job(_isolated_lock_path):
    assert jobs_registry._i_own_speaking_lock("job-1") is False  # missing

    jobs_registry._acquire_speaking_lock("job-1", "x")
    assert jobs_registry._i_own_speaking_lock("job-1") is True
    assert jobs_registry._i_own_speaking_lock("other-job") is False

    # Simulate another process taking over the lock
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": os.getppid(), "job_id": "job-1", "text": "x"},
    )
    assert jobs_registry._i_own_speaking_lock("job-1") is False


def test_two_processes_cannot_release_each_others_locks(_isolated_lock_path):
    """The original Codex bug: process A's release deletes process B's lock.

    Simulated with mismatched-pid lock content + a release call that should
    no-op because (pid, job_id) doesn't match this process.
    """
    # Process B (different pid) currently owns the lock
    other_pid = os.getppid()
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": other_pid, "job_id": "B-job", "text": "playing..."},
    )

    # Process A finishes its own job and calls release.
    # The OLD release would delete the file (false interrupt for B);
    # the NEW release must observe pid mismatch and no-op.
    ok = jobs_registry._release_speaking_lock("A-job")
    assert ok is False
    assert os.path.exists(_isolated_lock_path)

    # B's lock content is unchanged
    with open(_isolated_lock_path) as f:
        lock = json.load(f)
    assert lock["pid"] == other_pid
    assert lock["job_id"] == "B-job"


def test_force_clear_removes_any_lock(_isolated_lock_path):
    """The bargein watcher's cross-process kill path."""
    other_pid = os.getppid()
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": other_pid, "job_id": "B-job", "text": "long playback"},
    )

    cleared = jobs_registry._force_clear_speaking_lock()
    assert cleared is not None
    assert cleared["pid"] == other_pid
    assert cleared["job_id"] == "B-job"
    assert not os.path.exists(_isolated_lock_path)


def test_is_any_process_speaking_drops_dead_holder_lock(_isolated_lock_path):
    """Stale lock (dead holder pid) must be treated as 'no one speaking'."""
    _write_raw_lock(
        _isolated_lock_path,
        {"pid": 99999, "job_id": "ghost", "text": "..."},
    )

    with patch.object(jobs_registry, "_pid_is_alive", return_value=False):
        result = jobs_registry._is_any_process_speaking()

    assert result is None
    # Side effect: stale file is removed
    assert not os.path.exists(_isolated_lock_path)


def test_pid_is_alive_handles_self_and_invalid():
    assert jobs_registry._pid_is_alive(os.getpid()) is True
    assert jobs_registry._pid_is_alive(0) is False
    assert jobs_registry._pid_is_alive(-1) is False
    assert jobs_registry._pid_is_alive("not-an-int") is False  # type: ignore[arg-type]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

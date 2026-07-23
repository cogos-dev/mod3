"""Tests for job retention (feature 1) — the time-based prune in
jobs_registry.py's _jobs, and the merged /v1/jobs / /v1/jobs/{id} view in
http_api.py.

Regression coverage for the observability hole this closes: POST /v1/speak
enqueues through jobs_registry._start_speech, which tracks its own jobs in
jobs_registry._jobs — a separate dict from http_api.py's own _jobs (used by
/v1/synthesize, /v1/audio/speech, /v1/vad). GET /v1/jobs/{id} for a job
launched via /v1/speak — the only endpoint that actually plays audio — was
originally always "not found", even mid-playback, because http_api.py never
looked at the /v1/speak registry at all.

jobs_registry.py itself exists to fix a second-order bug in that first fix
(2026-07-23): http_api.py's original patch read the /v1/speak registry via
``from server import _jobs``, and since server.py also runs as ``__main__``
in production, that lazy cross-import silently re-executed server.py's
entire module body a second time under the name "server" — a separate job
registry, speech queue, and barge-in watcher thread from the one actually
driving playback. See jobs_registry.py's module docstring for the full
mechanism and tests/test_server_topology.py for the regression test that
reproduces the deployed subprocess shape.

Run: python3 -m pytest tests/test_job_retention.py -v
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# jobs_registry.py: time-based _prune_jobs
# ---------------------------------------------------------------------------


class TestPruneJobsRetention:
    def test_retention_window_is_at_least_ten_minutes(self):
        from jobs_registry import JOB_RETENTION_SECONDS

        assert JOB_RETENTION_SECONDS >= 600

    def test_finished_job_survives_within_the_retention_window(self):
        from jobs_registry import _jobs, _prune_jobs

        original = dict(_jobs)
        _jobs.clear()
        try:
            _jobs["recent"] = {"status": "done", "end_time": time.time() - 5}
            _prune_jobs()
            assert "recent" in _jobs
        finally:
            _jobs.clear()
            _jobs.update(original)

    def test_finished_job_is_evicted_after_the_retention_window(self):
        from jobs_registry import JOB_RETENTION_SECONDS, _jobs, _prune_jobs

        original = dict(_jobs)
        _jobs.clear()
        try:
            _jobs["stale"] = {"status": "done", "end_time": time.time() - JOB_RETENTION_SECONDS - 1}
            _prune_jobs()
            assert "stale" not in _jobs
        finally:
            _jobs.clear()
            _jobs.update(original)

    def test_in_flight_job_is_never_evicted_regardless_of_age(self):
        from jobs_registry import JOB_RETENTION_SECONDS, _jobs, _prune_jobs

        original = dict(_jobs)
        _jobs.clear()
        try:
            _jobs["ancient-but-speaking"] = {
                "status": "speaking",
                "submitted_time": time.time() - JOB_RETENTION_SECONDS * 10,
            }
            _prune_jobs()
            assert "ancient-but-speaking" in _jobs
        finally:
            _jobs.clear()
            _jobs.update(original)

    def test_missing_end_time_falls_back_to_submitted_time(self):
        """A finished job missing end_time (e.g. an older record) is timed off
        submitted_time instead of being evicted just for lacking the field."""
        from jobs_registry import _jobs, _prune_jobs

        original = dict(_jobs)
        _jobs.clear()
        try:
            _jobs["no-end-time"] = {"status": "done", "submitted_time": time.time()}
            _prune_jobs()
            assert "no-end-time" in _jobs
        finally:
            _jobs.clear()
            _jobs.update(original)


# ---------------------------------------------------------------------------
# http_api.py: merged /v1/jobs and /v1/jobs/{id}
# ---------------------------------------------------------------------------


class TestMergedJobsEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        import http_api

        return TestClient(http_api.app, base_url="http://localhost:7860")

    def test_speak_job_is_found_by_id(self, client):
        """A job that only exists in jobs_registry._jobs (as /v1/speak creates) must
        be resolvable via GET /v1/jobs/{id} — previously always 404."""
        import jobs_registry

        original = dict(jobs_registry._jobs)
        jobs_registry._jobs.clear()
        try:
            jobs_registry._jobs["speak-job-1"] = {
                "type": "speak",
                "status": "speaking",
                "engine": "kokoro",
                "voice": "bm_lewis",
                "text": "hello",
                "submitted_time": time.time(),
                "start_time": time.time(),
                "end_time": None,
                "metrics": None,
                "error": None,
                "player": object(),  # stand-in for a live AdaptivePlayer
            }
            r = client.get("/v1/jobs/speak-job-1")
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["job_id"] == "speak-job-1"
            assert body["status"] == "speaking"
            assert "player" not in body, "the live player object must not leak into the HTTP response"
        finally:
            jobs_registry._jobs.clear()
            jobs_registry._jobs.update(original)

    def test_unknown_job_still_404s(self, client):
        r = client.get("/v1/jobs/does-not-exist")
        assert r.status_code == 404

    def test_native_job_lookup_is_unaffected(self, client):
        """A job recorded by http_api.py's own _record_job still resolves
        exactly as before the merge (no regression on the pre-existing path)."""
        import http_api

        job_id = http_api._record_job({"type": "synthesize", "status": "complete"})
        try:
            r = client.get(f"/v1/jobs/{job_id}")
            assert r.status_code == 200, r.text
            assert r.json()["job_id"] == job_id
        finally:
            with http_api._jobs_lock:
                http_api._jobs.pop(job_id, None)

    def test_list_jobs_includes_speak_jobs(self, client):
        import jobs_registry

        original = dict(jobs_registry._jobs)
        jobs_registry._jobs.clear()
        try:
            jobs_registry._jobs["speak-list-1"] = {
                "type": "speak",
                "status": "done",
                "submitted_time": time.time(),
                "end_time": time.time(),
                "player": None,
            }
            r = client.get("/v1/jobs")
            assert r.status_code == 200, r.text
            ids = [j["job_id"] for j in r.json()["jobs"]]
            assert "speak-list-1" in ids
        finally:
            jobs_registry._jobs.clear()
            jobs_registry._jobs.update(original)

    def test_list_jobs_type_filter_applies_across_both_registries(self, client):
        import jobs_registry

        original = dict(jobs_registry._jobs)
        jobs_registry._jobs.clear()
        try:
            jobs_registry._jobs["speak-filter-1"] = {
                "type": "speak",
                "status": "done",
                "submitted_time": time.time(),
                "player": None,
            }
            r = client.get("/v1/jobs", params={"type": "speak"})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["jobs"], "expected at least the seeded speak job"
            assert all(j["type"] == "speak" for j in body["jobs"])
            assert any(j["job_id"] == "speak-filter-1" for j in body["jobs"])
        finally:
            jobs_registry._jobs.clear()
            jobs_registry._jobs.update(original)

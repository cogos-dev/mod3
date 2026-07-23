"""Tests for job retention (feature 1) — the time-based prune in server.py's
_jobs, and the merged /v1/jobs / /v1/jobs/{id} view in http_api.py.

Regression coverage for the observability hole this closes: POST /v1/speak
enqueues through server._start_speech, which tracks its own jobs in
server._jobs — a completely separate dict from http_api.py's own _jobs
(used by /v1/synthesize, /v1/audio/speech, /v1/vad). GET /v1/jobs/{id} for
a job launched via /v1/speak — the only endpoint that actually plays
audio — was therefore always "not found", even mid-playback.

Run: python3 -m pytest tests/test_job_retention.py -v
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# server.py: time-based _prune_jobs
# ---------------------------------------------------------------------------


class TestPruneJobsRetention:
    def test_retention_window_is_at_least_ten_minutes(self):
        from server import JOB_RETENTION_SECONDS

        assert JOB_RETENTION_SECONDS >= 600

    def test_finished_job_survives_within_the_retention_window(self):
        from server import _jobs, _prune_jobs

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
        from server import JOB_RETENTION_SECONDS, _jobs, _prune_jobs

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
        from server import JOB_RETENTION_SECONDS, _jobs, _prune_jobs

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
        from server import _jobs, _prune_jobs

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
        """A job that only exists in server._jobs (as /v1/speak creates) must
        be resolvable via GET /v1/jobs/{id} — previously always 404."""
        import server

        original = dict(server._jobs)
        server._jobs.clear()
        try:
            server._jobs["speak-job-1"] = {
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
            server._jobs.clear()
            server._jobs.update(original)

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
        import server

        original = dict(server._jobs)
        server._jobs.clear()
        try:
            server._jobs["speak-list-1"] = {
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
            server._jobs.clear()
            server._jobs.update(original)

    def test_list_jobs_type_filter_applies_across_both_registries(self, client):
        import server

        original = dict(server._jobs)
        server._jobs.clear()
        try:
            server._jobs["speak-filter-1"] = {
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
            server._jobs.clear()
            server._jobs.update(original)

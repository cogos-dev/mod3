"""Regression test for the deployed process topology (2026-07-23 incident).

Production runs ``python server.py --http --port 7860`` directly, which
makes server.py execute as ``__main__``. Before this fix, http_api.py read
shared state via ``from server import _bus`` (and several other names) —
and because a module run as ``__main__`` is never the same module object as
one later imported by its own name, that lazy cross-import silently
re-executed server.py's entire top-level body a SECOND time under the module
name "server": a second job registry, a second speech queue, a second
ModalityBus, and — critically — a second ``_bargein_watcher`` thread with
its own ``pipeline_state``. The watcher bound to ``__main__`` never saw the
real job's ``pipeline_state`` as speaking, so it misread the other
instance's live utterance as a foreign cross-process speaker and forcibly
cleared the shared speaking-lock file out from under it.

This test reproduces the actual deployed shape — a real subprocess running
server.py as ``__main__`` in HTTP mode — rather than an in-process import
(which never exhibits the bug: pytest always imports server.py by its own
name, so the topology under test differs from the topology in production).
It launches a real speak job (using the real Kokoro engine, gated on Apple
Silicon / mlx availability — no mock/null audio backend is worth the
complexity here since the engine and default output device are already
present in this environment; the assertions are on job-state consistency,
not on what came out of the speaker), polls GET /v1/jobs/{id} mid-flight
(the exact operation the field incident's operator was doing when playback
cut out), and asserts:

  1. the mid-job poll does not disturb the job (it keeps running / finishes
     cleanly, rather than being killed by a duplicate instance's confused
     barge-in watcher);
  2. the job reaches a genuine terminal state ("done"), not "error";
  3. GET /diagnostics reports topology.server_reimported == False — i.e.
     "server" never appears as its own key in the subprocess's sys.modules.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import HAS_MCP, HAS_MLX  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (HAS_MLX and HAS_MCP),
    reason="needs the real Kokoro engine (mlx, Apple Silicon) and the mcp package",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _running_daemon(tmp_path, port: int):
    """Launch server.py --http as a real subprocess (the deployed topology).

    Barge-in signal/lock paths and the LMS contention probe are redirected
    into tmp_path / disabled so this test can never read, write, or race the
    real daemon's files or the operator's actual LM Studio instance.
    """
    env = dict(os.environ)
    env.update(
        {
            "MOD3_INBOUND_ENABLED": "0",  # no real mic in a test
            # MOD3_PREWARM_TTS left at its production default (on): the first
            # Kokoro synthesis call in a *fresh, non-prewarmed* thread hits an
            # unrelated MLX/Metal quirk ("no Stream(gpu, 0) in current
            # thread") — a pre-existing environment interaction the prewarm
            # thread happens to paper over, not something this fix touches.
            # Disabling it here would make the test fail for a reason that
            # has nothing to do with the topology bug under test.
            "MOD3_ADAPTIVE_BUFFER_PROBE": "0",  # never touch a real LM Studio
            "MOD3_BARGEIN_SIGNAL_PATH": str(tmp_path / "mod3-barge-in.json"),
            "MOD3_SPEAKING_LOCK_PATH": str(tmp_path / "mod3-speaking.json"),
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "server.py", "--http", "--port", str(port), "--host", "127.0.0.1"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _wait_for_kokoro_ready(base_url: str, proc: subprocess.Popen, timeout: float = 45.0) -> None:
    """Wait for /health AND for Kokoro's prewarm synthesis to have completed.

    Firing a real /v1/speak before prewarm's dummy synthesis has run in its
    own thread hits an unrelated MLX/Metal cold-thread quirk (see the
    comment in ``_running_daemon``) — waiting for ``engines.kokoro ==
    "loaded"`` is a real readiness signal, not a fixed guess-and-sleep.
    """
    import httpx

    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(f"daemon exited early (code {proc.returncode}) before becoming ready:\n{out[-4000:]}")
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200 and r.json().get("engines", {}).get("kokoro") == "loaded":
                return
        except Exception as exc:  # noqa: BLE001 — server not up yet
            last_exc = exc
        time.sleep(0.3)
    pytest.fail(f"daemon never reported Kokoro loaded within {timeout}s (last error: {last_exc})")


class TestDeployedTopology:
    def test_no_second_server_module_instance(self, tmp_path):
        """topology.server_reimported must be False in the real subprocess shape."""
        import httpx

        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"

        with _running_daemon(tmp_path, port) as proc:
            _wait_for_kokoro_ready(base_url, proc)
            diag = httpx.get(f"{base_url}/diagnostics", timeout=5.0).json()

        assert "topology" in diag, "/diagnostics must report the topology sentinel"
        assert diag["topology"]["server_reimported"] is False, (
            "'server' appeared as its own module in sys.modules — the 2026-07-23 "
            "double-import regression is back (see jobs_registry.py's module docstring)"
        )

    def test_mid_job_poll_does_not_disturb_playback(self, tmp_path):
        """GET /v1/jobs/{id} mid-flight must not kill or corrupt an active job.

        This is the exact operator action ("a GET /v1/jobs/{id} poll hit the
        daemon") that coincided with the field-incident cutoff.
        """
        import httpx

        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"

        with _running_daemon(tmp_path, port) as proc:
            _wait_for_kokoro_ready(base_url, proc)

            resp = httpx.post(
                f"{base_url}/v1/speak",
                json={"text": "Testing the job registry topology fix.", "voice": "bm_lewis"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            job_id = body["job_id"]
            assert body["status"] in ("speaking", "queued"), body

            # Poll /v1/jobs/{id} repeatedly while the job is in flight — this
            # is the operation under test, not incidental to it.
            saw_in_flight = False
            terminal_statuses = {"done", "error", "interrupted", "cancelled"}
            deadline = time.monotonic() + 30.0
            last_status = None
            while time.monotonic() < deadline:
                r = httpx.get(f"{base_url}/v1/jobs/{job_id}", timeout=5.0)
                assert r.status_code == 200, (
                    f"GET /v1/jobs/{job_id} returned {r.status_code} — the merged-registry "
                    "regression this branch's earlier PR fixed"
                )
                job = r.json()
                last_status = job.get("status")
                if last_status not in terminal_statuses:
                    saw_in_flight = True
                if last_status in terminal_statuses:
                    break
                time.sleep(0.2)

            assert saw_in_flight, "job finished before a single in-flight poll landed — tighten the test, not the fix"
            assert last_status == "done", (
                f"job did not reach a clean 'done' — got {last_status!r}. A duplicate "
                "server module instance's confused barge-in watcher would surface here "
                "as an unexpected 'interrupted'/'error' despite no real barge-in signal."
            )
            assert job.get("error") is None
            assert job.get("metrics") is not None

            # Cross-check against /diagnostics too — before the fix this
            # endpoint only ever saw http_api's own (empty, for /v1/speak)
            # ledger, so a completed /v1/speak job was invisible here.
            diag = httpx.get(f"{base_url}/diagnostics", timeout=5.0).json()
            assert diag["jobs"]["total"] >= 1

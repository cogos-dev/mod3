"""Regression tests for Claude Code session_id ↔ mod3 seat binding.

The "close the loop" wiring landed 2026-05-19. Two paths bind a mod3 seat
to a real Claude Code session_id instead of collapsing into the legacy
"main" sentinel:

1. Manual launch: ``claude --dangerously-load-development-channels server:mod3``
   uses the repo's ``mcp.channel.json``. The ``env`` block substitutes
   ``${CLAUDE_CODE_SESSION_ID}`` so each Claude Code session gets a distinct
   ``MOD3_SESSION_ID`` and channel_client.py registers its seat under that id.

2. Dashboard spawn: ``sessions.html`` posts to ``/v1/claude-code/spawn`` and on
   success ``window.opener.postMessage`` notifies the dashboard tab, which
   calls ``acpTransport.sessionResume(session_id)`` to bind its ACP connection
   to the new session's seats.

These tests guard the static artefacts (mcp.channel.json + the three dashboard
files) — server-side ACP ``session/resume`` semantics already have coverage in
``test_acp_session_management.py``.

Run with::

    PYTHONPATH=. .venv/bin/python -m pytest tests/test_claude_session_id_binding.py -v
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Path 1 — channel_client resolves session_id from ~/.claude/sessions/PID.json
# ---------------------------------------------------------------------------
#
# Earlier shape (PR #103): mcp.channel.json used ${CLAUDE_CODE_SESSION_ID:-main}
# env-substitution. Empirically that var is NOT in the parent claude process's
# env at MCP-spawn time, so the substitution fell back to 'main' silently and
# the loop never closed. PR #105 replaced env-substitution with state-file
# resolution: channel_client.py walks up its parent chain and reads
# ~/.claude/sessions/<PID>.json (which Claude Code writes at startup).


class TestChannelClientSessionResolution:
    @pytest.fixture(scope="class")
    def cfg(self):
        return json.loads((REPO_ROOT / "mcp.channel.json").read_text())

    @pytest.fixture(scope="class")
    def channel_client_src(self):
        return (REPO_ROOT / "clients" / "channel_client.py").read_text()

    def test_mod3_server_entry_exists(self, cfg):
        assert "mcpServers" in cfg
        assert "mod3" in cfg["mcpServers"]

    def test_resolver_function_defined(self, channel_client_src):
        """channel_client must expose a resolver that reads the Claude Code
        session_id from ~/.claude/sessions/<PID>.json rather than relying on
        environment substitution at MCP-spawn time."""
        assert re.search(r"def\s+_resolve_claude_session_id\s*\(", channel_client_src), (
            "channel_client.py must define _resolve_claude_session_id()"
        )
        assert "~/.claude/sessions" in channel_client_src or ".claude/sessions" in channel_client_src
        assert "_CLAUDE_SESSIONS_DIR" in channel_client_src
        assert "sessionId" in channel_client_src, "resolver must extract the 'sessionId' field from the state file"

    def test_resolver_walks_parent_chain(self, channel_client_src):
        """The resolver must walk the parent PID chain (not just check getppid)
        — pre-warm-spare configurations interpose a wrapper between claude
        and the MCP child, so the immediate parent isn't always claude."""
        assert "os.getppid" in channel_client_src
        assert "_read_parent_pid" in channel_client_src or "ppid=" in channel_client_src

    def test_default_session_uses_resolver(self, channel_client_src):
        """The argparse --session default must thread through the resolver,
        with priority: explicit --session > MOD3_SESSION_ID env > state-file
        resolver > _DEFAULT_SESSION_ID."""
        # The composed expression should appear near the parser.add_argument
        # for --session. Match the resolver call within a few hundred chars of
        # the argparse setup.
        assert re.search(
            r"MOD3_SESSION_ID[\s\S]{0,400}_resolve_claude_session_id",
            channel_client_src,
        ), "default session must combine MOD3_SESSION_ID env with the state-file resolver"
        # Must still fall back to _DEFAULT_SESSION_ID as last resort.
        assert re.search(
            r"_resolve_claude_session_id[\s\S]{0,200}_DEFAULT_SESSION_ID",
            channel_client_src,
        ), "must fall back to _DEFAULT_SESSION_ID when neither env nor state-file yields a value"

    def test_resolver_handles_missing_state_dir(self, channel_client_src):
        """If ~/.claude/sessions doesn't exist (non-Claude MCP clients), the
        resolver must return None gracefully rather than crashing."""
        # The function body should guard the directory existence.
        match = re.search(
            r"def\s+_resolve_claude_session_id[\s\S]*?\n\s*def\s",
            channel_client_src,
        )
        assert match, "could not isolate the resolver function body"
        body = match.group(0)
        assert "is_dir" in body or "exists" in body, "resolver must guard against missing ~/.claude/sessions directory"
        assert "return None" in body, "resolver must return None when no state file is found"

    def test_resolver_polls_for_state_file(self, channel_client_src):
        """Claude Code writes ~/.claude/sessions/<PID>.json AFTER spawning
        its MCP children — empirically ~2 minutes later. A one-shot check
        returns None and falls back to 'main', breaking the binding. The
        resolver must poll with a deadline so it survives the startup race.
        """
        match = re.search(
            r"def\s+_resolve_claude_session_id[\s\S]*?\n\s*def\s",
            channel_client_src,
        )
        assert match
        body = match.group(0)
        # Look for a poll-shaped pattern: a deadline + sleep loop, or
        # equivalent retry mechanism.
        has_poll = bool(re.search(r"poll_timeout|deadline|monotonic|time\.sleep|while\s+True", body))
        assert has_poll, (
            "resolver must poll for the state file (Claude Code writes it "
            "after MCP children spawn — a single up-front check loses the race)"
        )

    def test_resolver_actually_resolves_with_simulated_late_state_file(self, tmp_path, monkeypatch):
        """Live-fire the resolver: simulate Claude Code's race by writing the
        state file ~200ms after the resolver starts. The resolver should
        succeed because of the poll. This catches regressions where someone
        removes the poll thinking it's belt-and-suspenders.
        """
        import importlib.util
        import threading

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fake_sessions = tmp_path / "claude_sessions"
        fake_sessions.mkdir()
        monkeypatch.setattr(mod, "_CLAUDE_SESSIONS_DIR", fake_sessions)
        # Force the resolver to walk the parent chain to a known PID.
        fake_pid = 99999
        monkeypatch.setattr(mod.os, "getppid", lambda: fake_pid)
        monkeypatch.setattr(mod, "_read_parent_pid", lambda _pid: 0)

        target_session = "deadbeef-1234-5678-9abc-def012345678"

        def write_state_file_later():
            import time

            time.sleep(0.2)
            (fake_sessions / f"{fake_pid}.json").write_text(json.dumps({"pid": fake_pid, "sessionId": target_session}))

        threading.Thread(target=write_state_file_later, daemon=True).start()

        # 2-second timeout is more than enough for the 200ms delay
        resolved = mod._resolve_claude_session_id(poll_timeout_s=2.0, poll_interval_s=0.05)
        assert resolved == target_session, (
            f"resolver must wait for the state file (expected {target_session}, got {resolved})"
        )

    def test_resolver_gives_up_after_timeout(self, tmp_path, monkeypatch):
        """If the state file truly never appears (non-Claude MCP client),
        the resolver must return None after its deadline rather than
        blocking indefinitely."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fake_sessions = tmp_path / "claude_sessions"
        fake_sessions.mkdir()
        monkeypatch.setattr(mod, "_CLAUDE_SESSIONS_DIR", fake_sessions)
        monkeypatch.setattr(mod.os, "getppid", lambda: 99999)
        monkeypatch.setattr(mod, "_read_parent_pid", lambda _pid: 0)

        # 200ms total timeout, no file appears
        resolved = mod._resolve_claude_session_id(poll_timeout_s=0.2, poll_interval_s=0.05)
        assert resolved is None

    def test_resolver_falls_back_to_most_recent_state_file(self, tmp_path, monkeypatch):
        """When the parent-chain ancestry doesn't include the claude process
        (e.g., pre-warm-spare wrapper interposed) but a fresh state file
        EXISTS in the directory, the resolver should fall back to using the
        most-recently-modified file rather than collapsing to 'main'.

        This catches the regression where PR #105/#106 worked only when the
        parent-chain happened to contain a claude PID — bg-spare re-parenting
        breaks that assumption.
        """
        import importlib.util
        import time

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fake_sessions = tmp_path / "claude_sessions"
        fake_sessions.mkdir()
        monkeypatch.setattr(mod, "_CLAUDE_SESSIONS_DIR", fake_sessions)
        # Parent chain hits a dead end without ever finding a state file.
        monkeypatch.setattr(mod.os, "getppid", lambda: 99999)
        monkeypatch.setattr(mod, "_read_parent_pid", lambda _pid: 0)
        # Both fake PIDs are "alive" — we're testing recency, not liveness.
        monkeypatch.setattr(mod, "_pid_is_alive", lambda _pid: True)

        # Two state files: an old one and a recent one. The resolver should
        # pick the recent one. Note: the parent-chain PID (99999) is NOT in
        # either filename — this verifies the fallback path.
        (fake_sessions / "11111.json").write_text(json.dumps({"pid": 11111, "sessionId": "old-session"}))
        # Backdate so mtime is far in the past
        old_path = fake_sessions / "11111.json"
        old_time = time.time() - 300
        import os as _os

        _os.utime(old_path, (old_time, old_time))

        (fake_sessions / "22222.json").write_text(json.dumps({"pid": 22222, "sessionId": "newest-session"}))

        # Short timeout — fallback should fire after parent-chain misses
        resolved = mod._resolve_claude_session_id(poll_timeout_s=0.2, poll_interval_s=0.05)
        assert resolved == "newest-session", f"resolver must fall back to most-recent state file; got {resolved!r}"

    def test_fallback_ignores_dead_pid_state_files(self, tmp_path, monkeypatch):
        """The fallback must skip state files whose owning PID is dead.
        Mtime isn't a reliable freshness signal — Claude Code only updates
        the file on status changes, so an active session can be quiet for
        many minutes. Liveness (kill -0) is the right check.

        Here we name a file with a definitely-dead PID (PID 1 doesn't have
        a claude state file; high random PIDs are very likely free) and
        confirm the resolver doesn't pick it.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fake_sessions = tmp_path / "claude_sessions"
        fake_sessions.mkdir()
        monkeypatch.setattr(mod, "_CLAUDE_SESSIONS_DIR", fake_sessions)
        monkeypatch.setattr(mod.os, "getppid", lambda: 99999)
        monkeypatch.setattr(mod, "_read_parent_pid", lambda _pid: 0)
        # Force liveness check to say "dead" for any PID we file under
        monkeypatch.setattr(mod, "_pid_is_alive", lambda _pid: False)

        (fake_sessions / "44444.json").write_text(json.dumps({"pid": 44444, "sessionId": "dead-session"}))

        resolved = mod._resolve_claude_session_id(poll_timeout_s=0.2, poll_interval_s=0.05)
        assert resolved is None, f"dead-PID state files must not be used; got {resolved!r}"

    def test_fallback_picks_alive_pid_over_dead_one(self, tmp_path, monkeypatch):
        """Two state files exist: dead older one is most-recently-modified;
        live newer one (in mtime order) should still be chosen because the
        dead one's PID fails the liveness check."""
        import importlib.util
        import os as _os
        import time

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fake_sessions = tmp_path / "claude_sessions"
        fake_sessions.mkdir()
        monkeypatch.setattr(mod, "_CLAUDE_SESSIONS_DIR", fake_sessions)
        monkeypatch.setattr(mod.os, "getppid", lambda: 99999)
        monkeypatch.setattr(mod, "_read_parent_pid", lambda _pid: 0)
        monkeypatch.setattr(mod, "_pid_is_alive", lambda pid: pid == 77777)

        # Dead PID 55555, recently modified; live PID 77777, older.
        (fake_sessions / "55555.json").write_text(json.dumps({"pid": 55555, "sessionId": "dead-but-recent"}))
        (fake_sessions / "77777.json").write_text(json.dumps({"pid": 77777, "sessionId": "alive-but-older"}))
        old_time = time.time() - 60
        _os.utime(fake_sessions / "77777.json", (old_time, old_time))

        resolved = mod._resolve_claude_session_id(poll_timeout_s=0.2, poll_interval_s=0.05)
        assert resolved == "alive-but-older", (
            f"resolver should skip dead-PID files even if they're more recent; got {resolved!r}"
        )

    def test_startup_log_writes_to_file(self, tmp_path, monkeypatch):
        """channel_client writes a startup diagnostic line to a known path
        on every launch, so operators can post-mortem what session_id the
        resolver picked even though Claude Code captures the MCP child's
        stderr internally."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        log_path = tmp_path / "startup.log"
        monkeypatch.setattr(mod, "_STARTUP_LOG_PATH", log_path)

        mod._write_startup_log(
            resolved_session_id="d4f02587",
            source="~/.claude/sessions/12345.json (parent chain)",
            ppid=12345,
        )
        content = log_path.read_text()
        assert "session_id=d4f02587" in content
        assert "ppid=12345" in content
        assert "source=~/.claude/sessions/12345.json" in content

    def test_startup_log_trims_to_max_lines(self, tmp_path, monkeypatch):
        """The startup log auto-trims to keep size bounded. Without this,
        a long-running operator who restarts claude many times would
        accumulate an unbounded log file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        log_path = tmp_path / "startup.log"
        monkeypatch.setattr(mod, "_STARTUP_LOG_PATH", log_path)
        monkeypatch.setattr(mod, "_STARTUP_LOG_MAX_LINES", 5)

        for i in range(20):
            mod._write_startup_log(
                resolved_session_id=f"sid-{i}",
                source="test",
                ppid=1000 + i,
            )

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) <= 5, f"log should be trimmed to 5 lines, got {len(lines)}"
        # Most recent entries are at the end
        assert "sid-19" in lines[-1]

    def test_startup_log_never_raises(self, tmp_path, monkeypatch):
        """The startup log is best-effort — channel_client must not crash
        if the log path is unwritable (e.g., no perms, read-only fs)."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Point at a path under a file (so mkdir fails)
        existing_file = tmp_path / "a_file"
        existing_file.write_text("x")
        monkeypatch.setattr(mod, "_STARTUP_LOG_PATH", existing_file / "subdir" / "log")

        # Should silently succeed
        mod._write_startup_log(
            resolved_session_id="x",
            source="y",
            ppid=1,
        )

    def test_resolver_prefers_resume_argv_over_state_file(self, tmp_path, monkeypatch):
        """When parent claude's argv contains ``--resume <id>``, the resolver
        must use THAT id directly — not the state file. Claude Code writes a
        placeholder session_id to the state file at startup, then rewrites
        it with the actual --resume target ~28s later. Trusting the state
        file alone gives the placeholder for the entire startup window.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fake_sessions = tmp_path / "claude_sessions"
        fake_sessions.mkdir()
        monkeypatch.setattr(mod, "_CLAUDE_SESSIONS_DIR", fake_sessions)
        monkeypatch.setattr(mod.os, "getppid", lambda: 50000)
        monkeypatch.setattr(mod, "_read_parent_pid", lambda _pid: 0)

        # State file says PLACEHOLDER (the bug we're fixing)
        (fake_sessions / "50000.json").write_text(
            json.dumps({"pid": 50000, "sessionId": "PLACEHOLDER-from-buggy-state-file"})
        )
        # argv says the REAL --resume target
        monkeypatch.setattr(
            mod,
            "_read_resume_arg_from_pid",
            lambda pid: "REAL-resume-target" if pid == 50000 else None,
        )

        resolved = mod._resolve_claude_session_id(poll_timeout_s=0.2, poll_interval_s=0.05)
        assert resolved == "REAL-resume-target", (
            f"resolver must prefer argv --resume value over state file content; got {resolved!r}"
        )

    def test_read_resume_arg_parses_session_id(self, tmp_path, monkeypatch):
        """The argv parser must extract the session_id following --resume."""
        import importlib.util
        import subprocess

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Mock subprocess.check_output to simulate ps output
        def fake_check_output(cmd, **kwargs):
            return b"claude --dangerously-skip-permissions --dangerously-load-development-channels server:mod3 --resume my-target-session-id\n"

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        result = mod._read_resume_arg_from_pid(12345)
        assert result == "my-target-session-id"

    def test_read_resume_arg_rejects_non_claude(self, monkeypatch):
        """The argv parser must NOT match --resume in unrelated processes."""
        import importlib.util
        import subprocess

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def fake_check_output(cmd, **kwargs):
            # A different tool that happens to use --resume
            return b"some_other_tool --resume abc-def\n"

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        assert mod._read_resume_arg_from_pid(12345) is None

    def test_read_resume_arg_rejects_shell_with_claude_in_string(self, monkeypatch):
        """A shell process whose argv contains the string ``claude --resume``
        (e.g., in an echo statement) must NOT match. The first token has to
        be the claude executable, not just a process that mentions it.
        """
        import importlib.util
        import subprocess

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def fake_check_output(cmd, **kwargs):
            # A shell running a command that mentions claude --resume
            return b"/bin/zsh -c echo 'claude --resume sneaky-session-id'\n"

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        assert mod._read_resume_arg_from_pid(12345) is None

    def test_read_resume_arg_accepts_absolute_claude_path(self, monkeypatch):
        """Full path to the claude binary (e.g., /usr/local/bin/claude) must
        still be recognized — we match on basename."""
        import importlib.util
        import subprocess

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def fake_check_output(cmd, **kwargs):
            return b"/home/example/.local/bin/claude --resume my-sid\n"

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        assert mod._read_resume_arg_from_pid(12345) == "my-sid"

    def test_read_resume_arg_handles_bare_resume(self, monkeypatch):
        """``claude --resume`` (no argument; interactive picker) should NOT
        be treated as a session_id. The token following --resume is a flag
        (or absent), not a session_id."""
        import importlib.util
        import subprocess

        spec = importlib.util.spec_from_file_location("cc", REPO_ROOT / "clients" / "channel_client.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def fake_check_output(cmd, **kwargs):
            return b"claude --resume --debug\n"

        monkeypatch.setattr(subprocess, "check_output", fake_check_output)
        assert mod._read_resume_arg_from_pid(12345) is None

    def test_default_poll_timeout_at_least_30s(self, channel_client_src):
        """The default poll_timeout_s must be at least 30 seconds — the
        observed delay between channel_client spawn and Claude Code writing
        its state file was ~35s on one machine. A tighter default risks
        the same race that broke PR #106."""
        match = re.search(
            r"def\s+_resolve_claude_session_id\s*\([\s\S]*?poll_timeout_s\s*:\s*float\s*=\s*([\d.]+)",
            channel_client_src,
        )
        assert match, "could not find poll_timeout_s default in resolver signature"
        default_timeout = float(match.group(1))
        assert default_timeout >= 30.0, (
            f"poll_timeout_s default is {default_timeout}s — observed startup delay was ~35s, "
            "default should be at least 30s (preferably 60s)"
        )

    def test_old_env_substitution_removed_from_config(self, cfg):
        """The original PR #103 used ${CLAUDE_CODE_SESSION_ID:-main} in an env
        block; that approach didn't actually fire because the var isn't in the
        parent claude's env at substitution time. PR #105 removes it to avoid
        confusion (the state-file resolver in channel_client.py is the real
        mechanism now). An env block may still exist for OTHER reasons, but
        MOD3_SESSION_ID specifically should not be set via env-substitution."""
        env = cfg["mcpServers"]["mod3"].get("env", {})
        mod3_session = env.get("MOD3_SESSION_ID", "")
        assert "${CLAUDE_CODE_SESSION_ID" not in mod3_session, (
            "MOD3_SESSION_ID should not rely on ${CLAUDE_CODE_SESSION_ID} substitution — "
            "that var is not in the parent claude process's env at MCP-spawn time. "
            "Use channel_client.py's state-file resolver instead."
        )


# ---------------------------------------------------------------------------
# Path 2 — dashboard postMessage bridge
# ---------------------------------------------------------------------------


class TestDashboardPostMessageBridge:
    @pytest.fixture(scope="class")
    def acp_transport_js(self):
        return (REPO_ROOT / "dashboard" / "acp-transport.js").read_text()

    @pytest.fixture(scope="class")
    def sessions_html(self):
        return (REPO_ROOT / "dashboard" / "sessions.html").read_text()

    @pytest.fixture(scope="class")
    def index_html(self):
        return (REPO_ROOT / "dashboard" / "index.html").read_text()

    def test_acp_transport_exposes_session_resume(self, acp_transport_js):
        """AcpTransport.sessionResume must exist and issue a session/resume
        JSON-RPC request, then store the sessionId for subsequent prompts."""
        assert re.search(r"\basync\s+sessionResume\s*\(", acp_transport_js), (
            "AcpTransport class must define an async sessionResume(sessionId) method"
        )
        assert "'session/resume'" in acp_transport_js or '"session/resume"' in acp_transport_js
        # The method must persist _sessionId so sessionPrompt() finds it.
        assert re.search(
            r"sessionResume[\s\S]{0,400}this\._sessionId\s*=",
            acp_transport_js,
        ), "sessionResume must assign this._sessionId from the result"

    def test_sessions_html_posts_to_opener(self, sessions_html):
        """sessions.html must postMessage to window.opener after a successful
        spawn so the dashboard tab can attach via ACP."""
        assert "window.opener" in sessions_html
        assert "postMessage" in sessions_html
        assert "mod3:claude-code-spawned" in sessions_html, (
            "sessions.html must use the 'mod3:claude-code-spawned' message type (matched by index.html's listener)"
        )
        # Same-origin guard — must scope the postMessage to window.location.origin
        # rather than '*' to avoid leaking the spawn details to unrelated frames.
        assert "window.location.origin" in sessions_html

    def test_index_html_listens_for_spawn_message(self, index_html):
        """index.html must register a message listener that filters on
        origin + message type and binds the ACP transport to the new session.

        Loose 'sessionResume appears anywhere in the file' would let a broken
        implementation that defines the method but never calls it from the
        listener pass. The wiring check below requires the message listener
        to actually invoke attachToClaudeCodeSession with the spawned
        session_id.
        """
        assert "mod3:claude-code-spawned" in index_html, (
            "index.html must listen for the 'mod3:claude-code-spawned' message type"
        )
        # Same-origin check is required on the listener side too.
        assert "ev.origin !== window.location.origin" in index_html or (
            "event.origin !== window.location.origin" in index_html
        )
        # Wiring check: the listener body must call attachToClaudeCodeSession
        # with data.session_id (not just reference sessionResume somewhere).
        # Scope to `window.addEventListener('message', ...)` — the dashboard
        # also has WebSocket-message listeners that match a looser regex.
        listener_match = re.search(
            r"window\.addEventListener\(\s*['\"]message['\"]\s*,\s*\([^)]*\)\s*=>\s*\{[\s\S]*?mod3:claude-code-spawned[\s\S]*?\}\s*\)\s*;",
            index_html,
        )
        assert listener_match, "index.html must register a window message listener with an arrow handler"
        listener_body = listener_match.group(0)
        assert "mod3:claude-code-spawned" in listener_body, (
            "the message listener body must filter on the 'mod3:claude-code-spawned' type"
        )
        assert "attachToClaudeCodeSession" in listener_body, (
            "the message listener body must actually call attachToClaudeCodeSession; "
            "merely defining sessionResume elsewhere is insufficient wiring"
        )
        # And the helper itself must invoke sessionResume.
        # Anchor on the global assignment that follows the helper definition,
        # since the helper body contains nested `}\n` lines that confuse a
        # naïve non-greedy match.
        helper_match = re.search(
            r"async\s+function\s+attachToClaudeCodeSession\s*\([\s\S]*?window\.__mod3AttachToClaudeCodeSession",
            index_html,
        )
        assert helper_match, "attachToClaudeCodeSession helper must exist"
        assert "sessionResume" in helper_match.group(0), (
            "attachToClaudeCodeSession must call the ACP transport's sessionResume"
        )

    def test_origin_mismatch_aborts_listener_path(self, index_html):
        """The same-origin guard must be an early return at the top of the
        message handler — not a conditional that still proceeds on mismatch.

        Without this, a broken refactor (e.g. an `if/else` that flips polarity)
        could let cross-origin messages reach attachToClaudeCodeSession.
        """
        match = re.search(
            r"window\.addEventListener\(\s*['\"]message['\"]\s*,\s*\([^)]*\)\s*=>\s*\{([\s\S]*?mod3:claude-code-spawned[\s\S]*?)\}\s*\)\s*;",
            index_html,
        )
        assert match, "could not locate the spawn-message listener body"
        body = match.group(1)
        # The origin check must appear before attachToClaudeCodeSession; otherwise
        # cross-origin messages could reach the attach logic.
        origin_idx = -1
        for needle in ("ev.origin", "event.origin"):
            i = body.find(needle)
            if i >= 0:
                origin_idx = i
                break
        assert origin_idx >= 0, "listener body must reference ev.origin or event.origin"
        attach_idx = body.find("attachToClaudeCodeSession")
        assert attach_idx > origin_idx, (
            "the origin check must appear before attachToClaudeCodeSession; "
            "otherwise cross-origin messages could trigger the attach"
        )

    def test_attach_helper_waits_for_seat(self, index_html):
        """The attach helper must poll for the channel-client seat to register
        before issuing session/resume, since the spawned subprocess needs a
        moment to launch and POST its seat. Otherwise session/prompt would
        fail with 'No channel-client seats attached' immediately after spawn."""
        # Match the fetch against the seats endpoint with the session_id.
        assert re.search(
            r"/v1/sessions/\$\{[^}]+\}/seats|/v1/sessions/.+seats",
            index_html,
        ), "attach helper must poll /v1/sessions/<id>/seats before calling sessionResume"

    def test_attach_helper_skips_resume_on_timeout(self, index_html):
        """If the seat-poll deadline expires without a seat appearing, the
        helper must NOT call sessionResume — otherwise the subsequent
        session/prompt fails with the cryptic 'no seats' error instead of
        the cause (spawn produced no working channel client)."""
        # Anchor on the global assignment that follows the helper definition,
        # since the helper body contains nested `}\n` lines that confuse a
        # naïve non-greedy match.
        helper_match = re.search(
            r"async\s+function\s+attachToClaudeCodeSession\s*\([\s\S]*?window\.__mod3AttachToClaudeCodeSession",
            index_html,
        )
        assert helper_match
        body = helper_match.group(0)
        # The helper must track whether a seat was found and skip resume when
        # it wasn't. Accept either an explicit boolean or an early-return.
        assert re.search(r"seatFound|no seat appeared", body), (
            "helper must track seat-found state and warn/abort when the poll deadline expires without a seat"
        )

    def test_attach_helper_has_supersede_guard(self, index_html):
        """When two postMessages for different session_ids arrive in rapid
        succession, only the latest attach attempt should call sessionResume.
        Without this guard, a slow first attach can overwrite a fast second
        bind, and the dashboard ends up addressing the wrong session.

        Implementation may use a monotonic token, an AbortController, or
        equivalent — just verify some form of newer-wins discipline exists.
        """
        # Anchor on the global assignment that follows the helper definition,
        # since the helper body contains nested `}\n` lines that confuse a
        # naïve non-greedy match.
        helper_match = re.search(
            r"async\s+function\s+attachToClaudeCodeSession\s*\([\s\S]*?window\.__mod3AttachToClaudeCodeSession",
            index_html,
        )
        assert helper_match
        body = helper_match.group(0)
        has_token_guard = bool(re.search(r"_attachAttemptToken|AbortController|attemptId|superseded", body))
        assert has_token_guard, (
            "attachToClaudeCodeSession must guard against newer attempts overwriting older ones; "
            "expected a token/AbortController/'superseded' pattern in the helper"
        )

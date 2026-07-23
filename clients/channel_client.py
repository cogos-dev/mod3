"""Mod³ Channel Client — standalone stdio MCP server for Claude Code channel integration.

Architecture
------------
This process is spawned by Claude Code as a child process (stdio MCP transport).
It authenticates with the Mod³ HTTP daemon (localhost:7860) and registers a
"seat" in a mod³ session.  Events from the session are forwarded to Claude Code
via MCP notifications; Claude Code can call mod3_dashboard_post and mod3_speak
to send outbound messages.

        Claude Code
           │ spawns (stdio)
           ▼
  channel_client.py  ← this file
           │ HTTP/SSE
           ▼
  mod3 HTTP daemon (port 7860)
           │ session with N seats (one per channel_client.py child)

Auth
----
Bearer token read from ~/.mod3/channel-client.token.
On first run (no token file) a UUID token is generated, written to that file,
and registered with the mod3 daemon via POST /v1/channel-tokens.

Seat lifecycle
--------------
  startup  → POST /v1/sessions/{session_id}/seats  → get seat_id
  running  → GET  /v1/sessions/{session_id}/seats/{seat_id}/events  (SSE)
             each event → notifications/claude/channel
  shutdown → DELETE /v1/sessions/{session_id}/seats/{seat_id}

MCP surface
-----------
  capabilities:
    experimental["claude/channel"] = {}
    experimental["claude/channel/permission"] = {}
    tools = {}
  instructions: channel-tag shape + outbound tools description
  tools:
    mod3_dashboard_post(text, role?)  → POST /v1/dashboard-chat
    mod3_speak(text, voice?, speed?, skip_playback?, format?)  → POST /v1/speak or /v1/synthesize

Usage
-----
  # Normal (spawned by Claude Code via mcp.channel.json):
  python3 clients/channel_client.py [--session SESSION_ID] [--server URL]

  # Connectivity smoke-test (exits 0 on success):
  python3 clients/channel_client.py --test [--server URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

# Sibling module; this file runs as a script, so clients/ is on sys.path.
# Import failure degrades to "no sink" rather than killing the channel.
try:
    import theseus_sink
except ImportError:  # pragma: no cover — packaging/layout drift
    theseus_sink = None  # type: ignore[assignment]

logger = logging.getLogger("mod3.channel_client")

# Strong refs so in-flight sink tasks aren't garbage-collected mid-write.
_SINK_TASKS: set = set()


def _sink_fire_and_forget(text: str, thread: str, from_id: str) -> None:
    """Best-effort THESEUS ledger sink for a conversational utterance.

    Runs theseus_sink.sink_turn in a worker thread and drops the handle —
    speech and dashboard posts never wait on a git push. Disabled cleanly
    when the module is missing, the repo is absent, or MOD3_THESEUS_SINK=0.
    """
    if theseus_sink is None or not theseus_sink.enabled():
        return

    async def _run() -> None:
        result = await asyncio.to_thread(
            theseus_sink.sink_turn, text, thread, from_id
        )
        if not result.get("ok"):
            logger.warning("theseus sink failed: %s", result.get("error"))
        else:
            logger.debug("theseus sink: %s", result)

    try:
        task = asyncio.get_running_loop().create_task(_run())
        _SINK_TASKS.add(task)
        task.add_done_callback(_SINK_TASKS.discard)
    except RuntimeError:  # no running loop (sync caller in tests)
        theseus_sink.sink_turn(text, thread, from_id)

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------

_DEFAULT_SERVER_URL = "http://localhost:7860"
_DEFAULT_SESSION_ID = "main"
_TOKEN_PATH = Path.home() / ".mod3" / "channel-client.token"
_CLAUDE_SESSIONS_DIR = Path.home() / ".claude" / "sessions"
_STARTUP_LOG_PATH = Path.home() / ".mod3" / "channel-client-startup.log"
_STARTUP_LOG_MAX_LINES = 200


def _resolve_claude_session_id(
    *,
    poll_timeout_s: float = 60.0,
    poll_interval_s: float = 0.2,
) -> str | None:
    """Find the harness session_id of the Claude Code process that spawned us.

    Resolution order (highest priority first):

    1. **Parse ``--resume <id>`` from the parent claude's argv.** When claude
       is launched with ``--resume <id>``, the session_id is right there in
       its command line. This is the most reliable source — beats the state
       file (which has a startup race), beats anything else.

    2. **Read ``~/.claude/sessions/<PID>.json``.** Claude Code writes this for
       every active claude process. We walk the parent-chain (because pre-warm
       spares can interpose a wrapper) and poll for any candidate's file to
       appear. Used for new sessions and ``--continue``.

    3. **Most-recently-modified live state file.** Fallback when the
       parent-chain ancestry doesn't trace back to claude.

    Empirically-observed quirks the design has to absorb:

    - **Startup race.** Channel_client spawns ~2s after claude; claude writes
      state file LATER (up to 35s observed). 60s poll covers it.
    - **State file rewrite for --resume.** Claude writes a PLACEHOLDER
      session_id first, then rewrites with the actual --resume target ~28s
      later. The argv-parse path (above) avoids this entirely by reading
      argv instead of trusting the file.
    - **Ancestry mismatch.** Pre-warm spares re-parent the MCP child away
      from the claude binary; parent-chain misses, fallback fires.

    Returns the sessionId string, or ``None`` if all sources are exhausted.
    """
    import time

    # Snapshot the parent-chain once. PPIDs don't change during this poll
    # (macOS re-parents only when a parent dies). Walk up to 6 levels —
    # usually claude is the direct parent, but pre-warm-spare configurations
    # can interpose a wrapper.
    candidates: list[int] = []
    pid = os.getppid()
    for _ in range(6):
        if pid <= 1:
            break
        candidates.append(pid)
        pid = _read_parent_pid(pid)

    # Priority 1: parse --resume <id> from any claude ancestor's argv. This
    # is immune to the placeholder-rewrite bug — argv is set at exec() time
    # and never changes for the life of the process.
    for cand_pid in candidates:
        sid = _read_resume_arg_from_pid(cand_pid)
        if sid:
            return sid

    if not _CLAUDE_SESSIONS_DIR.is_dir():
        return None

    deadline = time.monotonic() + poll_timeout_s
    while candidates:
        for cand_pid in candidates:
            sid = _read_session_id(_CLAUDE_SESSIONS_DIR / f"{cand_pid}.json")
            if sid:
                return sid
        if time.monotonic() >= deadline:
            break
        time.sleep(poll_interval_s)

    # Parent-chain poll didn't yield. Fall back to the most-recently-modified
    # state file in the directory. This handles bg-spare/wrapper interposition
    # where the channel_client's ancestry doesn't actually trace back to the
    # claude process currently driving it.
    return _most_recent_state_file_session_id()


def _read_session_id(session_file: "Path") -> str | None:
    if not session_file.is_file():
        return None
    try:
        payload = json.loads(session_file.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    sid = payload.get("sessionId")
    return sid if isinstance(sid, str) and sid else None


def _most_recent_state_file_session_id() -> str | None:
    """Find the most-recently-modified ``~/.claude/sessions/<PID>.json``
    whose PID is still alive, and return its sessionId. Liveness — not file
    mtime — is the freshness signal: Claude Code only updates the state
    file on status changes, so an active session can have an mtime of many
    minutes ago. We skip files whose owning PID has died (abandoned state).
    """
    try:
        files = list(_CLAUDE_SESSIONS_DIR.glob("*.json"))
    except OSError:
        return None
    if not files:
        return None
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files:
        # File name is <PID>.json
        try:
            pid = int(f.stem)
        except ValueError:
            continue
        if not _pid_is_alive(pid):
            continue
        sid = _read_session_id(f)
        if sid:
            return sid
    return None


def _session_id_source(session_id: str) -> str:
    """Classify where a resolved session_id came from for diagnostic logging."""
    if session_id == _DEFAULT_SESSION_ID:
        return f"DEFAULT ({_DEFAULT_SESSION_ID})"
    if os.environ.get("MOD3_SESSION_ID") == session_id:
        return "MOD3_SESSION_ID env"
    # Either parent-chain or fallback — distinguish by checking if the
    # parent's state file matches.
    ppid = os.getppid()
    direct = _read_session_id(_CLAUDE_SESSIONS_DIR / f"{ppid}.json")
    if direct == session_id:
        return f"~/.claude/sessions/{ppid}.json (parent chain)"
    return "~/.claude/sessions/*.json (most-recent fallback)"


def _write_startup_log(*, resolved_session_id: str, source: str, ppid: int) -> None:
    """Append a startup diagnostic line, then trim to last N lines.

    Format: ``YYYY-MM-DDTHH:MM:SS pid=<self> ppid=<parent> session_id=<id> source=<...>``

    Best-effort; never raises (the channel client must not die because the
    log directory can't be written).
    """
    import datetime as _dt

    try:
        _STARTUP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"{_dt.datetime.now().isoformat(timespec='seconds')} "
            f"pid={os.getpid()} ppid={ppid} "
            f"session_id={resolved_session_id} "
            f"source={source}\n"
        )
        try:
            existing = _STARTUP_LOG_PATH.read_text().splitlines()
        except (OSError, FileNotFoundError):
            existing = []
        trimmed = existing[-(_STARTUP_LOG_MAX_LINES - 1) :] + [line.rstrip()]
        _STARTUP_LOG_PATH.write_text("\n".join(trimmed) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with this PID exists. Sends signal 0
    (kernel-level check, no actual signal delivered)."""
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        # PermissionError means the process exists but we can't signal it;
        # that's still 'alive' for our purposes.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    except OSError:
        return False


def _read_parent_pid(pid: int) -> int:
    """Return the parent PID of ``pid``, or 0 if it can't be determined.

    Uses ``ps`` rather than /proc because macOS has no /proc filesystem.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        return 0
    text = out.decode("ascii", errors="ignore").strip()
    return int(text) if text.isdigit() else 0


def _read_resume_arg_from_pid(pid: int) -> str | None:
    """Look for ``--resume <session_id>`` in the argv of ``pid``.

    Returns the session_id if found, else ``None``. Only considers processes
    whose argv contains ``claude`` (otherwise we'd match unrelated processes
    that happen to use a ``--resume`` flag).

    This is the most reliable source of the harness session_id when claude
    is launched with ``--resume``: argv is set at exec() time and never
    changes, unlike the ``~/.claude/sessions/<PID>.json`` state file which
    gets rewritten by claude ~28s after startup with the actual --resume
    target (the initial write contains a placeholder).
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["ps", "-o", "command=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    argv = out.decode("utf-8", errors="ignore").strip()
    tokens = argv.split()
    if not tokens:
        return None
    # Only consider processes where claude is the actual executable (the
    # first token), not just a string appearing somewhere in argv (e.g. an
    # `echo "claude --resume ..."` in a shell). The first token's basename
    # must be ``claude`` (the binary name) — full path is fine.
    exe_basename = tokens[0].rsplit("/", 1)[-1]
    if exe_basename != "claude":
        return None
    for i, tok in enumerate(tokens):
        if tok == "--resume" and i + 1 < len(tokens):
            candidate = tokens[i + 1]
            # Reject the bare ``--resume`` (interactive picker) where the
            # next token is another flag rather than a session_id.
            if candidate.startswith("-"):
                return None
            return candidate
    return None


# ---------------------------------------------------------------------------
# Auth token helpers
# ---------------------------------------------------------------------------


def _load_or_create_token() -> str:
    """Return the persistent bearer token, generating one on first run."""
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _TOKEN_PATH.exists():
        token = _TOKEN_PATH.read_text().strip()
        if token:
            return token
    token = str(uuid.uuid4())
    _TOKEN_PATH.write_text(token + "\n")
    logger.info("Generated new channel-client token at %s", _TOKEN_PATH)
    return token


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register_seat(
    client: httpx.AsyncClient,
    server_url: str,
    session_id: str,
    token: str,
    device_uuid: str,
) -> dict[str, Any]:
    """POST /v1/sessions/{session_id}/seats — register a seat, create session if needed."""
    url = f"{server_url}/v1/sessions/{session_id}/seats"
    body = {"client_type": "claude-code-channel", "device_uuid": device_uuid}
    resp = await client.post(url, json=body, headers=_auth_headers(token), timeout=10.0)
    resp.raise_for_status()
    return resp.json()


async def _delete_seat(
    server_url: str,
    session_id: str,
    seat_id: str,
    token: str,
) -> None:
    """DELETE /v1/sessions/{session_id}/seats/{seat_id} — clean up on exit."""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{server_url}/v1/sessions/{session_id}/seats/{seat_id}"
            await client.delete(url, headers=_auth_headers(token), timeout=5.0)
            logger.info("Seat %s deleted", seat_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Seat delete failed (already gone?): %s", exc)


# ---------------------------------------------------------------------------
# Channel client core
# ---------------------------------------------------------------------------


class ChannelClient:
    """Coordinates seat registration, SSE subscription, and MCP server."""

    def __init__(self, server_url: str, session_id: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.session_id = session_id
        self.token = _load_or_create_token()
        self.device_uuid = self._load_or_create_device_uuid()
        self.seat_id: str | None = None
        self._notification_sender: Any = None  # set when MCP session starts

    def _load_or_create_device_uuid(self) -> str:
        uuid_path = _TOKEN_PATH.parent / "device-uuid"
        uuid_path.parent.mkdir(parents=True, exist_ok=True)
        if uuid_path.exists():
            val = uuid_path.read_text().strip()
            if val:
                return val
        val = str(uuid.uuid4())
        uuid_path.write_text(val + "\n")
        return val

    def set_notification_sender(self, sender: Any) -> None:
        self._notification_sender = sender

    async def send_channel_notification(self, content: str, meta: dict | None = None) -> None:
        """Forward a seat event to Claude Code as notifications/claude/channel."""
        if self._notification_sender is None:
            logger.debug("No notification sender yet; dropping event: %s", content[:80])
            return
        params: dict[str, Any] = {"content": content}
        if meta:
            params["meta"] = meta
        notification = JSONRPCNotification(
            jsonrpc="2.0",
            method="notifications/claude/channel",
            params=params,
        )
        try:
            await self._notification_sender(notification)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send channel notification: %s", exc)

    async def _stream_seat_events_once(self) -> None:
        """One pass over the SSE stream. Returns on clean stream end.

        Raises asyncio.CancelledError on intentional cancel; surfaces httpx
        errors to the caller so the reconnect loop can act on them.
        """
        url = f"{self.server_url}/v1/sessions/{self.session_id}/seats/{self.seat_id}/events"
        headers = {**_auth_headers(self.token), "Accept": "text/event-stream"}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=headers) as resp:
                resp.raise_for_status()
                event_type: str | None = None
                data_lines: list[str] = []
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        event_type = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].strip())
                    elif line == "":
                        if data_lines:
                            raw = "\n".join(data_lines)
                            await self._handle_sse_event(event_type, raw)
                        event_type = None
                        data_lines = []

    async def _reregister_seat(self) -> bool:
        """Re-POST the seat registration. Used after mod3 restart wipes state."""
        try:
            async with httpx.AsyncClient() as client:
                result = await _register_seat(
                    client,
                    self.server_url,
                    self.session_id,
                    self.token,
                    self.device_uuid,
                )
            new_seat_id = result.get("seat_id")
            if not new_seat_id:
                logger.warning("Re-register returned no seat_id: %r", result)
                return False
            self.seat_id = new_seat_id
            logger.info(
                "Seat re-registered after disconnect: seat_id=%s session_id=%s",
                self.seat_id,
                self.session_id,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Seat re-register failed: %s", exc)
            return False

    async def run_sse_subscription(self) -> None:
        """Stream seat events with reconnect-on-disconnect.

        Originally a one-shot subscriber — when mod3 restarted (or any
        connection blip dropped the stream), the seat was silently lost and
        the dashboard sidebar would not surface this Claude Code session
        until the MCP child was respawned. Now: on any disconnect we re-POST
        the seat (mod3 wiped its registry on restart, so the old seat_id is
        invalid) and resume streaming under the new seat_id.

        Backoff: 1s, 2s, 4s, ..., capped at 30s; reset to 1s after a stream
        survives longer than ``_HEALTHY_STREAM_S``.
        """
        if not self.seat_id:
            logger.error("No seat_id — cannot subscribe to events")
            return

        backoff = 1.0
        backoff_cap = 30.0
        _HEALTHY_STREAM_S = 5.0

        while True:
            logger.info(
                "Subscribing to seat events: session=%s seat=%s",
                self.session_id,
                self.seat_id,
            )
            started = asyncio.get_event_loop().time()
            stream_error: str | None = None
            try:
                await self._stream_seat_events_once()
                stream_error = "stream ended cleanly"
            except asyncio.CancelledError:
                logger.debug("SSE subscription cancelled")
                raise
            except httpx.HTTPStatusError as exc:
                stream_error = f"http {exc.response.status_code}"
            except httpx.RemoteProtocolError as exc:
                stream_error = f"remote protocol error: {exc}"
            except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout) as exc:
                stream_error = f"connect/read error: {exc}"
            except Exception as exc:  # noqa: BLE001
                stream_error = f"unexpected: {exc}"

            elapsed = asyncio.get_event_loop().time() - started
            logger.info(
                "SSE subscription ended after %.1fs: %s — attempting reconnect",
                elapsed,
                stream_error,
            )
            if elapsed >= _HEALTHY_STREAM_S:
                backoff = 1.0

            # Re-register with exponential backoff until success. Banging on
            # the stream while mod3 is still down just thrashes; the inner
            # loop sleeps + retries the cheap POST until mod3 responds.
            while True:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, backoff_cap)
                if await self._reregister_seat():
                    break
                # else: keep looping with increasing backoff

    async def _handle_sse_event(self, event_type: str | None, raw_data: str) -> None:
        """Parse an SSE event and forward to Claude Code."""
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = {"raw": raw_data}

        etype = event_type or payload.get("type", "event")

        if etype == "user_message":
            text = payload.get("content", "")
            meta = {
                "session_id": self.session_id,
                "seat_id": self.seat_id,
                "source": "mod3",
                "input_type": payload.get("input_type", "text"),
            }
            channel_content = (
                f'<channel source="mod3" session_id="{self.session_id}" '
                f'seat_id="{self.seat_id}" input_type="{meta["input_type"]}">'
                f"{text}</channel>"
            )
            await self.send_channel_notification(channel_content, meta)

        elif etype == "pairing_request":
            code = payload.get("code", "")
            identifier = payload.get("identifier", "")
            meta = {"session_id": self.session_id, "seat_id": self.seat_id, "source": "mod3"}
            channel_content = (
                f'<channel source="mod3" pairing_request="true" '
                f'identifier="{identifier}" code="{code}">'
                f"Pairing request from identifier '{identifier}'. "
                f"Run `/mod3:access pair {code}` to approve.</channel>"
            )
            await self.send_channel_notification(channel_content, meta)

        elif etype == "permission_request":
            meta = {"session_id": self.session_id, "seat_id": self.seat_id, "source": "mod3"}
            channel_content = (
                f'<channel source="mod3" permission_request="true" '
                f'session_id="{self.session_id}">'
                f"{json.dumps(payload)}</channel>"
            )
            await self.send_channel_notification(channel_content, meta)

        else:
            logger.debug("Unhandled SSE event type %r: %s", etype, raw_data[:200])


# ---------------------------------------------------------------------------
# MCP server setup
# ---------------------------------------------------------------------------

_CHANNEL_INSTRUCTIONS = """
Mod³ Channel Client — you are connected to the Mod³ TTS daemon as a channel seat.

## Inbound channel tags

Dashboard messages arrive as MCP notifications (method: notifications/claude/channel)
with a content field shaped like:

  <channel source="mod3" session_id="main" seat_id="seat-xxxx" input_type="text">
  User's message here
  </channel>

Pairing requests look like:

  <channel source="mod3" pairing_request="true" identifier="uuid" code="abcde">
  Pairing request from identifier 'uuid'. Run `/mod3:access pair abcde` to approve.
  </channel>

## Outbound tools

  mod3_dashboard_post(text, role?) — send text to the dashboard chat panel
  mod3_speak(text, voice?, speed?, skip_playback?, format?) — synthesize text to speech and play it (or return base64 bytes when skip_playback=True)

Always reply to user messages using mod3_dashboard_post so your response appears
in the dashboard.  Use mod3_speak when voice output is appropriate.
"""


def build_mcp_server(client: ChannelClient) -> FastMCP:
    """Build the FastMCP server with channel capabilities and tools."""
    mcp = FastMCP(
        "mod3-channel",
        instructions=_CHANNEL_INSTRUCTIONS,
    )

    # Declare channel capabilities (Anthropic Channels spec)
    mcp.experimental_capabilities = {
        "claude/channel": {},
        "claude/channel/permission": {},
    }

    @mcp.tool()
    async def mod3_dashboard_post(text: str, role: str = "assistant") -> str:
        """Send text to the Mod³ dashboard chat panel.

        Args:
            text: The message text to display in the dashboard.
            role: Message role — "assistant" (default) or "user".
        """
        url = f"{client.server_url}/v1/dashboard-chat"
        body = {
            "text": text,
            "role": role,
            "session_id": client.session_id,
            "seat_id": client.seat_id,
        }
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(url, json=body, headers=_auth_headers(client.token), timeout=10.0)
                if resp.status_code in (200, 404):
                    # 404 = endpoint not yet implemented on server; degrade gracefully
                    if resp.status_code == 404:
                        logger.debug("/v1/dashboard-chat not found — falling back to /ws broadcast")
                else:
                    resp.raise_for_status()
            # Durable half: the dashboard's message store is RAM-only, so the
            # seat's own posts also land in the THESEUS ledger. Only the
            # seat's voice ("assistant") is sunk — a relayed "user" turn is
            # someone else's words and must never be stamped origin=seat.
            if role == "assistant":
                _sink_fire_and_forget(
                    text,
                    os.environ.get("MOD3_THESEUS_THREAD", "dashboard"),
                    "seat-root-dashboard",
                )
            return "ok"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @mcp.tool()
    async def mod3_speak(
        text: str,
        voice: str = "eng_uk_m_davids",
        speed: float = 1.0,
        post_to_chat: bool = True,
        skip_playback: bool = False,
        format: str = "wav",
        ledger_thread: str = "",
    ) -> dict:
        """Synthesize text to speech and play it through the Mod³ daemon.

        When skip_playback=False (default): hits POST /v1/speak (queue-aware
        endpoint) and returns immediately with a job token — the daemon's drain
        thread owns all audio playback.

        When skip_playback=True: hits POST /v1/synthesize directly and returns
        the encoded audio as a base64 string under the key "audio_base64", with
        "media_type" set to the appropriate MIME type. No audio is played.
        Useful for retrieving OGG/Opus bytes to forward over a remote channel.

        By default also posts the spoken text to the dashboard chat panel as
        an ``assistant`` message so the visible transcript matches what the
        operator heard. Without this, audio plays but the chat pane shows
        nothing — confusing when the operator goes back later to reread.
        Pass ``post_to_chat=False`` for system sounds, audio cues, or other
        non-conversational speech that shouldn't appear in the transcript.

        Args:
            text: Text to synthesize and speak aloud.
            voice: Voice preset name (use list_voices to discover options).
                Default: eng_uk_m_davids (Chatterbox-Turbo, British male).
            speed: Playback speed multiplier (0.5–2.0). Default: 1.0.
            post_to_chat: When True (default), also POSTs to /v1/dashboard-chat
                under the current session_id so the spoken text appears in the
                dashboard chat panel and is persisted to per-session history.
            skip_playback: When True, call /v1/synthesize and return the audio
                bytes as base64 without playing. Default: False.
            format: Output audio format. One of "wav", "pcm", "ogg".
                "ogg" produces OGG/Opus at ~24 kbps. Only meaningful when
                skip_playback=True (playback always uses WAV internally).
                Default: "wav".
            ledger_thread: THESEUS conversation thread the spoken text is
                sunk to (durable ledger in the book repo). Empty (default)
                uses $MOD3_THESEUS_THREAD or "voice". Conversational speech
                during a THESEUS thread conversation should name that thread
                so the spoken and written halves land together. Sinking
                follows post_to_chat: system sounds and non-conversational
                speech (post_to_chat=False) are not sunk.

        Returns:
            skip_playback=False:
                {"job_id": str, "queue_position": int, "status": "speaking"|"queued"}
                Poll GET /v1/jobs/{job_id} for completion. Stop via POST /v1/stop.
            skip_playback=True:
                {"audio_base64": str, "media_type": str, "format": str,
                 "duration_sec": float, "sample_rate": int}
        """
        import base64

        try:
            async with httpx.AsyncClient() as http:
                if skip_playback:
                    synth_url = f"{client.server_url}/v1/synthesize"
                    synth_body: dict[str, Any] = {
                        "text": text,
                        "voice": voice,
                        "speed": speed,
                        "format": format,
                    }
                    resp = await http.post(
                        synth_url,
                        json=synth_body,
                        headers=_auth_headers(client.token),
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    audio_b64 = base64.b64encode(resp.content).decode("ascii")
                    media_type = resp.headers.get("content-type", "audio/wav").split(";")[0].strip()
                    duration_sec = float(resp.headers.get("X-Mod3-Duration-Sec", "0"))
                    sample_rate = int(resp.headers.get("X-Mod3-Sample-Rate", "0"))
                    return {
                        "audio_base64": audio_b64,
                        "media_type": media_type,
                        "format": format,
                        "duration_sec": duration_sec,
                        "sample_rate": sample_rate,
                    }

                speak_url = f"{client.server_url}/v1/speak"
                speak_body: dict[str, Any] = {
                    "text": text,
                    "voice": voice,
                    "speed": speed,
                }
                # Fan the same text to the dashboard chat panel first so the
                # transcript lands at roughly the same time as audio playback
                # begins. Best-effort: a failed post must not block speech.
                if post_to_chat:
                    try:
                        chat_url = f"{client.server_url}/v1/dashboard-chat"
                        chat_body = {
                            "text": text,
                            "role": "assistant",
                            "session_id": client.session_id,
                            "seat_id": client.seat_id,
                        }
                        await http.post(
                            chat_url,
                            json=chat_body,
                            headers=_auth_headers(client.token),
                            timeout=5.0,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("mod3_speak: chat mirror failed (non-fatal): %s", exc)

                resp = await http.post(speak_url, json=speak_body, headers=_auth_headers(client.token), timeout=30.0)
                resp.raise_for_status()
                # Durable half: sink the spoken text to the THESEUS ledger,
                # gated on the same flag as the visible transcript — speech
                # the operator is meant to reread is speech worth keeping.
                # After raise_for_status: only accepted speech is recorded.
                if post_to_chat:
                    _sink_fire_and_forget(
                        text,
                        ledger_thread
                        or os.environ.get("MOD3_THESEUS_THREAD", "voice"),
                        "seat-root-voice",
                    )
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    return mcp


# ---------------------------------------------------------------------------
# Main entrypoints
# ---------------------------------------------------------------------------


async def _run_channel(server_url: str, session_id: str) -> None:
    """Full channel client: register seat, subscribe SSE, serve MCP over stdio."""
    client = ChannelClient(server_url=server_url, session_id=session_id)
    mcp = build_mcp_server(client)

    # Register seat with mod3 HTTP daemon
    async with httpx.AsyncClient() as http:
        try:
            result = await _register_seat(
                http,
                server_url=client.server_url,
                session_id=client.session_id,
                token=client.token,
                device_uuid=client.device_uuid,
            )
            client.seat_id = result["seat_id"]
            logger.info("Registered seat %s in session %s", client.seat_id, client.session_id)
        except Exception as exc:
            logger.error("Failed to register seat: %s", exc)
            sys.exit(1)

    # Run SSE subscription in background
    sse_task = asyncio.create_task(client.run_sse_subscription())

    # Cleanup on exit
    async def _cleanup(signum=None):
        # Drain in-flight ledger sinks first (bounded): a session whose FINAL
        # tool call was mod3_speak would otherwise have its sink task swept by
        # asyncio.run()'s shutdown cancellation before the receipt is written,
        # silently dropping the last utterance from the ledger.
        pending = {t for t in _SINK_TASKS if not t.done()}
        if pending:
            done, still = await asyncio.wait(pending, timeout=15.0)
            if still:
                logger.warning(
                    "theseus sink: %d receipt(s) still in flight at shutdown "
                    "after 15s drain — may be lost",
                    len(still),
                )
        sse_task.cancel()
        try:
            await sse_task
        except asyncio.CancelledError:
            pass
        if client.seat_id:
            await _delete_seat(
                server_url=client.server_url,
                session_id=client.session_id,
                seat_id=client.seat_id,
                token=client.token,
            )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_cleanup()))

    # Run MCP server over stdio, wiring the notification sender
    async with stdio_server() as (read_stream, write_stream):
        # Capture the write stream so we can send notifications
        async def _send_notification(notification: JSONRPCNotification) -> None:
            msg = SessionMessage(message=JSONRPCMessage(notification))
            await write_stream.send(msg)

        client.set_notification_sender(_send_notification)

        try:
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
        finally:
            await _cleanup()


async def _run_test(server_url: str) -> int:
    """Smoke test: check mod3 health and seat registration, exit 0 on success."""
    token = _load_or_create_token()
    device_uuid = str(uuid.uuid4())  # ephemeral for test

    print(f"Testing connection to {server_url} ...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health check
            resp = await client.get(f"{server_url}/health")
            if resp.status_code != 200:
                print(f"FAIL: /health returned {resp.status_code}")
                return 1
            print(f"  /health OK ({resp.status_code})")

            # Seat registration
            result = await _register_seat(
                client,
                server_url=server_url,
                session_id="test",
                token=token,
                device_uuid=device_uuid,
            )
            seat_id = result.get("seat_id")
            if not seat_id:
                print(f"FAIL: no seat_id in response: {result}")
                return 1
            print(f"  seat registered: {seat_id}")

            # Cleanup
            await _delete_seat(
                server_url=server_url,
                session_id="test",
                seat_id=seat_id,
                token=token,
            )
            print(f"  seat deleted: {seat_id}")

        print("PASS: connectivity test succeeded")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("MOD3_LOG_LEVEL", "WARNING"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Mod³ Channel Client — stdio MCP server for Claude Code channel integration",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("MOD3_SERVER_URL", _DEFAULT_SERVER_URL),
        help="Mod³ HTTP server URL (default: %(default)s)",
    )
    # Default session_id resolution priority (most → least preferred):
    #   1. Explicit --session arg
    #   2. MOD3_SESSION_ID env var (e.g. injected by the kernel's spawn flow
    #      at /v1/claude-code/spawn → temp .mcp.json with --session <id>)
    #   3. ~/.claude/sessions/<parent-pid>.json (the canonical mechanism for
    #      manual-launch Claude Code: the harness writes its own state file
    #      at startup, keyed by PID)
    #   4. _DEFAULT_SESSION_ID ("main") — last-resort fallback for non-Claude
    #      MCP clients
    default_session = os.environ.get("MOD3_SESSION_ID") or _resolve_claude_session_id() or _DEFAULT_SESSION_ID
    parser.add_argument(
        "--session",
        default=default_session,
        help="Mod³ session ID to join (default: %(default)s)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run connectivity smoke test and exit (does not start stdio MCP server)",
    )
    args = parser.parse_args()

    # Belt-and-suspenders: if the resolved session_id is empty/whitespace
    # (e.g. MOD3_SESSION_ID="" or a malformed Claude session file), fall back
    # to the default rather than registering a seat under session_id="" which
    # would land at the malformed path /v1/sessions//seats.
    if not args.session or not args.session.strip():
        logger.warning(
            "channel_client: --session resolved to empty; falling back to %r",
            _DEFAULT_SESSION_ID,
        )
        args.session = _DEFAULT_SESSION_ID

    # Diagnostic startup log to a file the operator can read post-mortem.
    # Claude Code captures the MCP child's stderr internally and it's not
    # easy to retrieve, so without this we have no visibility into what
    # session_id the resolver picked. The startup log is small (one line
    # per invocation) and auto-rotates by trimming old entries to the
    # most recent 200 lines.
    _write_startup_log(
        resolved_session_id=args.session,
        source=_session_id_source(args.session),
        ppid=os.getppid(),
    )

    if args.session != _DEFAULT_SESSION_ID:
        logger.info(
            "channel_client: resolved session_id=%s (from %s)",
            args.session,
            "MOD3_SESSION_ID" if os.environ.get("MOD3_SESSION_ID") else "~/.claude/sessions/<ppid>.json",
        )

    if args.test:
        rc = asyncio.run(_run_test(args.server))
        sys.exit(rc)
    else:
        asyncio.run(_run_channel(server_url=args.server, session_id=args.session))


if __name__ == "__main__":
    main()

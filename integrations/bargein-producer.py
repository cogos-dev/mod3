#!/usr/bin/env python3
"""
Barge-in signal producer — detects SuperWhisper recording and writes
the signal file that Mod3's barge-in consumer watches.

DEPRECATED as a standalone script once mod3 absorbs this functionality.
Prefer the in-process provider at ``mod3.bargein.providers.superwhisper``
(enabled by setting ``MOD3_BARGEIN_PROVIDERS=superwhisper``), which calls
into mod3's barge-in consumer directly instead of going through the
``/tmp/mod3-barge-in.json`` file IPC. This script is retained so existing
launchd users (e.g. ``com.cogos.bargein-producer.plist``) continue to work
until they migrate.

Detection method:
  SuperWhisper creates a timestamped directory in its recordings folder
  the instant recording begins (the dir is empty). When recording finishes,
  it writes output.wav and meta.json into that directory. We poll for new
  empty directories to detect start, and for the appearance of output.wav
  (or a matching row in SuperWhisper's SQLite DB) to detect end.

Signal file: /tmp/mod3-barge-in.json
  Start:  {"event": "user_speaking_start", "timestamp": "...", "source": "superwhisper"}
  End:    {"event": "user_speaking_end",   "timestamp": "...", "source": "superwhisper"}

Usage:
  python3 bargein-producer.py          # foreground (logs to stderr)
  python3 bargein-producer.py &        # background
  launchctl load com.cogos.bargein-producer.plist  # launchd

Environment variables:
  BARGEIN_SIGNAL      — override signal file path (default: /tmp/mod3-barge-in.json)
  BARGEIN_POLL_MS     — poll interval in ms (default: 150)
  SW_RECORDINGS_DIR   — override recordings path
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIGNAL_FILE = os.environ.get("BARGEIN_SIGNAL", "/tmp/mod3-barge-in.json")
POLL_INTERVAL = int(os.environ.get("BARGEIN_POLL_MS", "150")) / 1000.0

# SuperWhisper recordings directory
_default_rec_dir = os.path.expanduser("~/Documents/superwhisper/recordings")
RECORDINGS_DIR = os.environ.get("SW_RECORDINGS_DIR", _default_rec_dir)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [bargein] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bargein-producer")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State:
    """Tracks whether a recording is currently active."""

    def __init__(self):
        self.recording = False
        # The folder name (timestamp) of the active recording
        self.active_folder: str | None = None
        # Set of known-completed folder names (avoid re-triggering)
        self.known_folders: set[str] = set()

    def start(self, folder: str):
        if self.recording and self.active_folder == folder:
            return  # already tracking
        self.recording = True
        self.active_folder = folder
        _write_signal("user_speaking_start")
        log.info("Recording started  (folder=%s)", folder)

    def end(self):
        if not self.recording:
            return
        folder = self.active_folder
        self.recording = False
        if self.active_folder:
            self.known_folders.add(self.active_folder)
        self.active_folder = None
        _write_signal("user_speaking_end")
        log.info("Recording finished (folder=%s)", folder)


# ---------------------------------------------------------------------------
# Signal file
# ---------------------------------------------------------------------------


def _write_signal(event: str):
    """Atomically write the barge-in signal file."""
    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "superwhisper",
    }
    tmp = SIGNAL_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, SIGNAL_FILE)
    except OSError as e:
        log.error("Failed to write signal file: %s", e)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def _is_empty_dir(path: Path) -> bool:
    """True if path is a directory with no children (except . and ..)."""
    try:
        return path.is_dir() and not any(path.iterdir())
    except OSError:
        return False


def _has_output(path: Path) -> bool:
    """True if the recording directory has output.wav (recording complete)."""
    return (path / "output.wav").exists() or (path / "meta.json").exists()


# SuperWhisper SQLite DB — secondary signal for recording completion
_SW_DB = os.path.expanduser("~/Library/Application Support/SuperWhisper/database/superwhisper.sqlite")


def _is_in_db(folder_name: str) -> bool:
    """Check if SuperWhisper has finished processing this recording.

    SuperWhisper writes to its SQLite DB only AFTER transcription completes.
    This is the structural ground truth — if the folder is in the DB, the
    recording is definitely done, regardless of filesystem state.
    """
    try:
        import sqlite3

        conn = sqlite3.connect(f"file:{_SW_DB}?mode=ro", uri=True, timeout=1.0)
        cursor = conn.execute(
            "SELECT 1 FROM recording WHERE folderName = ? LIMIT 1",
            (folder_name,),
        )
        found = cursor.fetchone() is not None
        conn.close()
        return found
    except Exception:
        return False


_last_dir_mtime: float = 0.0


def _scan(state: State, rec_dir: Path):
    """One poll cycle: scan the recordings directory for state changes.

    Optimisation: stat() the recordings dir first (~1us). Only do a full
    iterdir() when the directory mtime has changed (new folder created)
    OR when we're actively tracking a recording (need to check for output.wav).
    """
    global _last_dir_mtime

    # Fast path: if we're tracking an active recording, check completion signals
    if state.recording and state.active_folder:
        active_path = rec_dir / state.active_folder
        # Primary: filesystem (output.wav or meta.json appeared)
        if _has_output(active_path):
            state.end()
            return
        # Secondary: SuperWhisper DB (transcription complete — structural ground truth)
        if _is_in_db(state.active_folder):
            log.info("DB confirms recording complete (filesystem missed it)")
            state.end()
            return
        # Folder deleted/cancelled
        if not active_path.exists():
            log.warning("Active recording folder disappeared, clearing state")
            state.end()
            return
        # Don't return here — fall through to check if directory changed,
        # so we can detect if a newer recording superseded this one

    # Check if directory changed since last scan
    try:
        dir_mtime = os.stat(rec_dir).st_mtime
    except OSError:
        return
    if dir_mtime == _last_dir_mtime:
        return  # nothing changed — skip expensive iterdir()
    _last_dir_mtime = dir_mtime

    # Directory changed — scan for new entries
    try:
        candidates: list[Path] = []
        for entry in rec_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit() and entry.name not in state.known_folders:
                candidates.append(entry)
    except OSError:
        return

    candidates.sort(key=lambda p: p.name, reverse=True)
    for entry in candidates[:5]:
        name = entry.name

        if _is_empty_dir(entry):
            # New empty directory = recording just started
            state.start(name)
            return

        # Non-empty, not tracked — it's a completed recording we haven't seen
        state.known_folders.add(name)


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------

_STALE_TIMEOUT = 150  # 2.5 minutes — if a recording folder stays empty this long,
#                       assume SuperWhisper cancelled/crashed/system slept and clear state.
#                       User has long recordings (60s+), so this is generous.


def _check_stale(state: State, rec_dir: Path):
    """Clear recording state if the active folder has been empty too long.

    Before clearing, performs a final DB check to confirm the recording
    isn't still being actively transcribed. This prevents the staleness
    timeout from invalidating a legitimately long recording session.
    """
    if not state.recording or not state.active_folder:
        return
    folder = rec_dir / state.active_folder
    try:
        # Use folder creation time as the start timestamp
        ctime = folder.stat().st_birthtime
    except (OSError, AttributeError):
        return
    if time.time() - ctime > _STALE_TIMEOUT:
        # Final gateway: check DB before declaring stale
        if _is_in_db(state.active_folder):
            log.info("Stale timeout hit but DB confirms completion — ending normally")
            state.end()
        elif _has_output(folder):
            log.info("Stale timeout hit but output files present — ending normally")
            state.end()
        else:
            # Neither DB nor filesystem confirm completion — truly stale
            log.warning(
                "Stale recording (>%ds), no DB entry, no output files — clearing as cancelled/crashed",
                _STALE_TIMEOUT,
            )
            state.end()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    rec_dir = Path(RECORDINGS_DIR)
    if not rec_dir.is_dir():
        log.error("Recordings directory not found: %s", rec_dir)
        log.error("Is SuperWhisper installed and configured?")
        sys.exit(1)

    state = State()

    # Pre-populate known folders so we don't false-trigger on startup.
    # Only treat an empty folder as "currently recording" if it was created
    # within the last 30 seconds — older empties are stale/cancelled.
    _STARTUP_FRESH_SECS = 30
    now = time.time()
    newest_empty: tuple[str, float] | None = None
    try:
        for entry in rec_dir.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                if _has_output(entry):
                    state.known_folders.add(entry.name)
                elif _is_empty_dir(entry):
                    try:
                        age = now - entry.stat().st_birthtime
                    except (OSError, AttributeError):
                        age = float("inf")
                    if age < _STARTUP_FRESH_SECS:
                        # Possibly active — track the newest one
                        if newest_empty is None or entry.name > newest_empty[0]:
                            newest_empty = (entry.name, age)
                    else:
                        # Stale empty folder — ignore it
                        state.known_folders.add(entry.name)
                        log.debug("Ignoring stale empty folder: %s (age=%.0fs)", entry.name, age)
    except OSError as e:
        log.warning("Error during startup scan: %s", e)

    if newest_empty:
        state.start(newest_empty[0])
        log.info("Detected in-progress recording on startup (age=%.1fs)", newest_empty[1])

    log.info(
        "Barge-in producer started (poll=%dms, signal=%s, recordings=%s, known=%d)",
        POLL_INTERVAL * 1000,
        SIGNAL_FILE,
        RECORDINGS_DIR,
        len(state.known_folders),
    )

    stale_counter = 0
    try:
        while True:
            _scan(state, rec_dir)

            # Check for stale recordings every ~2 seconds
            stale_counter += 1
            if stale_counter >= int(2.0 / POLL_INTERVAL):
                _check_stale(state, rec_dir)
                stale_counter = 0

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        log.info("Shutting down")
        # Clean up signal file on exit
        if state.recording:
            state.end()


if __name__ == "__main__":
    main()

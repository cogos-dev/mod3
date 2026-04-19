"""SuperWhisper barge-in provider.

Watches the SuperWhisper recordings directory and its SQLite DB for
recording start/end, emitting ``BargeinEvent``s through the registered
callback. This is the in-process replacement for the standalone
``integrations/bargein-producer.py`` script: same detection logic, but
instead of writing ``/tmp/mod3-barge-in.json`` it calls directly into
mod3's barge-in consumer.

Detection:
  * Start: a new empty timestamped folder appears under the recordings dir.
  * End (any of):
      - ``output.wav`` or ``meta.json`` appears in that folder, OR
      - a matching row appears in ``superwhisper.sqlite`` (structural ground
        truth — written only after transcription completes), OR
      - the folder disappears (cancellation), OR
      - the staleness timeout elapses without the above (crash / sleep).

Environment variables:
  SW_RECORDINGS_DIR   — override recordings path
  BARGEIN_POLL_MS     — poll interval in ms (default: 150)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .base import BargeinProvider

log = logging.getLogger("bargein.superwhisper")


class SuperWhisperProvider(BargeinProvider):
    """Barge-in provider backed by SuperWhisper's recordings folder + DB."""

    source = "superwhisper"

    # Default ~/Documents/superwhisper/recordings, overridable via env.
    _DEFAULT_REC_DIR = os.path.expanduser("~/Documents/superwhisper/recordings")
    # SuperWhisper SQLite DB — secondary "recording finished" signal.
    _SW_DB = os.path.expanduser("~/Library/Application Support/SuperWhisper/database/superwhisper.sqlite")
    # 2.5 minutes — recordings can legitimately run 60s+; be generous
    # before declaring a stuck folder stale.
    _STALE_TIMEOUT = 150
    _STARTUP_FRESH_SECS = 30

    def __init__(self, on_event, recordings_dir: str | None = None, poll_ms: int | None = None):
        super().__init__(on_event)
        self.recordings_dir = Path(recordings_dir or os.environ.get("SW_RECORDINGS_DIR", self._DEFAULT_REC_DIR))
        poll_ms = poll_ms if poll_ms is not None else int(os.environ.get("BARGEIN_POLL_MS", "150"))
        self._poll_interval = poll_ms / 1000.0

        # Mutable state (touched only from the provider thread)
        self._recording = False
        self._active_folder: str | None = None
        self._known_folders: set[str] = set()
        self._last_dir_mtime: float = 0.0

    # ------------------------------------------------------------------
    # State transitions (emit events through the callback)
    # ------------------------------------------------------------------

    def _start_recording(self, folder: str) -> None:
        if self._recording and self._active_folder == folder:
            return
        self._recording = True
        self._active_folder = folder
        log.info("Recording started (folder=%s)", folder)
        self._emit("user_speaking_start", {"folder": folder})

    def _end_recording(self, reason: str) -> None:
        if not self._recording:
            return
        folder = self._active_folder
        self._recording = False
        if folder:
            self._known_folders.add(folder)
        self._active_folder = None
        log.info("Recording finished (folder=%s, reason=%s)", folder, reason)
        self._emit("user_speaking_end", {"folder": folder, "reason": reason})

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_empty_dir(path: Path) -> bool:
        try:
            return path.is_dir() and not any(path.iterdir())
        except OSError:
            return False

    @staticmethod
    def _has_output(path: Path) -> bool:
        return (path / "output.wav").exists() or (path / "meta.json").exists()

    @classmethod
    def _is_in_db(cls, folder_name: str) -> bool:
        """True if SuperWhisper's DB has a ``recording`` row for this folder.

        SuperWhisper writes the row only after transcription completes, so a
        hit here is a definitive "recording is done" signal regardless of
        filesystem state.
        """
        try:
            import sqlite3

            conn = sqlite3.connect(f"file:{cls._SW_DB}?mode=ro", uri=True, timeout=1.0)
            cursor = conn.execute(
                "SELECT 1 FROM recording WHERE folderName = ? LIMIT 1",
                (folder_name,),
            )
            found = cursor.fetchone() is not None
            conn.close()
            return found
        except Exception:
            return False

    def _scan(self) -> None:
        """One poll cycle: detect state changes in the recordings dir."""
        rec_dir = self.recordings_dir

        # Fast path: if we're tracking an active recording, check completion signals
        if self._recording and self._active_folder:
            active_path = rec_dir / self._active_folder
            if self._has_output(active_path):
                self._end_recording(reason="output_files")
                return
            if self._is_in_db(self._active_folder):
                log.info("DB confirms recording complete (filesystem missed it)")
                self._end_recording(reason="db")
                return
            if not active_path.exists():
                log.warning("Active recording folder disappeared, clearing state")
                self._end_recording(reason="folder_gone")
                return
            # Fall through so we can detect a newer recording superseding this one

        # Stat-then-iterdir: skip the expensive scan if mtime is unchanged
        try:
            dir_mtime = os.stat(rec_dir).st_mtime
        except OSError:
            return
        if dir_mtime == self._last_dir_mtime:
            return
        self._last_dir_mtime = dir_mtime

        try:
            candidates: list[Path] = []
            for entry in rec_dir.iterdir():
                if entry.is_dir() and entry.name.isdigit() and entry.name not in self._known_folders:
                    candidates.append(entry)
        except OSError:
            return

        candidates.sort(key=lambda p: p.name, reverse=True)
        for entry in candidates[:5]:
            name = entry.name
            if self._is_empty_dir(entry):
                self._start_recording(name)
                return
            # Non-empty, previously unseen — completed recording we missed
            self._known_folders.add(name)

    def _check_stale(self) -> None:
        """Clear stuck recording state if the active folder has been empty too long.

        Before clearing, double-check the DB so legitimately long recordings
        aren't thrown away when they finally land.
        """
        if not self._recording or not self._active_folder:
            return
        folder = self.recordings_dir / self._active_folder
        try:
            ctime = folder.stat().st_birthtime
        except (OSError, AttributeError):
            return
        if time.time() - ctime <= self._STALE_TIMEOUT:
            return

        if self._is_in_db(self._active_folder):
            log.info("Stale timeout hit but DB confirms completion — ending normally")
            self._end_recording(reason="db_after_stale")
        elif self._has_output(folder):
            log.info("Stale timeout hit but output files present — ending normally")
            self._end_recording(reason="output_after_stale")
        else:
            log.warning(
                "Stale recording (>%ds), no DB entry, no output files — clearing as cancelled/crashed",
                self._STALE_TIMEOUT,
            )
            self._end_recording(reason="stale")

    # ------------------------------------------------------------------
    # Startup scan: handle recordings that existed before we started
    # ------------------------------------------------------------------

    def _startup_scan(self) -> None:
        now = time.time()
        newest_empty: tuple[str, float] | None = None
        try:
            for entry in self.recordings_dir.iterdir():
                if not (entry.is_dir() and entry.name.isdigit()):
                    continue
                if self._has_output(entry):
                    self._known_folders.add(entry.name)
                elif self._is_empty_dir(entry):
                    try:
                        age = now - entry.stat().st_birthtime
                    except (OSError, AttributeError):
                        age = float("inf")
                    if age < self._STARTUP_FRESH_SECS:
                        if newest_empty is None or entry.name > newest_empty[0]:
                            newest_empty = (entry.name, age)
                    else:
                        self._known_folders.add(entry.name)
        except OSError as e:
            log.warning("Startup scan error: %s", e)

        if newest_empty:
            log.info("Detected in-progress recording on startup (age=%.1fs)", newest_empty[1])
            self._start_recording(newest_empty[0])

    # ------------------------------------------------------------------
    # Provider contract
    # ------------------------------------------------------------------

    def _run(self) -> None:
        rec_dir = self.recordings_dir
        if not rec_dir.is_dir():
            log.warning(
                "SuperWhisper recordings directory not found: %s (provider inactive)",
                rec_dir,
            )
            return

        self._startup_scan()
        log.info(
            "SuperWhisper provider running (poll=%dms, recordings=%s, known=%d)",
            self._poll_interval * 1000,
            rec_dir,
            len(self._known_folders),
        )

        stale_every = max(1, int(2.0 / self._poll_interval))
        stale_counter = 0
        while not self._stop.is_set():
            try:
                self._scan()
                stale_counter += 1
                if stale_counter >= stale_every:
                    self._check_stale()
                    stale_counter = 0
            except Exception:
                log.exception("SuperWhisper poll cycle raised; continuing")
            # Use Event.wait for responsive shutdown
            if self._stop.wait(self._poll_interval):
                break

        if self._recording:
            # Emit a synthetic end so consumers don't stay in "speaking" forever
            self._end_recording(reason="shutdown")

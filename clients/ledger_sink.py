"""Ledger sink -- the durable half of the seat's mouths.

Every conversational utterance the seat makes through the mod3 channel client
(spoken via mod3_speak, posted via mod3_dashboard_post) is also written to a
durable conversation ledger, so the spoken half of a conversation stops being
lost on daemon restart. mod3's message store is RAM-only by design (see the
2026-07-23 seat/channel tomography, gap #6); this module is the additive fix
at the mouth, not a change to the daemon.

Mechanism -- receipts, never direct ledger writes:
    The ledger repo's conversations/ lane already has a multi-writer protocol:
    a client commits a receipt {"count": N, "turns": [...]} into
    conversations/inbox/, and the conversations-ingest workflow upserts each
    turn into ledger.json by id (idempotent) and settles the receipt in the
    same commit. Receipts are new files with unique names (ms timestamp +
    writer + entropy suffix), so writers never contend on CONTENT -- and
    local git-level contention (index.lock is fail-fast, not a queue) is
    serialized by an interprocess flock per repo (_repo_lock), so several
    channel-client processes and both of this process's mouths can sink
    concurrently without losing turns. This is why the sink uses receipts
    rather than appending to ledger.json directly, which WOULD race the
    seat and the ingest bot on content.

Identity -- declared, never inferred:
    Every turn this sink writes carries origin="seat" (the field the
    downstream wake-line watcher's self-echo suppression keys on, 2026-07-23
    fix) and a seat-* `from` (which the watcher's author fallback
    prefix-matches). Both ends, so a sunk turn can never wake the seat with
    its own voice. Trust remains commit identity, per conversations/README.md
    in the ledger repo -- the `from` field is a mouth label, not an
    authority claim.

Failure posture -- speech never blocks on the ledger:
    Callers fire this through asyncio.to_thread and drop the handle. Any
    failure here is logged and swallowed; a commit that fails to push stays
    local and rides out with the next successful push. The kill switch is
    MOD3_LEDGER_SINK=0.

Env:
    MOD3_LEDGER_SINK    "0" disables the sink entirely (default: enabled).
                        Falls back to MOD3_THESEUS_SINK if set, for
                        compatibility with earlier deployments.
    MOD3_LEDGER_REPO    path to the ledger repo (default:
                        $MYRGIC_REPOS_ROOT/ledger-repo, MYRGIC_REPOS_ROOT
                        defaulting to ~/workspaces/myrgic). Falls back to
                        MOD3_THESEUS_REPO if set. There is no useful literal
                        default here -- point this at your own ledger repo's
                        clone; when the target directory doesn't exist,
                        enabled() safely reports the sink as off.
    MOD3_LEDGER_THREAD  default thread for sunk turns (default: "voice" for
                        speech, "dashboard" for dashboard posts -- see
                        callers). Falls back to MOD3_THESEUS_THREAD if set.

CLI (for verification and manual sinking):
    python3 clients/ledger_sink.py --text "..." [--thread voice]
                                   [--from seat-root-voice] [--no-push]
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import time
import uuid

logger = logging.getLogger("mod3.ledger_sink")

_GIT_TIMEOUT = 30  # seconds per git subprocess
_PUSH_RETRIES = 2  # pull --rebase && push attempts after a rejected push
_LOCK_TIMEOUT = 45  # seconds to wait for the interprocess sink lock
_LOCK_DIR = pathlib.Path.home() / ".mod3"


def _env(new_name: str, old_name: str, default: str | None = None) -> str | None:
    """Read new_name, falling back to the pre-rename old_name, then default.

    Keeps early deployments that already set MOD3_THESEUS_* working after
    the MOD3_LEDGER_* rename.
    """
    if new_name in os.environ:
        return os.environ[new_name]
    if old_name in os.environ:
        return os.environ[old_name]
    return default


@contextlib.contextmanager
def _repo_lock(repo: pathlib.Path):
    """Interprocess lock serializing the whole git sequence for one repo.

    Several channel-client processes (one per Claude session) plus this
    process's own two mouths can sink into the same clone concurrently.
    git's index.lock is fail-fast, not a queue -- an unserialized racer
    loses its turn outright. One flock per repo path makes every sink's
    write→add→commit→push (including any pull --rebase) atomic with
    respect to its siblings, which also guarantees a `rebase --abort` in
    the push-retry path can only ever abort a rebase this call started.
    Raises TimeoutError if the lock isn't acquired in _LOCK_TIMEOUT.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(repo.resolve()).encode()).hexdigest()[:16]
    lock_path = _LOCK_DIR / f"ledger-sink-{digest}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"sink lock busy for {_LOCK_TIMEOUT}s: {lock_path}")
                time.sleep(0.2)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _repo_path() -> pathlib.Path:
    root = os.environ.get("MYRGIC_REPOS_ROOT", os.path.expanduser("~/workspaces/myrgic"))
    return pathlib.Path(_env("MOD3_LEDGER_REPO", "MOD3_THESEUS_REPO", os.path.join(root, "ledger-repo")))


def enabled() -> bool:
    """True when the sink should run: not switched off, and the repo exists."""
    if _env("MOD3_LEDGER_SINK", "MOD3_THESEUS_SINK", "1") == "0":
        return False
    repo = _repo_path()
    return (repo / "conversations" / "inbox").is_dir()


def _git(repo: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )


def sink_turn(
    text: str,
    thread: str,
    from_id: str,
    push: bool = True,
) -> dict:
    """Write one turn as a committed receipt in the ledger repo's inbox.

    Returns {"ok": True, "receipt": <relpath>, "commit": <sha>, "pushed": bool}
    or {"ok": False, "error": "..."}. Never raises past this function -- the
    caller is a fire-and-forget thread and has nowhere to put an exception.
    """
    try:
        if not text.strip():
            return {"ok": False, "error": "empty text"}
        if not from_id.startswith("seat"):
            # The wake line's self-echo suppression prefix-matches seat-*;
            # a non-seat from would re-echo the seat's own voice back at it.
            # Enforced here, not by caller convention.
            return {"ok": False, "error": f"from_id must start with 'seat': {from_id!r}"}
        repo = _repo_path()
        inbox = repo / "conversations" / "inbox"
        if not inbox.is_dir():
            return {"ok": False, "error": f"no inbox at {inbox}"}

        with _repo_lock(repo):
            # id shape stays <ms>-<writer> for readability, with a short
            # entropy suffix so two processes in the same millisecond can
            # never mint the same id (the ingest upserts by id -- a collision
            # would silently merge two distinct utterances).
            ms = int(time.time() * 1000)
            turn_id = f"{ms}-{from_id}-{uuid.uuid4().hex[:6]}"
            receipt = {
                "count": 1,
                "turns": [
                    {
                        "id": turn_id,
                        "thread": thread or "voice",
                        "from": from_id,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "text": text,
                        # Declared by the writer, never inferred by a reader --
                        # the wake line's self-echo suppression keys on this.
                        "origin": "seat",
                    }
                ],
            }
            rel = f"conversations/inbox/{turn_id}.json"
            path = repo / rel
            path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

            # Targeted add + commit: only the receipt, never the rest of the
            # tree.
            r = _git(repo, "add", "--", rel)
            if r.returncode != 0:
                return {"ok": False, "error": f"git add: {r.stderr.strip()}"}
            r = _git(repo, "commit", "-m", f"chat receipt from {from_id}", "--", rel)
            if r.returncode != 0:
                return {
                    "ok": False,
                    "error": f"git commit: {r.stderr.strip() or r.stdout.strip()}",
                }
            sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()

            pushed = False
            if push:
                for attempt in range(_PUSH_RETRIES + 1):
                    r = _git(repo, "push", "origin", "main")
                    if r.returncode == 0:
                        pushed = True
                        break
                    if attempt < _PUSH_RETRIES:
                        # Receipts are new files and never conflict; a
                        # rejected push just means the remote moved (ingest
                        # bot, another mouth, another device). Rebase and
                        # retry. Under _repo_lock, any in-progress rebase is
                        # OURS, so the abort below can never tear down a
                        # sibling's.
                        rb = _git(repo, "pull", "--rebase", "origin", "main")
                        if rb.returncode != 0:
                            _git(repo, "rebase", "--abort")
                            logger.warning(
                                "ledger_sink: rebase failed (%s); receipt %s stays local",
                                rb.stderr.strip(),
                                rel,
                            )
                            break
                if not pushed:
                    logger.warning(
                        "ledger_sink: push failed; receipt %s committed locally, rides out with the next push",
                        rel,
                    )

        return {"ok": True, "receipt": rel, "commit": sha, "pushed": pushed}
    except Exception as exc:  # noqa: BLE001 -- fire-and-forget boundary
        logger.warning("ledger_sink: %s", exc)
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", required=True)
    parser.add_argument("--thread", default=_env("MOD3_LEDGER_THREAD", "MOD3_THESEUS_THREAD", "voice"))
    parser.add_argument("--from", dest="from_id", default="seat-root-voice")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if not enabled():
        print("sink disabled (MOD3_LEDGER_SINK=0 or repo missing)")
        return 1
    result = sink_turn(args.text, args.thread, args.from_id, push=not args.no_push)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

"""THESEUS ledger sink — the durable half of the seat's mouths.

Every conversational utterance the seat makes through the mod3 channel client
(spoken via mod3_speak, posted via mod3_dashboard_post) is also written to the
THESEUS conversation ledger, so the spoken half of a conversation stops being
lost on daemon restart. mod3's message store is RAM-only by design (see the
2026-07-23 seat/channel tomography, gap #6); this module is the additive fix
at the mouth, not a change to the daemon.

Mechanism — receipts, never direct ledger writes:
    The book repo's conversations/ lane already has a multi-writer protocol:
    a client commits a receipt {"count": N, "turns": [...]} into
    conversations/inbox/, and the conversations-ingest workflow upserts each
    turn into ledger.json by id (idempotent) and settles the receipt in the
    same commit. Receipts are new files with unique names, so concurrent
    writers never conflict — this sink uses that path rather than appending
    to ledger.json directly, which WOULD race the seat and the ingest bot.

Identity — declared, never inferred:
    Every turn this sink writes carries origin="seat" (the field the THESEUS
    wake-line watcher's self-echo suppression keys on, 2026-07-23 fix) and a
    seat-* `from` (which the watcher's author fallback prefix-matches). Both
    ends, so a sunk turn can never wake the seat with its own voice. Trust
    remains commit identity, per conversations/README.md — the `from` field
    is a mouth label, not an authority claim.

Failure posture — speech never blocks on the ledger:
    Callers fire this through asyncio.to_thread and drop the handle. Any
    failure here is logged and swallowed; a commit that fails to push stays
    local and rides out with the next successful push. The kill switch is
    MOD3_THESEUS_SINK=0.

Env:
    MOD3_THESEUS_SINK    "0" disables the sink entirely (default: enabled)
    MOD3_THESEUS_REPO    book repo path (default:
                         $MYRGIC_REPOS_ROOT/thinking-through-distinction-internal,
                         MYRGIC_REPOS_ROOT defaulting to ~/workspaces/myrgic)
    MOD3_THESEUS_THREAD  default thread for sunk turns (default: "voice" for
                         speech, "dashboard" for dashboard posts — see callers)

CLI (for verification and manual sinking):
    python3 clients/theseus_sink.py --text "..." [--thread voice]
                                    [--from seat-root-voice] [--no-push]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import subprocess
import time

logger = logging.getLogger("mod3.theseus_sink")

_GIT_TIMEOUT = 30  # seconds per git subprocess
_PUSH_RETRIES = 2  # pull --rebase && push attempts after a rejected push


def _repo_path() -> pathlib.Path:
    root = os.environ.get(
        "MYRGIC_REPOS_ROOT", os.path.expanduser("~/workspaces/myrgic")
    )
    return pathlib.Path(
        os.environ.get(
            "MOD3_THESEUS_REPO",
            os.path.join(root, "thinking-through-distinction-internal"),
        )
    )


def enabled() -> bool:
    """True when the sink should run: not switched off, and the repo exists."""
    if os.environ.get("MOD3_THESEUS_SINK", "1") == "0":
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
    """Write one turn as a committed receipt in the book repo's inbox.

    Returns {"ok": True, "receipt": <relpath>, "commit": <sha>, "pushed": bool}
    or {"ok": False, "error": "..."}. Never raises past this function — the
    caller is a fire-and-forget thread and has nowhere to put an exception.
    """
    try:
        if not text.strip():
            return {"ok": False, "error": "empty text"}
        repo = _repo_path()
        inbox = repo / "conversations" / "inbox"
        if not inbox.is_dir():
            return {"ok": False, "error": f"no inbox at {inbox}"}

        ms = int(time.time() * 1000)
        turn_id = f"{ms}-{from_id}"
        receipt = {
            "count": 1,
            "turns": [
                {
                    "id": turn_id,
                    "thread": thread or "voice",
                    "from": from_id,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "text": text,
                    # Declared by the writer, never inferred by a reader —
                    # the wake line's self-echo suppression keys on this.
                    "origin": "seat",
                }
            ],
        }
        rel = f"conversations/inbox/{turn_id}.json"
        path = repo / rel
        # Unique-name new file: two sinks in the same millisecond from the
        # same from_id would collide, so bump until free (bounded).
        bump = 0
        while path.exists() and bump < 100:
            bump += 1
            turn_id = f"{ms + bump}-{from_id}"
            receipt["turns"][0]["id"] = turn_id
            rel = f"conversations/inbox/{turn_id}.json"
            path = repo / rel
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

        # Targeted add + commit: only the receipt, never the rest of the tree.
        r = _git(repo, "add", "--", rel)
        if r.returncode != 0:
            return {"ok": False, "error": f"git add: {r.stderr.strip()}"}
        r = _git(repo, "commit", "-m", f"chat receipt from {from_id}", "--", rel)
        if r.returncode != 0:
            return {"ok": False, "error": f"git commit: {r.stderr.strip() or r.stdout.strip()}"}
        sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip()

        pushed = False
        if push:
            for attempt in range(_PUSH_RETRIES + 1):
                r = _git(repo, "push", "origin", "main")
                if r.returncode == 0:
                    pushed = True
                    break
                if attempt < _PUSH_RETRIES:
                    # Receipts are new files and never conflict; a rejected
                    # push just means the remote moved (ingest bot, the seat,
                    # his phone). Rebase and retry.
                    rb = _git(repo, "pull", "--rebase", "origin", "main")
                    if rb.returncode != 0:
                        _git(repo, "rebase", "--abort")
                        logger.warning(
                            "theseus_sink: rebase failed (%s); receipt %s stays local",
                            rb.stderr.strip(),
                            rel,
                        )
                        break
            if not pushed:
                logger.warning(
                    "theseus_sink: push failed; receipt %s committed locally, "
                    "rides out with the next push",
                    rel,
                )

        return {"ok": True, "receipt": rel, "commit": sha, "pushed": pushed}
    except Exception as exc:  # noqa: BLE001 — fire-and-forget boundary
        logger.warning("theseus_sink: %s", exc)
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", required=True)
    parser.add_argument("--thread", default=os.environ.get("MOD3_THESEUS_THREAD", "voice"))
    parser.add_argument("--from", dest="from_id", default="seat-root-voice")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if not enabled():
        print("sink disabled (MOD3_THESEUS_SINK=0 or repo missing)")
        return 1
    result = sink_turn(args.text, args.thread, args.from_id, push=not args.no_push)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

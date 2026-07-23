"""Tests for clients/theseus_sink.py — the THESEUS ledger sink.

Uses a real bare-origin + clone git pair per test, so the commit/push/rebase
path is exercised against actual git, not mocks. The receipt protocol under
test is the book repo's conversations/ lane: new-file receipts in
conversations/inbox/, one commit each, pushed to origin.
"""

import json
import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "clients"))

import theseus_sink  # noqa: E402


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )


@pytest.fixture()
def repo_pair(tmp_path, monkeypatch):
    """A bare origin and a clone shaped like the book repo's conversations lane."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(origin)],
        capture_output=True,
        check=True,
    )
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", str(origin), str(clone)], capture_output=True, check=True
    )
    _git(clone, "config", "user.email", "test@test")
    _git(clone, "config", "user.name", "test")
    inbox = clone / "conversations" / "inbox"
    inbox.mkdir(parents=True)
    (inbox / ".gitkeep").write_text("")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-m", "init")
    _git(clone, "push", "origin", "main")
    monkeypatch.setenv("MOD3_THESEUS_REPO", str(clone))
    monkeypatch.delenv("MOD3_THESEUS_SINK", raising=False)
    return origin, clone


def test_disabled_by_env(repo_pair, monkeypatch):
    monkeypatch.setenv("MOD3_THESEUS_SINK", "0")
    assert not theseus_sink.enabled()


def test_disabled_when_repo_missing(monkeypatch):
    monkeypatch.setenv("MOD3_THESEUS_REPO", "/nonexistent/nowhere")
    assert not theseus_sink.enabled()


def test_sink_writes_receipt_and_pushes(repo_pair):
    origin, clone = repo_pair
    assert theseus_sink.enabled()
    result = theseus_sink.sink_turn("hello from the voice", "voice", "seat-root-voice")
    assert result["ok"], result
    assert result["pushed"]

    receipt_path = clone / result["receipt"]
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["count"] == 1
    turn = receipt["turns"][0]
    assert turn["text"] == "hello from the voice"
    assert turn["thread"] == "voice"
    assert turn["from"] == "seat-root-voice"
    # The two self-echo guards, both declared by the writer:
    assert turn["origin"] == "seat"
    assert turn["from"].startswith("seat-")
    # id convention: <ms>-<from>
    assert turn["id"].endswith("-seat-root-voice")

    # Pushed: origin's main carries the receipt commit.
    log = subprocess.run(
        ["git", "-C", str(origin), "log", "--oneline", "-1", "main"],
        capture_output=True,
        text=True,
    ).stdout
    assert "chat receipt from seat-root-voice" in log


def test_sink_only_commits_the_receipt(repo_pair):
    """A dirty tree must not be swept into the receipt commit."""
    _, clone = repo_pair
    stray = clone / "stray.txt"
    stray.write_text("uncommitted work")
    result = theseus_sink.sink_turn("clean commit", "voice", "seat-root-voice")
    assert result["ok"], result
    shown = _git(clone, "show", "--stat", "--name-only", "HEAD").stdout
    assert "stray.txt" not in shown
    assert result["receipt"] in shown
    assert stray.exists()  # untouched


def test_sink_rebases_past_remote_movement(repo_pair, tmp_path):
    """A rejected push (remote moved) rebases and succeeds — the ingest-bot race."""
    origin, clone = repo_pair
    other = tmp_path / "other"
    subprocess.run(
        ["git", "clone", str(origin), str(other)], capture_output=True, check=True
    )
    _git(other, "config", "user.email", "bot@test")
    _git(other, "config", "user.name", "bot")
    (other / "settled.txt").write_text("bot moved main")
    _git(other, "add", "-A")
    _git(other, "commit", "-m", "bot: settle")
    _git(other, "push", "origin", "main")

    result = theseus_sink.sink_turn("raced the bot", "voice", "seat-root-voice")
    assert result["ok"], result
    assert result["pushed"], result
    log = subprocess.run(
        ["git", "-C", str(origin), "log", "--oneline", "main"],
        capture_output=True,
        text=True,
    ).stdout
    assert "chat receipt from seat-root-voice" in log
    assert "bot: settle" in log


def test_push_failure_stays_local(repo_pair):
    """Unreachable origin: the receipt commit lands locally, ok=True, pushed=False."""
    _, clone = repo_pair
    _git(clone, "remote", "set-url", "origin", "/nonexistent/origin.git")
    result = theseus_sink.sink_turn("stranded", "voice", "seat-root-voice")
    assert result["ok"], result
    assert not result["pushed"]
    log = _git(clone, "log", "--oneline", "-1").stdout
    assert "chat receipt from seat-root-voice" in log


def test_empty_text_refused(repo_pair):
    result = theseus_sink.sink_turn("   ", "voice", "seat-root-voice")
    assert not result["ok"]


def test_same_millisecond_ids_do_not_collide(repo_pair, monkeypatch):
    """Two sinks in the same millisecond get distinct receipt files."""
    monkeypatch.setattr(theseus_sink.time, "time", lambda: 1700000000.0)
    r1 = theseus_sink.sink_turn("first", "voice", "seat-root-voice", push=False)
    r2 = theseus_sink.sink_turn("second", "voice", "seat-root-voice", push=False)
    assert r1["ok"] and r2["ok"]
    assert r1["receipt"] != r2["receipt"]


def test_fire_and_forget_sync_path(repo_pair):
    """channel_client's no-running-loop fallback calls sink_turn synchronously."""
    import channel_client

    channel_client._sink_fire_and_forget("sync path", "voice", "seat-root-voice")
    _, clone = repo_pair
    log = _git(clone, "log", "--oneline", "-1").stdout
    assert "chat receipt from seat-root-voice" in log

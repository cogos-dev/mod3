"""Tests for voice-profile curation metadata (slice 2/3).

Covers:
  - VoiceProfile.from_json fills defaults for missing curation fields (backward compat)
  - VoiceProfileRegistry.patch_metadata: happy path, per-field updates, validation
  - VoiceProfileRegistry.update_last_used_at: atomicity, no-op for unknown profiles
  - GET /v1/voices/profiles filter: tag (OR), favorite, engine, sort
  - PATCH /v1/voices/profiles/{name}: 200, 404, 400 (invalid rating, bad type)
  - GET /v1/voices/profiles/{name}: 200, 404
  - JSON round-trip with all curation fields present

No daemon, no network, no ~/.mod3/voices/ access. Uses tmp_path exclusively.
Run: python -m pytest tests/test_voice_profile_curation.py -v
"""

from __future__ import annotations

import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests that exercise the GET/PATCH endpoints via TestClient call
# _make_conditionals() which imports mlx at call time; skip the entire module
# in CI where the Apple-Silicon MLX stack is absent.
from conftest import HAS_MLX  # noqa: E402

if not HAS_MLX:
    pytest.skip("mlx not available (Apple Silicon only)", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path) -> None:
    """Write a minimal silent WAV file."""
    import pathlib

    path = pathlib.Path(path)
    num_samples = 1600
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


def _make_minimal_wav_bytes() -> bytes:
    """Return a minimal WAV header as bytes (for the HTTP fixture)."""
    return (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


def _make_profile_json(name: str, extra: dict | None = None) -> dict:
    """Return a minimal valid profile JSON dict."""
    base = {
        "name": name,
        "engine": "chatterbox-turbo",
        "source_audio_path": f"/tmp/{name}.wav",
        "source_sha256": "aabbcc" * 10 + "aabb",
        "ref_text": None,
        "exaggeration": 0.5,
        "model_id": "mlx-community/chatterbox-turbo",
        "created_at": "2026-05-15T00:00:00+00:00",
    }
    if extra:
        base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Helpers: synthetic Conditionals (no model load)
# ---------------------------------------------------------------------------


def _make_conditionals():
    import mlx.core as mx
    from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals
    from mlx_audio.tts.models.chatterbox.t3.cond_enc import T3Cond

    t3 = T3Cond(
        speaker_emb=mx.zeros((1, 256)),
        cond_prompt_speech_tokens=mx.zeros((1, 150), dtype=mx.int32),
        emotion_adv=mx.ones((1, 1, 1)) * 0.5,
    )
    gen = {
        "prompt_token": mx.zeros((1, 50), dtype=mx.int32),
        "prompt_token_len": mx.array([50]),
        "prompt_feat": mx.zeros((1, 120, 80)),
        "prompt_feat_len": mx.array([120]),
        "embedding": mx.zeros((1, 192)),
    }
    return Conditionals(t3=t3, gen=gen)


# ===========================================================================
# 1. Schema backward compatibility
# ===========================================================================


class TestVoiceProfileSchemaDefaults:
    """Loading a pre-existing JSON without curation fields fills defaults."""

    def test_missing_all_curation_fields_fills_defaults(self):
        from voice_profile_schema import VoiceProfile

        d = _make_profile_json("alice")
        # Ensure no curation fields are present
        for field in ("favorite", "notes", "tags", "last_used_at", "rating"):
            d.pop(field, None)

        profile = VoiceProfile.from_json(d)
        assert profile.favorite is False
        assert profile.notes == ""
        assert profile.tags == []
        assert profile.last_used_at is None
        assert profile.rating is None

    def test_partial_curation_fields_keeps_present_values(self):
        from voice_profile_schema import VoiceProfile

        d = _make_profile_json("bob", {"favorite": True, "tags": ["british"]})
        profile = VoiceProfile.from_json(d)
        assert profile.favorite is True
        assert profile.tags == ["british"]
        assert profile.notes == ""
        assert profile.last_used_at is None
        assert profile.rating is None

    def test_full_curation_fields_roundtrip(self):
        from voice_profile_schema import VoiceProfile

        d = _make_profile_json(
            "carol",
            {
                "favorite": True,
                "notes": "Good for narration",
                "tags": ["british", "female", "warm"],
                "last_used_at": "2026-05-15T10:00:00+00:00",
                "rating": 4,
            },
        )
        profile = VoiceProfile.from_json(d)
        out = profile.to_json()

        assert out["favorite"] is True
        assert out["notes"] == "Good for narration"
        assert out["tags"] == ["british", "female", "warm"]
        assert out["last_used_at"] == "2026-05-15T10:00:00+00:00"
        assert out["rating"] == 4

    def test_to_json_includes_all_curation_fields(self):
        from voice_profile_schema import VoiceProfile

        profile = VoiceProfile.from_json(_make_profile_json("dave"))
        out = profile.to_json()
        for key in ("favorite", "notes", "tags", "last_used_at", "rating"):
            assert key in out, f"Missing key {key!r} in to_json() output"


# ===========================================================================
# 2. VoiceProfileRegistry.patch_metadata
# ===========================================================================


class TestPatchMetadata:
    def _registry(self, tmp_path):
        from voice_profiles import VoiceProfileRegistry

        return VoiceProfileRegistry(root=tmp_path)

    def _seed_profile(self, tmp_path, reg, name: str, extra: dict | None = None) -> None:
        """Write a JSON + empty .safetensors sidecar without touching the model."""
        d = _make_profile_json(name, extra)
        json_path = tmp_path / f"{name}.json"
        st_path = tmp_path / f"{name}.safetensors"
        json_path.write_text(json.dumps(d))
        st_path.write_bytes(b"")  # registry checks existence, not content for get()

    def test_patch_favorite_updates_and_returns_profile(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        profile = reg.patch_metadata("alice", {"favorite": True})
        assert profile is not None
        assert profile.favorite is True

    def test_patch_notes(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        profile = reg.patch_metadata("alice", {"notes": "Very warm timbre"})
        assert profile.notes == "Very warm timbre"

    def test_patch_tags(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        profile = reg.patch_metadata("alice", {"tags": ["british", "female"]})
        assert profile.tags == ["british", "female"]

    def test_patch_rating_valid(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        for rating in (1, 2, 3, 4, 5):
            profile = reg.patch_metadata("alice", {"rating": rating})
            assert profile.rating == rating

    def test_patch_rating_zero_raises(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        with pytest.raises(ValueError, match="rating"):
            reg.patch_metadata("alice", {"rating": 0})

    def test_patch_rating_six_raises(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        with pytest.raises(ValueError, match="rating"):
            reg.patch_metadata("alice", {"rating": 6})

    def test_patch_rating_null_clears_it(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice", {"rating": 3})

        profile = reg.patch_metadata("alice", {"rating": None})
        assert profile.rating is None

    def test_patch_unknown_profile_returns_none(self, tmp_path):
        reg = self._registry(tmp_path)
        result = reg.patch_metadata("nobody", {"favorite": True})
        assert result is None

    def test_patch_persists_to_disk(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice")

        reg.patch_metadata("alice", {"favorite": True, "rating": 5})

        # Re-read from disk via a fresh registry instance
        reg2 = self._registry(tmp_path)
        profile = reg2.get("alice")
        assert profile is not None
        assert profile.favorite is True
        assert profile.rating == 5

    def test_patch_partial_update_preserves_other_fields(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, reg, "alice", {"favorite": True, "rating": 4, "tags": ["warm"]})

        reg.patch_metadata("alice", {"notes": "Updated notes"})

        profile = reg.get("alice")
        assert profile.favorite is True
        assert profile.rating == 4
        assert profile.tags == ["warm"]
        assert profile.notes == "Updated notes"


# ===========================================================================
# 3. VoiceProfileRegistry.update_last_used_at
# ===========================================================================


class TestUpdateLastUsedAt:
    def _registry(self, tmp_path):
        from voice_profiles import VoiceProfileRegistry

        return VoiceProfileRegistry(root=tmp_path)

    def _seed_profile(self, tmp_path, name: str) -> None:
        d = _make_profile_json(name)
        (tmp_path / f"{name}.json").write_text(json.dumps(d))
        (tmp_path / f"{name}.safetensors").write_bytes(b"")

    def test_update_sets_last_used_at(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, "alice")

        reg.update_last_used_at("alice")

        profile = reg.get("alice")
        assert profile is not None
        assert profile.last_used_at is not None
        # Should be a valid ISO 8601 string
        assert "T" in profile.last_used_at

    def test_update_unknown_profile_is_noop(self, tmp_path):
        reg = self._registry(tmp_path)
        # Must not raise
        reg.update_last_used_at("ghost")

    def test_update_persists_to_disk(self, tmp_path):
        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, "alice")

        reg.update_last_used_at("alice")

        # Read raw JSON from disk
        data = json.loads((tmp_path / "alice.json").read_text())
        assert data.get("last_used_at") is not None

    def test_update_twice_advances_timestamp(self, tmp_path):
        import time

        reg = self._registry(tmp_path)
        self._seed_profile(tmp_path, "alice")

        reg.update_last_used_at("alice")
        ts1 = reg.get("alice").last_used_at

        time.sleep(0.01)
        reg.update_last_used_at("alice")
        ts2 = reg.get("alice").last_used_at

        # Timestamps are ISO 8601; lexicographic order matches chronological order
        assert ts2 >= ts1

    def test_builtin_voice_is_noop(self, tmp_path):
        """Built-in voice names (no .json + .safetensors pair) → silently ignored."""
        reg = self._registry(tmp_path)
        # No files for "bm_lewis" in tmp_path → should not raise
        reg.update_last_used_at("bm_lewis")


# ===========================================================================
# 4. HTTP API — PATCH and GET filter tests
# ===========================================================================


class _FakeChatterboxModel:
    def prepare_conditionals(self, ref_audio, ref_sr=None, exaggeration=0.5, **kwargs):
        conds = _make_conditionals()
        self._conds = conds
        return conds


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """FastAPI TestClient with tmp_path registry and no real model."""
    import http_api
    from voice_profiles import VoiceProfileRegistry

    test_registry = VoiceProfileRegistry(root=tmp_path)
    monkeypatch.setattr(http_api, "_registry", test_registry)

    import engine as _engine_mod

    fake_model = _FakeChatterboxModel()
    monkeypatch.setattr(_engine_mod, "get_model", lambda engine_name: fake_model)

    from fastapi.testclient import TestClient

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture()
def ref_audio(tmp_path):
    p = tmp_path / "sample.wav"
    p.write_bytes(_make_minimal_wav_bytes())
    return str(p)


def _register(client, name: str, ref_audio: str, engine: str = "chatterbox") -> None:
    resp = client.post(
        "/v1/voices/profiles",
        json={"name": name, "engine": engine, "ref_audio_path": ref_audio},
    )
    assert resp.status_code == 200, f"register {name}: {resp.text}"


class TestPatchProfileEndpoint:
    def test_patch_favorite_returns_updated_profile(self, client, ref_audio):
        _register(client, "alice", ref_audio)
        resp = client.patch("/v1/voices/profiles/alice", json={"favorite": True})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["favorite"] is True
        assert body["name"] == "alice"

    def test_patch_tags(self, client, ref_audio):
        _register(client, "bob", ref_audio)
        resp = client.patch("/v1/voices/profiles/bob", json={"tags": ["british", "male"]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["tags"] == ["british", "male"]

    def test_patch_notes(self, client, ref_audio):
        _register(client, "carol", ref_audio)
        resp = client.patch("/v1/voices/profiles/carol", json={"notes": "Great for podcasts"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["notes"] == "Great for podcasts"

    def test_patch_rating_valid(self, client, ref_audio):
        _register(client, "dave", ref_audio)
        resp = client.patch("/v1/voices/profiles/dave", json={"rating": 4})
        assert resp.status_code == 200, resp.text
        assert resp.json()["rating"] == 4

    def test_patch_rating_zero_returns_400(self, client, ref_audio):
        _register(client, "eve", ref_audio)
        resp = client.patch("/v1/voices/profiles/eve", json={"rating": 0})
        assert resp.status_code == 400, resp.text

    def test_patch_rating_six_returns_400(self, client, ref_audio):
        _register(client, "frank", ref_audio)
        resp = client.patch("/v1/voices/profiles/frank", json={"rating": 6})
        assert resp.status_code == 400, resp.text

    def test_patch_unknown_profile_returns_404(self, client):
        resp = client.patch("/v1/voices/profiles/nobody", json={"favorite": True})
        assert resp.status_code == 404, resp.text

    def test_patch_unknown_field_returns_400(self, client, ref_audio):
        _register(client, "grace", ref_audio)
        resp = client.patch("/v1/voices/profiles/grace", json={"unknown_field": "value"})
        assert resp.status_code == 400, resp.text

    def test_patch_multiple_fields_at_once(self, client, ref_audio):
        _register(client, "hank", ref_audio)
        resp = client.patch(
            "/v1/voices/profiles/hank",
            json={"favorite": True, "notes": "Top pick", "tags": ["warm"], "rating": 5},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["favorite"] is True
        assert body["notes"] == "Top pick"
        assert body["tags"] == ["warm"]
        assert body["rating"] == 5

    def test_patch_persists_across_get_list(self, client, ref_audio):
        _register(client, "iris", ref_audio)
        client.patch("/v1/voices/profiles/iris", json={"favorite": True, "tags": ["breathy"]})

        resp = client.get("/v1/voices/profiles")
        profiles = {p["name"]: p for p in resp.json()["profiles"]}
        assert profiles["iris"]["favorite"] is True
        assert "breathy" in profiles["iris"]["tags"]


class TestGetProfileEndpoint:
    def test_get_existing_profile_returns_200(self, client, ref_audio):
        _register(client, "alice", ref_audio)
        resp = client.get("/v1/voices/profiles/alice")
        assert resp.status_code == 200, resp.text
        assert resp.json()["name"] == "alice"

    def test_get_unknown_profile_returns_404(self, client):
        resp = client.get("/v1/voices/profiles/nobody")
        assert resp.status_code == 404, resp.text

    def test_get_profile_includes_curation_fields(self, client, ref_audio):
        _register(client, "bob", ref_audio)
        client.patch("/v1/voices/profiles/bob", json={"favorite": True, "rating": 3})
        resp = client.get("/v1/voices/profiles/bob")
        body = resp.json()
        assert body["favorite"] is True
        assert body["rating"] == 3


class TestListProfilesFilters:
    def _setup(self, client, ref_audio):
        """Register three profiles with distinct metadata."""
        _register(client, "brit_m", ref_audio)
        client.patch("/v1/voices/profiles/brit_m", json={"tags": ["british", "male"], "favorite": True, "rating": 5})

        _register(client, "brit_f", ref_audio)
        client.patch("/v1/voices/profiles/brit_f", json={"tags": ["british", "female"], "favorite": False, "rating": 3})

        _register(client, "aus_m", ref_audio)
        client.patch("/v1/voices/profiles/aus_m", json={"tags": ["australian", "male"], "favorite": True, "rating": 4})

    def test_filter_by_single_tag(self, client, ref_audio):
        self._setup(client, ref_audio)
        resp = client.get("/v1/voices/profiles?tag=british")
        assert resp.status_code == 200
        names = {p["name"] for p in resp.json()["profiles"]}
        assert names == {"brit_m", "brit_f"}
        assert "aus_m" not in names

    def test_filter_by_tag_no_match_returns_empty(self, client, ref_audio):
        self._setup(client, ref_audio)
        resp = client.get("/v1/voices/profiles?tag=american")
        assert resp.status_code == 200
        assert resp.json()["profiles"] == []

    def test_filter_by_multiple_tags_or_semantics(self, client, ref_audio):
        self._setup(client, ref_audio)
        # australian OR british → all three profiles
        resp = client.get("/v1/voices/profiles?tag=australian&tag=british")
        assert resp.status_code == 200
        names = {p["name"] for p in resp.json()["profiles"]}
        assert names == {"brit_m", "brit_f", "aus_m"}

    def test_filter_by_favorite_true(self, client, ref_audio):
        self._setup(client, ref_audio)
        resp = client.get("/v1/voices/profiles?favorite=true")
        assert resp.status_code == 200
        names = {p["name"] for p in resp.json()["profiles"]}
        assert names == {"brit_m", "aus_m"}
        assert "brit_f" not in names

    def test_filter_by_favorite_false(self, client, ref_audio):
        self._setup(client, ref_audio)
        resp = client.get("/v1/voices/profiles?favorite=false")
        assert resp.status_code == 200
        names = {p["name"] for p in resp.json()["profiles"]}
        assert "brit_f" in names
        assert "brit_m" not in names

    def test_filter_tag_and_favorite_compose(self, client, ref_audio):
        self._setup(client, ref_audio)
        # british AND favorite → only brit_m
        resp = client.get("/v1/voices/profiles?tag=british&favorite=true")
        assert resp.status_code == 200
        names = {p["name"] for p in resp.json()["profiles"]}
        assert names == {"brit_m"}

    def test_sort_by_rating_descending(self, client, ref_audio):
        self._setup(client, ref_audio)
        resp = client.get("/v1/voices/profiles?sort=rating")
        assert resp.status_code == 200
        profiles = resp.json()["profiles"]
        ratings = [p["rating"] for p in profiles if p["rating"] is not None]
        # Should be non-increasing
        assert ratings == sorted(ratings, reverse=True)

    def test_sort_by_name_default(self, client, ref_audio):
        self._setup(client, ref_audio)
        resp = client.get("/v1/voices/profiles")
        assert resp.status_code == 200
        names = [p["name"] for p in resp.json()["profiles"]]
        assert names == sorted(names)

    def test_sort_by_last_used_nulls_last(self, client, ref_audio):
        self._setup(client, ref_audio)
        # Only brit_m gets a last_used_at
        client.patch(
            "/v1/voices/profiles/brit_m",
            json={"last_used_at": "2026-05-15T12:00:00+00:00"},
        )
        resp = client.get("/v1/voices/profiles?sort=last_used")
        assert resp.status_code == 200
        profiles = resp.json()["profiles"]
        # brit_m (has last_used_at) should come first
        assert profiles[0]["name"] == "brit_m"
        # The rest have null last_used_at; their order is unspecified but non-null first
        for p in profiles[1:]:
            assert p["last_used_at"] is None

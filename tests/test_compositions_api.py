"""FastAPI surface tests for /v1/voices/compositions and /v1/voices/segments.

These cover the in-flight composition object the voice lab uses for iteration:
creating drafts, listing, updating, deleting, and registering a draft into a
profile. The segment-audio endpoint's path-allowlist behavior is also covered.
"""

import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import HAS_MLX  # noqa: E402

if not HAS_MLX:
    pytest.skip("mlx not available (Apple Silicon only)", allow_module_level=True)


def _make_synthetic_conditionals():
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


class _FakeChatterboxModel:
    def prepare_conditionals(self, ref_audio, ref_sr=None, exaggeration=0.5, **kwargs):
        conds = _make_synthetic_conditionals()
        self._conds = conds
        return conds


def _write_wav_24k_mono(path, seconds):
    sr = 24000
    samples = (8000 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * sr)) / sr)).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import http_api
    from compositions import CompositionRegistry
    from voice_profiles import VoiceProfileRegistry

    monkeypatch.setattr(http_api, "_registry", VoiceProfileRegistry(root=tmp_path / "voices"))
    monkeypatch.setattr(http_api, "_compositions", CompositionRegistry(root=tmp_path / "compositions"))

    import engine as _engine_mod

    monkeypatch.setattr(_engine_mod, "get_model", lambda engine_name: _FakeChatterboxModel())

    monkeypatch.setenv("HOME", str(tmp_path))

    from fastapi.testclient import TestClient

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture()
def two_clips(tmp_path):
    # Put them under ~/.claude/jobs/ (an allowed segment root) so the
    # segment endpoint can serve them under the HOME override.
    seg_root = tmp_path / ".claude" / "jobs" / "test-job" / "diar"
    seg_root.mkdir(parents=True, exist_ok=True)
    a = seg_root / "001.wav"
    b = seg_root / "002.wav"
    _write_wav_24k_mono(a, 1.0)
    _write_wav_24k_mono(b, 0.5)
    return [str(a), str(b)]


# ---------------------------------------------------------------------------
# Composition CRUD
# ---------------------------------------------------------------------------


class TestCompositionCrud:
    def test_create_returns_201_shape(self, client, two_clips):
        r = client.post(
            "/v1/voices/compositions",
            json={
                "name": "draft_a",
                "segments": [{"path": two_clips[0], "label": "first"}],
                "engine": "chatterbox-turbo",
                "exaggeration": 0.7,
                "gap_sec": 0.2,
                "notes": "first try",
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["name"] == "draft_a"
        assert data["segments"][0]["label"] == "first"
        assert data["notes"] == "first try"
        assert data["created_at"] and data["updated_at"]

    def test_duplicate_name_returns_409(self, client, two_clips):
        body = {"name": "dup", "segments": [{"path": two_clips[0]}]}
        assert client.post("/v1/voices/compositions", json=body).status_code == 200
        r = client.post("/v1/voices/compositions", json=body)
        assert r.status_code == 409

    def test_list_returns_all(self, client, two_clips):
        for n in ("a", "b", "c"):
            client.post(
                "/v1/voices/compositions",
                json={"name": n, "segments": [{"path": two_clips[0]}]},
            )
        r = client.get("/v1/voices/compositions")
        assert r.status_code == 200
        names = {c["name"] for c in r.json()["compositions"]}
        assert names == {"a", "b", "c"}

    def test_get_missing_returns_404(self, client):
        assert client.get("/v1/voices/compositions/nope").status_code == 404

    def test_patch_updates_segments_and_bumps_updated_at(self, client, two_clips):
        client.post(
            "/v1/voices/compositions",
            json={"name": "patch_target", "segments": [{"path": two_clips[0]}]},
        )
        initial = client.get("/v1/voices/compositions/patch_target").json()
        r = client.patch(
            "/v1/voices/compositions/patch_target",
            json={"segments": [{"path": two_clips[1], "label": "swapped"}], "notes": "v2"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["segments"][0]["label"] == "swapped"
        assert data["notes"] == "v2"
        assert data["created_at"] == initial["created_at"]
        assert data["updated_at"] >= initial["updated_at"]

    def test_patch_missing_returns_404(self, client):
        r = client.patch("/v1/voices/compositions/nope", json={"notes": "x"})
        assert r.status_code == 404

    def test_delete_removes_then_404(self, client, two_clips):
        client.post(
            "/v1/voices/compositions",
            json={"name": "doomed", "segments": [{"path": two_clips[0]}]},
        )
        assert client.delete("/v1/voices/compositions/doomed").status_code == 200
        assert client.get("/v1/voices/compositions/doomed").status_code == 404
        assert client.delete("/v1/voices/compositions/doomed").status_code == 404

    def test_invalid_name_returns_400(self, client):
        r = client.post("/v1/voices/compositions", json={"name": "has space"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Composition → Profile registration
# ---------------------------------------------------------------------------


class TestRegisterFromComposition:
    def test_register_creates_a_profile_with_same_name(self, client, two_clips):
        client.post(
            "/v1/voices/compositions",
            json={
                "name": "for_register",
                "segments": [{"path": p} for p in two_clips],
                "engine": "chatterbox-turbo",
            },
        )
        r = client.post("/v1/voices/compositions/for_register/register")
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "for_register"
        # And the profile is now in the profile registry:
        profiles = client.get("/v1/voices/profiles").json()["profiles"]
        assert any(p["name"] == "for_register" for p in profiles)

    def test_register_with_alt_name_via_query(self, client, two_clips):
        client.post(
            "/v1/voices/compositions",
            json={
                "name": "base_draft",
                "segments": [{"path": p} for p in two_clips],
                "engine": "chatterbox-turbo",
            },
        )
        r = client.post(
            "/v1/voices/compositions/base_draft/register",
            params={"profile_name": "ab_test_2"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["name"] == "ab_test_2"

    def test_register_missing_composition_returns_404(self, client):
        r = client.post("/v1/voices/compositions/nope/register")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Segment audio passthrough
# ---------------------------------------------------------------------------


class TestSegmentAudio:
    def test_serves_wav_within_allowed_root(self, client, two_clips):
        r = client.get("/v1/voices/segments", params={"path": two_clips[0]})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")
        assert len(r.content) > 44  # has at least the WAV header

    def test_rejects_paths_outside_allowed_roots(self, client, tmp_path):
        outside = tmp_path / "outside.wav"
        _write_wav_24k_mono(outside, 0.2)
        # outside.wav is NOT under ~/.mod3, ~/.claude/jobs, or /tmp/voice_lab
        r = client.get("/v1/voices/segments", params={"path": str(outside)})
        assert r.status_code == 403

    def test_404_for_nonexistent(self, client, tmp_path):
        bogus = tmp_path / ".claude" / "jobs" / "fake.wav"
        r = client.get("/v1/voices/segments", params={"path": str(bogus)})
        assert r.status_code == 404

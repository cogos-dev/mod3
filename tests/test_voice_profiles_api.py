"""FastAPI surface tests for the voice-profile HTTP endpoints.

Covers:
  POST   /v1/voices/profiles  — register
  GET    /v1/voices/profiles  — list
  DELETE /v1/voices/profiles/{name}  — remove
  GET    /v1/voices  — grouping of custom profiles alongside built-ins

No real model is loaded. The slow ``get_model`` step is monkeypatched to a
fast fake. The ``_registry`` singleton in http_api is monkeypatched to a
tmp_path-rooted instance so tests never touch ~/.mod3/voices/.

Run:
  ~/workspaces/myrgic/mod3/.venv/bin/python -m pytest tests/test_voice_profiles_api.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Synthetic Conditionals factory (no model load)
# ---------------------------------------------------------------------------


def _make_synthetic_conditionals():
    """Build a minimal Conditionals dataclass using only mlx primitives.

    Shape matches what chatterbox produces in practice; the important thing is
    that save_conditionals() can serialise it without error.
    """
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


# ---------------------------------------------------------------------------
# Fake model — replaces the real chatterbox so tests are < 1 s
# ---------------------------------------------------------------------------


class _FakeChatterboxModel:
    """Drop-in for what get_model() returns for a cloning-capable engine."""

    def prepare_conditionals(self, ref_audio, ref_sr=None, exaggeration=0.5, **kwargs):
        """Return a synthetic Conditionals; also stash in _conds for turbo path."""
        conds = _make_synthetic_conditionals()
        self._conds = conds
        return conds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """FastAPI TestClient wired to a tmp_path registry and fake model."""
    import http_api
    from voice_profiles import VoiceProfileRegistry

    # Redirect the global registry away from ~/.mod3/voices/
    test_registry = VoiceProfileRegistry(root=tmp_path)
    monkeypatch.setattr(http_api, "_registry", test_registry)

    # Prevent any real model from loading — both the top-level import path
    # and the local-import path inside register_profile use engine.get_model.
    import engine as _engine_mod

    fake_model = _FakeChatterboxModel()
    monkeypatch.setattr(_engine_mod, "get_model", lambda engine_name: fake_model)

    from fastapi.testclient import TestClient

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture()
def ref_audio(tmp_path):
    """A tiny WAV-like file that passes the path-existence check."""
    p = tmp_path / "sample.wav"
    # Minimal 44-byte WAV header — content doesn't matter for existence checks.
    p.write_bytes(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return str(p)


# ---------------------------------------------------------------------------
# POST /v1/voices/profiles
# ---------------------------------------------------------------------------


class TestRegisterProfile:
    def test_happy_path_returns_200_with_profile_shape(self, client, ref_audio):
        """Successful registration returns 200 and a dict matching VoiceProfile.to_json()."""
        resp = client.post(
            "/v1/voices/profiles",
            json={"name": "alice", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # VoiceProfile.to_json() == dataclasses.asdict(profile)
        assert body["name"] == "alice"
        assert body["engine"] == "chatterbox"
        assert "source_audio_path" in body
        assert "source_sha256" in body
        assert "exaggeration" in body
        assert "model_id" in body
        assert "created_at" in body

    def test_duplicate_name_returns_409(self, client, ref_audio):
        """Registering the same name twice yields 409."""
        payload = {"name": "bob", "engine": "chatterbox", "ref_audio_path": ref_audio}
        first = client.post("/v1/voices/profiles", json=payload)
        assert first.status_code == 200, first.text
        second = client.post("/v1/voices/profiles", json=payload)
        assert second.status_code == 409, second.text

    def test_non_cloning_engine_returns_400(self, client, ref_audio):
        """Engines without supports_cloning (e.g. kokoro) are rejected with 400."""
        resp = client.post(
            "/v1/voices/profiles",
            json={"name": "carol", "engine": "kokoro", "ref_audio_path": ref_audio},
        )
        assert resp.status_code == 400, resp.text
        assert "cloning" in resp.json()["detail"].lower() or "cloning" in resp.text.lower()

    def test_missing_ref_audio_returns_404(self, client, tmp_path):
        """A ref_audio_path that does not exist yields 404."""
        missing = str(tmp_path / "does_not_exist.wav")
        resp = client.post(
            "/v1/voices/profiles",
            json={"name": "dave", "engine": "chatterbox", "ref_audio_path": missing},
        )
        assert resp.status_code == 404, resp.text

    def test_invalid_name_special_chars_returns_400(self, client, ref_audio):
        """Names with special characters (spaces, dots, slashes) yield 400."""
        for bad_name in ["my voice", "voice.v1", "voice/clone", "voice@1"]:
            resp = client.post(
                "/v1/voices/profiles",
                json={"name": bad_name, "engine": "chatterbox", "ref_audio_path": ref_audio},
            )
            assert resp.status_code == 400, f"Expected 400 for name {bad_name!r}, got {resp.status_code}"

    def test_turbo_engine_also_succeeds(self, client, ref_audio):
        """chatterbox-turbo (turbo code path) also registers successfully."""
        resp = client.post(
            "/v1/voices/profiles",
            json={"name": "turbo_test", "engine": "chatterbox-turbo", "ref_audio_path": ref_audio},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["engine"] == "chatterbox-turbo"

    def test_exaggeration_stored_in_profile(self, client, ref_audio):
        """Custom exaggeration value is preserved in the returned profile."""
        resp = client.post(
            "/v1/voices/profiles",
            json={"name": "emo", "engine": "chatterbox", "ref_audio_path": ref_audio, "exaggeration": 0.8},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["exaggeration"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# GET /v1/voices/profiles
# ---------------------------------------------------------------------------


class TestListProfiles:
    def test_empty_registry_returns_empty_list(self, client):
        """Fresh registry returns {profiles: []}."""
        resp = client.get("/v1/voices/profiles")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"profiles": []}

    def test_after_register_profile_appears_in_list(self, client, ref_audio):
        """Registered profile shows up in the listing."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "listed", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.get("/v1/voices/profiles")
        assert resp.status_code == 200, resp.text
        names = [p["name"] for p in resp.json()["profiles"]]
        assert "listed" in names

    def test_list_contains_full_profile_shape(self, client, ref_audio):
        """Each item in the profiles list has the full VoiceProfile shape."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "full_shape", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.get("/v1/voices/profiles")
        profile = next(p for p in resp.json()["profiles"] if p["name"] == "full_shape")
        for key in ("name", "engine", "source_audio_path", "source_sha256", "exaggeration", "model_id", "created_at"):
            assert key in profile, f"Missing key {key!r}"

    def test_multiple_profiles_all_listed(self, client, ref_audio):
        """All registered profiles appear in the list."""
        for name in ("p1", "p2", "p3"):
            client.post(
                "/v1/voices/profiles",
                json={"name": name, "engine": "chatterbox", "ref_audio_path": ref_audio},
            )
        resp = client.get("/v1/voices/profiles")
        names = {p["name"] for p in resp.json()["profiles"]}
        assert {"p1", "p2", "p3"}.issubset(names)


# ---------------------------------------------------------------------------
# DELETE /v1/voices/profiles/{name}
# ---------------------------------------------------------------------------


class TestDeleteProfile:
    def test_existing_profile_deleted_returns_200(self, client, ref_audio):
        """Deleting a registered profile returns 200 with {deleted: true}."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "to_delete", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.delete("/v1/voices/profiles/to_delete")
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] is True

    def test_after_delete_profile_absent_from_list(self, client, ref_audio):
        """After deletion the profile no longer appears in the list."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "gone", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        client.delete("/v1/voices/profiles/gone")
        resp = client.get("/v1/voices/profiles")
        names = [p["name"] for p in resp.json()["profiles"]]
        assert "gone" not in names

    def test_missing_profile_returns_404_with_deleted_false(self, client):
        """Deleting a non-existent profile returns 404 with {deleted: false, ...}."""
        resp = client.delete("/v1/voices/profiles/no_such_profile")
        assert resp.status_code == 404, resp.text
        body = resp.json()
        assert body["deleted"] is False
        # error key should be present
        assert "error" in body or "detail" in body

    def test_delete_then_reregister_succeeds(self, client, ref_audio):
        """A deleted profile name can be re-registered without 409."""
        payload = {"name": "recycle", "engine": "chatterbox", "ref_audio_path": ref_audio}
        client.post("/v1/voices/profiles", json=payload)
        client.delete("/v1/voices/profiles/recycle")
        resp = client.post("/v1/voices/profiles", json=payload)
        assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# GET /v1/voices — grouping of profiles
# ---------------------------------------------------------------------------


class TestVoicesGrouping:
    def test_no_profiles_custom_voices_absent(self, client):
        """With no profiles registered, custom_voices key is absent for all engines."""
        resp = client.get("/v1/voices")
        assert resp.status_code == 200, resp.text
        for engine_data in resp.json()["engines"].values():
            assert "custom_voices" not in engine_data

    def test_registered_profile_appears_in_engine_voices(self, client, ref_audio):
        """After registration the profile name appears in the engine's voices list."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "myvoice", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.get("/v1/voices")
        assert resp.status_code == 200, resp.text
        chatterbox = resp.json()["engines"]["chatterbox"]
        assert "myvoice" in chatterbox["voices"]

    def test_registered_profile_appears_in_custom_voices(self, client, ref_audio):
        """After registration the profile name appears in custom_voices (not just voices)."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "custom1", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.get("/v1/voices")
        chatterbox = resp.json()["engines"]["chatterbox"]
        assert "custom_voices" in chatterbox
        assert "custom1" in chatterbox["custom_voices"]

    def test_builtins_still_present_alongside_profiles(self, client, ref_audio):
        """Built-in voices are not displaced when a profile is added."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "extra", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.get("/v1/voices")
        chatterbox = resp.json()["engines"]["chatterbox"]
        assert "chatterbox" in chatterbox["voices"], "built-in 'chatterbox' voice should still be present"

    def test_custom_voices_does_not_contain_builtins(self, client, ref_audio):
        """custom_voices contains only profile names, not the built-in 'chatterbox' voice."""
        client.post(
            "/v1/voices/profiles",
            json={"name": "profileonly", "engine": "chatterbox", "ref_audio_path": ref_audio},
        )
        resp = client.get("/v1/voices")
        chatterbox = resp.json()["engines"]["chatterbox"]
        custom = chatterbox.get("custom_voices", [])
        assert "chatterbox" not in custom, "built-in voice must not appear in custom_voices"
        assert "profileonly" in custom

    def test_multiple_profiles_all_in_custom_voices(self, client, ref_audio):
        """Multiple registered profiles all appear in custom_voices."""
        for name in ("voice_a", "voice_b"):
            client.post(
                "/v1/voices/profiles",
                json={"name": name, "engine": "chatterbox", "ref_audio_path": ref_audio},
            )
        resp = client.get("/v1/voices")
        custom = resp.json()["engines"]["chatterbox"].get("custom_voices", [])
        assert "voice_a" in custom
        assert "voice_b" in custom

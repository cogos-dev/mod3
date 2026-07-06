"""FastAPI surface tests for POST /v1/voices/profiles/compose.

Same shape as test_voice_profiles_api.py: the slow model load is replaced
with a synthetic Conditionals factory, and the global registry is redirected
to a tmp_path-rooted instance so tests never touch ~/.mod3/voices/.

Run:
  ~/workspaces/myrgic/mod3/.venv/bin/python -m pytest tests/test_voice_profile_compose_api.py -v
"""

import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# _make_synthetic_conditionals() imports mlx at call time; skip the entire
# module in CI where the Apple-Silicon MLX stack is absent.
from conftest import HAS_MLX  # noqa: E402

if not HAS_MLX:
    pytest.skip("mlx not available (Apple Silicon only)", allow_module_level=True)


# ---------------------------------------------------------------------------
# Shared fakes (copied from test_voice_profiles_api so the two test files
# can run independently; small enough not to merit a conftest)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# WAV helpers — produce real on-disk WAVs the endpoint will read
# ---------------------------------------------------------------------------


def _write_wav(path, samples, sr=24000, nch=1, sampwidth=2):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def _tone(seconds, freq=440, sr=24000, amp=8000):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import http_api
    from voice_profiles import VoiceProfileRegistry

    monkeypatch.setattr(http_api, "_registry", VoiceProfileRegistry(root=tmp_path / "voices"))

    import engine as _engine_mod

    monkeypatch.setattr(_engine_mod, "get_model", lambda engine_name: _FakeChatterboxModel())

    # Redirect ~/.mod3 to a tmp dir so the source WAV write doesn't pollute.
    monkeypatch.setenv("HOME", str(tmp_path))

    from fastapi.testclient import TestClient

    return TestClient(http_api.app, base_url="http://localhost:7860")


@pytest.fixture()
def two_segments(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _write_wav(a, _tone(1.0, freq=300))  # 1 s
    _write_wav(b, _tone(0.5, freq=600))  # 0.5 s
    return [str(a), str(b)]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestComposeProfile:
    def test_happy_path_returns_200_with_profile_shape(self, client, two_segments):
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "kronk_test",
                "engine": "chatterbox-turbo",
                "segment_paths": two_segments,
                "gap_sec": 0.1,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] == "kronk_test"
        assert data["engine"] == "chatterbox-turbo"

    def test_combined_wav_has_expected_duration(self, client, two_segments, tmp_path):
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "duration_test",
                "engine": "chatterbox-turbo",
                "segment_paths": two_segments,
                "gap_sec": 0.2,
            },
        )
        assert resp.status_code == 200, resp.text
        composed = tmp_path / ".mod3" / "voices" / "sources" / "duration_test.wav"
        assert composed.exists()
        with wave.open(str(composed), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
        # 1.0 + 0.2 (gap) + 0.5 = 1.7 s, allow tolerance for int rounding
        assert abs(duration - 1.7) < 0.01
        with wave.open(str(composed), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    def test_single_segment_is_a_valid_compose(self, client, two_segments):
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "single_seg",
                "engine": "chatterbox-turbo",
                "segment_paths": [two_segments[0]],
            },
        )
        assert resp.status_code == 200, resp.text

    def test_stereo_segment_is_downmixed_not_rejected(self, client, tmp_path):
        stereo = tmp_path / "stereo.wav"
        # Two interleaved channels of equal-amplitude tone.
        mono = _tone(0.4)
        interleaved = np.repeat(mono, 2)
        _write_wav(stereo, interleaved, nch=2)
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "stereo_ok",
                "engine": "chatterbox-turbo",
                "segment_paths": [str(stereo)],
            },
        )
        assert resp.status_code == 200, resp.text

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_empty_segments_returns_400(self, client):
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={"name": "x", "engine": "chatterbox-turbo", "segment_paths": []},
        )
        assert resp.status_code == 400
        assert "at least one" in resp.json()["detail"].lower()

    def test_missing_segment_returns_404(self, client, two_segments, tmp_path):
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "missing",
                "engine": "chatterbox-turbo",
                "segment_paths": [two_segments[0], str(tmp_path / "does_not_exist.wav")],
            },
        )
        assert resp.status_code == 404
        assert "segment not found" in resp.json()["detail"]

    def test_wrong_sample_rate_returns_400(self, client, tmp_path):
        bad = tmp_path / "16k.wav"
        _write_wav(bad, _tone(0.5, sr=16000), sr=16000)
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "sr_test",
                "engine": "chatterbox-turbo",
                "segment_paths": [str(bad)],
            },
        )
        assert resp.status_code == 400
        assert "24000" in resp.json()["detail"]

    def test_non_cloning_engine_returns_400(self, client, two_segments):
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "engine_test",
                "engine": "kokoro",
                "segment_paths": two_segments,
            },
        )
        assert resp.status_code == 400
        assert "cloning" in resp.json()["detail"].lower()

    def test_duplicate_name_returns_409(self, client, two_segments):
        client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "dup",
                "engine": "chatterbox-turbo",
                "segment_paths": two_segments,
            },
        )
        resp = client.post(
            "/v1/voices/profiles/compose",
            json={
                "name": "dup",
                "engine": "chatterbox-turbo",
                "segment_paths": two_segments,
            },
        )
        assert resp.status_code == 409

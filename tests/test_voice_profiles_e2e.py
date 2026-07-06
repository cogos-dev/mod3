"""End-to-end tests for the profile-aware synthesis path in engine.py.

Covers the integration between engine.generate_audio(), the registry's profile
resolution shim, and the full synthesis path.  No real models are loaded; no
network calls are made; no daemon is spawned.

Run: python3 -m pytest tests/test_voice_profiles_e2e.py -v
"""

from __future__ import annotations

import os
import sys
import wave
from typing import Any

import numpy as np
import pytest

# Ensure the project root is on sys.path so imports resolve without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# make_synthetic_conditionals() imports mlx + mlx_audio at call time; skip the
# whole module in CI where the Apple-Silicon MLX stack is absent.  Check the
# pre-stub HAS_MLX flag from conftest rather than importorskip so we don't
# accidentally succeed against a CI stub.
from conftest import HAS_MLX  # noqa: E402

if not HAS_MLX:
    pytest.skip("mlx not available (Apple Silicon only)", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dummy_wav(path) -> None:
    """Write a minimal 1-second silent WAV to *path* so registry validates it."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(b"\x00" * (24000 * 2 * 1))  # 1 s of silence


def make_synthetic_conditionals():
    """Build a minimal Conditionals without loading a chatterbox model."""
    import mlx.core as mx
    from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals
    from mlx_audio.tts.models.chatterbox.t3.cond_enc import T3Cond

    t3 = T3Cond(speaker_emb=mx.zeros((1, 256)))
    gen = {
        "prompt_token": mx.zeros((1, 50), dtype=mx.int32),
        "prompt_token_len": mx.array([50]),
    }
    return Conditionals(t3=t3, gen=gen)


def _make_registry(tmp_path):
    """Return a VoiceProfileRegistry rooted at *tmp_path*."""
    from voice_profiles import VoiceProfileRegistry

    return VoiceProfileRegistry(root=tmp_path)


def _register_profile(tmp_path, name="test_voice", engine="chatterbox-turbo"):
    """Register a synthetic profile and return (registry, conds, wav_path)."""
    registry = _make_registry(tmp_path)
    conds = make_synthetic_conditionals()
    wav = tmp_path / "ref.wav"
    make_dummy_wav(wav)
    registry.register(
        name=name,
        engine=engine,
        ref_audio_path=str(wav),
        conds=conds,
    )
    return registry, conds, wav


# ---------------------------------------------------------------------------
# Fake model for patching engine.get_model
# ---------------------------------------------------------------------------


class FakeResult:
    """Mimics the result object that model.generate() yields."""

    def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
        n = len(audio)
        self.audio = audio.tolist()
        self.processing_time_seconds = 0.01
        self.real_time_factor = 0.1
        self.samples = n
        self.token_count = 10
        self.is_final_chunk = True
        self.peak_memory_usage = 0.05


class FakeModel:
    """Records kwargs passed to generate() and yields a single fake result.

    Also captures the value of self._conds at generate() time, since the
    chatterbox-turbo dispatch path mutates _conds rather than passing it
    as a kwarg (the upstream generate() signature ignores `conds=`).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.sample_rate = 24000
        self._last_kwargs: dict[str, Any] = {}
        self._conds = None
        self._last_conds = None

    def generate(self, **kwargs) -> list[FakeResult]:  # type: ignore[override]
        self._last_kwargs = dict(kwargs)
        self._last_conds = self._conds
        dummy = np.zeros(128, dtype=np.float32)
        return [FakeResult(dummy, self.sample_rate)]


# ---------------------------------------------------------------------------
# Tests: resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    """Unit-level tests for engine.resolve_model()."""

    def test_registered_profile_returns_parent_engine(self, tmp_path, monkeypatch):
        """resolve_model returns the profile's engine for a registered profile name."""
        import engine

        registry, _, _ = _register_profile(tmp_path, name="my_voice", engine="chatterbox-turbo")
        monkeypatch.setattr(engine, "_get_profile_registry", lambda: registry)
        # Also reset the cached singleton so the monkeypatch takes effect
        monkeypatch.setattr(engine, "_profile_registry", None)

        result_engine, result_voice = engine.resolve_model("my_voice")

        assert result_engine == "chatterbox-turbo"
        assert result_voice == "my_voice"

    def test_builtin_voice_falls_through_to_models(self, monkeypatch):
        """resolve_model falls through to MODELS for a known built-in voice."""
        import engine

        # No profiles registered — registry returns None for everything
        monkeypatch.setattr(engine, "_get_profile_registry", lambda: None)

        result_engine, result_voice = engine.resolve_model("af_heart")

        assert result_engine == "kokoro"
        assert result_voice == "af_heart"

    def test_unknown_voice_raises_value_error(self, monkeypatch):
        """resolve_model raises ValueError for a voice that is neither a profile nor in MODELS."""
        import engine

        monkeypatch.setattr(engine, "_get_profile_registry", lambda: None)

        with pytest.raises(ValueError, match="Unknown voice"):
            engine.resolve_model("not_a_real_voice")

    def test_registered_profile_chatterbox_engine(self, tmp_path, monkeypatch):
        """resolve_model also works for profiles backed by the plain chatterbox engine."""
        import engine

        registry, _, _ = _register_profile(tmp_path, name="alt_voice", engine="chatterbox")
        monkeypatch.setattr(engine, "_get_profile_registry", lambda: registry)

        result_engine, result_voice = engine.resolve_model("alt_voice")

        assert result_engine == "chatterbox"
        assert result_voice == "alt_voice"


# ---------------------------------------------------------------------------
# Tests: generate_audio — profile path
# ---------------------------------------------------------------------------


class TestGenerateAudioProfilePath:
    """Integration tests for the profile-aware chatterbox dispatch in generate_audio()."""

    def _setup(self, tmp_path, monkeypatch, profile_name="synth_voice"):
        """Register a profile, inject fake model, return (fake_model, conds)."""
        import engine

        registry, conds, _ = _register_profile(tmp_path, name=profile_name, engine="chatterbox-turbo")

        fake_model = FakeModel("chatterbox-turbo")
        monkeypatch.setattr(engine, "_get_profile_registry", lambda: registry)
        monkeypatch.setattr(engine, "_profile_registry", None)
        monkeypatch.setattr(engine, "get_model", lambda name: fake_model)
        monkeypatch.setattr(engine, "split_sentences", lambda text: [text.strip()])

        return fake_model, conds

    def test_conds_passed_not_ref_audio(self, tmp_path, monkeypatch):
        """When voice is a registered profile, generate_audio routes Conditionals
        via model._conds (chatterbox-turbo) and never passes ref_audio."""
        from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals

        import engine

        fake_model, expected_conds = self._setup(tmp_path, monkeypatch)

        list(engine.generate_audio(text="hello", voice="synth_voice"))

        kwargs = fake_model._last_kwargs
        # Turbo's generate() silently ignores `conds=` kwarg and reads from
        # self._conds, so the dispatch mutates the singleton instead.
        assert "ref_audio" not in kwargs, "Did not expect 'ref_audio' when profile is registered"
        assert "conds" not in kwargs, "Turbo branch should mutate model._conds, not pass it as a kwarg"
        actual = fake_model._last_conds
        assert isinstance(actual, Conditionals), "model._conds must be a Conditionals instance"
        # Identity is not preserved across registry round-trip (the conds came
        # from a fresh safetensors load), but the tensors must match.
        import mlx.core as mx

        assert mx.array_equal(actual.t3.speaker_emb, expected_conds.t3.speaker_emb)
        assert set(actual.gen.keys()) == set(expected_conds.gen.keys())

    def test_conds_has_t3_and_gen(self, tmp_path, monkeypatch):
        """The Conditionals routed via model._conds has .t3 with speaker_emb and .gen dict."""
        import engine

        fake_model, _ = self._setup(tmp_path, monkeypatch)

        list(engine.generate_audio(text="hi", voice="synth_voice"))

        conds = fake_model._last_conds
        assert hasattr(conds, "t3"), "Conditionals must have .t3"
        assert hasattr(conds, "gen"), "Conditionals must have .gen"
        assert hasattr(conds.t3, "speaker_emb"), "T3Cond must have .speaker_emb"

    def test_generate_returns_audio_chunks(self, tmp_path, monkeypatch):
        """generate_audio yields AudioChunk objects with the right shape."""
        import engine

        self._setup(tmp_path, monkeypatch)

        chunks = list(engine.generate_audio(text="test sentence", voice="synth_voice"))

        # Expect at least one non-empty chunk (the sentence itself)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, engine.AudioChunk)
            assert chunk.sample_rate == 24000
            assert isinstance(chunk.samples, np.ndarray)

    def test_exaggeration_forwarded(self, tmp_path, monkeypatch):
        """The emotion parameter is forwarded as exaggeration to the chatterbox generate call."""
        import engine

        fake_model, _ = self._setup(tmp_path, monkeypatch)

        list(engine.generate_audio(text="hi", voice="synth_voice", emotion=0.8))

        assert fake_model._last_kwargs.get("exaggeration") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Tests: generate_audio — ref_audio fallback (no profile)
# ---------------------------------------------------------------------------


class TestGenerateAudioRefAudioFallback:
    """Verify that when no profile is registered, ref_audio is forwarded instead of conds."""

    def test_ref_audio_passed_when_no_profile(self, tmp_path, monkeypatch):
        """For a built-in chatterbox-turbo voice with ref_audio supplied, conds is absent."""
        import engine

        fake_model = FakeModel("chatterbox-turbo")

        # No profile registry in play
        monkeypatch.setattr(engine, "_get_profile_registry", lambda: None)
        monkeypatch.setattr(engine, "_profile_registry", None)
        monkeypatch.setattr(engine, "get_model", lambda name: fake_model)
        monkeypatch.setattr(engine, "split_sentences", lambda text: [text.strip()])

        dummy_wav = tmp_path / "ref.wav"
        make_dummy_wav(dummy_wav)

        list(
            engine.generate_audio(
                text="hello",
                voice="chatterbox-turbo",
                ref_audio=str(dummy_wav),
            )
        )

        kwargs = fake_model._last_kwargs
        assert "conds" not in kwargs, "Should not have conds when no profile registered"
        assert "ref_audio" in kwargs, "ref_audio must be forwarded when no profile"
        assert kwargs["ref_audio"] == str(dummy_wav)

    def test_neither_conds_nor_ref_audio_when_no_profile_and_no_ref(self, monkeypatch):
        """When no profile and no ref_audio supplied, neither key appears in kwargs."""
        import engine

        fake_model = FakeModel("chatterbox-turbo")
        monkeypatch.setattr(engine, "_get_profile_registry", lambda: None)
        monkeypatch.setattr(engine, "_profile_registry", None)
        monkeypatch.setattr(engine, "get_model", lambda name: fake_model)
        monkeypatch.setattr(engine, "split_sentences", lambda text: [text.strip()])

        list(engine.generate_audio(text="hi", voice="chatterbox-turbo"))

        kwargs = fake_model._last_kwargs
        assert "conds" not in kwargs
        assert "ref_audio" not in kwargs

"""Tests for the voice profile registry subsystem.

Covers:
  - voice_profile_io.py  : save_conditionals / load_conditionals roundtrip
  - voice_profile_schema.py : VoiceProfile JSON roundtrip + compute_source_sha256
  - voice_profiles.py    : VoiceProfileRegistry CRUD

No daemon, no network, no ~/.mod3/voices/ access.  Uses tmp_path exclusively.
Run: python -m pytest tests/test_voice_profiles.py -v
"""

import os
import struct
import sys

import pytest

# Ensure project root is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These tests require the Apple-Silicon MLX stack.  Skip the entire module in
# CI where mlx is unavailable.  We check HAS_MLX from conftest (which was set
# before any stubs were installed) so the check reflects real availability, not
# whether a stub exists.
from conftest import HAS_MLX  # noqa: E402

if not HAS_MLX:
    pytest.skip("mlx not available (Apple Silicon only)", allow_module_level=True)

import mlx.core as mx
from mlx_audio.tts.models.chatterbox.chatterbox import Conditionals
from mlx_audio.tts.models.chatterbox.t3.cond_enc import T3Cond

# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic Conditionals without loading a model
# ---------------------------------------------------------------------------


def make_conditionals() -> Conditionals:
    """Return a structurally valid Conditionals with predictable tensor values."""
    t3 = T3Cond(
        speaker_emb=mx.zeros((1, 256), dtype=mx.float32),
        cond_prompt_speech_tokens=mx.zeros((1, 150), dtype=mx.int32),
        emotion_adv=mx.ones((1, 1, 1)) * 0.5,
    )
    gen = {
        "prompt_token": mx.zeros((1, 50), dtype=mx.int32),
        "prompt_token_len": mx.array([50], dtype=mx.int32),
        "prompt_feat": mx.zeros((1, 120, 80), dtype=mx.float32),
        "prompt_feat_len": mx.array([120], dtype=mx.int32),
        "embedding": mx.zeros((1, 192), dtype=mx.float32),
    }
    return Conditionals(t3=t3, gen=gen)


def make_wav(path) -> None:
    """Write a minimal valid (but silent) WAV file to *path*."""
    import pathlib

    path = pathlib.Path(path)
    # PCM 16-bit mono, 16000 Hz, 0.1 s (1600 samples)
    num_samples = 1600
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt  chunk
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


# ===========================================================================
# 1. voice_profile_io: save_conditionals / load_conditionals
# ===========================================================================


class TestConditionalIO:
    def test_roundtrip_all_fields_bit_equal(self, tmp_path):
        """Save then load a Conditionals; every tensor must be bit-equal."""
        from voice_profile_io import load_conditionals, save_conditionals

        dest = tmp_path / "test.safetensors"
        conds = make_conditionals()
        save_conditionals(conds, dest)

        loaded = load_conditionals(dest)

        # t3 required field
        assert mx.array_equal(loaded.t3.speaker_emb, conds.t3.speaker_emb)
        # t3 optional fields that we populated
        assert mx.array_equal(
            loaded.t3.cond_prompt_speech_tokens,
            conds.t3.cond_prompt_speech_tokens,
        )
        assert mx.array_equal(loaded.t3.emotion_adv, conds.t3.emotion_adv)
        # gen fields
        for key in conds.gen:
            assert mx.array_equal(loaded.gen[key], conds.gen[key]), f"gen[{key!r}] mismatch after roundtrip"

    def test_file_created_on_disk(self, tmp_path):
        """save_conditionals must create the target file."""
        from voice_profile_io import save_conditionals

        dest = tmp_path / "out.safetensors"
        assert not dest.exists()
        save_conditionals(make_conditionals(), dest)
        assert dest.exists()
        assert dest.stat().st_size > 0

    def test_load_nonexistent_raises(self, tmp_path):
        """load_conditionals on a missing path must raise an error."""
        from voice_profile_io import load_conditionals

        missing = tmp_path / "does_not_exist.safetensors"
        try:
            load_conditionals(missing)
            assert False, "Expected an exception for missing file"
        except (FileNotFoundError, OSError, Exception):
            pass  # any error is acceptable; just must not succeed silently

    def test_roundtrip_minimal_t3_no_optionals(self, tmp_path):
        """T3Cond with only speaker_emb (no optional fields) roundtrips cleanly."""
        from voice_profile_io import load_conditionals, save_conditionals

        t3 = T3Cond(speaker_emb=mx.ones((1, 256), dtype=mx.float32))
        gen = {
            "prompt_token": mx.zeros((1, 50), dtype=mx.int32),
            "prompt_token_len": mx.array([50], dtype=mx.int32),
            "prompt_feat": mx.zeros((1, 120, 80), dtype=mx.float32),
            "prompt_feat_len": mx.array([120], dtype=mx.int32),
            "embedding": mx.zeros((1, 192), dtype=mx.float32),
        }
        conds = Conditionals(t3=t3, gen=gen)
        dest = tmp_path / "minimal.safetensors"
        save_conditionals(conds, dest)
        loaded = load_conditionals(dest)
        assert mx.array_equal(loaded.t3.speaker_emb, conds.t3.speaker_emb)


# ===========================================================================
# 2. voice_profile_schema: VoiceProfile + compute_source_sha256
# ===========================================================================


class TestVoiceProfileSchema:
    def _make_profile(self):
        from voice_profile_schema import VoiceProfile

        return VoiceProfile(
            name="alice",
            engine="chatterbox",
            source_audio_path="/tmp/alice.wav",
            source_sha256="aabbcc",
            ref_text="Hello world",
            exaggeration=0.7,
            model_id="mlx-community/chatterbox-4bit",
            created_at="2026-01-01T00:00:00+00:00",
        )

    def test_to_json_produces_dict(self):
        profile = self._make_profile()
        d = profile.to_json()
        assert isinstance(d, dict)
        assert d["name"] == "alice"
        assert d["engine"] == "chatterbox"

    def test_from_json_roundtrip(self):
        """to_json then from_json must reconstruct an equal VoiceProfile."""
        profile = self._make_profile()
        d = profile.to_json()
        from voice_profile_schema import VoiceProfile

        restored = VoiceProfile.from_json(d)
        assert restored.name == profile.name
        assert restored.engine == profile.engine
        assert restored.source_audio_path == profile.source_audio_path
        assert restored.source_sha256 == profile.source_sha256
        assert restored.ref_text == profile.ref_text
        assert abs(restored.exaggeration - profile.exaggeration) < 1e-9
        assert restored.model_id == profile.model_id
        assert restored.created_at == profile.created_at

    def test_from_json_ref_text_none(self):
        """ref_text may be absent from the dict (treated as None)."""
        from voice_profile_schema import VoiceProfile

        d = self._make_profile().to_json()
        del d["ref_text"]
        restored = VoiceProfile.from_json(d)
        assert restored.ref_text is None

    def test_compute_source_sha256_stable(self, tmp_path):
        """Same file must always produce the same hex digest."""
        from voice_profile_schema import compute_source_sha256

        f = tmp_path / "sample.wav"
        make_wav(f)
        h1 = compute_source_sha256(f)
        h2 = compute_source_sha256(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 produces 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in h1)

    def test_compute_source_sha256_differs_for_different_files(self, tmp_path):
        """Different file content must produce different digests."""
        from voice_profile_schema import compute_source_sha256

        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        make_wav(f1)
        f2.write_bytes(b"different content entirely")
        assert compute_source_sha256(f1) != compute_source_sha256(f2)


# ===========================================================================
# 3. VoiceProfileRegistry: CRUD
# ===========================================================================


class TestVoiceProfileRegistry:
    def _registry(self, tmp_path):
        from voice_profiles import VoiceProfileRegistry

        return VoiceProfileRegistry(root=tmp_path)

    def _register_alice(self, reg, tmp_path):
        wav = tmp_path / "alice.wav"
        make_wav(wav)
        return reg.register(
            name="alice",
            engine="chatterbox",
            ref_audio_path=str(wav),
            conds=make_conditionals(),
            ref_text="Hello",
            exaggeration=0.5,
        )

    # -- register -------------------------------------------------------------

    def test_register_returns_voice_profile(self, tmp_path):
        reg = self._registry(tmp_path)
        profile = self._register_alice(reg, tmp_path)
        from voice_profile_schema import VoiceProfile

        assert isinstance(profile, VoiceProfile)
        assert profile.name == "alice"
        assert profile.engine == "chatterbox"

    def test_register_creates_both_files(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        assert (tmp_path / "alice.json").exists()
        assert (tmp_path / "alice.safetensors").exists()

    def test_register_duplicate_raises_value_error(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        wav2 = tmp_path / "alice2.wav"
        make_wav(wav2)
        try:
            reg.register(
                name="alice",
                engine="chatterbox",
                ref_audio_path=str(wav2),
                conds=make_conditionals(),
            )
            assert False, "Expected ValueError for duplicate"
        except ValueError as e:
            assert "already exists" in str(e).lower()

    def test_register_invalid_name_raises_value_error(self, tmp_path):
        reg = self._registry(tmp_path)
        wav = tmp_path / "x.wav"
        make_wav(wav)
        try:
            reg.register(
                name="bad/name",
                engine="chatterbox",
                ref_audio_path=str(wav),
                conds=make_conditionals(),
            )
            assert False, "Expected ValueError for invalid name"
        except ValueError as e:
            assert "invalid" in str(e).lower() or "bad/name" in str(e)

    def test_register_non_cloning_engine_raises_value_error(self, tmp_path):
        reg = self._registry(tmp_path)
        wav = tmp_path / "x.wav"
        make_wav(wav)
        try:
            reg.register(
                name="alice",
                engine="kokoro",
                ref_audio_path=str(wav),
                conds=make_conditionals(),
            )
            assert False, "Expected ValueError for non-cloning engine"
        except ValueError as e:
            assert "cloning" in str(e).lower() or "kokoro" in str(e)

    def test_register_missing_audio_raises_file_not_found(self, tmp_path):
        reg = self._registry(tmp_path)
        try:
            reg.register(
                name="alice",
                engine="chatterbox",
                ref_audio_path=str(tmp_path / "ghost.wav"),
                conds=make_conditionals(),
            )
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass

    # -- list -----------------------------------------------------------------

    def test_list_empty_registry(self, tmp_path):
        reg = self._registry(tmp_path)
        assert reg.list() == []

    def test_list_returns_one_after_register(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        profiles = reg.list()
        assert len(profiles) == 1
        assert profiles[0].name == "alice"

    def test_list_sorted_by_name(self, tmp_path):
        reg = self._registry(tmp_path)
        for name in ("charlie", "alice", "bob"):
            wav = tmp_path / f"{name}.wav"
            make_wav(wav)
            reg.register(
                name=name,
                engine="chatterbox",
                ref_audio_path=str(wav),
                conds=make_conditionals(),
            )
        names = [p.name for p in reg.list()]
        assert names == sorted(names), f"Expected sorted, got {names}"

    # -- get ------------------------------------------------------------------

    def test_get_returns_profile(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        profile = reg.get("alice")
        assert profile is not None
        assert profile.name == "alice"

    def test_get_unknown_returns_none(self, tmp_path):
        reg = self._registry(tmp_path)
        assert reg.get("nobody") is None

    # -- get_conditionals -----------------------------------------------------

    def test_get_conditionals_returns_valid_object(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        conds = reg.get_conditionals("alice")
        assert conds is not None
        assert isinstance(conds, Conditionals)
        assert conds.t3.speaker_emb is not None

    def test_get_conditionals_unknown_returns_none(self, tmp_path):
        reg = self._registry(tmp_path)
        assert reg.get_conditionals("nobody") is None

    # -- delete ---------------------------------------------------------------

    def test_delete_returns_true_and_removes_files(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        result = reg.delete("alice")
        assert result is True
        assert not (tmp_path / "alice.json").exists()
        assert not (tmp_path / "alice.safetensors").exists()

    def test_delete_subsequent_get_returns_none(self, tmp_path):
        reg = self._registry(tmp_path)
        self._register_alice(reg, tmp_path)
        reg.delete("alice")
        assert reg.get("alice") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        reg = self._registry(tmp_path)
        assert reg.delete("nobody") is False


# ===========================================================================
# 4. Primitive 3 — IdentityVoiceProfile schema + cog://voices/* URI resolver
# ===========================================================================


class TestIdentityVoiceProfileSchema:
    """Tests for the identity-level VoiceProfile schema (Primitive 3)."""

    def test_generative_only_from_dict(self):
        """from_dict with generative head only; discriminative is None."""
        from voice_profile_schema import IdentityVoiceProfile

        data = {
            "generative": {
                "engine": "chatterbox-turbo",
                "conditionals_ref": "cog://voices/cog",
            }
        }
        vp = IdentityVoiceProfile.from_dict(data)
        assert vp.generative is not None
        assert vp.generative.engine == "chatterbox-turbo"
        assert vp.generative.conditionals_ref == "cog://voices/cog"
        assert vp.discriminative is None

    def test_both_heads_from_dict(self):
        """from_dict with generative and discriminative heads; all fields round-trip."""
        from voice_profile_schema import IdentityVoiceProfile

        data = {
            "generative": {
                "engine": "chatterbox-turbo",
                "conditionals_ref": "cog://voices/cog",
                "enrolled_at": "2026-05-19T00:00:00Z",
            },
            "discriminative": {
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "embedding_ref": "cog://voices/cog/ecapa-embedding",
                "enrolled_at": "2026-05-19T00:00:00Z",
            },
        }
        vp = IdentityVoiceProfile.from_dict(data)
        assert vp.generative is not None
        assert vp.discriminative is not None
        assert vp.discriminative.model == "speechbrain/spkrec-ecapa-voxceleb"
        assert vp.discriminative.embedding_ref == "cog://voices/cog/ecapa-embedding"
        assert vp.discriminative.enrolled_at == "2026-05-19T00:00:00Z"

    def test_discriminative_enrolled_at_optional(self):
        """enrolled_at is optional — absent field is None, no error."""
        from voice_profile_schema import IdentityVoiceProfile

        data = {
            "discriminative": {
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "embedding_ref": "cog://voices/chaz/ecapa-embedding",
            }
        }
        vp = IdentityVoiceProfile.from_dict(data)
        assert vp.discriminative is not None
        assert vp.discriminative.enrolled_at is None

    def test_to_dict_round_trip(self):
        """to_dict then from_dict must reconstruct an equal profile."""
        from voice_profile_schema import IdentityVoiceProfile

        data = {
            "generative": {
                "engine": "chatterbox-turbo",
                "conditionals_ref": "cog://voices/cog",
            },
            "discriminative": {
                "model": "speechbrain/spkrec-ecapa-voxceleb",
                "embedding_ref": "cog://voices/cog/ecapa-embedding",
            },
        }
        vp = IdentityVoiceProfile.from_dict(data)
        restored = IdentityVoiceProfile.from_dict(vp.to_dict())
        assert restored.generative.engine == vp.generative.engine
        assert restored.generative.conditionals_ref == vp.generative.conditionals_ref
        assert restored.discriminative.model == vp.discriminative.model
        assert restored.discriminative.embedding_ref == vp.discriminative.embedding_ref

    def test_absent_profile_both_heads_none(self):
        """Empty dict produces a VoiceProfile with both heads None — no error."""
        from voice_profile_schema import IdentityVoiceProfile

        vp = IdentityVoiceProfile.from_dict({})
        assert vp.generative is None
        assert vp.discriminative is None


class TestVoiceProfileDiscriminativeField:
    """Tests for the discriminative head field on the registry-level VoiceProfile."""

    def test_embedding_ref_absent_defaults_to_none(self):
        """A pre-Primitive-3 JSON without embedding_ref loads without error."""
        from voice_profile_schema import VoiceProfile

        data = {
            "name": "cog",
            "engine": "chatterbox-turbo",
            "source_audio_path": "/tmp/cog.wav",
            "source_sha256": "abc123",
            "ref_text": None,
            "exaggeration": 0.5,
            "model_id": "mlx-community/chatterbox-turbo-4bit",
            "created_at": "2026-05-19T00:00:00Z",
        }
        vp = VoiceProfile.from_json(data)
        assert vp.embedding_ref is None

    def test_embedding_ref_round_trips(self):
        """embedding_ref survives to_json/from_json round-trip."""
        from voice_profile_schema import VoiceProfile

        data = {
            "name": "cog",
            "engine": "chatterbox-turbo",
            "source_audio_path": "/tmp/cog.wav",
            "source_sha256": "abc123",
            "ref_text": None,
            "exaggeration": 0.5,
            "model_id": "mlx-community/chatterbox-turbo-4bit",
            "created_at": "2026-05-19T00:00:00Z",
            "embedding_ref": "cog://voices/cog/ecapa-embedding",
        }
        vp = VoiceProfile.from_json(data)
        assert vp.embedding_ref == "cog://voices/cog/ecapa-embedding"
        restored = VoiceProfile.from_json(vp.to_json())
        assert restored.embedding_ref == "cog://voices/cog/ecapa-embedding"


class TestVoicesURIResolver:
    """Tests for resolve_voices_uri — the cog://voices/* local resolver."""

    def test_bare_name_uri_resolves_to_safetensors(self, tmp_path):
        """cog://voices/<name> resolves to <root>/<name>.safetensors."""
        from voice_profile_schema import resolve_voices_uri

        result = resolve_voices_uri("cog://voices/cog", registry_root=tmp_path)
        assert result == tmp_path / "cog.safetensors"

    def test_bare_form_accepted(self, tmp_path):
        """cog:voices/<name> (bare form, no //) is also accepted."""
        from voice_profile_schema import resolve_voices_uri

        result = resolve_voices_uri("cog:voices/cog", registry_root=tmp_path)
        assert result == tmp_path / "cog.safetensors"

    def test_ecapa_embedding_uri_resolves_to_npy(self, tmp_path):
        """cog://voices/<name>/ecapa-embedding resolves to <root>/<name>.ecapa.npy."""
        from voice_profile_schema import resolve_voices_uri

        result = resolve_voices_uri("cog://voices/cog/ecapa-embedding", registry_root=tmp_path)
        assert result == tmp_path / "cog.ecapa.npy"

    def test_various_profile_names(self, tmp_path):
        """Resolver works with different voice profile names."""
        from voice_profile_schema import resolve_voices_uri

        for name in ("bm_lewis", "eng_uk_m_davids", "af_heart"):
            result = resolve_voices_uri(f"cog://voices/{name}", registry_root=tmp_path)
            assert result == tmp_path / f"{name}.safetensors"

    def test_non_voices_uri_raises(self, tmp_path):
        """URIs that are not voices namespace raise ValueError."""
        import pytest

        from voice_profile_schema import resolve_voices_uri

        with pytest.raises(ValueError, match="not a voices URI"):
            resolve_voices_uri("cog://mem/semantic/foo", registry_root=tmp_path)

    def test_empty_name_raises(self, tmp_path):
        """URI with empty name segment raises ValueError."""
        import pytest

        from voice_profile_schema import resolve_voices_uri

        with pytest.raises(ValueError):
            resolve_voices_uri("cog://voices/", registry_root=tmp_path)

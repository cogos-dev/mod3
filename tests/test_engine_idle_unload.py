"""Tests for engine.py idle-unload subsystem (MOD3_TTS_IDLE_UNLOAD_SECONDS).

These tests exercise the pure-Python logic of the subsystem without requiring
MLX or any real TTS model — all model loading is monkeypatched.

Run: python3 -m pytest tests/test_engine_idle_unload.py -v
"""

import os
import sys
import time

# Ensure the repo root is on sys.path so `import engine` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal model stub — has a sample_rate and a generate method."""

    sample_rate = 24000

    def generate(self, **_kwargs):
        return iter([])


def _fresh_engine_module(monkeypatch, idle_seconds: str = "0"):
    """Return a freshly-imported engine module with the given env var set.

    We must re-import the module each time because the idle-unload subsystem
    state is initialised at *import time* via _start_idle_unload_watcher_if_enabled().
    """
    monkeypatch.setenv("MOD3_TTS_IDLE_UNLOAD_SECONDS", idle_seconds)
    # Remove cached module so the next import re-runs module-level code.
    sys.modules.pop("engine", None)
    import engine as eng

    return eng


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestParseIdleUnloadSeconds:
    def test_default_disabled(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "0")
        assert eng._idle_unload_enabled is False
        assert eng._idle_unload_seconds == 0.0

    def test_positive_value_enables(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "300")
        assert eng._idle_unload_enabled is True
        assert eng._idle_unload_seconds == 300.0

    def test_negative_clamped_to_zero(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "-10")
        assert eng._idle_unload_enabled is False

    def test_non_numeric_warns_and_disables(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "garbage")
        assert eng._idle_unload_enabled is False


class TestGetModelTimestampUpdate:
    def test_last_use_ts_updated_on_get_model(self, monkeypatch):
        """_last_use_ts should be stamped when idle-unload is enabled."""
        eng = _fresh_engine_module(monkeypatch, "300")

        # Patch mlx_audio.tts.load so we don't need a real model.
        fake_mlx = type(sys)("mlx_audio")
        fake_mlx.tts = type(sys)("mlx_audio.tts")
        fake_mlx.tts.load = lambda _id: _FakeModel()
        sys.modules["mlx_audio"] = fake_mlx
        sys.modules["mlx_audio.tts"] = fake_mlx.tts

        before = time.time()
        model = eng.get_model("kokoro")
        after = time.time()

        assert eng._last_use_ts >= before
        assert eng._last_use_ts <= after
        assert isinstance(model, _FakeModel)

    def test_last_use_ts_not_updated_when_disabled(self, monkeypatch):
        """When idle-unload is off, _last_use_ts stays at 0."""
        eng = _fresh_engine_module(monkeypatch, "0")

        fake_mlx = type(sys)("mlx_audio")
        fake_mlx.tts = type(sys)("mlx_audio.tts")
        fake_mlx.tts.load = lambda _id: _FakeModel()
        sys.modules["mlx_audio"] = fake_mlx
        sys.modules["mlx_audio.tts"] = fake_mlx.tts

        eng.get_model("kokoro")
        assert eng._last_use_ts == 0.0


class TestUnloadAllModels:
    def test_unload_clears_models_dict(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "300")

        # Seed _models directly (no real load needed).
        with eng._model_lock:
            eng._models["kokoro"] = _FakeModel()
            eng._models["chatterbox"] = _FakeModel()

        evicted = eng._unload_all_models()
        assert set(evicted) == {"kokoro", "chatterbox"}
        assert eng._models == {}

    def test_unload_idempotent_on_empty(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "300")
        evicted = eng._unload_all_models()
        assert evicted == []

    def test_reload_after_unload(self, monkeypatch):
        """After eviction, get_model() should reload transparently."""
        eng = _fresh_engine_module(monkeypatch, "300")

        load_calls = []

        def fake_load(_id):
            load_calls.append(_id)
            return _FakeModel()

        fake_mlx = type(sys)("mlx_audio")
        fake_mlx.tts = type(sys)("mlx_audio.tts")
        fake_mlx.tts.load = fake_load
        sys.modules["mlx_audio"] = fake_mlx
        sys.modules["mlx_audio.tts"] = fake_mlx.tts

        # First load.
        eng.get_model("kokoro")
        assert len(load_calls) == 1

        # Evict.
        eng._unload_all_models()
        assert eng._models == {}

        # Reload — should call load() again.
        eng.get_model("kokoro")
        assert len(load_calls) == 2


class TestGetLoadedEnginesAfterUnload:
    def test_get_loaded_engines_empty_after_unload(self, monkeypatch):
        eng = _fresh_engine_module(monkeypatch, "0")
        with eng._model_lock:
            eng._models["kokoro"] = _FakeModel()
        eng._unload_all_models()
        assert eng.get_loaded_engines() == []

"""Tests that the runtime-reported version matches the package metadata.

Acceptance criterion from issue #16:
  A test asserts importlib.metadata.version("mod3") matches what /health returns.
  Catches future drift where pyproject.toml is bumped but code is not updated.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_heavy_deps():
    """Inject lightweight stubs for ML/audio deps not available in CI.

    http_api imports engine, vad, and several audio modules that all pull in
    numpy / mlx-audio / sounddevice.  We stub them at the sys.modules level so
    that importing http_api succeeds without the native libraries.
    """
    # Stub numpy (engine.py imports it at module level)
    if "numpy" not in sys.modules:
        numpy_mock = types.ModuleType("numpy")
        numpy_mock.ndarray = object
        numpy_mock.float32 = float
        sys.modules["numpy"] = numpy_mock

    for mod_name in (
        "mlx",
        "mlx.core",
        "mlx_audio",
        "mlx_audio.tts",
        "mlx_audio.tts.models",
        "mlx_audio.tts.models.kokoro",
        "mlx_whisper",
        "sounddevice",
        "pysbd",
        "misaki",
        "misaki.en",
        "num2words",
        "espeakng_loader",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # engine module stubs — must expose MODELS, generate_audio, get_loaded_engines
    if "engine" not in sys.modules:
        engine_mod = types.ModuleType("engine")
        engine_mod.MODELS = {}
        engine_mod.generate_audio = MagicMock()
        engine_mod.get_loaded_engines = MagicMock(return_value=[])
        sys.modules["engine"] = engine_mod

    # vad module stubs
    if "vad" not in sys.modules:
        vad_mod = types.ModuleType("vad")
        vad_mod.detect_speech_file = MagicMock()
        vad_mod.is_hallucination = MagicMock(return_value=False)
        vad_mod.is_model_loaded = MagicMock(return_value=False)
        # http_api imports these two at module level (F2 pipecat wrapper)
        vad_mod.is_pipecat_vad_available = MagicMock(return_value=False)
        vad_mod.voice_confidence = MagicMock(return_value=0.0)
        sys.modules["vad"] = vad_mod

    # modality stubs
    if "modality" not in sys.modules:
        sys.modules["modality"] = MagicMock()

    # remaining local modules that http_api imports
    for local in ("audio_subscribers", "bus", "session_registry"):
        if local not in sys.modules:
            sys.modules[local] = MagicMock()

    # modules.text / modules.voice
    if "modules" not in sys.modules:
        sys.modules["modules"] = MagicMock()
    for sub in ("modules.text", "modules.voice"):
        if sub not in sys.modules:
            sys.modules[sub] = MagicMock()


# ---------------------------------------------------------------------------
# _version module
# ---------------------------------------------------------------------------


def test_version_module_returns_string():
    """_version.__version__ must be a non-empty string."""
    from _version import __version__

    assert isinstance(__version__, str)
    assert __version__, "_version.__version__ must not be empty"
    assert __version__ != "unknown", (
        "__version__ resolved to 'unknown' — either the package is not installed "
        "and pyproject.toml could not be found, or tomllib failed to parse it."
    )


def test_version_module_matches_pyproject():
    """_version.__version__ must equal the version declared in pyproject.toml."""
    import tomllib
    from pathlib import Path

    from _version import __version__

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)

    expected = data["project"]["version"]
    assert __version__ == expected, (
        f"_version.__version__ is {__version__!r} but pyproject.toml declares {expected!r}. "
        "Run `pip install -e .` (or `uv pip install -e .`) so importlib.metadata is in sync."
    )


# ---------------------------------------------------------------------------
# /health and /capabilities endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def http_app(restore_sys_modules):
    """Import http_api.app with heavy native deps stubbed out.

    Uses restore_sys_modules so the stubs injected by _stub_heavy_deps() are
    cleaned up after each test.  Without this, the fake engine/vad/etc. modules
    linger in sys.modules for the rest of the pytest session and cause
    ImportError in test files that later import the real modules.

    Scope is function (not module) so each test gets a fresh import against a
    clean stub set; the overhead is negligible because no heavy I/O occurs.
    """
    _stub_heavy_deps()
    # Force a fresh import so http_api is loaded against the current stub set
    if "http_api" in sys.modules:
        del sys.modules["http_api"]
    import http_api as _http_api

    return _http_api.app


def test_health_version_matches_package_version(http_app):
    """/health JSON must report the same version as importlib.metadata / pyproject.toml."""
    from fastapi.testclient import TestClient

    from _version import __version__

    client = TestClient(http_app, raise_server_exceptions=False)
    response = client.get("/health")

    # /health may return 200 or 500 when engines aren't loaded; either way the
    # version field must match.
    body = response.json()
    reported = body.get("version")
    assert reported == __version__, (
        f"/health reports version {reported!r} but package version is {__version__!r}. "
        "A hardcoded version literal may have been re-introduced in http_api.py."
    )


def test_capabilities_version_matches_package_version(http_app):
    """/capabilities JSON must report the same version as importlib.metadata / pyproject.toml."""
    from fastapi.testclient import TestClient

    from _version import __version__

    client = TestClient(http_app, raise_server_exceptions=False)
    response = client.get("/capabilities")

    assert response.status_code == 200, f"/capabilities returned {response.status_code}: {response.text}"
    body = response.json()
    reported = body.get("version")
    assert reported == __version__, (
        f"/capabilities reports version {reported!r} but package version is {__version__!r}. "
        "A hardcoded version literal may have been re-introduced in http_api.py."
    )

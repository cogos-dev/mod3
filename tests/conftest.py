"""Shared pytest fixtures for the mod3 test suite.

Key responsibilities:
  1. sys.modules stub isolation — any test that injects fake modules into
     sys.modules must restore the original state after the test (or module)
     completes.  Without this, a stub written by test_version.py's
     _stub_heavy_deps() persists for the rest of the session and causes
     ImportError / AttributeError in unrelated tests that later try to import
     the real engine, vad, or other modules.

  2. testpaths is declared in pyproject.toml [tool.pytest.ini_options] so
     pytest never walks the worktree root or vendor/ dirs during collection.

  3. CI-mode stubs for unavailable native libs — when torch, sounddevice, mlx,
     etc. are absent (linux CI, or dev machines without Apple Silicon stack),
     pytest collection must still succeed.  The _ci_native_stubs() function
     inserts lightweight MagicMock stubs for any missing native package so that
     module-level imports in production code do not abort collection.  Tests
     that exercise the real behaviour of these libs should use
     pytest.importorskip at module level to skip themselves when absent.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# CI-mode stubs for unavailable native libs
# ---------------------------------------------------------------------------


def _stub_if_missing(name: str, mod: object | None = None) -> None:
    """Insert *mod* (default: a fresh MagicMock) into sys.modules for *name*
    if and only if the real package cannot be imported.
    """
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = mod if mod is not None else MagicMock()


def _install_ci_native_stubs() -> None:
    """Stub out native libs that are unavailable in CI (Linux / non-Apple).

    Called once at session start so that module-level imports in production
    code (torch in vad.py, sounddevice in capture.py, etc.) resolve without
    ImportError during collection or test setup.

    Only affects packages that are genuinely absent — if torch is installed,
    the real torch is used.
    """
    # torch + torchaudio (vad.py imports torch at module level)
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_mod = types.ModuleType("torch")
            torch_mod.Tensor = MagicMock  # type: ignore[attr-defined]
            torch_mod.hub = MagicMock()  # type: ignore[attr-defined]
            torch_mod.from_numpy = MagicMock()  # type: ignore[attr-defined]
            torch_mod.tensor = MagicMock()  # type: ignore[attr-defined]
            sys.modules["torch"] = torch_mod
            sys.modules["torchaudio"] = MagicMock()
            sys.modules["torchaudio.functional"] = MagicMock()

    # sounddevice (capture.py, session_registry.py, server.py)
    _stub_if_missing("sounddevice")

    # soundfile (http_api.py, server.py — lazy import, but mlx_audio may pull it)
    _stub_if_missing("soundfile")

    # scipy (pulled in by some audio helpers)
    _stub_if_missing("scipy")
    _stub_if_missing("scipy.signal")

    # mlx stack — Apple Silicon only
    for mod_name in (
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_audio",
        "mlx_audio.tts",
        "mlx_audio.tts.models",
        "mlx_audio.tts.models.kokoro",
        "mlx_whisper",
    ):
        _stub_if_missing(mod_name)

    # mcp (server.py and channel_client.py import from mcp at module level)
    if "mcp" not in sys.modules:
        try:
            import mcp  # noqa: F401
        except ImportError:
            mcp_mod = MagicMock()
            sys.modules["mcp"] = mcp_mod
            # Register subpackages as real module objects so that dotted
            # imports (from mcp.server.fastmcp import FastMCP, etc.) resolve.
            for sub in (
                "mcp.server",
                "mcp.server.fastmcp",
                "mcp.server.stdio",
                "mcp.types",
                "mcp.shared",
                "mcp.shared.context",
                "mcp.shared.message",
            ):
                sys.modules[sub] = MagicMock()

    # pysbd (engine.py — usually installable, but stub just in case)
    _stub_if_missing("pysbd")

    # misaki + related NLP deps
    for mod_name in ("misaki", "misaki.en", "num2words", "espeakng_loader"):
        _stub_if_missing(mod_name)


# Track real availability before any stubs are installed, so importorskip
# guards in individual test files can gate on the *real* lib rather than the
# stub.  These booleans are importable from conftest or checked via the
# skip markers below.
def _real_importable(name: str) -> bool:
    """Return True if *name* can be imported WITHOUT relying on a stub."""
    if name in sys.modules:
        # Already loaded — check whether it is a stub (MagicMock) or real.
        mod = sys.modules[name]
        return not isinstance(mod, MagicMock)
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_MLX = _real_importable("mlx")
HAS_MCP = _real_importable("mcp")
HAS_TORCH = _real_importable("torch")

# Reusable skip markers for test modules that hard-import Apple-Silicon / native libs.
skip_without_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not available (Apple Silicon only)")
skip_without_mcp = pytest.mark.skipif(not HAS_MCP, reason="mcp package not installed")

# Install stubs once at import time (i.e. when conftest is loaded by pytest,
# before any test module is collected).
_install_ci_native_stubs()


# ---------------------------------------------------------------------------
# sys.modules save/restore fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def restore_sys_modules() -> Generator[None, None, None]:
    """Save and restore sys.modules around a single test.

    Usage — request this fixture in any test or fixture that injects stubs:

        def test_something(restore_sys_modules):
            sys.modules["mymod"] = fake_mod
            ...  # sys.modules is restored when the test exits

    The fixture saves the *keys* that existed before the test and removes any
    new keys added during the test.  It also restores the original object for
    any key that was present before and got overwritten, ensuring that a stub
    written by one test does not leak into the next.
    """
    snapshot: dict[str, object] = dict(sys.modules)
    yield
    # Remove keys added during the test
    added = set(sys.modules) - set(snapshot)
    for key in added:
        del sys.modules[key]
    # Restore keys that were overwritten or deleted
    for key, mod in snapshot.items():
        if sys.modules.get(key) is not mod:
            sys.modules[key] = mod

"""Integration tests for Mod3 bus wiring.

Verify that the ModalityBus singleton in jobs_registry.py is correctly
instantiated and wired through to http_api.py, with a VoiceModule
registered and all key APIs returning expected structures.

Run: python3 -m pytest tests/test_bus_wiring.py -v
"""

import json
import os
import sys

import pytest

# Ensure the project root is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_import_fails(module_name: str):
    """Return a pytest skip decorator if the given module cannot be imported."""
    try:
        __import__(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"{module_name} unavailable: {e}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bus_singleton_exists():
    """jobs_registry._bus exists and is a ModalityBus instance."""
    from bus import ModalityBus
    from jobs_registry import _bus

    assert _bus is not None, "_bus should not be None"
    assert isinstance(_bus, ModalityBus), f"_bus should be a ModalityBus, got {type(_bus).__name__}"


def test_bus_has_voice_module():
    """The server bus has a VoiceModule registered under ModalityType.VOICE."""
    from jobs_registry import _bus
    from modality import ModalityType
    from modules.voice import VoiceModule

    modules = getattr(_bus, "_modules", {})
    assert ModalityType.VOICE in modules, "Bus should have a VOICE module registered"
    voice_module = modules[ModalityType.VOICE]
    assert isinstance(voice_module, VoiceModule), (
        f"VOICE module should be VoiceModule, got {type(voice_module).__name__}"
    )
    # The server uses PlaceholderDecoder (no heavy model deps)
    assert voice_module.gate is not None, "VoiceModule should have a gate"
    assert voice_module.decoder is not None, "VoiceModule should have a decoder"
    assert voice_module.encoder is not None, "VoiceModule should have an encoder"


def test_bus_health_returns_dict():
    """_bus.health() returns a dict with modules, channels, queues, event_count."""
    from jobs_registry import _bus

    health = _bus.health()
    assert isinstance(health, dict), f"health() should return a dict, got {type(health).__name__}"

    expected_keys = {"modules", "channels", "queues", "event_count"}
    assert expected_keys.issubset(health.keys()), f"health() missing keys: {expected_keys - health.keys()}"

    # modules should contain at least 'voice'
    assert "voice" in health["modules"], "health() modules should include 'voice'"

    voice_health = health["modules"]["voice"]
    assert "has_gate" in voice_health, "voice health should report has_gate"
    assert "has_decoder" in voice_health, "voice health should report has_decoder"
    assert "has_encoder" in voice_health, "voice health should report has_encoder"
    assert voice_health["has_gate"] is True
    assert voice_health["has_decoder"] is True
    assert voice_health["has_encoder"] is True


def test_bus_hud_returns_dict():
    """_bus.hud() returns a dict with modules, channels, queues, recent_events."""
    from jobs_registry import _bus

    hud = _bus.hud()
    assert isinstance(hud, dict), f"hud() should return a dict, got {type(hud).__name__}"

    expected_keys = {"modules", "channels", "queues", "recent_events"}
    assert expected_keys.issubset(hud.keys()), f"hud() missing keys: {expected_keys - hud.keys()}"

    # modules should contain 'voice' with status info
    assert "voice" in hud["modules"], "hud() modules should include 'voice'"
    voice_hud = hud["modules"]["voice"]
    assert "status" in voice_hud, "voice HUD entry should have 'status'"
    assert voice_hud["status"] == "idle", f"voice status should be 'idle', got {voice_hud['status']}"

    # timestamp should be present and numeric
    assert "timestamp" in hud, "hud() should include a timestamp"
    assert isinstance(hud["timestamp"], (int, float)), "timestamp should be numeric"

    # recent_events should be a list
    assert isinstance(hud["recent_events"], list), "recent_events should be a list"

    # channels and queues should be dicts
    assert isinstance(hud["channels"], dict), "channels should be a dict"
    assert isinstance(hud["queues"], dict), "queues should be a dict"


def test_diagnostics_includes_bus():
    """The diagnostics() MCP tool response includes a 'bus' key with health and hud."""
    from jobs_registry import diagnostics

    raw = diagnostics()
    data = json.loads(raw)

    assert "bus" in data, "diagnostics() should include a 'bus' key"
    bus_data = data["bus"]

    assert "health" in bus_data, "bus section should include 'health'"
    assert "hud" in bus_data, "bus section should include 'hud'"

    # Verify nested structure is populated
    assert "modules" in bus_data["health"], "bus.health should have 'modules'"
    assert "modules" in bus_data["hud"], "bus.hud should have 'modules'"
    assert "voice" in bus_data["health"]["modules"], "bus health modules should include 'voice'"
    assert "voice" in bus_data["hud"]["modules"], "bus hud modules should include 'voice'"


def test_http_api_imports_bus():
    """http_api.py can import the bus from jobs_registry without circular import errors."""
    # This import itself is the test: http_api does
    #   from jobs_registry import _bus as _shared_bus
    # If there's a circular import, this will raise ImportError.
    # http_api also imports engine which may not be available, so we
    # handle that gracefully.
    try:
        from http_api import _bus as http_bus
    except ImportError as e:
        # If engine or another heavy dep is missing, that's OK for this test
        # as long as it's not a circular import error.
        if "circular" in str(e).lower():
            pytest.fail(f"Circular import detected: {e}")
        pytest.skip(f"http_api import failed (likely missing dep): {e}")
    except Exception as e:
        # Some deps (engine, sounddevice, etc.) may fail on import.
        # The key assertion is that it doesn't fail due to circular imports
        # between server.py and http_api.py.
        if "circular" in str(e).lower():
            pytest.fail(f"Circular import detected: {e}")
        pytest.skip(f"http_api import failed (non-circular): {e}")

    from bus import ModalityBus

    assert isinstance(http_bus, ModalityBus), f"http_api._bus should be a ModalityBus, got {type(http_bus).__name__}"

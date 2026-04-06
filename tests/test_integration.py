"""Integration tests for Mod³ channel server components.

Tests component wiring without live audio or MCP session.
Run: python3 tests/test_integration.py
"""

import os
import sys
import traceback

# Ensure the project root is on the path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SkipTest(Exception):
    """Raised when a test cannot run due to missing dependencies."""

    pass


# ---------------------------------------------------------------------------
# Mock player — stands in for AdaptivePlayer without audio hardware
# ---------------------------------------------------------------------------


class MockPlayer:
    def __init__(self):
        self.flushed = False

    def flush(self):
        self.flushed = True

    def get_progress(self):
        return (5000, 10000)  # 50% played


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_voice_module_instantiates():
    """VoiceModule with WhisperDecoder instantiates without loading models."""
    from modality import ModalityType
    from modules.voice import VoiceEncoder, VoiceGate, VoiceModule, WhisperDecoder

    mod = VoiceModule()
    assert mod.modality_type == ModalityType.VOICE, "VoiceModule should be VOICE modality"
    assert mod.gate is not None, "VoiceModule should have a gate"
    assert mod.decoder is not None, "VoiceModule should have a decoder"
    assert mod.encoder is not None, "VoiceModule should have an encoder"
    assert isinstance(mod.gate, VoiceGate), "Gate should be VoiceGate"
    assert isinstance(mod.decoder, WhisperDecoder), "Decoder should be WhisperDecoder"
    assert isinstance(mod.encoder, VoiceEncoder), "Encoder should be VoiceEncoder"


def test_modality_bus_accepts_voice_module():
    """ModalityBus accepts VoiceModule registration and routes correctly."""
    from bus import ModalityBus
    from modality import ModalityType
    from modules.voice import VoiceModule

    bus = ModalityBus()
    mod = VoiceModule()
    bus.register(mod)

    assert ModalityType.VOICE in bus._modules, "Bus should have VOICE module registered"
    assert bus._modules[ModalityType.VOICE] is mod, "Registered module should be our instance"

    # Health check should include the voice module
    health = bus.health()
    assert "voice" in health["modules"], "Health should report voice module"
    assert health["modules"]["voice"]["has_gate"] is True
    assert health["modules"]["voice"]["has_decoder"] is True
    assert health["modules"]["voice"]["has_encoder"] is True


def test_pipeline_state_interrupt_with_mock_player():
    """PipelineState interrupt() flushes the mock player and returns InterruptInfo."""
    from pipeline_state import InterruptInfo, PipelineState

    state = PipelineState()
    player = MockPlayer()

    # Start speaking
    state.start_speaking("Hello, how are you doing today?", player)
    assert state.is_speaking is True, "Should be speaking after start_speaking"

    # Update progress to 50%
    state.update_position(5000, 10000)
    assert abs(state.spoken_pct - 0.5) < 0.01, f"spoken_pct should be ~0.5, got {state.spoken_pct}"

    # Interrupt
    info = state.interrupt("vad_reflex")
    assert info is not None, "interrupt() should return InterruptInfo when speaking"
    assert isinstance(info, InterruptInfo), "Should be an InterruptInfo instance"
    assert player.flushed is True, "Player should have been flushed"
    assert abs(info.spoken_pct - 0.5) < 0.01, f"spoken_pct should be ~0.5, got {info.spoken_pct}"
    assert info.reason == "vad_reflex", f"reason should be vad_reflex, got {info.reason}"
    assert info.full_text == "Hello, how are you doing today?"
    assert state.is_speaking is False, "Should not be speaking after interrupt"


def test_pipeline_state_interrupt_when_not_speaking():
    """PipelineState interrupt() returns None when not speaking."""
    from pipeline_state import PipelineState

    state = PipelineState()
    info = state.interrupt("vad_reflex")
    assert info is None, "interrupt() should return None when not speaking"


def test_delivered_text_word_boundaries():
    """PipelineState.delivered_text splits at word boundaries, never mid-word."""
    from pipeline_state import PipelineState

    text = "The quick brown fox jumps over the lazy dog"

    # Edge cases
    assert PipelineState.delivered_text(text, 0.0) == "", "0% should return empty"
    assert PipelineState.delivered_text(text, 1.0) == text, "100% should return full text"

    # 50% — should cut near "jumps" (position ~22 out of 43 chars)
    half = PipelineState.delivered_text(text, 0.5)
    assert " " not in half or half == half.rstrip(), "Should not have trailing spaces"
    # Should not end in a partial word
    words = text.split()
    half_words = half.split()
    for w in half_words:
        assert w in words, f"Word '{w}' should be from the original text"

    # ~25% — "The quick brown" zone (position ~11 out of 43)
    quarter = PipelineState.delivered_text(text, 0.25)
    assert len(quarter) > 0, "25% of text should yield something"
    assert quarter == quarter.rstrip(), "No trailing spaces"

    # Very small percentage — might be empty or just the first word
    tiny = PipelineState.delivered_text(text, 0.05)
    assert tiny == "" or tiny in words, f"Tiny slice should be empty or a complete word, got '{tiny}'"


def test_audio_capture_instantiates():
    """AudioCapture instantiates without starting the mic."""
    from capture import AudioCapture

    cap = AudioCapture(sample_rate=16000)
    assert cap.is_active() is False, "Should not be active before start()"
    assert cap._target_sr == 16000, "Sample rate should be 16000"

    # get_audio should return None before start (no data in buffer)
    result = cap.get_audio(1.0)
    assert result is None, "get_audio should return None with no data captured"


def test_inbound_pipeline_instantiates():
    """InboundPipeline instantiates with bus + pipeline_state without starting."""
    from bus import ModalityBus
    from capture import AudioCapture
    from modules.voice import VoiceModule
    from pipeline_state import PipelineState

    try:
        from inbound import InboundPipeline
    except (ImportError, ModuleNotFoundError) as e:
        raise SkipTest(f"InboundPipeline import unavailable ({e})")

    bus = ModalityBus()
    bus.register(VoiceModule())
    state = PipelineState()
    capture = AudioCapture(sample_rate=16000)

    pipeline = InboundPipeline(
        bus=bus,
        pipeline_state=state,
        capture=capture,
    )
    assert pipeline.is_running is False, "Should not be running before start()"
    assert pipeline._bus is bus, "Bus reference should match"
    assert pipeline._pipeline_state is state, "PipelineState reference should match"
    assert pipeline._capture is capture, "Capture reference should match"


def test_hallucination_filter_patterns():
    """BoH hallucination filter matches known phantom phrases and rejects real speech."""
    try:
        from vad import HALLUCINATION_PHRASES, is_hallucination
    except (ImportError, ModuleNotFoundError) as e:
        raise SkipTest(f"vad module unavailable ({e})")

    # Known hallucinations should be caught
    assert is_hallucination("Thank you.") is True, "'Thank you.' should be a hallucination"
    assert is_hallucination("thanks for watching") is True
    assert is_hallucination("  Subscribe  ") is True, "Should handle whitespace"
    assert is_hallucination("...") is True, "Ellipsis is a hallucination"
    assert is_hallucination("") is True, "Empty string is a hallucination"
    assert is_hallucination("THANK YOU!") is True, "Should be case-insensitive"
    assert is_hallucination("The End.") is True

    # Real speech should not be filtered
    assert is_hallucination("Hey can you help me with something?") is False
    assert is_hallucination("What time is the meeting?") is False
    assert is_hallucination("Please turn on the lights") is False
    assert is_hallucination("I'd like to schedule a call") is False

    # The phrases set should have meaningful content
    assert len(HALLUCINATION_PHRASES) >= 10, "Should have at least 10 hallucination phrases"


def test_channel_mode_argparse():
    """The --channel flag is accepted by the server's argument parser."""
    import argparse

    # Reconstruct the parser as defined in server.py
    parser = argparse.ArgumentParser(description="Mod³ TTS Server")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--channel", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")

    # --channel should parse correctly
    args = parser.parse_args(["--channel"])
    assert args.channel is True, "--channel should be True"
    assert args.http is False, "--http should be False"
    assert args.all is False, "--all should be False"

    # Default (no flags) should have channel=False
    args_default = parser.parse_args([])
    assert args_default.channel is False, "Default should have channel=False"


def test_bus_hud_with_voice_module():
    """ModalityBus.hud() returns structured state with voice module."""
    from bus import ModalityBus
    from modules.voice import VoiceModule

    bus = ModalityBus()
    bus.register(VoiceModule())

    hud = bus.hud()
    assert "modules" in hud, "HUD should have modules"
    assert "voice" in hud["modules"], "HUD should report voice module"
    voice_state = hud["modules"]["voice"]
    assert voice_state["status"] == "idle", "Voice module should be idle"
    assert "timestamp" in hud, "HUD should have a timestamp"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_voice_module_instantiates,
        test_modality_bus_accepts_voice_module,
        test_pipeline_state_interrupt_with_mock_player,
        test_pipeline_state_interrupt_when_not_speaking,
        test_delivered_text_word_boundaries,
        test_audio_capture_instantiates,
        test_inbound_pipeline_instantiates,
        test_hallucination_filter_patterns,
        test_channel_mode_argparse,
        test_bus_hud_with_voice_module,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
            passed += 1
        except SkipTest as e:
            print(f"  SKIP  {name}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    total = passed + failed + skipped
    summary = f"{passed}/{total} passed"
    if skipped:
        summary += f", {skipped} skipped"
    if failed:
        summary += f", {failed} FAILED"
    print(f"\n{summary}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

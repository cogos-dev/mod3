#!/usr/bin/env bash
# Mod³ — Model Modality Modulator
# One-command setup for the TTS MCP server on Apple Silicon
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Mod³ Setup ==="

# Check Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: Mod³ is optimized for Apple Silicon (arm64). Performance may vary on other architectures."
fi

# System dependency: espeak-ng (required for Kokoro phonemizer)
if ! command -v espeak-ng &>/dev/null; then
    echo "Installing espeak-ng via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install espeak-ng
    else
        echo "Error: espeak-ng is required but Homebrew is not installed."
        echo "Install Homebrew (https://brew.sh) then run: brew install espeak-ng"
        exit 1
    fi
fi

# Python venv
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
fi

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Models are downloaded on first use. Available engines:"
echo "  - Kokoro (82M)      — fast, speed control, emphasis via caps"
echo "  - Voxtral (4B)      — highest quality, no control surfaces"
echo "  - Chatterbox (~1B)  — emotion/exaggeration control"
echo "  - Spark (0.5B)      — pitch + speed + gender control"
echo ""
echo "To use with Claude Code, add to your .mcp.json:"
echo "  {\"mcpServers\": {\"tts\": {\"command\": \"$VENV_DIR/bin/python\", \"args\": [\"$SCRIPT_DIR/server.py\"]}}}"
echo ""
echo "Or copy .mcp.json to your project root and update paths."

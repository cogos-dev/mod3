# Contributing to Mod³

Thanks for your interest in Mod³. This document covers local development, testing, and PR workflow.

## Development setup

```sh
git clone https://github.com/cogos-dev/mod3.git
cd mod3
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ruff pyright pytest
```

Requirements: Python 3.12+, macOS (Apple Silicon recommended for the TTS engines).

## Running tests

```sh
ruff check .
ruff format --check .
pyright --pythonversion 3.12 engine.py vad.py http_api.py server.py
pytest tests/ -v
```

CI runs all of the above on every PR (`.github/workflows/ci.yml`). Match the versions used there.

## Project layout

- `engine.py` — TTS engine dispatch (kokoro, piper, espeak-ng, system)
- `vad.py` — voice-activity detection + hallucination filtering
- `http_api.py` — FastAPI app exposing `/v1/jobs`, `/v1/voices`, `/v1/filter`
- `server.py` — MCP server entry point
- `tests/` — pytest suite
- `integrations/openclaw/` — OpenClaw plugin manifest and extension

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run the full test suite above
4. Update `CHANGELOG.md` under the Unreleased section
5. Open a pull request using the org PR template

Commit messages: conventional-commit style (`feat:`, `fix:`, `chore:`, etc.) is preferred but not enforced.

## Reporting issues

Use the org-level [Bug Report](https://github.com/cogos-dev/mod3/issues/new?template=bug.yml) or [Feature Request](https://github.com/cogos-dev/mod3/issues/new?template=feature.yml) forms.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

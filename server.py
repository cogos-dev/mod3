"""Mod³ TTS Server — CLI entrypoint for the MCP process.

Multi-model support: Voxtral, Kokoro, Chatterbox, Spark.
Voice presets are resolved to the correct engine automatically.

Interfaces:
  HTTP (--http):  REST API + HTTP-MCP at /mcp (canonical transport)
  HTTP (default): same as --http when invoked without flags (stdio deprecated)
  stdio (--all, no-args): deprecated — see issue #11 and README

Channel client: use clients/channel_client.py (separate stdio process) — see CHANNELS.md.

This file is intentionally thin: argument parsing, the HTTP bootstrap
(``_run_http``), and mounting FastMCP's streamable-HTTP transport
(``install_mcp_route``). Everything stateful — the job registry, the speech
queue, the modality bus, pipeline (barge-in) state, and the MCP tool
definitions themselves (speak, stop, speech_status, diagnostics, ...) — lives
in jobs_registry.py.

That split exists because this file is also the ``__main__`` entrypoint
(``python server.py --http``), and http_api.py needs to read/act on the same
runtime state. A module run as ``__main__`` is never the same module object
as one imported normally by its own name — importing "from server import X"
from http_api.py used to silently re-execute this entire file's top level as
a second, independent module, with its own job registry, speech queue, and
barge-in watcher thread (2026-07-23 incident; see jobs_registry.py's module
docstring for the full mechanism). jobs_registry.py is never run as
``__main__``, so it is only ever imported once, under one name, regardless
of who imports it first — that is what makes it safe to share.
"""

import logging
import os
import threading
import time
import warnings
from typing import Any

from jobs_registry import _bargein_registry, _bus, logger, mcp, pipeline_state


def install_mcp_route(app) -> None:
    """Mount FastMCP's streamable-HTTP transport at /mcp and start its session manager.

    FastMCP's StreamableHTTPSessionManager needs an async task group entered inside
    the app lifespan; mounting alone yields 500s with "Task group is not initialized".
    http_api.app uses a `lifespan=` context manager, which makes legacy
    `@app.on_event("startup")` hooks silent no-ops — so we wrap the existing
    lifespan instead. Tested by tests/test_mcp_route.py.

    Note: the `mcp` instance is a module-level singleton, and
    `session_manager.run()` is not re-entrant. Calling this helper a second time
    would raise a cryptic RuntimeError at lifespan startup, so we guard with an
    explicit error here. Tests that need a fresh app should reuse the same
    install — a module-scoped fixture is the canonical pattern.

    Note: the FastMCP sub-app returned by `streamable_http_app()` has its own
    lifespan that calls `session_manager.run()`, but Starlette does not
    propagate mounted sub-app lifespans to the parent — that is why we wrap
    explicitly. If a future FastMCP release moves session startup outside the
    sub-app lifespan (or adds parent-lifespan propagation), this helper will
    need to be revisited.
    """
    from contextlib import asynccontextmanager

    if getattr(mcp, "_mod3_route_installed", False):
        raise RuntimeError(
            "install_mcp_route() called more than once on the same FastMCP "
            "singleton — session_manager.run() is not re-entrant. Reuse the "
            "first-installed app (e.g. via a module-scoped pytest fixture)."
        )

    app.mount("/mcp", mcp.streamable_http_app())

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _combined_lifespan(application):
        async with mcp.session_manager.run():
            async with original_lifespan(application):
                yield

    app.router.lifespan_context = _combined_lifespan
    setattr(mcp, "_mod3_route_installed", True)  # noqa: B010 — dynamic sentinel, not a declared attribute


def _start_inbound_pipeline_if_enabled() -> Any | None:
    """Start the server-side inbound mic → VAD → STT pipeline (env-gated).

    Default-on so the MCP/HTTP-tier path gets bidirectional audio without
    manual opt-in; the dashboard already has its own in-browser VAD so this
    is purely additive. Disable with ``MOD3_INBOUND_ENABLED=0``.

    Returns the started InboundPipeline (so the caller can stop it on
    shutdown) or ``None`` when disabled / import failed. Mic access errors
    are swallowed at the warning level — server startup must not abort just
    because no microphone is present.
    """
    raw = os.environ.get("MOD3_INBOUND_ENABLED", "1").strip().lower()
    enabled = raw not in {"0", "false", "no", "off"}
    if not enabled:
        logger.info("MOD3_INBOUND_ENABLED=%s — inbound pipeline disabled", raw)
        return None

    try:
        from inbound import InboundPipeline

        pipeline = InboundPipeline(
            bus=_bus,
            pipeline_state=pipeline_state,
            bargein_registry=_bargein_registry,
        )
        pipeline.start()
        logger.info("inbound voice pipeline started (mic → VAD → STT)")
        return pipeline
    except Exception:
        logger.warning("inbound pipeline failed to start; continuing without mic input", exc_info=True)
        return None


def _prewarm_tts_if_enabled() -> None:
    """Fire-and-forget Kokoro pre-warm so the first real synthesize call is fast.

    First-time Kokoro init can take ~60s; deferring that to the first
    user-facing synthesize causes the OutputQueue to stall and the per-job
    delivery timer to fire on older jobs. Doing one throwaway synthesize on
    a background thread at boot pays the cold-start cost up front. Disable
    with ``MOD3_PREWARM_TTS=0``.
    """
    raw = os.environ.get("MOD3_PREWARM_TTS", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        logger.info("MOD3_PREWARM_TTS=%s — Kokoro pre-warm disabled", raw)
        return

    def _warm():
        try:
            from engine import synthesize

            t0 = time.time()
            synthesize("warmup", voice="bm_lewis", speed=1.25)
            logger.info("Kokoro pre-warm complete in %.1fs", time.time() - t0)
        except Exception:
            logger.warning("Kokoro pre-warm failed; first real synthesize may be slow", exc_info=True)

    threading.Thread(target=_warm, name="kokoro-prewarm", daemon=True).start()


def _run_http(host: str = "127.0.0.1", port: int = 7860):
    """Start the HTTP API server with MCP streamable-HTTP mounted at /mcp."""
    import uvicorn

    from http_api import app

    install_mcp_route(app)
    inbound_pipeline: Any | None = None
    try:
        inbound_pipeline = _start_inbound_pipeline_if_enabled()
        _prewarm_tts_if_enabled()
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        if inbound_pipeline is not None:
            try:
                inbound_pipeline.stop()
            except Exception:
                logger.debug("inbound pipeline stop raised", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mod³ TTS Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP API only")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run both MCP (stdio) and HTTP [deprecated — use HTTP-MCP: python server.py --http, then connect via /mcp]",
    )
    parser.add_argument("--dashboard", action="store_true", help="Run HTTP API with voice/text dashboard (no MCP)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="HTTP bind address (default: 127.0.0.1 loopback; use 0.0.0.0 to expose on LAN)",
    )
    args = parser.parse_args()

    _STDIO_DEPRECATION_MSG = (
        "stdio MCP transport is deprecated; prefer HTTP-MCP at /mcp (see README). "
        "This path will be removed in a future release."
    )

    if args.http:
        _run_http(host=args.host, port=args.port)
    elif args.all:
        # HTTP in background thread, MCP on stdio
        warnings.warn(_STDIO_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        http_thread = threading.Thread(
            target=_run_http,
            kwargs={"host": args.host, "port": args.port},
            daemon=True,
        )
        http_thread.start()
        mcp.run()
    elif args.dashboard:
        # Dashboard mode: HTTP server with WebSocket voice/text chat
        # Swap PlaceholderDecoder → WhisperDecoder for real STT
        from modules.text import TextModule
        from modules.voice import VoiceModule, WhisperDecoder

        _bus._modules.clear()
        _bus.register(VoiceModule(decoder=WhisperDecoder()))
        _bus.register(TextModule())
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting dashboard mode (WhisperDecoder enabled)")
        _run_http(host=args.host, port=args.port)
    else:
        warnings.warn(_STDIO_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        mcp.run()

"""Regression test for the /mcp streamable-HTTP route wiring.

Guards against the failure mode introduced when http_api.app migrated from
@app.on_event hooks to a lifespan= context manager (commit ba5e8e9): the
legacy startup hook that entered FastMCP's session_manager.run() was silently
ignored, so every POST to /mcp/ returned 500 with
"RuntimeError: Task group is not initialized. Make sure to use run()."

This test calls server.install_mcp_route() — the same helper _run_http() uses
in production — and exercises the MCP initialize handshake through TestClient.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def mcp_client():
    """Module-scoped: FastMCP's session_manager.run() is not re-entrant on the
    shared singleton, so we mount once for the whole test module."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from server import install_mcp_route

    app = FastAPI()
    install_mcp_route(app)

    # FastMCP's StreamableHTTP transport auto-enables DNS-rebinding protection
    # whose allowlist patterns ("127.0.0.1:*") require a port in the Host
    # header. TestClient's default base_url ("http://testserver") and a
    # port-less host both fail with 421. Pin to 127.0.0.1 with any port.
    with TestClient(app, base_url="http://127.0.0.1:7860") as client:
        yield client


def test_mcp_initialize_handshake_succeeds(mcp_client):
    """POST /mcp/ initialize must return 200, not 500.

    Failure mode: if install_mcp_route() stops entering FastMCP's
    session_manager during lifespan startup, the StreamableHTTPSessionManager
    raises "Task group is not initialized" on every request.
    """
    response = mcp_client.post(
        "/mcp/",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "regression-test", "version": "1"},
            },
        },
    )

    assert response.status_code == 200, (
        f"MCP initialize returned {response.status_code}: {response.text}\n"
        "Likely cause: FastMCP session_manager.run() was not entered during "
        "lifespan startup. See server.install_mcp_route()."
    )
    assert "mcp-session-id" in response.headers, (
        "MCP server did not mint a session id; session manager probably never entered its task group."
    )
    assert "Task group is not initialized" not in response.text


def test_mcp_route_mounted(mcp_client):
    """The /mcp redirect must be present (sanity: helper actually mounted)."""
    response = mcp_client.get("/mcp", follow_redirects=False)
    assert response.status_code in (200, 307), response.text

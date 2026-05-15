"""Thin MCP server — wraps agent.tools over the MCP stdio transport.

Zero analysis logic lives here. `agent.tools.build_tools` already
produces the catalog and `as_mcp_specs` the spec shape; this module
only bridges that to the `mcp` package's server API and routes calls
back to each Tool's Python callable.

Behind the [mcp] extra. `from otter_docs.mcp import serve` raises
ImportError with an actionable message when `mcp` isn't installed —
the CLI catches that and tells the user how to fix it.

This is the in-repo equivalent of the separate `otter-docs-mcp`
package mentioned in the spec; shipping it inside the library for
v0.1 avoids a second release pipeline. If it grows past "thin", it
moves out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def serve(repo_root: Path) -> None:
    """Run an MCP stdio server exposing otter-docs tools for `repo_root`.

    Blocks until the transport closes. Tools that need an LLM/embedder
    are advertised but raise on call (the library is offline-only from
    the MCP surface in v0.1 — hosts that want enrichment drive the
    library directly).
    """
    try:
        import anyio
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        import mcp.types as types
    except ImportError as e:  # pragma: no cover - exercised via CLI message
        raise ImportError(
            "The MCP server needs the optional `mcp` package.\n"
            "Install with:  pip install otter-docs[mcp]"
        ) from e

    from otter_docs import Repo
    from otter_docs.agent.tools import build_tools

    repo = Repo(repo_root)
    tools = {t.name: t for t in build_tools(repo)}
    server: Any = Server("otter-docs")

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=t.name,
                description=t.description,
                inputSchema=t.input_schema,
            )
            for t in tools.values()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        tool = tools.get(name)
        if tool is None:
            return [types.TextContent(type="text", text=f"unknown tool: {name}")]
        try:
            result = tool.call(**(arguments or {}))
        except Exception as e:  # noqa: BLE001 — surface tool errors to the host
            return [types.TextContent(type="text", text=f"error: {type(e).__name__}: {e}")]
        import json
        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    try:
        anyio.run(_run)
    finally:
        repo.close()

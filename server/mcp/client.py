"""MCP (Model Context Protocol) client wrapper.

Connects to MCP servers via Streamable HTTP (current spec), falling back to
the legacy SSE transport when the URL or handshake requires it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from server.mcp.transport import (
    STREAMABLE_HTTP,
    mcp_tool_requires_confirmation,
    mcp_transport_order,
    sanitize_mcp_identifier,
)

logger = logging.getLogger(__name__)

# Bound remote MCP tool metadata before it reaches the model's tool list.
_MAX_TOOL_DESCRIPTION_CHARS = 2000
_MAX_TOOL_SCHEMA_BYTES = 16384
_CONNECT_TIMEOUT_SECONDS = 15

# Global cache for MCP tools (synchronous access). None = never refreshed.
_mcp_tools_cache: list[dict[str, Any]] | None = None
# Global cache for MCP server info (server_id -> info dict)
_mcp_server_info_cache: dict[int, dict[str, Any]] = {}
_refresh_lock: asyncio.Lock | None = None


def _get_refresh_lock() -> asyncio.Lock:
    global _refresh_lock
    if _refresh_lock is None:
        _refresh_lock = asyncio.Lock()
    return _refresh_lock


def _guarded_httpx_client_factory():
    """httpx factory for MCP: pins every request to a validated resolved IP.

    The MCP SDK builds its own httpx clients; injecting a guarded client here
    closes the DNS-rebinding window between ``validate_fetch_url`` and the
    actual connection, and disables redirects the SDK enables by default.
    ``auth`` is forwarded to the client (httpx applies it per request).
    """

    def factory(headers=None, timeout=None, auth=None):
        import httpx

        from server.utils.ssrf import build_guarded_http_client

        kwargs: dict = {}
        if headers is not None:
            kwargs["headers"] = headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        if auth is not None:
            kwargs["auth"] = auth
        return build_guarded_http_client(**kwargs)

    return factory


async def _enter_transport(stack: AsyncExitStack, url: str, transport_name: str):
    """Open a read/write MCP transport and enter it on ``stack``."""
    factory = _guarded_httpx_client_factory()
    if transport_name == STREAMABLE_HTTP:
        from mcp.client.streamable_http import streamablehttp_client

        try:
            transport = await stack.enter_async_context(
                streamablehttp_client(
                    url,
                    timeout=_CONNECT_TIMEOUT_SECONDS,
                    httpx_client_factory=factory,
                )
            )
        except TypeError:
            transport = await stack.enter_async_context(
                streamablehttp_client(
                    url,
                    timeout=_CONNECT_TIMEOUT_SECONDS,
                    httpx_client_factory=factory,
                )
            )
        return transport[0], transport[1]

    from mcp.client.sse import sse_client

    try:
        transport = await stack.enter_async_context(
            sse_client(
                url,
                timeout=_CONNECT_TIMEOUT_SECONDS,
                httpx_client_factory=factory,
            )
        )
    except TypeError:
        transport = await stack.enter_async_context(
            sse_client(
                url,
                timeout=_CONNECT_TIMEOUT_SECONDS,
                httpx_client_factory=factory,
            )
        )
    return transport[0], transport[1]


class McpServerClient:
    """Client for a single MCP server (Streamable HTTP, with SSE fallback)."""

    def __init__(self, server_config: dict[str, Any]) -> None:
        """Initialize the MCP server client.

        Args:
            server_config: Server configuration from mcp_manager
        """
        self.server_config = server_config
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._tools_cache: list[dict[str, Any]] | None = None
        self._server_info: dict[str, Any] | None = None
        self._transport_name: str | None = None

    async def connect(self) -> bool:
        """Connect to the MCP server, trying Streamable HTTP then SSE.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            url = self.server_config.get("url")
            if not url:
                logger.error(f"No URL configured for MCP server '{self.server_config['name']}'")
                return False

            # SSRF guard: allow local/LAN MCP servers (legitimate), but block
            # non-http(s) schemes and cloud-metadata/link-local targets.
            from server.utils.ssrf import validate_fetch_url

            await asyncio.to_thread(validate_fetch_url, url)

            last_error: Exception | None = None
            for transport_name in mcp_transport_order(url):
                stack = AsyncExitStack()
                try:
                    read, write = await _enter_transport(stack, url, transport_name)
                    session = await stack.enter_async_context(ClientSession(read, write))
                    init_result = await session.initialize()

                    self.exit_stack = stack
                    self.session = session
                    self._transport_name = transport_name

                    if init_result and hasattr(init_result, "serverInfo"):
                        self._server_info = {
                            "name": getattr(init_result.serverInfo, "name", "Unknown"),
                            "version": getattr(init_result.serverInfo, "version", ""),
                        }
                    else:
                        self._server_info = {
                            "name": self.server_config.get("name", "Unknown"),
                            "version": "",
                        }

                    server_id = self.server_config.get("id")
                    if server_id and self._server_info:
                        global _mcp_server_info_cache
                        _mcp_server_info_cache[server_id] = self._server_info

                    logger.info(
                        "Connected to MCP server '%s' (%s v%s) via %s",
                        self.server_config["name"],
                        self._server_info.get("name", "Unknown"),
                        self._server_info.get("version", "?"),
                        transport_name,
                    )
                    return True
                except Exception as error:
                    last_error = error
                    logger.info(
                        "MCP %s handshake failed for '%s': %s",
                        transport_name,
                        self.server_config.get("name"),
                        error,
                    )
                    with contextlib.suppress(Exception):
                        await stack.aclose()

            logger.error(
                "Failed to connect to MCP server '%s': %s",
                self.server_config["name"],
                last_error,
            )
            return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{self.server_config['name']}': {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            await self.exit_stack.aclose()
            self.session = None
            self._tools_cache = None
            self._transport_name = None
            logger.info(f"Disconnected from MCP server '{self.server_config['name']}'")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server.

        Returns:
            List of tool definitions
        """
        if not self.session:
            await self.connect()

        if not self.session:
            return []

        try:
            response = await self.session.list_tools()
            tools = []
            server_slug = sanitize_mcp_identifier(self.server_config["name"])

            for tool in response.tools:
                # Never pass remote MCP metadata through verbatim: the
                # description and schema go straight into the model's tool
                # list. Malformed values break the OpenAI-compatible API and
                # oversized ones bloat the prompt (or smuggle instructions).
                description = str(tool.description or "").strip()
                if len(description) > _MAX_TOOL_DESCRIPTION_CHARS:
                    description = description[:_MAX_TOOL_DESCRIPTION_CHARS] + "…"

                input_schema = tool.inputSchema
                if not isinstance(input_schema, dict):
                    input_schema = {"type": "object", "properties": {}}
                else:
                    try:
                        serialized = len(json.dumps(input_schema))
                    except (TypeError, ValueError):
                        serialized = _MAX_TOOL_SCHEMA_BYTES + 1
                    if serialized > _MAX_TOOL_SCHEMA_BYTES:
                        logger.warning(
                            "MCP tool '%s' schema too large (%d bytes); "
                            "replaced with an empty object schema",
                            tool.name,
                            serialized,
                        )
                        input_schema = {"type": "object", "properties": {}}

                tool_slug = sanitize_mcp_identifier(tool.name)
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"mcp_{server_slug}_{tool_slug}",
                            "description": description,
                            "parameters": input_schema,
                        },
                        "_mcp_server_id": self.server_config["id"],
                        "_mcp_tool_name": tool.name,
                        "_mcp_requires_confirmation": mcp_tool_requires_confirmation(tool),
                    }
                )

            self._tools_cache = tools
            return tools

        except Exception as e:
            logger.error(f"Error listing tools from MCP server: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call (without the mcp_ prefix)
            arguments: Arguments to pass to the tool

        Returns:
            The tool's response
        """
        if not self.session:
            await self.connect()

        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        try:
            response = await self.session.call_tool(tool_name, arguments)
            return response
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on MCP server: {e}")
            raise

    def get_server_info(self) -> dict[str, Any] | None:
        """Get the server info from the initialization response.

        Returns:
            Dict with 'name' and 'version' keys, or None if not connected
        """
        return self._server_info


async def get_mcp_tools() -> list[dict[str, Any]]:
    """Get all available tools from enabled MCP servers.

    Returns:
        List of tool definitions from all enabled MCP servers
    """
    from server.database.config.mcp_manager import mcp_config_manager

    tools = []
    servers = mcp_config_manager.get_enabled_servers()

    for server_config in servers:
        client = McpServerClient(server_config)
        try:
            server_tools = await client.list_tools()
            tools.extend(server_tools)
        except Exception as e:
            logger.error(f"Failed to get tools from MCP server '{server_config['name']}': {e}")
        finally:
            await client.disconnect()

    global _mcp_tools_cache
    _mcp_tools_cache = tools

    return tools


def get_mcp_tools_sync() -> list[dict[str, Any]]:
    """Get MCP tools from the synchronous cache.

    Returns:
        List of cached tool definitions (empty if never refreshed)
    """
    return list(_mcp_tools_cache or [])


async def refresh_mcp_tools_cache() -> None:
    """Refresh the global MCP tools cache."""
    await get_mcp_tools()


async def ensure_mcp_tools_cache(*, force: bool = False) -> None:
    """Fill the MCP tools cache if it has never been loaded.

    Safe to call on every chat turn: a completed refresh (even one that found
    zero tools) is not repeated unless ``force`` is set.
    """
    global _mcp_tools_cache
    if not force and _mcp_tools_cache is not None:
        return
    async with _get_refresh_lock():
        if not force and _mcp_tools_cache is not None:
            return
        try:
            await get_mcp_tools()
        except Exception:
            logger.exception("Failed to refresh MCP tools cache")
            if _mcp_tools_cache is None:
                _mcp_tools_cache = []


async def call_mcp_tool(server_id: int, tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call a tool on a specific MCP server.

    Args:
        server_id: ID of the MCP server
        tool_name: Name of the tool to call (without namespace prefix)
        arguments: Arguments to pass to the tool

    Returns:
        The tool's response
    """
    from server.database.config.mcp_manager import mcp_config_manager

    server_config = mcp_config_manager.get_server(server_id)
    if not server_config:
        raise ValueError(f"MCP server {server_id} not found")

    client = McpServerClient(server_config)
    try:
        if not await client.connect():
            raise RuntimeError(f"Failed to connect to MCP server {server_id}")

        return await client.call_tool(tool_name, arguments)
    finally:
        await client.disconnect()

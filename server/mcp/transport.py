"""MCP transport selection helpers (no MCP SDK import)."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

STREAMABLE_HTTP = "streamable_http"
SSE = "sse"


def sanitize_mcp_identifier(name: str) -> str:
    """Make a server or tool name safe for OpenAI-style function names."""
    ascii_name = (name or "").lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    ascii_name = re.sub(r"[^a-z0-9_]", "_", ascii_name)
    ascii_name = re.sub(r"_+", "_", ascii_name).strip("_")
    return ascii_name or "tool"


def _annotation_flag(annotations: Any, name: str) -> Any:
    if annotations is None:
        return None
    if isinstance(annotations, dict):
        return annotations.get(name)
    return getattr(annotations, name, None)


def mcp_tool_requires_confirmation(tool: Any) -> bool:
    """True when the tool may not be implicitly read-only.

    Fail closed: most MCP servers ship no annotations at all, so the tool's
    effect is unknown. An unknown tool must not run without the user's
    explicit approval (LLM01/independent-of-frameworks: an LLM calling an
    unclassified MCP tool is an under-specified authorization boundary). Only
    an explicit ``readOnlyHint: true`` allows the tool to run directly.
    """
    annotations = getattr(tool, "annotations", None)
    if annotations is None:
        return True
    if _annotation_flag(annotations, "destructiveHint") is True:
        return True
    read_only = _annotation_flag(annotations, "readOnlyHint")
    if read_only is True:
        return False
    # readOnlyHint absent or non-boolean: unknown -> require confirmation.
    return True


def mcp_transport_order(url: str) -> list[str]:
    """Preferred MCP transports for ``url``.

    Streamable HTTP is the current MCP default. URLs whose path ends in
    ``/sse`` are treated as classic SSE servers and tried that way first.
    """
    path = urlparse(url or "").path.lower().rstrip("/")
    if path.endswith("/sse"):
        return [SSE, STREAMABLE_HTTP]
    return [STREAMABLE_HTTP, SSE]

"""OpenAI Responses API adapter.

The Responses API is the recommended OpenAI generation API.  Phlox keeps its
small internal ``{"message": ...}`` result shape so the chat/tool engine does
not need provider-specific branches.  Streaming remains on the Chat
Completions adapter because its chunks are already consumed by the UI.
"""

from __future__ import annotations

import json
from typing import Any


def _response_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or tool
    result = {
        "type": "function",
        "name": function.get("name") or "",
        "description": function.get("description") or "",
        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
    }
    if "strict" in function:
        result["strict"] = function["strict"]
    return result


def _response_content(content: Any) -> Any:
    """Translate Chat Completions multimodal blocks to Responses input blocks."""
    if not isinstance(content, list):
        return content or ""
    converted: list[Any] = []
    for block in content:
        if isinstance(block, str):
            converted.append({"type": "input_text", "text": block})
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            converted.append({"type": "input_text", "text": block.get("text") or ""})
        elif block.get("type") == "image_url":
            image_url = block.get("image_url") or {}
            converted.append({"type": "input_image", "image_url": image_url.get("url") or ""})
    return converted or ""


def _response_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert legacy chat tool and multimodal messages to Responses items."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system" or role == "developer":
            content = message.get("content")
            if isinstance(content, str) and content:
                instructions.append(content)
            continue
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = message.get("content") or ""
            if content:
                items.append({"role": "assistant", "content": content})
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            continue
        # Normal user/assistant messages are accepted as Responses input items.
        items.append({"role": role or "user", "content": _response_content(message.get("content"))})
    return "\n\n".join(instructions), items


def _output_to_message(response: Any, model: str) -> dict[str, Any]:
    text = getattr(response, "output_text", None) or ""
    tool_calls: list[dict[str, Any]] = []
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type != "function_call":
            continue
        get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
        tool_calls.append(
            {
                "id": get("call_id", "") or get("id", ""),
                "type": "function",
                "function": {
                    "name": get("name", ""),
                    "arguments": get("arguments", "{}") or "{}",
                },
            }
        )
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": model, "message": message}


async def openai_responses_chat(
    client,
    model: str,
    messages: list[dict[str, Any]],
    format: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    instructions, input_items = _response_input(messages)
    params: dict[str, Any] = {"model": model, "input": input_items}
    if instructions:
        params["instructions"] = instructions
    options = options or {}
    if "temperature" in options:
        params["temperature"] = options["temperature"]
    if "max_tokens" in options or "num_predict" in options:
        params["max_output_tokens"] = int(options.get("max_tokens") or options["num_predict"])
    if "stop" in options:
        params["stop"] = options["stop"]
    if format:
        params["text"] = {
            "format": {
                "type": "json_schema",
                "name": format.get("title", "response"),
                "schema": format,
                "strict": True,
            }
        }
    if tools:
        params["tools"] = [_response_tool(tool) for tool in tools]
        if options.get("force_tools"):
            params["tool_choice"] = "required"
    response = await client.responses.create(**params)
    return _output_to_message(response, model)

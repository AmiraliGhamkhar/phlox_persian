"""Native Anthropic Messages API provider.

Uses httpx against ``/v1/messages`` so we do not add an extra SDK dependency.
Streaming, tools, vision blocks, and JSON-schema-style structured output
(via a system instruction) are supported.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from server.utils.ssrf import build_guarded_http_client

from server.utils.http_retry import (
    ProviderHTTPError,
    sanitize_provider_error,
    with_retries,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:"):
        return None
    header, separator, data = url.partition(",")
    if not separator or not data:
        return None
    media_type = header[5:].split(";")[0].strip().lower() or "image/png"
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    if media_type not in _ALLOWED_IMAGE_TYPES:
        media_type = "image/png"
    return media_type, data


def _image_url_to_block(url: str) -> dict[str, Any] | None:
    parsed = _parse_data_url(url)
    if parsed:
        media_type, data = parsed
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _user_content_to_anthropic(content: Any) -> Any:
    """Keep OpenAI-style vision blocks as Anthropic image sources."""
    if isinstance(content, str) or content is None:
        return content or ""
    if not isinstance(content, list):
        return _content_to_text(content)
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item:
                blocks.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = str(item.get("text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif item_type == "image_url":
            url = str((item.get("image_url") or {}).get("url") or item.get("url") or "")
            block = _image_url_to_block(url)
            if block:
                blocks.append(block)
        elif item_type == "image":
            blocks.append(item)
    return blocks or ""


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Split OpenAI-style messages into Anthropic system + messages."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            system_parts.append(_content_to_text(message.get("content")))
            continue
        if role == "tool":
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "",
                "content": _content_to_text(message.get("content")),
            }
            # Anthropic wants every tool_result for a turn in a single user
            # message. Merge consecutive tool role messages.
            if (
                converted
                and converted[-1]["role"] == "user"
                and isinstance(converted[-1].get("content"), list)
                and converted[-1]["content"]
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in converted[-1]["content"]
                )
            ):
                converted[-1]["content"].append(tool_result_block)
            else:
                converted.append({"role": "user", "content": [tool_result_block]})
            continue
        if role == "assistant":
            content_blocks: list[dict[str, Any]] = []
            text = _content_to_text(message.get("content"))
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments") or "{}"
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    parsed = {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id") or "",
                        "name": function.get("name") or "",
                        "input": parsed or {},
                    }
                )
            converted.append({"role": "assistant", "content": content_blocks or ""})
            continue
        converted.append(
            {"role": "user", "content": _user_content_to_anthropic(message.get("content"))}
        )

    # Anthropic requires the first message to be from the user.
    while converted and converted[0]["role"] != "user":
        converted.pop(0)
    return "\n\n".join(part for part in system_parts if part), converted


def convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for tool in tools:
        function = tool.get("function") or tool
        converted.append(
            {
                "name": function.get("name") or "",
                "description": function.get("description") or "",
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    body = sanitize_provider_error(response.text)
    status = response.status_code
    if status == 401:
        raise ProviderHTTPError(
            f"Anthropic authentication failed (401): {body}", status_code=status
        )
    if status == 403:
        raise ProviderHTTPError(f"Anthropic request forbidden (403): {body}", status_code=status)
    if status == 429:
        raise ProviderHTTPError(f"Anthropic rate limited (429): {body}", status_code=status)
    if status >= 500:
        raise ProviderHTTPError(f"Anthropic provider error ({status}): {body}", status_code=status)
    raise ProviderHTTPError(f"Anthropic request failed ({status}): {body}", status_code=status)


def _message_to_ollama(model: str, data: dict[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    result: dict[str, Any] = {"model": model, "message": message}
    return result


async def anthropic_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    format: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    stream: bool = False,
    timeout: int = 80,
) -> dict[str, Any] | AsyncGenerator:
    """Send a chat request to the Anthropic Messages API."""
    system, converted = convert_messages(messages)
    if format:
        schema_text = json.dumps(format)
        instruction = (
            "Respond with a JSON object that matches this schema. "
            "Do not wrap the JSON in markdown. Schema: "
            f"{schema_text}"
        )
        system = f"{system}\n\n{instruction}" if system else instruction

    options = options or {}
    payload: dict[str, Any] = {
        "model": model,
        "messages": converted or [{"role": "user", "content": "Hello"}],
        "max_tokens": int(
            options.get("max_tokens") or options.get("num_predict") or DEFAULT_MAX_TOKENS
        ),
        "stream": stream,
    }
    if system:
        payload["system"] = system
    if "temperature" in options:
        payload["temperature"] = options["temperature"]
    if "stop" in options:
        payload["stop_sequences"] = (
            options["stop"] if isinstance(options["stop"], list) else [options["stop"]]
        )
    anthropic_tools = convert_tools(tools)
    if anthropic_tools:
        payload["tools"] = anthropic_tools
        if options.get("force_tools"):
            payload["tool_choice"] = {"type": "any"}

    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = _anthropic_headers(api_key)

    if stream:

        async def response_generator():
            tool_blocks: dict[int, dict[str, str]] = {}
            try:
                async with (
                    build_guarded_http_client(timeout=timeout) as client,
                    client.stream("POST", url, json=payload, headers=headers) as response,
                ):
                    _raise_for_status(response)
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        event_type = event.get("type")
                        if event_type == "content_block_start":
                            block = event.get("content_block") or {}
                            if block.get("type") == "tool_use":
                                tool_blocks[int(event.get("index") or 0)] = {
                                    "id": str(block.get("id") or ""),
                                    "name": str(block.get("name") or ""),
                                    "json": "",
                                }
                        elif event_type == "content_block_delta":
                            delta = event.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                yield {
                                    "model": model,
                                    "message": {
                                        "role": "assistant",
                                        "content": delta["text"],
                                    },
                                }
                            elif delta.get("type") == "input_json_delta":
                                index = int(event.get("index") or 0)
                                if index in tool_blocks:
                                    tool_blocks[index]["json"] += str(
                                        delta.get("partial_json") or ""
                                    )
                        elif event_type == "message_stop" and tool_blocks:
                            yield {
                                "model": model,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": item["id"],
                                            "type": "function",
                                            "function": {
                                                "name": item["name"],
                                                "arguments": item["json"] or "{}",
                                            },
                                        }
                                        for item in tool_blocks.values()
                                    ],
                                },
                            }
                        elif event_type == "error":
                            error = event.get("error") or {}
                            raise ValueError(
                                f"Anthropic stream error: {error.get('message') or event}"
                            )
            except httpx.HTTPStatusError as error:
                _raise_for_status(error.response)
            except Exception as error:
                logger.error("Error in Anthropic streaming chat: %s", error)
                raise

        return response_generator()

    async def _post() -> dict[str, Any]:
        async with build_guarded_http_client(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            _raise_for_status(response)
            return _message_to_ollama(model, response.json())

    try:
        return await with_retries(_post, operation_name="anthropic")
    except Exception as error:
        logger.error("Error in Anthropic chat request: %s", error)
        raise

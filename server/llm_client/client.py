"""
Main unified LLM client supporting OpenAI-compatible providers and Anthropic.

This module provides AsyncLLMClient, a unified interface for:
- OpenAI-compatible APIs (Ollama, LM Studio, llama.cpp, 9Router, OmniRoute, OpenAI, Fireworks)
- Native Anthropic Messages API
- Local models via bundled llama.cpp server (exposed through an OpenAI-style API)
"""

import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Any, Union

from server.database.config.manager import config_manager
from server.locale_policy import add_persian_output_instruction
from server.utils.url_utils import normalize_openai_base_url

from .providers.anthropic import anthropic_chat
from .providers.openai import openai_compatible_chat
from .utils import repair_json

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """A unified client interface for OpenAI-compatible, Anthropic, and local providers."""

    def __init__(
        self,
        provider_type: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 80,
        protocol: str = "openai_compatible",
    ):
        """
        Initialize the LLM client.

        Args:
            provider_type: Canonical provider id (ollama, openai, anthropic, ...)
            base_url: Base URL for the API
            api_key: API key (required for some providers)
            timeout: Request timeout in seconds
            protocol: ``openai_compatible`` or ``anthropic``
        """
        self.provider_type = provider_type.lower()
        self.protocol = protocol
        self.timeout = timeout
        self.api_key = api_key or "not-needed"

        if base_url:
            self.base_url = (
                base_url.rstrip("/")
                if protocol == "anthropic"
                else normalize_openai_base_url(base_url)
            )
        else:
            self.base_url = None

        self.extra_body = None
        extra_body_env = os.getenv("LLM_EXTRA_BODY")
        if extra_body_env:
            try:
                self.extra_body = json.loads(extra_body_env)
            except json.JSONDecodeError:
                logger.error(
                    "Failed to parse LLM_EXTRA_BODY environment variable: %s", extra_body_env
                )

        if not self.base_url:
            raise ValueError("base_url is required for the selected LLM provider")

        self._client = None
        if self.protocol != "anthropic":
            try:
                import httpx

                from openai import AsyncOpenAI

                from server.utils.ssrf import build_guarded_http_client

                # Guarded transport: DNS is resolved once, validated, and the
                # connection is pinned to that IP (Host/SNI preserved), so a
                # rebinding hostname cannot bypass the SSRF guard.
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=f"{self.base_url}/v1",
                    timeout=timeout,
                    max_retries=2,
                    http_client=build_guarded_http_client(timeout=httpx.Timeout(timeout)),
                )
            except ImportError as error:
                raise ImportError(
                    "OpenAI client not installed. Install with 'pip install openai'"
                ) from error

    async def chat_with_structured_output(
        self,
        model: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a chat completion request with structured output.

        Args:
            model: Model name
            messages: List of message dictionaries
            schema: JSON schema for structured output
            options: Additional options for the model

        Returns:
            JSON string response
        """
        response = await self.chat(model=model, messages=messages, format=schema, options=options)

        # chat() with stream=False always returns dict
        if isinstance(response, dict):
            message_content = response["message"]["content"]  # ty: ignore
        else:
            raise RuntimeError("Expected dict response, got async generator")

        # Handle emdashes and en-dashes (can cause JSON parsing issues)
        # Preserve UTF-8 characters for international language support
        response_str = message_content.replace("—", "-").replace("–", "-")

        return repair_json(response_str)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        format: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> Union[dict[str, Any], AsyncGenerator]:
        """Send a chat completion request."""
        from .utils import ensure_system_messages_first

        messages = ensure_system_messages_first(messages)
        # Enforce the product language for every generation path (chat,
        # summaries, letters, reasoning, templates and transcription cleanup).
        # The instruction explicitly preserves English medical terms in mixed
        # Persian/English input and never changes JSON keys or identifiers.
        messages = add_persian_output_instruction(messages)

        from server.utils.request_context import get_request_id

        started = time.perf_counter()

        try:
            if self.protocol == "anthropic":
                result = await anthropic_chat(
                    self.base_url or "",
                    self.api_key,
                    model,
                    messages,
                    format,
                    options,
                    tools,
                    stream,
                    self.timeout,
                )
            else:
                result = await openai_compatible_chat(
                    self._client,
                    model,
                    messages,
                    format,
                    options,
                    tools,
                    stream,
                    self.extra_body,
                )
        except Exception:
            logger.warning(
                "ai_llm provider=%s protocol=%s model=%s stream=%s duration_ms=%d "
                "status=error request_id=%s",
                self.provider_type,
                self.protocol,
                model,
                stream,
                int((time.perf_counter() - started) * 1000),
                get_request_id(),
            )
            raise

        if stream:

            async def _logged_stream(generator):
                status = "ok"
                try:
                    async for chunk in generator:
                        yield chunk
                except Exception:
                    status = "error"
                    raise
                finally:
                    logger.info(
                        "ai_llm provider=%s protocol=%s model=%s stream=true "
                        "duration_ms=%d status=%s request_id=%s",
                        self.provider_type,
                        self.protocol,
                        model,
                        int((time.perf_counter() - started) * 1000),
                        status,
                        get_request_id(),
                    )

            return _logged_stream(result)

        logger.info(
            "ai_llm provider=%s protocol=%s model=%s stream=false duration_ms=%d "
            "status=ok request_id=%s",
            self.provider_type,
            self.protocol,
            model,
            int((time.perf_counter() - started) * 1000),
            get_request_id(),
        )
        return result


def get_llm_client(timeout: int = 80):
    """Create and return an LLM client with configuration from config manager.

    Args:
        timeout: Request timeout in seconds (default: 80)
    """
    from server.utils.providers import resolve_llm_connection

    config = config_manager.get_config()
    connection = resolve_llm_connection(config)

    return AsyncLLMClient(
        provider_type=connection["provider"],
        base_url=connection["base_url"],
        api_key=connection["api_key"],
        timeout=timeout,
        protocol=connection["protocol"],
    )

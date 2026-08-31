"""
Embedding providers for the Phlox vector store.

Usage:
    provider = OpenAICompatibleProvider(base_url=..., api_key=..., model_name=...)
    vectors = provider(["hello world", "second text"])
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider:
    """Embedding provider using an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        import httpx

        from openai import OpenAI

        from server.utils.ssrf import build_guarded_http_client

        # Guarded transport: resolve once, validate, pin to the resolved IP so
        # a DNS-rebind after validation cannot reach a blocked target.
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=80,
            max_retries=2,
            http_client=build_guarded_http_client(timeout=httpx.Timeout(80.0)),
        )
        self.model_name = model_name
        self._dimension: int | None = None

    def __call__(self, texts: list[str] | str) -> list[list[float]]:
        """Embed a list of texts (or a single string)."""
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(model=self.model_name, input=texts)
        result = [item.embedding for item in response.data]

        if self._dimension is None and result:
            self._dimension = len(result[0])
            logger.info("Detected embedding dimension: %d", self._dimension)

        return result

    @property
    def dimension(self) -> int:
        """Return the embedding dimension, probing with a dummy call if unknown."""
        if self._dimension is None:
            logger.info("Probing embedding dimension with test call...")
            result = self(["test"])
            self._dimension = len(result[0])
        return self._dimension

import logging

import httpx
from fastapi import APIRouter

from server.utils.providers import (
    resolve_asr_connection,
    resolve_embedding_connection,
    resolve_llm_connection,
)
from server.utils.url_utils import build_openai_v1_url, build_whisper_v1_url

router = APIRouter()


def _get_llm_status_url(config: dict) -> str | None:
    """Determine the LLM status check URL based on provider configuration."""
    connection = resolve_llm_connection(config)
    provider = connection["provider"]
    base_url = connection["base_url"]
    if provider == "anthropic":
        return f"{base_url.rstrip('/')}/v1/models"
    if base_url:
        return build_openai_v1_url(base_url, "models")
    return None


def _get_whisper_status_url(config: dict) -> str | None:
    """Determine the Whisper status check URL based on configuration."""
    connection = resolve_asr_connection(config)
    provider = connection["provider"]
    model = connection["model"]
    protocol = connection["protocol"]

    # ONNX runtimes (Shenava, Parakeet) have no HTTP sidecar.
    if provider == "local" and (
        str(model).startswith("shenava-") or str(model).startswith("parakeet-")
    ):
        return None

    if protocol == "speechmatics":
        # Speechmatics has no public models listing; a configured key is enough.
        return None

    if protocol == "fireworks":
        return None

    if provider == "local" and not connection["base_url"]:
        from server.utils.allocated_ports import get_whisper_port

        return f"http://127.0.0.1:{get_whisper_port()}/health"

    if connection["base_url"]:
        return build_whisper_v1_url(connection["base_url"], "models")

    return None


def _get_embedding_status_url(config: dict) -> str | None:
    """Determine the embedding server status check URL."""
    connection = resolve_embedding_connection(config)
    if connection["provider"] == "local":
        from server.utils.allocated_ports import get_embedding_port

        return f"http://127.0.0.1:{get_embedding_port()}/health"
    if connection["base_url"]:
        return build_openai_v1_url(connection["base_url"], "models")
    return None


@router.get("/status")
async def get_server_status():
    """Check the status of LLM, Whisper, and embedding servers."""
    from server.database.config.manager import config_manager

    config = config_manager.get_config()
    # embedding defaults to None: only set to True/False when a distinct local
    # embedding server exists.
    status = {"llm": False, "whisper": False, "embedding": None}

    try:
        llm_url = _get_llm_status_url(config)
        if llm_url:
            headers = {}
            connection = resolve_llm_connection(config)
            if connection["provider"] == "anthropic":
                headers = {
                    "x-api-key": connection["api_key"],
                    "anthropic-version": "2023-06-01",
                }
            elif connection["api_key"] and connection["api_key"] not in {
                "not-needed",
                "ollama",
                "lm-studio",
            }:
                headers = {"Authorization": f"Bearer {connection['api_key']}"}
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(llm_url, timeout=2.0, headers=headers)
                    status["llm"] = response.status_code in [200, 401, 403]
                except Exception:
                    logging.debug("LLM status check failed (service unreachable)")

        whisper_url = _get_whisper_status_url(config)
        asr = resolve_asr_connection(config)
        if asr["provider"] in {"speechmatics", "fireworks"}:
            # Cloud providers are "up" when a key is configured. Speechmatics
            # keys are product-scoped, so also accept the Batch key.
            batch_key = config.get("ASR_BATCH_KEY") or config.get("WHISPER_BATCH_KEY")
            status["whisper"] = bool(asr["api_key"] or batch_key)
        elif asr["provider"] == "local" and str(asr["model"]).startswith(("shenava-", "parakeet-")):
            from server.utils.whisper_models import asr_model_manager

            status["whisper"] = asr_model_manager.get_model_path(asr["model"]) is not None
        elif whisper_url:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(whisper_url, timeout=2.0)
                    status["whisper"] = response.status_code in [200, 401, 403]
                except Exception:
                    logging.debug("Whisper status check failed (service unreachable)")

        embedding_url = _get_embedding_status_url(config)
        if embedding_url:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(embedding_url, timeout=2.0)
                    status["embedding"] = response.status_code in [200, 401, 403]
                except Exception:
                    logging.debug("Embedding status check failed (service unreachable)")

        return status
    except Exception as e:
        logging.error(f"Error checking server status: {str(e)}")
        return status

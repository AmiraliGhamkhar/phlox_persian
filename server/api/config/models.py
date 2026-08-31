import logging

import httpx
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from server.constants import IS_DOCKER
from server.database.config.manager import config_manager
from server.utils.llama_models import llama_model_manager
from server.utils.ssrf import build_guarded_http_client
from server.utils.providers import (
    ASR_PROVIDERS,
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    list_providers,
    looks_like_embedding_model,
    normalize_provider_id,
)
from server.utils.url_utils import build_openai_v1_url, build_whisper_v1_url

router = APIRouter()


def _extract_model_ids(data) -> list[str]:
    """Be tolerant of common OpenAI-compatible /v1/models response shapes."""
    model_list: list[str] = []
    if isinstance(data, dict):
        entries = data.get("data") if isinstance(data.get("data"), list) else None
        if entries is None and isinstance(data.get("models"), list):
            entries = data["models"]
        if entries is None:
            entries = [data] if data.get("id") or data.get("name") else []
    elif isinstance(data, list):
        entries = data
    else:
        entries = []
    for model in entries:
        if isinstance(model, dict):
            model_id = model.get("id") or model.get("name")
            if model_id:
                model_list.append(str(model_id))
        elif isinstance(model, str):
            model_list.append(model)
    return list(dict.fromkeys(model_list))


@router.get("/providers")
def get_providers():
    """Return the LLM, ASR, and embedding provider catalogs for the settings UI."""
    return list_providers()


@router.get("/options")
def get_options():
    """Retrieve all options configuration."""
    prompts_and_options = config_manager.get_prompts_and_options()
    return JSONResponse(content=prompts_and_options["options"])


@router.post("/options/reset-to-defaults")
def reset_options_to_defaults():
    """Reset all model configuration options to their default values."""
    config_manager.reset_options_to_defaults()
    return {"message": "Options reset to defaults successfully"}


@router.post("/options/{category}")
def update_options(category: str, data: dict = Body(...)):
    """Update options for a specific category."""
    config_manager.update_options(category, data)
    return {"message": f"{category} options updated successfully"}


@router.get("/llm/models")
async def get_llm_models(
    provider: str = Query(..., description="LLM provider type"),
    baseUrl: str = Query(None, description="The base URL for the LLM API"),
    apiKey: str = Query(
        None, description="Optional API key for authenticated OpenAI-compatible endpoints"
    ),
):
    """Fetch available models from the configured LLM provider."""
    try:
        provider_id = normalize_provider_id(provider, "llm")
        info = LLM_PROVIDERS.get(provider_id, LLM_PROVIDERS["openai_compatible"])

        if provider_id == "local":
            if IS_DOCKER:
                return {
                    "models": [],
                    "error": "Local models not available in Docker",
                }

            try:
                models = llama_model_manager.get_downloaded_models()
                return {"models": [model["name"] for model in models]}
            except Exception as e:
                logging.error(f"Error fetching local models: {e}")
                return {"models": [], "error": "Failed to fetch local models"}

        effective_url = (baseUrl or info.get("default_base_url") or "").strip()
        if not effective_url:
            raise HTTPException(
                status_code=400,
                detail="baseUrl is required for this provider",
            )

        effective_key = apiKey or config_manager.get_config().get("LLM_API_KEY")
        catalog_defaults = list(info.get("default_models") or [])

        if provider_id == "anthropic":
            headers = {
                "x-api-key": effective_key or "",
                "anthropic-version": info.get("anthropic_version", "2023-06-01"),
            }
            models_url = f"{effective_url.rstrip('/')}/v1/models"
        else:
            headers = {"Authorization": f"Bearer {effective_key}"} if effective_key else {}
            models_url = build_openai_v1_url(effective_url, "models")

        async with build_guarded_http_client(headers=headers) as client:
            try:
                response = await client.get(models_url, timeout=5.0)
                if response.status_code == 200:
                    model_list = _extract_model_ids(response.json())
                    if not model_list:
                        model_list = catalog_defaults
                    return {"models": model_list}
                if response.status_code in [401, 403]:
                    return {
                        "models": catalog_defaults,
                        "error": "Authentication failed",
                    }
                return {"models": catalog_defaults}
            except Exception:
                return {"models": catalog_defaults}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching LLM models: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@router.get("/embedding/models")
async def get_embedding_models(
    provider: str = Query(None, description="Embedding provider type"),
    baseUrl: str = Query(None, description="The base URL for the embedding API"),
    apiKey: str = Query(None, description="Optional API key"),
):
    """List embedding models for the selected provider."""
    try:
        provider_id = normalize_provider_id(provider, "embedding")
        info = EMBEDDING_PROVIDERS.get(provider_id, EMBEDDING_PROVIDERS["openai_compatible"])
        catalog_defaults = list(info.get("default_models") or [])

        if provider_id == "local":
            return {"models": catalog_defaults or ["Qwen3-Embedding-0.6B-Q8_0"]}

        effective_url = (baseUrl or info.get("default_base_url") or "").strip()
        if not effective_url:
            return {"models": catalog_defaults}

        effective_key = (
            apiKey
            or config_manager.get_config().get("EMBEDDING_API_KEY")
            or config_manager.get_config().get("LLM_API_KEY")
        )
        headers = {"Authorization": f"Bearer {effective_key}"} if effective_key else {}
        models_url = build_openai_v1_url(effective_url, "models")
        async with build_guarded_http_client(headers=headers) as client:
            try:
                response = await client.get(models_url, timeout=5.0)
                if response.status_code == 200:
                    model_list = [
                        model_id
                        for model_id in _extract_model_ids(response.json())
                        if looks_like_embedding_model(model_id)
                    ]
                    if not model_list:
                        model_list = catalog_defaults or _extract_model_ids(response.json())
                    return {"models": model_list}
            except Exception:
                return {"models": catalog_defaults}
        return {"models": catalog_defaults}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching embedding models: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e


@router.get("/asr/models")
@router.get("/whisper/models")
async def get_whisper_models(
    whisperEndpoint: str | None = Query(None, description="Legacy ASR endpoint parameter"),
    asrEndpoint: str | None = Query(None, description="The endpoint for the ASR API"),
    provider: str | None = Query(None, description="Named ASR provider"),
    apiKey: str | None = Query(None, description="Optional API key"),
):
    """Fetch available automatic speech recognition models from the configured endpoint.

    Accepts endpoints with or without a terminal /v1 segment.
    Only works if the instance exposes a compatible /v1/models endpoint
    (for example an OpenAI-compatible Whisper or faster-whisper server). The
    endpoint is treated as an ASR endpoint, so multilingual model ids are
    accepted instead of filtering only ids containing the word Whisper.
    """
    try:
        provider_id = normalize_provider_id(provider, "asr") if provider else ""
        if provider_id in {"speechmatics", "fireworks"}:
            info = ASR_PROVIDERS.get(provider_id) or {}
            return {
                "models": list(info.get("default_models") or []),
                "listAvailable": True,
            }

        endpoint = (asrEndpoint or whisperEndpoint or "").strip()
        if not endpoint and provider_id:
            endpoint = str((ASR_PROVIDERS.get(provider_id) or {}).get("default_base_url") or "")
        if not endpoint:
            raise HTTPException(status_code=422, detail="ASR endpoint is required")

        headers = {}
        effective_key = (
            apiKey
            or config_manager.get_config().get("ASR_KEY")
            or config_manager.get_config().get("WHISPER_KEY")
        )
        if effective_key:
            headers["Authorization"] = f"Bearer {effective_key}"

        # First try to fetch models from the endpoint
        async with build_guarded_http_client(headers=headers) as client:
            try:
                url = build_whisper_v1_url(endpoint, "models")
                response = await client.get(url, timeout=5.0)

                if response.status_code == 200:
                    # Parse the response based on the expected format
                    data = response.json()
                    # Extract model names depending on the API structure
                    models = []
                    entries = (
                        data
                        if isinstance(data, list)
                        else data.get("data", [])
                        if isinstance(data, dict)
                        else []
                    )
                    models = []
                    for model in entries:
                        if isinstance(model, str):
                            model_id = model
                        elif isinstance(model, dict):
                            model_id = model.get("id", model.get("name", ""))
                        else:
                            model_id = ""
                        if model_id:
                            models.append(str(model_id))

                    # If we found some models, return them
                    if models:
                        return {"models": models, "listAvailable": True}
            except Exception as e:
                logging.warning(f"Could not fetch Whisper models from endpoint: {e}")

        # Named cloud/local servers with a curated catalog can fall back to
        # those ids. Custom OpenAI-compatible endpoints keep a free-text field
        # when they do not expose /v1/models.
        if provider_id and provider_id not in {"openai_compatible", "local"}:
            defaults = list((ASR_PROVIDERS.get(provider_id) or {}).get("default_models") or [])
            if defaults:
                return {"models": defaults, "listAvailable": True}
        return {"models": [], "listAvailable": False}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in get_whisper_models: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e

import asyncio
import logging

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from server.database.config.manager import config_manager
from server.utils.ssrf import validate_fetch_url

router = APIRouter()


SENSITIVE_KEYS = {
    "LLM_API_KEY",
    "WHISPER_KEY",
    "WHISPER_BATCH_KEY",
    "ASR_KEY",
    "ASR_BATCH_KEY",
    "EMBEDDING_API_KEY",
}
MASK_BULLET = "•"

# Keys a client may write through POST /api/config/global. Anything else is
# rejected (API3:2023 mass assignment): unknown keys must not be able to seed
# arbitrary configuration (provider URLs, keys, internal toggles). This is the
# union of the config-table seeds (migrations v1/v2/v3/v5/v7/v10) and the
# keys the settings UI / server read path actually consume
# (server/chat/config, transcription/audio.py, api/config/system.py).
ALLOWED_CONFIG_KEYS = {
    # LLM
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "PRIMARY_MODEL",
    "SECONDARY_MODEL",
    "REASONING_MODEL",
    "REASONING_ENABLED",
    "DAILY_SUMMARY",
    # Embeddings
    "EMBEDDING_PROVIDER",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    # ASR / Whisper (canonical ASR_* + legacy WHISPER_* compat + batch keys)
    "ASR_PROVIDER",
    "ASR_BASE_URL",
    "ASR_MODEL",
    "ASR_KEY",
    "ASR_BATCH_URL",
    "ASR_BATCH_KEY",
    "ASR_LANGUAGE",
    "WHISPER_BASE_URL",
    "WHISPER_MODEL",
    "WHISPER_KEY",
    "WHISPER_LANGUAGE",
    "WHISPER_BATCH_URL",
    "WHISPER_BATCH_KEY",
    # Runtime/server-managed keys that round-trip through the settings UI
    # (GET returns them, autosave POSTs them back unchanged; the server
    # ignores writes to the VISION_* probe cache).
    "AUDIT_RETENTION_DAYS",
    "DOCUMENT_IMAGE_PROCESSING_MODE",
    "VISION_CAPABILITY_CACHE",
    "VISION_CAPABILITY_CACHE_KEY",
    "VISION_MODEL_CAPABLE",
}

# Base-URL keys the server fetches from; every non-empty value must pass the
# SSRF guard before it is stored (the guarded client re-checks per request).
URL_CONFIG_KEYS = {
    "LLM_BASE_URL",
    "EMBEDDING_BASE_URL",
    "ASR_BASE_URL",
    "ASR_BATCH_URL",
    "WHISPER_BASE_URL",
    "WHISPER_BATCH_URL",
}

# Writes to server-owned state must not clobber runtime results.
SERVER_OWNED_KEYS = {
    "VISION_CAPABILITY_CACHE",
    "VISION_CAPABILITY_CACHE_KEY",
    "VISION_MODEL_CAPABLE",
}

LANGUAGE_CONFIG_KEYS = {"ASR_LANGUAGE", "WHISPER_LANGUAGE"}
ALLOWED_LANGUAGES = {"fa", "en", "auto"}


def mask_key(key):
    """Partially mask a secret for display: first 3 + bullets + last 4."""
    if not key:
        return key
    if len(key) < 12:
        return MASK_BULLET * len(key)
    return key[:3] + MASK_BULLET * 4 + key[-4:]


@router.get("/global")
def get_config():
    """Retrieve the current global configuration."""
    config = config_manager.get_config()
    masked = dict(config)
    for sensitive_key in SENSITIVE_KEYS:
        if sensitive_key in masked:
            masked[sensitive_key] = mask_key(masked[sensitive_key])
    return JSONResponse(content=masked)


@router.post("/global")
async def update_config(data: dict = Body(...)):
    """Update configuration items with provided data.

    Only allowlisted keys are accepted (mass-assignment guard); URL fields
    must pass the SSRF guard before being stored; sensitive key fields
    containing mask bullets (•) are stripped to avoid overwriting the stored
    secret with a masked display value.
    """
    unknown = set(data) - ALLOWED_CONFIG_KEYS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown configuration keys: {', '.join(sorted(unknown))}",
        )

    for url_key in URL_CONFIG_KEYS:
        value = data.get(url_key)
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            continue
        try:
            await asyncio.to_thread(validate_fetch_url, value)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=f"{url_key}: {error}") from error

    for lang_key in LANGUAGE_CONFIG_KEYS:
        value = data.get(lang_key)
        if value is None:
            continue
        value = str(value).strip().lower()
        if value and value not in ALLOWED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"{lang_key} must be one of: fa, en, auto",
            )
        data[lang_key] = value

    filtered = dict(data)
    # Masked display values must never overwrite the stored secret.
    for sensitive_key in SENSITIVE_KEYS:
        if sensitive_key in filtered and MASK_BULLET in str(filtered[sensitive_key]):
            del filtered[sensitive_key]
    # Server-owned runtime state (vision probe cache) is not client-editable;
    # accept-and-ignore so the autosave round-trip does not corrupt it.
    for owned_key in SERVER_OWNED_KEYS:
        filtered.pop(owned_key, None)

    config_manager.update_config(filtered)

    try:
        from server.rag.vector_store import get_vector_store_manager

        vector_store_mgr = get_vector_store_manager()
        if vector_store_mgr is not None:
            vector_store_mgr._reload_embedding_function()
    except Exception:
        logging.debug("Vector store reload skipped during config update")

    return {"message": "config.js updated successfully"}

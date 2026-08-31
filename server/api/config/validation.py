import asyncio
import logging

import httpx
from fastapi import APIRouter, HTTPException, Query

from server.utils.ssrf import build_guarded_http_client, validate_fetch_url
from server.utils.url_utils import build_openai_v1_url, build_whisper_v1_url

router = APIRouter()


_OPENAI_COMPATIBLE_TYPES = {
    "openai",
    "openai_compatible",
    "ollama",
    "lmstudio",
    "llamacpp",
    "ninerouter",
    "omniroute",
    "fireworks",
    "local",
    "embedding",
}
_ASR_TYPES = {"whisper", "asr", "whispercpp", "speechmatics", "fireworks"}
_ANTHROPIC_TYPES = {"anthropic"}


def _normalize_validation_type(request_type: str) -> str:
    """Normalize URL validation types to openai, anthropic, or whisper."""
    normalized = request_type.lower().strip()

    if normalized in _ASR_TYPES:
        return "whisper"
    if normalized in _ANTHROPIC_TYPES:
        return "anthropic"
    if normalized in _OPENAI_COMPATIBLE_TYPES:
        return "openai"

    raise HTTPException(status_code=400, detail="Invalid URL type")


@router.get("/validate-url")
async def validate_url(
    url: str = Query(..., description="URL to validate"),
    type: str = Query(
        ...,
        description="Type of URL (whisper or openai)",
    ),
):
    """Validate if a URL is accessible and returns a valid response."""
    try:
        validation_type = _normalize_validation_type(type)

        # SSRF guard: scheme allowlist + block link-local/metadata targets.
        # Loopback/private remain allowed because local model servers are the
        # primary legitimate use of this validator.
        await asyncio.to_thread(validate_fetch_url, url)

        # follow_redirects stays False (httpx default) so a validated host
        # cannot bounce the request somewhere else.
        async with build_guarded_http_client() as client:
            if validation_type == "whisper":
                # For Whisper, try to access the audio/transcriptions endpoint with a minimal request.
                # Accept endpoints with or without a terminal /v1 segment.
                validate_url = build_whisper_v1_url(url, "audio/transcriptions")
                form_data = {"model": "whisper-1"}

                try:
                    response = await client.post(
                        validate_url,
                        data=form_data,
                        headers={},
                        timeout=3.0,
                    )

                    # 400/422 => endpoint exists but request payload is minimal/invalid (expected)
                    # 401/403 => endpoint exists but requires authentication
                    # 200 => valid and accessible
                    return {"valid": response.status_code in [200, 400, 401, 403, 422]}
                except Exception as error:
                    logging.error(f"Error validating Whisper URL: {error}")
                    return {
                        "valid": False,
                        "error": "An internal error has occurred while validating the URL.",
                    }

            if validation_type == "anthropic":
                models_url = f"{url.rstrip('/')}/v1/models"
                try:
                    response = await client.get(
                        models_url,
                        headers={"anthropic-version": "2023-06-01"},
                        timeout=3.0,
                    )
                    return {"valid": response.status_code in [200, 401, 403]}
                except Exception as error:
                    logging.error(f"Error validating Anthropic URL: {error}")
                    return {
                        "valid": False,
                        "error": "An internal error has occurred while validating the URL.",
                    }

            # OpenAI-compatible validation (accept base URLs with or without /v1)
            models_url = build_openai_v1_url(url, "models")
            try:
                response = await client.get(models_url, timeout=3.0)
                # 401/403 often indicates valid endpoint with auth required
                # 404 may occur on compatible servers that don't expose model listing
                if response.status_code in [200, 401, 403, 404]:
                    return {"valid": True}
            except Exception:
                logging.debug("Models endpoint unreachable, falling through to chat probe")

            chat_url = build_openai_v1_url(url, "chat/completions")
            try:
                response = await client.post(
                    chat_url,
                    json={"model": "test", "messages": []},
                    timeout=3.0,
                )
                # 400/422 often means endpoint exists but request schema/model is invalid
                # 401/403 means auth is required
                # 404 accepted for some proxy/provider edge cases
                return {"valid": response.status_code in [200, 400, 401, 403, 404, 422]}
            except Exception:
                return {"valid": False}

    except ValueError as error:
        # SSRF guard rejection (bad scheme / metadata target) -> client error
        return {"valid": False, "error": str(error)}
    except HTTPException:
        raise
    except Exception as error:
        logging.error(f"Error validating URL: {error}")
        return {
            "valid": False,
            "error": "An internal error has occurred while validating the URL.",
        }

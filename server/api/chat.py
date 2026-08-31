import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from server.chat import ChatEngine
from server.constants import DATA_DIR
from server.database.config.manager import config_manager
from server.llm_client.client import AsyncLLMClient, get_llm_client
from server.nlp_tools.document_processing import extract_text_from_document
from server.schemas.chat import ChatRequest, ChatResponse
from server.schemas.documents import VisualDocumentPage
from server.utils.request_limits import MAX_IMAGE_UPLOAD_BYTES, read_upload_limited
from server.utils.ssrf import validate_fetch_url

router = APIRouter()


class VisualDocumentRequest(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    strategy: str = "vision"
    pages: list[VisualDocumentPage] = Field(default_factory=list)
    fallback_text: str | None = None
    extraction_info: dict | None = None


class VisualDocumentResponse(BaseModel):
    text: str
    filename: str
    content_type: str
    page_count: int


class VisionCapabilityProbeRequest(BaseModel):
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class VisionCapabilityProbeResponse(BaseModel):
    vision_capable: bool
    status_code: int
    detail: str


class DirectVisualChatRequest(BaseModel):
    prompt: str | None = None
    filename: str | None = None
    content_type: str | None = None
    pages: list[VisualDocumentPage] = Field(default_factory=list)


class DirectVisualChatResponse(BaseModel):
    answer: str
    filename: str
    content_type: str
    page_count: int


class VisionCurrentCapabilityResponse(BaseModel):
    vision_capable: bool
    status_code: int
    detail: str
    cache_key: str
    source: str
    probed_at: str | None = None


def _build_vision_cache_key(provider: str, base_url: str, model: str) -> str:
    normalized_provider = (provider or "openai").strip().lower()
    normalized_base = (base_url or "").strip().lower().rstrip("/")
    normalized_model = (model or "").strip().lower()
    return f"{normalized_provider}|{normalized_base}|{normalized_model}"


def _is_local_vision_capable(config: dict) -> bool:
    """Local (Tauri) builds always run vision-capable VLMs with a projector."""
    if config.get("LLM_BASE_URL"):
        return False  # remote mode — use the normal probe/cache path
    return any((DATA_DIR / "llm_models").glob("*mmproj*.gguf"))


def _get_vision_capability_cache(config: dict) -> dict:
    cache = config.get("VISION_CAPABILITY_CACHE", {})
    return cache if isinstance(cache, dict) else {}


def _store_vision_probe_result(
    *,
    provider: str,
    base_url: str,
    model: str,
    vision_capable: bool,
    status_code: int,
    detail: str,
):
    cache_key = _build_vision_cache_key(provider, base_url, model)
    current_config = config_manager.get_config()
    cache = _get_vision_capability_cache(current_config)
    cache[cache_key] = {
        "vision_capable": bool(vision_capable),
        "status_code": int(status_code),
        "detail": detail,
        "probed_at": datetime.now(UTC).isoformat(),
    }

    config_manager.update_config(
        {
            "VISION_CAPABILITY_CACHE": cache,
            "VISION_CAPABILITY_CACHE_KEY": cache_key,
            "VISION_MODEL_CAPABLE": bool(vision_capable),
        }
    )


def _build_visual_user_content(
    pages: list[VisualDocumentPage],
    instruction: str,
    max_pages: int = 8,
) -> tuple[list[dict[str, Any]], int]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    added_images = 0

    for page in pages[:max_pages]:
        if not isinstance(page.data_url, str) or not page.data_url.startswith("data:image/"):
            continue
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": page.data_url},
            }
        )
        added_images += 1

    return user_content, added_images


def _get_chat_engine():
    return ChatEngine()


class PendingActionDecision(BaseModel):
    """Request body for confirming or cancelling a pending tool action."""

    action_id: str


@router.post("/confirm-action")
async def confirm_pending_action(payload: PendingActionDecision):
    """Execute a previously parked mutating tool after explicit user approval.

    The tool call was captured (not executed) by the executor's confirmation
    gate; this endpoint is the only way it ever runs.
    """
    from server.chat.tools.accumulator import ToolResultAccumulator
    from server.chat.tools.executor import execute_tool_streaming
    from server.chat.tools.pending_actions import pop_pending_action
    from server.utils.request_context import get_request_actor

    action = pop_pending_action(payload.action_id)
    if action is None:
        raise HTTPException(
            status_code=404,
            detail="Pending action not found (it may have expired or already been handled)",
        )

    # Object-level authorization (API1:2023): only the identity that queued the
    # action may confirm it. Without this, any authenticated proxy user could
    # approve a record write another user (or another session) queued.
    if action.actor is not None and get_request_actor() != action.actor:
        logging.warning(
            f"Pending action confirm rejected: actor mismatch "
            f"({get_request_actor()!r} vs {action.actor!r})"
        )
        raise HTTPException(status_code=403, detail="This action belongs to another user")

    # Mark as approved so the executor's gate lets it through exactly once.
    action.tool_call["_approved"] = True

    # Re-register the patient captured when the action was parked: the
    # confirm endpoint runs outside stream_chat, so the ContextVar would
    # otherwise be empty here.
    from server.chat.tools.sanitization import set_active_patient_context

    set_active_patient_context(action.patient_context)

    async def _run():
        accumulator = ToolResultAccumulator()
        stream = execute_tool_streaming(
            tool_call=action.tool_call,
            llm_client=action.llm_client,
            config=action.config,
            message_list=action.message_list,
            context_question_options=action.context_question_options,
            vector_store_manager=action.vector_store_manager,
            conversation_history=action.conversation_history,
            raw_transcription=action.raw_transcription,
        )
        result, citations = await accumulator.consume_stream(stream)
        return result, citations, accumulator.artifacts

    try:
        result, citations, artifacts = await asyncio.wait_for(_run(), timeout=180)
    except Exception as e:
        logging.error(f"Confirmed action {action.tool_name} failed: {e}")
        raise HTTPException(
            status_code=500, detail="The approved action failed while executing"
        ) from e

    return JSONResponse(
        content={
            "result": result,
            "citations": citations,
            "tool": action.tool_name,
            "artifacts": artifacts,
        }
    )


@router.post("/cancel-action")
async def cancel_pending_action(payload: PendingActionDecision):
    """Discard a pending tool action without executing it."""
    from server.chat.tools.pending_actions import pop_pending_action
    from server.utils.request_context import get_request_actor

    action = pop_pending_action(payload.action_id)
    if action is None:
        raise HTTPException(status_code=404, detail="Pending action not found")
    if action.actor is not None and get_request_actor() != action.actor:
        logging.warning(
            f"Pending action cancel rejected: actor mismatch "
            f"({get_request_actor()!r} vs {action.actor!r})"
        )
        raise HTTPException(status_code=403, detail="This action belongs to another user")
    logging.info(f"Pending action cancelled: {action.tool_name} ({action.id})")
    return {"cancelled": True, "tool": action.tool_name}


@router.post("", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    chat_engine: ChatEngine = Depends(_get_chat_engine),
):
    """
    Process a chat request and return a streaming response.

    This endpoint accepts a chat request containing a conversation history and uses
    ChatEngine to generate responses asynchronously. The response chunks are streamed as
    Server Side Events.

    Args:
        chat_request (ChatRequest): The incoming chat request containing chat messages.
        chat_engine (ChatEngine): The chat engine used to process the chat request.

    Returns:
        StreamingResponse: An SSE streaming response that yields response chunks
                           formatted as JSON with the prefix "data: " and separated by newlines.

    Raises:
        HTTPException: If an error occurs during processing, a 500 error is raised with details.
    """
    try:
        logging.info("Received chat request")
        logging.debug(f"Chat request: {chat_request}")

        conversation_history = chat_request.messages
        raw_transcription = chat_request.raw_transcription
        patient_context = (
            chat_request.patient_context.model_dump() if chat_request.patient_context else None
        )

        async def generate():
            chunk_count = 0
            async for chunk in chat_engine.stream_chat(
                conversation_history,
                raw_transcription=raw_transcription,
                patient_context=patient_context,
            ):
                chunk_count += 1
                yield f"data: {json.dumps(chunk)}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Generic image/document upload endpoint for chat.

    Accepts images (png, jpg, etc.) and PDFs.
    Currently uses OCR to extract text; designed to be extensible for multimodal LLM.

    Returns extracted text for the LLM to interpret.
    """
    try:
        logging.info(f"Received image upload: {file.filename}, content_type: {file.content_type}")

        content = await read_upload_limited(file, MAX_IMAGE_UPLOAD_BYTES, "Image upload")
        content_type = file.content_type

        # OCR extract text using existing pipeline
        # (handles both images and PDFs - converts PDFs to images internally)
        extracted_text = await extract_text_from_document(content, content_type or "")

        logging.info(f"Successfully extracted {len(extracted_text)} characters from image")

        return {"text": extracted_text, "content_type": content_type, "filename": file.filename}
    except RuntimeError as e:
        # OCR dependencies not available
        logging.error(f"OCR not available: {e}")
        raise HTTPException(status_code=503, detail="OCR dependencies not available") from e
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/analyze-document-visual", response_model=VisualDocumentResponse)
async def analyze_document_visual(payload: VisualDocumentRequest):
    """
    Analyze image/PDF page payloads with a multimodal model and return extracted text.

    This endpoint is intended for frontend-first PDF handling:
    - frontend tries direct PDF text extraction
    - if insufficient, frontend renders PDF pages to images
    - this endpoint sends page images to a visual-capable model
    """
    try:
        if not payload.pages:
            raise HTTPException(status_code=400, detail="No page images supplied")

        max_pages = min(len(payload.pages), 8)

        user_content, added_images = _build_visual_user_content(
            pages=payload.pages,
            instruction=(
                "Extract all readable text from these document page images. "
                "Preserve clinical terminology and formatting where possible. "
                "If parts are unreadable, continue with best-effort extraction."
            ),
            max_pages=max_pages,
        )

        if payload.fallback_text and payload.fallback_text.strip():
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Fallback text from an earlier extraction attempt "
                        "(may be partial/noisy):\n\n" + payload.fallback_text.strip()
                    ),
                }
            )

        if added_images == 0:
            raise HTTPException(status_code=400, detail="No valid image data URLs supplied")

        config = config_manager.get_config()
        prompts = config_manager.get_prompts_and_options()
        options = prompts["options"]["general"].copy()
        options.pop("stop", None)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical document extraction assistant. "
                    "Return only the extracted document text."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        client = get_llm_client()
        response = await client.chat(
            model=config["PRIMARY_MODEL"],
            messages=messages,
            options=options,
        )

        extracted_text = (response.get("message", {}).get("content", "") or "").strip()

        # Last-resort fallback to any text already available from frontend attempt
        if not extracted_text and payload.fallback_text:
            extracted_text = payload.fallback_text.strip()

        return {
            "text": extracted_text,
            "filename": payload.filename or "uploaded-document",
            "content_type": payload.content_type or "application/pdf",
            "page_count": max_pages,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error analyzing visual document payload: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/vision-capability/current", response_model=VisionCurrentCapabilityResponse)
def get_current_vision_capability():
    """Return cached vision capability for the currently selected provider/base_url/model."""
    config = config_manager.get_config()
    provider = config.get("LLM_PROVIDER", "openai")
    base_url = config.get("LLM_BASE_URL", "")
    model = config.get("PRIMARY_MODEL", "")

    cache_key = _build_vision_cache_key(provider, base_url, model)
    cache = _get_vision_capability_cache(config)
    cached_result = cache.get(cache_key)

    # Local (Tauri) builds ship VLMs with a projector — vision is always available.
    if _is_local_vision_capable(config):
        return {
            "vision_capable": True,
            "status_code": 200,
            "detail": "Local vision model with projector loaded.",
            "cache_key": cache_key,
            "source": "local_assumed",
            "probed_at": None,
        }

    if cached_result:
        return {
            "vision_capable": bool(cached_result.get("vision_capable", False)),
            "status_code": int(cached_result.get("status_code", 200)),
            "detail": cached_result.get("detail", "Cached capability result found."),
            "cache_key": cache_key,
            "source": "cache",
            "probed_at": cached_result.get("probed_at"),
        }

    # Backward compatibility fallback to global flag
    return {
        "vision_capable": bool(config.get("VISION_MODEL_CAPABLE", False)),
        "status_code": 200,
        "detail": "No cache entry for current model endpoint; using global flag fallback.",
        "cache_key": cache_key,
        "source": "global_flag",
        "probed_at": None,
    }


@router.post("/vision-capability", response_model=VisionCapabilityProbeResponse)
async def probe_vision_capability(payload: VisionCapabilityProbeRequest):
    """
    Probe whether the configured/selected model endpoint accepts image inputs.

    Heuristic:
    - If the call succeeds, assume vision-capable.
    - If it fails with a 400-style unsupported-image error, assume not vision-capable.
    """
    config = config_manager.get_config()
    model = payload.model or config.get("PRIMARY_MODEL", "")
    probe_config = dict(config)
    if payload.base_url:
        probe_config["LLM_BASE_URL"] = payload.base_url
    if payload.api_key:
        probe_config["LLM_API_KEY"] = payload.api_key

    from server.utils.providers import resolve_llm_connection

    connection = resolve_llm_connection(probe_config)
    base_url = connection["base_url"]
    api_key = connection["api_key"]

    if base_url:
        # SSRF guard: probe targets must be http(s) and not metadata endpoints.
        await asyncio.to_thread(validate_fetch_url, base_url)

    # 1x1 black PNG
    black_square_data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+XcMsAAAAASUVORK5CYII="
    )

    try:
        client = AsyncLLMClient(
            provider_type=connection["provider"],
            base_url=base_url,
            api_key=api_key,
            protocol=connection["protocol"],
        )

        messages = [
            {
                "role": "system",
                "content": "You are a capability probe. Respond with 'ok' only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with 'ok' if you can see this image."},
                    {"type": "image_url", "image_url": {"url": black_square_data_url}},
                ],
            },
        ]

        await client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0},
        )

        result_payload = {
            "vision_capable": True,
            "status_code": 200,
            "detail": "Vision input accepted by endpoint/model.",
        }

        _store_vision_probe_result(
            provider=connection["provider"],
            base_url=base_url or "",
            model=model,
            vision_capable=result_payload["vision_capable"],
            status_code=result_payload["status_code"],
            detail=result_payload["detail"],
        )

        return result_payload
    except ValueError as e:
        # SSRF guard rejection -> client error, not a vision-capability result
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        error_text = str(e)
        lowered = error_text.lower()
        logging.error("Vision probe failed: %s", error_text, exc_info=True)

        looks_like_unsupported_vision = (
            "400" in lowered
            or "unsupported" in lowered
            or ("image" in lowered and "not" in lowered)
            or ("multimodal" in lowered and "not" in lowered)
        )

        result_payload = {
            "vision_capable": False,
            "status_code": 400 if looks_like_unsupported_vision else 500,
            "detail": "Vision model not supported"
            if looks_like_unsupported_vision
            else "Internal server error",
        }

        _store_vision_probe_result(
            provider=connection["provider"],
            base_url=base_url or "",
            model=model,
            vision_capable=result_payload["vision_capable"],
            status_code=result_payload["status_code"],
            detail=result_payload["detail"],
        )

        return result_payload


@router.post("/respond-visual", response_model=DirectVisualChatResponse)
async def respond_visual(payload: DirectVisualChatRequest):
    """
    Generate a direct chat response from image inputs + user prompt using a visual model.

    This is intended for chat flows where image reasoning should happen directly
    (not OCR text extraction first).
    """
    try:
        if not payload.pages:
            raise HTTPException(status_code=400, detail="No page images supplied")

        max_pages = min(len(payload.pages), 8)
        prompt = (payload.prompt or "").strip() or "Please analyze this image and respond."

        user_content, added_images = _build_visual_user_content(
            pages=payload.pages,
            instruction=prompt,
            max_pages=max_pages,
        )

        if added_images == 0:
            raise HTTPException(status_code=400, detail="No valid image data URLs supplied")

        config = config_manager.get_config()
        prompts = config_manager.get_prompts_and_options()
        options = prompts["options"]["general"].copy()
        options.pop("stop", None)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. Analyze the provided images and answer "
                    "the user's request directly and concisely."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        client = get_llm_client()
        response = await client.chat(
            model=config["PRIMARY_MODEL"],
            messages=messages,
            options=options,
        )

        answer = (response.get("message", {}).get("content", "") or "").strip()

        return {
            "answer": answer,
            "filename": payload.filename or "uploaded-image",
            "content_type": payload.content_type or "image/*",
            "page_count": max_pages,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error generating direct visual response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing the request."
        ) from None

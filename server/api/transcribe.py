import json
import logging
import secrets
import time

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from server.nlp_tools.document_processing import (
    _extract_demographics_from_text,
    extract_demographics_from_document,
    extract_demographics_from_visual_pages,
    process_document_text_with_template,
    process_document_with_template,
    process_visual_document_with_template,
)
from server.schemas.documents import VisualDocumentPage
from server.schemas.patient import TranscribeResponse
from server.transcription.audio import transcribe_audio
from server.transcription.text import process_transcription
from server.utils.request_limits import (
    MAX_AUDIO_UPLOAD_BYTES,
    MAX_DOCUMENT_UPLOAD_BYTES,
    read_upload_limited,
)

router = APIRouter()


def _format_patient_display_name(name: str | None) -> str:
    """Format a "Last, First" patient name into "First Last" for display."""
    if not name:
        return "N/A"
    parts = name.split(",")
    last_name = parts[0].strip()
    first_name = parts[1].strip() if len(parts) > 1 else ""
    full = f"{first_name} {last_name}".strip()
    return full or "N/A"


class ProcessDocumentFromTextRequest(BaseModel):
    extracted_text: str
    name: str | None = None
    gender: str | None = None
    dob: str | None = None
    templateKey: str = Field(..., description="Template key is required for document processing")


class ProcessVisualDocumentRequest(BaseModel):
    pages: list[VisualDocumentPage]
    filename: str | None = None
    content_type: str | None = None
    name: str | None = None
    gender: str | None = None
    dob: str | None = None
    templateKey: str = Field(..., description="Template key is required for document processing")


class ExtractDemographicsFromTextRequest(BaseModel):
    extracted_text: str


class ExtractDemographicsVisualRequest(BaseModel):
    pages: list[VisualDocumentPage]


# WebSocket handshake subprotocol for live transcription. The request token is
# carried as a second offered subprotocol so it never appears in the request
# URL/query string (query strings are written to uvicorn access logs and, via
# the desktop process manager, to the on-disk app log — A09:2025).
LIVE_WS_SUBPROTOCOL = "phlox-live"


def _authorize_live_socket(websocket: WebSocket) -> bool:
    """Accept a Bearer header or ``Sec-WebSocket-Protocol`` subprotocol.

    Browsers cannot set an Authorization header on the WebSocket handshake,
    so the desktop client offers ``phlox-live,<token>`` as subprotocols; the
    token is compared with constant-time equality and is never placed in the
    URL or query string.
    """
    from server.constants import is_docker_runtime
    from server.utils.local_request_token import get_request_token

    if is_docker_runtime():
        return True
    expected = get_request_token()
    if not expected:
        return False
    header = websocket.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return secrets.compare_digest(header[7:].strip(), expected)

    offered = websocket.headers.get("sec-websocket-protocol") or ""
    protocols = [p.strip() for p in offered.split(",") if p.strip()]
    if LIVE_WS_SUBPROTOCOL not in protocols:
        return False
    return any(
        secrets.compare_digest(protocol, expected)
        for protocol in protocols
        if protocol != LIVE_WS_SUBPROTOCOL
    )


@router.websocket("/live")
async def live_transcribe(websocket: WebSocket):
    """Stream PCM audio and receive partial / final transcripts.

    Client frames:
    - binary: 16-bit little-endian mono PCM at 16 kHz
    - text JSON ``{"type": "stop"}`` to finish
    Server frames (JSON text):
    - ``{"type": "partial"|"final"|"error"|"ready", "text"?: str, "message"?: str}``
    """
    if not _authorize_live_socket(websocket):
        await websocket.close(code=4401)
        return
    # Echo the app subprotocol (never the token) so the client can confirm the
    # negotiated connection.
    offered = websocket.headers.get("sec-websocket-protocol") or ""
    subprotocol = (
        LIVE_WS_SUBPROTOCOL
        if LIVE_WS_SUBPROTOCOL in [p.strip() for p in offered.split(",") if p.strip()]
        else None
    )
    await websocket.accept(subprotocol=subprotocol)

    from server.database.config.manager import config_manager
    from server.transcription.live import create_live_session, live_is_authoritative

    config = config_manager.get_config()

    async def emit(event: dict) -> None:
        try:
            await websocket.send_text(json.dumps(event))
        except Exception:
            logging.debug("Live transcript emit failed", exc_info=True)

    session = create_live_session(config, emit)
    try:
        await session.start()
        await emit({"type": "ready", "authoritative": live_is_authoritative(config)})
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                await session.feed_pcm(data)
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "stop":
                break
    except WebSocketDisconnect:
        logging.debug("Live transcription client disconnected")
    except Exception as error:
        logging.error("Live transcription failed: %s", error)
        try:
            await emit({"type": "error", "message": str(error)})
        except Exception:
            logging.debug("Could not send live transcription error", exc_info=True)
    finally:
        try:
            final_text = await session.stop()
            if final_text:
                await emit({"type": "final", "text": final_text})
        except Exception:
            logging.debug("Live transcription shutdown failed", exc_info=True)
        try:
            await websocket.close()
        except Exception:
            logging.debug("Live transcription websocket already closed", exc_info=True)


@router.post("/audio", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    gender: str | None = Form(None),
    dob: str | None = Form(None),
    templateKey: str | None = Form(None),
    isAmbient: bool = Form(True),
    noteId: int | None = Form(None),
):
    """Transcribes audio and processes the transcription."""
    try:
        # Read the audio file
        audio_buffer = await read_upload_limited(file, MAX_AUDIO_UPLOAD_BYTES, "Audio upload")

        # Process the name if provided
        formatted_name = _format_patient_display_name(name)

        # Perform transcription
        transcription_result = await transcribe_audio(audio_buffer)
        transcript_text = str(transcription_result["text"])
        transcription_duration = float(transcription_result["transcriptionDuration"])

        # Get template fields if template key is provided
        template_fields = []
        if templateKey:
            from server.database.repositories.templates import get_template_fields

            template_fields = get_template_fields(templateKey)

        # Look up primary condition for returning patients
        primary_condition = None
        if noteId:
            from server.database.repositories.encounter import get_patient_by_id

            existing_patient = get_patient_by_id(noteId)
            if existing_patient and existing_patient.get("primary_condition"):
                primary_condition = existing_patient["primary_condition"]

        # Create patient context
        patient_context = {"name": formatted_name, "dob": dob, "gender": gender}

        # Process the transcription with template fields
        processing_result = await process_transcription(
            transcript_text=transcript_text,
            template_fields=template_fields,
            patient_context=patient_context,
            is_ambient=isAmbient,
            primary_condition=primary_condition,
        )

        # Return the response in the expected format
        return TranscribeResponse(
            fields=dict(processing_result["fields"]),
            rawTranscription=transcript_text,
            transcriptionDuration=transcription_duration,
            processDuration=float(processing_result["process_duration"]),
        )

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/dictate")
async def dictate(file: UploadFile = File(...)):
    """Transcribes the dictated audio."""
    try:
        # Read the audio file
        audio_buffer = await read_upload_limited(file, MAX_AUDIO_UPLOAD_BYTES, "Audio upload")

        # Perform transcription
        transcription_result = await transcribe_audio(audio_buffer)
        transcript_text = str(transcription_result["text"])
        transcription_duration = float(transcription_result["transcriptionDuration"])

        # Return the response
        return {
            "transcription": transcript_text,
            "transcriptionDuration": transcription_duration,
        }
    except Exception as e:
        logging.error(f"Error occurred during dictation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/reprocess", response_model=TranscribeResponse)
async def reprocess_transcription(
    transcript_text: str = Form(...),
    name: str | None = Form(None),
    gender: str | None = Form(None),
    dob: str | None = Form(None),
    original_transcription_duration: float | None = Form(0),
    templateKey: str | None = Form(None),
    isAmbient: bool = Form(True),
    noteId: int | None = Form(None),
):
    """Reprocesses an existing transcription."""
    try:
        # Process the name if provided
        formatted_name = _format_patient_display_name(name)

        # Get template fields if template key is provided
        template_fields = []
        if templateKey:
            from server.database.repositories.templates import get_template_fields

            template_fields = get_template_fields(templateKey)

        # Look up primary condition for returning patients
        primary_condition = None
        if noteId:
            from server.database.repositories.encounter import get_patient_by_id

            existing_patient = get_patient_by_id(noteId)
            if existing_patient and existing_patient.get("primary_condition"):
                primary_condition = existing_patient["primary_condition"]

        # Create patient context
        patient_context = {"name": formatted_name, "dob": dob, "gender": gender}

        # Process the transcription with template fields
        processing_result = await process_transcription(
            transcript_text=transcript_text,
            template_fields=template_fields,
            patient_context=patient_context,
            is_ambient=isAmbient,
            primary_condition=primary_condition,
        )

        # Return the response in the expected format
        return TranscribeResponse(
            fields=dict(processing_result["fields"]),
            rawTranscription=transcript_text,
            transcriptionDuration=original_transcription_duration or 0.0,
            processDuration=float(processing_result["process_duration"]),
        )

    except Exception as e:
        logging.error(f"Error occurred during reprocessing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/process-document", response_model=TranscribeResponse)  # Changed response model
async def process_document(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    gender: str | None = Form(None),
    dob: str | None = Form(None),
    templateKey: str = Form(..., description="Template key is required for document processing"),
):
    """Processes a document to extract information and fill template fields."""
    try:
        # Read the document file
        document_buffer = await read_upload_limited(
            file, MAX_DOCUMENT_UPLOAD_BYTES, "Document upload"
        )

        # Get the file type
        content_type = file.content_type

        # Process the name if provided
        formatted_name = _format_patient_display_name(name)

        from server.database.repositories.templates import get_template_fields

        template_fields = get_template_fields(templateKey)

        # Create patient context
        patient_context = {"name": formatted_name, "dob": dob, "gender": gender}

        # Process the document
        process_start = time.perf_counter()
        result = await process_document_with_template(
            document_buffer, content_type or "", template_fields, patient_context
        )
        process_end = time.perf_counter()
        process_duration = process_end - process_start

        # The result is already in the format of field key-value pairs
        return TranscribeResponse(
            fields=result,
            rawTranscription="",  # We don't include raw transcription for document uploads
            transcriptionDuration=0,  # No transcription for documents
            processDuration=process_duration,
        )
    except Exception as e:
        logging.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/extract-demographics")
async def extract_demographics(file: UploadFile = File(...)):
    """Extract patient demographics from an uploaded document (referral, ID, etc.)."""
    try:
        document_buffer = await read_upload_limited(
            file, MAX_DOCUMENT_UPLOAD_BYTES, "Document upload"
        )
        result = await extract_demographics_from_document(document_buffer, file.content_type or "")
        return result
    except Exception as e:
        logging.error(f"Error extracting demographics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/extract-demographics-from-text")
async def extract_demographics_from_text(payload: ExtractDemographicsFromTextRequest):
    """Extract patient demographics from already-extracted document text."""
    try:
        extracted_text = (payload.extracted_text or "").strip()
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No extracted_text provided")
        return await _extract_demographics_from_text(extracted_text)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error extracting demographics from text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/extract-demographics-visual")
async def extract_demographics_visual(payload: ExtractDemographicsVisualRequest):
    """Extract patient demographics from rendered document page images."""
    try:
        if not payload.pages:
            raise HTTPException(status_code=400, detail="No visual pages provided")
        visual_pages = [page.model_dump() for page in payload.pages]
        return await extract_demographics_from_visual_pages(visual_pages)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error extracting demographics from visual: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/process-document-visual", response_model=TranscribeResponse)
async def process_document_visual(payload: ProcessVisualDocumentRequest):
    """Processes visual document pages directly with multimodal field extraction."""
    try:
        if not payload.pages:
            raise HTTPException(status_code=400, detail="No visual pages provided")

        # Process the name if provided
        formatted_name = _format_patient_display_name(payload.name)

        from server.database.repositories.templates import get_template_fields

        template_fields = get_template_fields(payload.templateKey)

        # Create patient context
        patient_context = {
            "name": formatted_name,
            "dob": payload.dob,
            "gender": payload.gender,
        }

        # Convert pydantic models to dicts expected by visual processor
        visual_pages = [page.model_dump() for page in payload.pages]

        process_start = time.perf_counter()
        result = await process_visual_document_with_template(
            visual_pages=visual_pages,
            template_fields=template_fields,
            patient_context=patient_context,
        )
        process_end = time.perf_counter()
        process_duration = process_end - process_start

        return TranscribeResponse(
            fields=result,
            rawTranscription="",
            transcriptionDuration=0,
            processDuration=process_duration,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing visual document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/process-document-from-text", response_model=TranscribeResponse)
async def process_document_from_text(payload: ProcessDocumentFromTextRequest):
    """Processes already-extracted document text and fills template fields."""
    try:
        extracted_text = (payload.extracted_text or "").strip()
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No extracted_text provided")

        # Process the name if provided
        formatted_name = _format_patient_display_name(payload.name)

        from server.database.repositories.templates import get_template_fields

        template_fields = get_template_fields(payload.templateKey)

        # Create patient context
        patient_context = {
            "name": formatted_name,
            "dob": payload.dob,
            "gender": payload.gender,
        }

        # Process extracted text directly (no file/OCR step)
        process_start = time.perf_counter()
        result = await process_document_text_with_template(
            extracted_text=extracted_text,
            template_fields=template_fields,
            patient_context=patient_context,
        )
        process_end = time.perf_counter()
        process_duration = process_end - process_start

        return TranscribeResponse(
            fields=result,
            rawTranscription="",
            transcriptionDuration=0,
            processDuration=process_duration,
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing extracted document text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

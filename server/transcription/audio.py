import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import wave
from pathlib import Path
from typing import Union

import httpx

from server.utils.ssrf import build_guarded_http_client

from server.database.config.manager import config_manager
from server.transcription.language import normalize_persian_text, resolve_asr_language

logger = logging.getLogger(__name__)


async def _post_audio(client: httpx.AsyncClient, url: str, **kwargs):
    """POST audio with retries on 429/5xx/network errors only."""
    from server.utils.http_retry import (
        ProviderHTTPError,
        is_retryable_status,
        sanitize_provider_error,
        with_retries,
    )

    async def _do():
        response = await client.post(url, **kwargs)
        if is_retryable_status(response.status_code):
            raise ProviderHTTPError(
                f"ASR provider error ({response.status_code}): "
                f"{sanitize_provider_error(response.text)}",
                status_code=response.status_code,
            )
        return response

    return await with_retries(_do, operation_name="asr")


def _get_whisper_port() -> str:
    """Get the whisper server port from global state."""
    from server.utils.allocated_ports import get_whisper_port

    return str(get_whisper_port())


def _validate_local_model_language(model_id: str, language: str) -> None:
    """Reject language/model combinations the local engines cannot do.

    The local catalog only ships three engines:
    - Whisper large-v3-turbo: multilingual — fa, en, and mixed (``auto``).
    - Parakeet TDT 0.6B v3: **not** a Persian model (25 European languages).
    - Shenava Koochik: **Persian-only**.

    Running Persian/auto through Parakeet or English through Shenava silently
    produces garbage text, so fail with an actionable message instead.
    """
    if not model_id or not model_id.startswith(("shenava-", "parakeet-")):
        return
    if model_id.startswith("parakeet-") and language in {"fa", "auto"}:
        raise ValueError(
            "Parakeet is an English/European-language model and cannot transcribe "
            "Persian or mixed Persian/English speech. Select a Whisper large-v3-turbo "
            "model, Shenava (Persian only), or an online provider for this language."
        )
    if model_id.startswith("shenava-") and language == "en":
        raise ValueError(
            "Shenava is a Persian-only model and cannot transcribe English. "
            "Select a Whisper large-v3-turbo model or an online provider for English."
        )


async def transcribe_audio(audio_buffer: bytes) -> dict[str, Union[str, float]]:
    """
    Transcribe an audio buffer using an OpenAI-compatible ASR endpoint.

    The endpoint is instructed to transcribe (never translate) and receives
    either Persian (``fa``), English (``en``), or no language hint (``auto``)
    for mixed Persian/English recordings.
    """
    try:
        config = config_manager.get_config()

        from server.utils.providers import resolve_asr_connection

        connection = resolve_asr_connection(config)
        provider = connection["provider"]
        protocol = connection["protocol"]
        model_id = str(
            connection.get("model") or config.get("ASR_MODEL") or config.get("WHISPER_MODEL") or ""
        ).strip()

        if provider == "local" or protocol == "local":
            if not model_id:
                try:
                    from server.utils.whisper_models import asr_model_manager

                    model_id = asr_model_manager.get_selected_model_id() or ""
                except Exception:
                    model_id = ""
            _validate_local_model_language(model_id, resolve_asr_language(config))
            if model_id.startswith("shenava-"):
                logger.info("Using local Shenava ASR for transcription")
                return await _transcribe_local_shenava(audio_buffer, config)
            if model_id.startswith("parakeet-"):
                logger.info("Using local Parakeet ASR for transcription")
                return await _transcribe_local_parakeet(audio_buffer, config)
            logger.info("Using local Whisper.cpp ASR for transcription")
            return await _transcribe_local_whisper(audio_buffer, config)
        if protocol == "speechmatics" or provider == "speechmatics":
            logger.info("Using Speechmatics Batch REST API for file transcription")
            return await _transcribe_speechmatics(audio_buffer, config)
        if protocol == "fireworks" or provider == "fireworks":
            logger.info("Using Fireworks ASR for transcription")
            return await _transcribe_fireworks(audio_buffer, config)

        logger.info("Using external OpenAI-compatible ASR API for transcription")
        return await _transcribe_external_api(audio_buffer, config)
    except Exception as error:
        logger.error(f"Error in transcribe_audio function: {error}")
        raise


async def _transcribe_local_whisper(
    audio_buffer: bytes, _config: dict
) -> dict[str, Union[str, float]]:
    """Transcribe using the local whisper.cpp OpenAI-compatible server."""
    whisper_port = _get_whisper_port()
    whisper_url = f"http://127.0.0.1:{whisper_port}/v1/audio/transcriptions"

    logger.info(f"Sending audio to local STT server at {whisper_url}")

    filename, content_type = _detect_audio_format(audio_buffer)

    async with build_guarded_http_client(timeout=httpx.Timeout(600.0)) as client:
        files = {"file": (filename, audio_buffer, content_type)}
        language = resolve_asr_language(_config)
        data = {
            "response_format": "verbose_json",
            "temperature": "0.0",
            "task": "transcribe",
        }
        # Whisper.cpp can receive an explicit language hint. Omitting it keeps
        # auto-detection available for mixed Persian/English recordings.
        if language != "auto":
            data["language"] = language

        transcription_start = time.perf_counter()

        try:
            response = await _post_audio(client, whisper_url, data=data, files=files)
            transcription_end = time.perf_counter()
            transcription_duration = transcription_end - transcription_start

            if response.status_code != 200:
                from server.utils.http_retry import sanitize_provider_error

                raise ValueError(
                    f"Whisper local server error: {sanitize_provider_error(response.text)}"
                )

            try:
                result = response.json()
            except Exception as e:
                raise ValueError(f"Failed to parse response: {e}") from e

            if "text" not in result:
                raise ValueError("No text in whisper.cpp response")

            if "segments" in result:
                transcript_text = "\n".join(
                    segment["text"].strip() for segment in result["segments"]
                )
            else:
                transcript_text = result["text"]

            # Clean repetitive text patterns
            transcript_text = normalize_persian_text(_clean_repetitive_text(transcript_text))

            return {
                "text": transcript_text,
                "transcriptionDuration": float(f"{transcription_duration:.2f}"),
            }
        except httpx.RequestError as e:
            raise ValueError(f"Cannot connect to local ASR server: {e}") from e


def _read_pcm_wav(audio_buffer: bytes) -> tuple[bytes, int]:
    """Return mono 16-bit PCM frames and their sample rate from a WAV buffer."""
    try:
        with wave.open(io.BytesIO(audio_buffer), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
                raise ValueError("ASR audio must be uncompressed 16-bit PCM WAV")
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError("Shenava requires a valid uncompressed 16-bit PCM WAV recording") from error

    if channels == 1:
        return frames, sample_rate
    if channels != 2:
        raise ValueError("ASR audio must have one or two channels")

    # The browser recorder is mono, but downmix stereo recordings defensively.
    import array

    samples = array.array("h")
    samples.frombytes(frames)
    mono = array.array(
        "h", ((left + right) // 2 for left, right in zip(samples[::2], samples[1::2], strict=True))
    )
    return mono.tobytes(), sample_rate


# Speechmatics Batch REST API (per https://docs.speechmatics.com/batch.yaml).
# The SaaS batch endpoint is EU1 for all customers; regional/enterprise or
# on-prem runtimes can be configured via ASR_BATCH_URL / SPEECHMATICS_BATCH_URL.
SPEECHMATICS_BATCH_DEFAULT_URL = "https://eu1.asr.api.speechmatics.com/v2"
# Uploads can be long (e.g. clinic visits), so allow a generous window but
# still make sure the request cannot hang forever.
SPEECHMATICS_BATCH_POLL_SECONDS = 900
SPEECHMATICS_BATCH_POLL_WAIT = 20


def speechmatics_batch_url(config: dict) -> str:
    """Resolve the Speechmatics Batch REST base URL for file transcription.

    Priority: ``ASR_BATCH_URL`` config → ``SPEECHMATICS_BATCH_URL`` env →
    an http(s) ``ASR_BASE_URL`` (custom/on-prem host) → the documented SaaS EU1
    endpoint. A ``wss://`` realtime URL must NOT be reused: Batch and Realtime
    are separate product surfaces with separate hosts.
    """
    url = str(config.get("ASR_BATCH_URL") or config.get("WHISPER_BATCH_URL") or "").strip()
    if url:
        return url.rstrip("/")
    url = os.environ.get("SPEECHMATICS_BATCH_URL") or ""
    if url.strip():
        return url.strip().rstrip("/")
    base = str(config.get("ASR_BASE_URL") or config.get("WHISPER_BASE_URL") or "").strip()
    if base.lower().startswith(("http://", "https://")):
        return base.rstrip("/")
    return SPEECHMATICS_BATCH_DEFAULT_URL


def _speechmatics_batch_key(config: dict) -> str:
    """Return the Batch-scoped API key.

    Speechmatics API keys are product-scoped (``type=rt`` vs ``type=batch``),
    so the Realtime key may not work for Batch and vice versa. A dedicated
    ``ASR_BATCH_KEY`` wins; the primary ``ASR_KEY`` is the fallback.
    """
    return str(
        config.get("ASR_BATCH_KEY")
        or config.get("WHISPER_BATCH_KEY")
        or config.get("ASR_KEY")
        or config.get("WHISPER_KEY")
        or ""
    ).strip()


async def _transcribe_speechmatics(
    audio_buffer: bytes, config: dict
) -> dict[str, Union[str, float]]:
    """Transcribe a recording through the speechmatics Batch REST API.

    Used for the after-the-fact file path (``/api/transcribe/audio``). Live
    mic streaming uses ``server/transcription/live.py`` instead.

    Flow (per batch.yaml):
      1. POST ``/jobs`` (multipart: ``config`` JSON + ``data_file``)
      2. GET  ``/jobs/{id}/transcript?format=txt&wait=…`` until 200
      3. GET  ``/jobs/{id}`` for the audio duration
    """
    filename, content_type = _detect_audio_format(audio_buffer)
    api_key = _speechmatics_batch_key(config)
    if not api_key:
        raise ValueError(
            "A Speechmatics Batch API key is required for file transcription "
            "(set ASR_BATCH_KEY or ASR_KEY in Settings)"
        )
    base_url = speechmatics_batch_url(config)

    # Batch supports automatic language identification (``auto``), unlike
    # Realtime. Pin the expected languages so the medical Persian/English mix
    # is never transcribed as a third language, and fall back to Persian when
    # confidence is low.
    language = resolve_asr_language(config)
    model = str(config.get("ASR_MODEL") or "enhanced").strip().lower()
    if model not in {"standard", "enhanced", "melia-1"}:
        model = "enhanced"
    # Melia 1 is multilingual and switches languages automatically, but it does
    # NOT support ``language``/``auto`` in the lang-identification sense: it
    # accepts an ISO hint only. Also it is Batch-only (not available on the
    # Realtime WebSocket SDK used for live transcription).
    if model == "melia-1" and language == "auto":
        language = "fa"
        logger.info("Melia 1 does not support language identification; using Persian hint 'fa'")
    job_config: dict[str, object] = {
        "type": "transcription",
        "transcription_config": {
            "language": language,
            "model": model,
            "enable_entities": True,
        },
    }
    if language == "auto":
        job_config["language_identification_config"] = {
            "expected_languages": ["fa", "en"],
            "low_confidence_action": "use_default_language",
            "default_language": "fa",
        }

    headers = {"Authorization": f"Bearer {api_key}"}
    started = time.perf_counter()
    try:
        async with build_guarded_http_client(timeout=httpx.Timeout(60.0)) as client:
            try:
                response = await _post_audio(
                    client,
                    f"{base_url}/jobs",
                    data={"config": json.dumps(job_config)},
                    files={"data_file": (filename, audio_buffer, content_type)},
                    headers=headers,
                )
            except Exception as error:
                raise ValueError(f"Speechmatics batch submit failed: {error}") from error

            from server.utils.http_retry import sanitize_provider_error

            if response.status_code != 201:
                detail = sanitize_provider_error(response.text)
                if response.status_code == 401:
                    raise ValueError(
                        "Speechmatics authentication failed (401): the API key is not "
                        f"valid for the Batch API. Create a key with type=batch. {detail}"
                    )
                if response.status_code == 403:
                    raise ValueError(f"Speechmatics request forbidden (403): {detail}")
                if response.status_code == 429:
                    raise ValueError(f"Speechmatics rate limited (429): {detail}")
                raise ValueError(
                    f"Speechmatics batch job rejected ({response.status_code}): {detail}"
                )

            try:
                job_id = str(response.json()["id"])
            except Exception as error:
                raise ValueError(f"Speechmatics batch response missing job id: {error}") from error

            # Poll the transcript endpoint. ``wait`` blocks server-side, so the
            # loop is quiet while the job is being processed.
            transcript_text: str | None = None
            deadline = time.monotonic() + SPEECHMATICS_BATCH_POLL_SECONDS
            while time.monotonic() < deadline:
                transcript_response = await client.get(
                    f"{base_url}/jobs/{job_id}/transcript",
                    params={"format": "txt", "wait": SPEECHMATICS_BATCH_POLL_WAIT},
                    headers=headers,
                )
                if transcript_response.status_code == 200:
                    transcript_text = transcript_response.text
                    break
                if transcript_response.status_code in (404, 423):
                    await asyncio.sleep(0.5)
                    continue
                if transcript_response.status_code == 429:
                    await asyncio.sleep(2.0)
                    continue
                detail = sanitize_provider_error(transcript_response.text)
                if transcript_response.status_code == 401:
                    raise ValueError(
                        "Speechmatics authentication failed (401) while fetching "
                        f"the batch transcript: {detail}"
                    )
                raise ValueError(
                    "Speechmatics batch transcript fetch failed "
                    f"({transcript_response.status_code}): {detail}"
                )
            if transcript_text is None:
                raise ValueError(
                    "Speechmatics batch transcription timed out after "
                    f"{SPEECHMATICS_BATCH_POLL_SECONDS}s; the job may still be running"
                )

            # Report the real audio duration when the provider gives it.
            duration = 0.0
            try:
                job_response = await client.get(f"{base_url}/jobs/{job_id}", headers=headers)
                if job_response.status_code == 200:
                    duration = float(
                        (job_response.json().get("job") or {}).get("duration") or 0
                    )
            except Exception:
                logger.debug("Speechmatics job details fetch failed", exc_info=True)
    except httpx.RequestError as error:
        raise ValueError(f"Speechmatics batch transcription failed: {error}") from error

    transcript_text = normalize_persian_text(_clean_repetitive_text(transcript_text))
    if not transcript_text:
        raise ValueError("Speechmatics returned no transcript")
    return {
        "text": transcript_text,
        "transcriptionDuration": duration
        or float(f"{time.perf_counter() - started:.2f}"),
    }


_SHENAVA_CACHE: tuple[object, list[str], tuple[str, str]] | None = None
_SHENAVA_LOCK = threading.Lock()


def _load_shenava_runtime(model_path: str, tokens_path: str):
    """Load the optional Shenava ONNX runtime once per model bundle."""
    global _SHENAVA_CACHE
    with _SHENAVA_LOCK:
        if _SHENAVA_CACHE and _SHENAVA_CACHE[2] == (model_path, tokens_path):
            return _SHENAVA_CACHE[0], _SHENAVA_CACHE[1]
        try:
            import importlib.util

            if importlib.util.find_spec("numpy") is None:
                raise ImportError("numpy is not installed")
            import onnxruntime as ort
        except ImportError as error:
            raise ValueError(
                "Shenava support requires the local ONNX runtime in this server build"
            ) from error

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        tokens = Path(tokens_path).read_text(encoding="utf-8").splitlines()
        _SHENAVA_CACHE = (session, tokens, (model_path, tokens_path))
        return session, tokens


def _shenava_log_mel(pcm: bytes, sample_rate: int):
    """Build the 80-bin NeMo-style log-mel input expected by Shenava."""
    import numpy as np

    if sample_rate != 16000:
        raise ValueError("Shenava requires 16 kHz audio")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    n_fft, win_length, hop = 512, 400, 160
    if samples.size < win_length:
        samples = np.pad(samples, (0, win_length - samples.size))
    frame_count = 1 + max(0, (samples.size - win_length) // hop)
    padded_size = (frame_count - 1) * hop + win_length
    samples = np.pad(samples, (0, max(0, padded_size - samples.size)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, win_length)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(win_length), n=n_fft)) ** 2

    # HTK mel filter bank, matching the usual FastConformer preprocessing.
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(8000), 82))
    bins = np.floor((n_fft + 1) * mel_points / sample_rate).astype(int)
    filters = np.zeros((80, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, 81):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            filters[m - 1, left:center] = np.arange(left, center) - left
            filters[m - 1, left:center] /= center - left
        if right > center:
            filters[m - 1, center:right] = right - np.arange(center, right)
            filters[m - 1, center:right] /= right - center
    mel = np.maximum(spectrum @ filters.T, 1e-10)
    return np.log(mel).T.astype(np.float32)


def _run_shenava_inference(audio_buffer: bytes, config: dict) -> str:
    """Run Shenava's cache-aware CTC graph over 121-frame chunks."""
    import numpy as np

    from server.utils.whisper_models import asr_model_manager

    model_id = str(config.get("ASR_MODEL") or config.get("WHISPER_MODEL") or "")
    model_path = asr_model_manager.get_model_path(model_id)
    info = asr_model_manager.get_available_models()
    model_info = next((model for model in info if model["id"] == model_id), None)
    if not model_path or not model_info or model_info.get("runtime") != "shenava_onnx":
        raise ValueError("The selected Shenava model is not downloaded")
    tokens_path = asr_model_manager.models_dir / "shenava-koochik-v1.0-tokens.txt"
    if not tokens_path.exists():
        raise ValueError("Shenava vocabulary file is missing; download the model again")

    pcm, sample_rate = _read_pcm_wav(audio_buffer)
    mel = _shenava_log_mel(pcm, sample_rate)
    session, tokens = _load_shenava_runtime(str(model_path), str(tokens_path))
    inputs = session.get_inputs()
    cache_inputs = []
    cache_values: dict[str, np.ndarray] = {}
    for item in inputs:
        name = item.name.lower()
        if "audio_signal" not in name and name != "length":
            cache_inputs.append(item)
            shape = [int(dim) if isinstance(dim, int) else 1 for dim in item.shape]
            dtype = np.int64 if "len" in name else np.float32
            cache_values[item.name] = np.zeros(shape, dtype=dtype)

    previous_token = None
    output_tokens: list[str] = []
    frame_start = 0
    while frame_start < mel.shape[1] or frame_start == 0:
        frame_end = min(frame_start + 121, mel.shape[1])
        valid_frames = max(1, frame_end - frame_start)
        chunk = np.zeros((80, 121), dtype=np.float32)
        if frame_end > frame_start:
            chunk[:, :valid_frames] = mel[:, frame_start:frame_end]
        feed: dict[str, np.ndarray] = {}
        for item in inputs:
            name = item.name.lower()
            if "audio_signal" in name:
                feed[item.name] = chunk[None, :, :]
            elif name == "length" or name.endswith("_length") or name.endswith("length"):
                feed[item.name] = np.array([valid_frames], dtype=np.int64)
            else:
                feed[item.name] = cache_values[item.name]

        outputs = session.run(None, feed)
        output_meta = session.get_outputs()
        logits_index = next(
            (index for index, item in enumerate(output_meta) if len(item.shape) >= 3),
            0,
        )
        logits = np.asarray(outputs[logits_index])
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise ValueError("Unexpected Shenava logits shape")
        if logits.shape[0] > logits.shape[1]:
            logits = logits.T
        for token_id in np.argmax(logits, axis=1).tolist():
            if token_id == 1024:
                previous_token = token_id
                continue
            if token_id == previous_token:
                continue
            if 0 <= token_id < len(tokens):
                output_tokens.append(tokens[token_id])
            previous_token = token_id

        # Feed returned cache tensors into the next chunk. ONNX exports do not
        # agree on whether the output is called ``cache_last_channel_next`` or
        # simply ``cache_last_channel``; match by stem first and fall back to
        # the documented input/output cache order.
        cache_outputs = [
            (index, item)
            for index, item in enumerate(output_meta)
            if index != logits_index and "cache" in item.name.lower()
        ]
        used_outputs: set[int] = set()
        for input_item in cache_inputs:
            input_name = input_item.name.lower()
            match = next(
                (
                    (index, item)
                    for index, item in cache_outputs
                    if index not in used_outputs
                    and (
                        input_name in item.name.lower() or item.name.lower().startswith(input_name)
                    )
                ),
                None,
            )
            if match is None:
                match = next(
                    ((index, item) for index, item in cache_outputs if index not in used_outputs),
                    None,
                )
            if match is not None:
                index, _ = match
                cache_values[input_item.name] = np.asarray(outputs[index])
                used_outputs.add(index)
        frame_start += 112

    text = "".join(token.replace("▁", " ") for token in output_tokens).strip()
    return normalize_persian_text(_clean_repetitive_text(text))


async def _transcribe_local_shenava(
    audio_buffer: bytes, config: dict
) -> dict[str, Union[str, float]]:
    """Transcribe with Shenava without requiring a running C++ sidecar."""
    started = time.perf_counter()
    text = await asyncio.to_thread(_run_shenava_inference, audio_buffer, config)
    if not text:
        raise ValueError("Shenava returned no transcript")
    return {
        "text": text,
        "transcriptionDuration": float(f"{time.perf_counter() - started:.2f}"),
    }


async def _transcribe_local_parakeet(
    audio_buffer: bytes, config: dict
) -> dict[str, Union[str, float]]:
    """Transcribe with local Parakeet TDT ONNX (no whisper.cpp sidecar)."""
    from server.transcription.parakeet import run_parakeet_inference

    started = time.perf_counter()
    text = await asyncio.to_thread(run_parakeet_inference, audio_buffer, config)
    if not text:
        raise ValueError("Parakeet returned no transcript")
    return {
        "text": text,
        "transcriptionDuration": float(f"{time.perf_counter() - started:.2f}"),
    }


def _fireworks_batch_url(config: dict) -> str:
    """Return the Fireworks batch ASR host for the selected model."""
    from server.utils.providers import ASR_PROVIDERS

    model = str(config.get("ASR_MODEL") or config.get("WHISPER_MODEL") or "").strip()
    info = ASR_PROVIDERS["fireworks"]
    batch_urls = info.get("batch_urls") or {}
    if model in batch_urls:
        return str(batch_urls[model])
    if "turbo" in model:
        return str(batch_urls.get("whisper-v3-turbo") or info["default_base_url"])
    configured = str(config.get("ASR_BASE_URL") or config.get("WHISPER_BASE_URL") or "").strip()
    return configured or str(info["default_base_url"])


async def _transcribe_fireworks(audio_buffer: bytes, config: dict) -> dict[str, Union[str, float]]:
    """Transcribe via Fireworks batch Whisper v3 / turbo HTTP API."""
    filename, content_type = _detect_audio_format(audio_buffer)
    model = str(config.get("ASR_MODEL") or config.get("WHISPER_MODEL") or "whisper-v3")
    # Streaming-only Fireworks ASR models still accept a Whisper v3 batch fallback.
    if model.startswith("fireworks-asr"):
        model = "whisper-v3"
    api_key = str(config.get("ASR_KEY") or config.get("WHISPER_KEY") or "").strip()
    if not api_key:
        raise ValueError("A Fireworks API key is required for the selected ASR provider")

    language = resolve_asr_language(config)
    if language == "auto":
        # Fireworks' documented language list has no ``auto``; omitting the
        # field can fall back to the provider default (English), which would
        # mangle Persian. This app is Persian-first, so pin ``fa`` and keep
        # true mixed fa/en detection to Speechmatics Batch or local Whisper.
        language = "fa"
    data = {
        "model": model,
        "temperature": "0.0",
        "response_format": "verbose_json",
        "task": "transcribe",
        "language": language,
    }

    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = _fireworks_batch_url(config).rstrip("/")
    if base_url.lower().endswith("/v1"):
        base_url = base_url[:-3]

    transcription_start = time.perf_counter()
    async with build_guarded_http_client(timeout=httpx.Timeout(600.0)) as client:
        try:
            response = await _post_audio(
                client,
                f"{base_url}/v1/audio/transcriptions",
                data=data,
                files={"file": (filename, audio_buffer, content_type)},
                headers=headers,
            )
        except httpx.RequestError as error:
            raise ValueError(f"Fireworks transcription failed: {error}") from error

    from server.utils.http_retry import sanitize_provider_error

    body = sanitize_provider_error(response.text)
    if response.status_code == 401:
        raise ValueError(f"Fireworks authentication failed (401): {body}")
    if response.status_code == 403:
        raise ValueError(f"Fireworks request forbidden (403): {body}")
    if response.status_code == 429:
        raise ValueError(f"Fireworks rate limited (429): {body}")
    if response.status_code != 200:
        raise ValueError(f"Fireworks transcription failed ({response.status_code}): {body}")

    try:
        result = response.json()
    except Exception as error:
        raise ValueError(f"Failed to parse Fireworks response: {error}") from error
    if "text" not in result:
        raise ValueError("Fireworks returned no transcript")
    if "segments" in result:
        transcript_text = "\n".join(segment["text"].strip() for segment in result["segments"])
    else:
        transcript_text = result["text"]
    transcript_text = normalize_persian_text(_clean_repetitive_text(transcript_text))
    return {
        "text": transcript_text,
        "transcriptionDuration": float(f"{time.perf_counter() - transcription_start:.2f}"),
    }


async def _transcribe_external_api(
    audio_buffer: bytes, config: dict
) -> dict[str, Union[str, float]]:
    """Transcribe using an external OpenAI-compatible ASR API."""
    filename, content_type = _detect_audio_format(audio_buffer)
    async with build_guarded_http_client(timeout=httpx.Timeout(600.0)) as client:
        files = {"file": (filename, audio_buffer, content_type)}
        language = resolve_asr_language(config)
        data = {
            "model": config.get("ASR_MODEL") or config.get("WHISPER_MODEL", "whisper-1"),
            "temperature": "0.1",
            "vad_filter": "true",
            "task": "transcribe",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        # Omitting language lets multilingual engines detect a mixed
        # Persian/English recording. ``fa`` gives Persian-first decoding.
        if language != "auto":
            data["language"] = language

        transcription_start = time.perf_counter()

        headers = {}
        whisper_key = (config.get("ASR_KEY") or config.get("WHISPER_KEY") or "").strip()
        if whisper_key:
            headers["Authorization"] = f"Bearer {whisper_key}"

        try:
            whisper_base_url = (
                (config.get("ASR_BASE_URL") or config.get("WHISPER_BASE_URL") or "")
                .strip()
                .rstrip("/")
            )
            if whisper_base_url.lower().endswith("/v1"):
                whisper_base_url = whisper_base_url[:-3]

            response = await _post_audio(
                client,
                f"{whisper_base_url}/v1/audio/transcriptions",
                data=data,
                files=files,
                headers=headers,
            )
        except httpx.RequestError as e:
            raise ValueError(f"Transcription failed: {e}") from e

        transcription_end = time.perf_counter()
        transcription_duration = transcription_end - transcription_start

        if response.status_code != 200:
            from server.utils.http_retry import sanitize_provider_error

            raise ValueError(f"Transcription failed: {sanitize_provider_error(response.text)}")

        try:
            result = response.json()
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}") from e

        if "text" not in result:
            raise ValueError("Transcription failed, no text in response")

        if "segments" in result:
            # Extract text from each segment and join with newlines
            transcript_text = "\n".join(segment["text"].strip() for segment in result["segments"])
        else:
            transcript_text = result["text"]

        # Clean repetitive text patterns and normalize Arabic/Persian Unicode.
        transcript_text = normalize_persian_text(_clean_repetitive_text(transcript_text))

        return {
            "text": transcript_text,
            "transcriptionDuration": float(f"{transcription_duration:.2f}"),
        }


def _clean_repetitive_text(text: str) -> str:
    """
    Clean up repetitive text patterns that might appear in transcripts.

    Args:
        text (str): The text to clean

    Returns:
        str: Cleaned text
    """
    # Pattern to find repetitions of the same word/phrase 3+ times in succession
    pattern = r"(\b\w+[\s\w]*?\b)(\s+\1){3,}"

    # Replace with just two instances
    cleaned_text = re.sub(pattern, r"\1 \1", text)

    # If the text changed, recursively clean again (for nested repetitions)
    if cleaned_text != text:
        return _clean_repetitive_text(cleaned_text)

    return cleaned_text


def _detect_audio_format(audio_buffer):
    """
    Simple audio format detection based on file signatures (magic numbers).
    """
    # Check file signatures for common audio formats
    if audio_buffer.startswith(b"ID3") or audio_buffer.startswith(b"\xff\xfb"):
        return "recording.mp3", "audio/mpeg"
    elif audio_buffer.startswith(b"RIFF") and b"WAVE" in audio_buffer[0:12]:
        return "recording.wav", "audio/wav"
    elif audio_buffer.startswith(b"OggS"):
        return "recording.ogg", "audio/ogg"
    elif audio_buffer.startswith(b"fLaC"):
        return "recording.flac", "audio/flac"
    elif b"ftyp" in audio_buffer[0:20]:  # M4A/MP4 format
        return "recording.m4a", "audio/mp4"
    # Default to WAV if we can't determine
    return "recording.wav", "audio/wav"

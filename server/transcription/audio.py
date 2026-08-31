import asyncio
import io
import logging
import re
import threading
import time
import wave
from pathlib import Path
from typing import Union

import httpx

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
            if model_id.startswith("shenava-"):
                logger.info("Using local Shenava ASR for transcription")
                return await _transcribe_local_shenava(audio_buffer, config)
            if model_id.startswith("parakeet-"):
                logger.info("Using local Parakeet ASR for transcription")
                return await _transcribe_local_parakeet(audio_buffer, config)
            logger.info("Using local Whisper.cpp ASR for transcription")
            return await _transcribe_local_whisper(audio_buffer, config)
        if protocol == "speechmatics" or provider == "speechmatics":
            logger.info("Using Speechmatics realtime ASR for transcription")
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

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
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
        raise ValueError("Speechmatics and Shenava require a valid WAV recording") from error

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


async def _transcribe_speechmatics(
    audio_buffer: bytes, config: dict
) -> dict[str, Union[str, float]]:
    """Transcribe a complete recording through Speechmatics Realtime."""
    try:
        from speechmatics.rt import (
            AsyncClient,
            AudioEncoding,
            AudioFormat,
            Model,
            ServerMessageType,
            TranscriptionConfig,
            TranscriptResult,
        )
    except ImportError as error:
        raise ValueError("Speechmatics support is not installed in this server build") from error

    api_key = str(config.get("ASR_KEY") or config.get("WHISPER_KEY") or "").strip()
    if not api_key:
        raise ValueError("A Speechmatics API key is required for the selected ASR provider")

    pcm, sample_rate = _read_pcm_wav(audio_buffer)
    language = resolve_asr_language(config)
    # Speechmatics supports automatic language identification for the mixed
    # Persian/English workflow. Keep an explicit ``fa`` or ``en`` hint when the
    # user chooses one in settings.
    speechmatics_language = "auto" if language == "auto" else language
    # Map the configured operating point onto the v1 ``Model`` enum; any
    # unrecognised value falls back to the default ``enhanced`` model.
    model_name = str(config.get("ASR_MODEL") or "enhanced").strip().lower()
    model = Model.STANDARD if model_name == "standard" else Model.ENHANCED

    transcript_parts: list[str] = []

    def handle_final(message: dict) -> None:
        try:
            result = TranscriptResult.from_message(message)
            text = result.metadata.transcript
            if text:
                transcript_parts.append(text)
        except (KeyError, TypeError, AttributeError):
            logger.debug("Ignoring malformed Speechmatics transcript event", exc_info=True)

    client_url = str(config.get("ASR_BASE_URL") or "").strip() or None
    client = AsyncClient(api_key=api_key, url=client_url)
    client.on(ServerMessageType.ADD_TRANSCRIPT, handle_final)
    started = time.perf_counter()
    try:
        await client.transcribe(
            io.BytesIO(pcm),
            transcription_config=TranscriptionConfig(
                language=speechmatics_language,
                model=model,
                enable_partials=False,
            ),
            audio_format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE,
                sample_rate=sample_rate,
                chunk_size=4096,
            ),
            timeout=600.0,
        )
    except Exception as error:
        raise ValueError(f"Speechmatics transcription failed: {error}") from error
    finally:
        await client.close()

    transcript_text = normalize_persian_text(_clean_repetitive_text("\n".join(transcript_parts)))
    if not transcript_text:
        raise ValueError("Speechmatics returned no transcript")
    return {
        "text": transcript_text,
        "transcriptionDuration": float(f"{time.perf_counter() - started:.2f}"),
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
    data = {
        "model": model,
        "temperature": "0.0",
        "response_format": "verbose_json",
        "task": "transcribe",
    }
    if language != "auto":
        data["language"] = language

    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = _fireworks_batch_url(config).rstrip("/")
    if base_url.lower().endswith("/v1"):
        base_url = base_url[:-3]

    transcription_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
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

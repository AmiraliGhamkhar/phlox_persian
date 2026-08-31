"""Language × provider × mode matrix tests for the ASR workflow.

Verifies the guarantees the rest of the app relies on:
- Local engines can't silently decode the wrong language (Parakeet ≠ Persian,
  Shenava ≠ English).
- Online providers with no ``auto`` option get an explicit, Persian-first code
  when the user left the default ``auto``.
- Melia 1 (Batch-only, multilingual) is never sent ``language=auto``.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from server.transcription.audio import (
    _transcribe_fireworks,
    _transcribe_speechmatics,
    _validate_local_model_language,
)


def test_local_model_language_guards():
    """Declared engine capabilities must match the configured language."""
    # Whisper: fa / en / auto all fine.
    _validate_local_model_language("whisper-large-v3-turbo-q5_0", "fa")
    _validate_local_model_language("whisper-large-v3-turbo-q5_0", "en")
    _validate_local_model_language("whisper-large-v3-turbo-q5_0", "auto")

    # Parakeet: English OK, Persian/mixed rejected.
    _validate_local_model_language("parakeet-tdt-0.6b-v3-int8", "en")
    with pytest.raises(ValueError, match="cannot transcribe Persian"):
        _validate_local_model_language("parakeet-tdt-0.6b-v3-int8", "fa")
    with pytest.raises(ValueError, match="cannot transcribe Persian"):
        _validate_local_model_language("parakeet-tdt-0.6b-v3-int8", "auto")

    # Shenava: Persian OK, English rejected.
    _validate_local_model_language("shenava-koochik-v1.0-int4", "fa")
    with pytest.raises(ValueError, match="Persian-only"):
        _validate_local_model_language("shenava-koochik-v1.0-int4", "en")


def _fake_response(status_code: int, json_body=None, text: str = ""):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    response.json.return_value = json_body or {}
    return response


def _mock_client(responses):
    mock_client = AsyncMock()
    mock_client.post.return_value = responses[0]
    mock_client.get.side_effect = responses[1:]
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
async def test_fireworks_batch_maps_auto_to_persian():
    """Fireworks documents no ``auto``; omission may default to English."""
    config = {
        "ASR_PROVIDER": "fireworks",
        "ASR_KEY": "fw-key",
        "ASR_MODEL": "whisper-v3",
        "ASR_LANGUAGE": "auto",
    }
    responses = [
        _fake_response(200, {"text": "سلام"}),
    ]
    mock_client = _mock_client(responses)
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _transcribe_fireworks(b"RIFF....WAVEdata", config)
    assert result["text"] == "سلام"
    data = mock_client.post.call_args.kwargs["data"]
    assert data["language"] == "fa"
    assert data["model"] == "whisper-v3"


@pytest.mark.asyncio
async def test_speechmatics_batch_melia1_gets_language_hint():
    """Melia 1 supports no ``auto``; use a Persian hint and skip lang-id."""
    config = {
        "ASR_PROVIDER": "speechmatics",
        "ASR_BATCH_KEY": "batch-key",
        "ASR_MODEL": "melia-1",
        "ASR_LANGUAGE": "auto",
    }
    responses = [
        _fake_response(201, {"id": "job-m"}),
        _fake_response(200, text="سلام"),
        _fake_response(200, {"job": {"id": "job-m", "duration": 4}}),
    ]
    mock_client = _mock_client(responses)
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _transcribe_speechmatics(b"RIFF....WAVEdata", config)
    assert result["text"] == "سلام"
    sent = json.loads(mock_client.post.call_args.kwargs["data"]["config"])
    assert sent["transcription_config"]["language"] == "fa"
    assert sent["transcription_config"]["model"] == "melia-1"
    assert "language_identification_config" not in sent

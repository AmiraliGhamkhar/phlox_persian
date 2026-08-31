"""Tests for the Speechmatics Batch REST file-transcription path.

Reference: https://docs.speechmatics.com/batch.yaml — the file (post-recording)
path must use POST /jobs + GET /jobs/{id}/transcript with a Batch-scoped key
(API keys are product-scoped: ``type=rt`` vs ``type=batch``).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from server.transcription.audio import (
    SPEECHMATICS_BATCH_DEFAULT_URL,
    _transcribe_speechmatics,
    speechmatics_batch_url,
)


def test_batch_url_resolution():
    assert speechmatics_batch_url({}) == SPEECHMATICS_BATCH_DEFAULT_URL
    assert (
        speechmatics_batch_url({"ASR_BATCH_URL": "https://eu2.asr.api.speechmatics.com/v2/"})
        == "https://eu2.asr.api.speechmatics.com/v2"
    )
    # A wss:// realtime URL must never be reused for batch.
    assert (
        speechmatics_batch_url({"ASR_BASE_URL": "wss://us.rt.speechmatics.com/v2"})
        == SPEECHMATICS_BATCH_DEFAULT_URL
    )
    # An https ASR_BASE_URL is treated as a custom/on-prem batch host.
    assert (
        speechmatics_batch_url({"ASR_BASE_URL": "https://asr.internal.example/v2"})
        == "https://asr.internal.example/v2"
    )


def _fake_response(status_code: int, json_body=None, text: str = ""):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    response.json.return_value = json_body or {}
    return response


def _mock_client(responses):
    """AsyncClient whose .post/.get return the given responses in order."""
    mock_client = AsyncMock()
    mock_client.post.return_value = responses[0]
    get_responses = responses[1:]
    mock_client.get.side_effect = get_responses
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@pytest.mark.asyncio
async def test_transcribe_speechmatics_uses_batch_rest_and_polls():
    """Submit → poll transcript (404 then 200) → fetch duration."""
    config = {
        "ASR_PROVIDER": "speechmatics",
        "ASR_BATCH_KEY": "batch-key",
        "ASR_MODEL": "enhanced",
        "ASR_LANGUAGE": "auto",
    }
    responses = [
        _fake_response(201, {"id": "job-123"}),
        _fake_response(404),  # transcript not ready yet
        _fake_response(200, text="سلام این یک آزمایش است."),
        _fake_response(200, {"job": {"id": "job-123", "duration": 12}}),
    ]
    mock_client = _mock_client(responses)

    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("server.transcription.audio._detect_audio_format") as mock_detect,
    ):
        mock_detect.return_value = ("recording.wav", "audio/wav")

        result = await _transcribe_speechmatics(b"RIFF....WAVEdata", config)

    assert result["text"] == "سلام این یک آزمایش است."
    assert result["transcriptionDuration"] == 12.0

    # Submit call: multipart with config JSON + data_file, batch key.
    post_call = mock_client.post.call_args
    assert post_call.args[0] == f"{SPEECHMATICS_BATCH_DEFAULT_URL}/jobs"
    assert post_call.kwargs["headers"] == {"Authorization": "Bearer batch-key"}
    sent_config = json.loads(post_call.kwargs["data"]["config"])
    assert sent_config["type"] == "transcription"
    assert sent_config["transcription_config"]["language"] == "auto"
    assert sent_config["transcription_config"]["model"] == "enhanced"
    # Batch supports language identification: pin fa/en with a Persian fallback.
    assert sent_config["language_identification_config"] == {
        "expected_languages": ["fa", "en"],
        "low_confidence_action": "use_default_language",
        "default_language": "fa",
    }
    assert "data_file" in post_call.kwargs["files"]

    transcript_url = f"{SPEECHMATICS_BATCH_DEFAULT_URL}/jobs/job-123/transcript"
    get_calls = [call.args[0] for call in mock_client.get.call_args_list]
    assert transcript_url in get_calls
    assert f"{SPEECHMATICS_BATCH_DEFAULT_URL}/jobs/job-123" in get_calls


@pytest.mark.asyncio
async def test_batch_key_fallback_and_explicit_language():
    """ASR_KEY fallback works, and a non-auto language skips lang-id config."""
    config = {
        "ASR_PROVIDER": "speechmatics",
        "ASR_KEY": "primary-key",
        "ASR_LANGUAGE": "fa",
    }
    responses = [
        _fake_response(201, {"id": "job-1"}),
        _fake_response(200, text="سلام"),
        _fake_response(200, {"job": {"id": "job-1", "duration": 3}}),
    ]
    mock_client = _mock_client(responses)
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _transcribe_speechmatics(b"RIFF....WAVEdata", config)

    post_call = mock_client.post.call_args
    assert post_call.kwargs["headers"] == {"Authorization": "Bearer primary-key"}
    sent_config = json.loads(post_call.kwargs["data"]["config"])
    assert sent_config["transcription_config"]["language"] == "fa"
    assert "language_identification_config" not in sent_config
    assert result["text"] == "سلام"


@pytest.mark.asyncio
async def test_batch_authentication_failure_is_explicit():
    """A realtime-scoped key must produce a clear message for the batch path."""
    config = {"ASR_PROVIDER": "speechmatics", "ASR_KEY": "rt-only-key"}
    responses = [
        _fake_response(401, text='{"code": 401, "error": "Permission Denied"}'),
    ]
    mock_client = _mock_client(responses)
    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(ValueError, match="type=batch"):
            await _transcribe_speechmatics(b"RIFF....WAVEdata", config)


@pytest.mark.asyncio
async def test_batch_requires_a_key():
    with pytest.raises(ValueError, match="Batch API key"):
        await _transcribe_speechmatics(b"RIFF....WAVEdata", {"ASR_PROVIDER": "speechmatics"})

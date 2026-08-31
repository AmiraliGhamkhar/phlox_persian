"""Regression tests for the live (streaming) ASR adapters.

These target the two provider-protocol issues that silently broke live
transcription with online providers:

1. ``language=auto`` was sent to Speechmatics/Fireworks *streaming* engines,
   which only support automatic language identification in Batch mode.
2. Fireworks streaming sends ``segments`` deltas, which the adapter ignored.
"""

import asyncio
import json

import pytest

from server.transcription.language import streaming_asr_language
from server.transcription.live import (
    FireworksLiveSession,
    SPEECHMATICS_DEFAULT_URL,
    SpeechmaticsLiveSession,
    speechmatics_rt_url,
)


def test_streaming_language_maps_auto_to_fa():
    """``auto`` is the persisted default but streaming engines need an ISO code."""
    assert streaming_asr_language({}) == "fa"
    assert streaming_asr_language({"ASR_LANGUAGE": "auto"}) == "fa"
    assert streaming_asr_language({"WHISPER_LANGUAGE": "auto"}) == "fa"
    assert streaming_asr_language({"ASR_LANGUAGE": "fa"}) == "fa"
    assert streaming_asr_language({"ASR_LANGUAGE": "en"}) == "en"


def test_speechmatics_endpoint_falls_back_to_global_default():
    """Never connect to the SDK's EU2-only default for self-service accounts."""
    assert speechmatics_rt_url({}) == SPEECHMATICS_DEFAULT_URL
    custom = "wss://us.rt.speechmatics.com/v2"
    assert speechmatics_rt_url({"ASR_BASE_URL": custom}) == custom
    assert speechmatics_rt_url({"WHISPER_BASE_URL": custom}) == custom
    # ASR_BASE_URL wins over the legacy alias.
    assert (
        speechmatics_rt_url(
            {"ASR_BASE_URL": custom, "WHISPER_BASE_URL": "wss://eu.rt.speechmatics.com/v2"}
        )
        == custom
    )


@pytest.mark.asyncio
async def test_speechmatics_session_builds_config_not_auto(monkeypatch):
    """The live session must not carry language=auto into StartRecognition."""
    import types

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self._handlers: dict[object, object] = {}

        def on(self, event, callback=None):
            if callback is not None:
                self._handlers[event] = callback
            return callback

        async def transcribe(self, _source, **kwargs):
            captured["config"] = kwargs["transcription_config"]
            captured["audio_format"] = kwargs["audio_format"]
            # simulate Speechmatics accepting the session
            started_handler = self._handlers.get("RecognitionStarted")
            if started_handler:
                started_handler({})

        async def close(self):
            return None

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _FakeAudioFormat:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_rt = types.ModuleType("speechmatics.rt")
    fake_rt.AsyncClient = _FakeClient
    fake_rt.AudioEncoding = type("AudioEncoding", (), {"PCM_S16LE": "pcm_s16le"})
    fake_rt.AudioFormat = _FakeAudioFormat
    fake_rt.Model = type("Model", (), {"STANDARD": "standard", "ENHANCED": "enhanced"})
    fake_rt.ServerMessageType = type(
        "ServerMessageType",
        (),
        {
            "ADD_PARTIAL_TRANSCRIPT": "AddPartialTranscript",
            "ADD_TRANSCRIPT": "AddTranscript",
            "RECOGNITION_STARTED": "RecognitionStarted",
        },
    )
    fake_rt.TranscriptionConfig = _FakeConfig
    fake_rt.TranscriptResult = type("TranscriptResult", (), {})
    monkeypatch.setitem(__import__("sys").modules, "speechmatics.rt", fake_rt)

    events = []

    async def emit(event):
        events.append(event)

    session = SpeechmaticsLiveSession(
        {"ASR_PROVIDER": "speechmatics", "ASR_KEY": "k", "ASR_MODEL": "enhanced"},
        emit,
    )
    await asyncio.wait_for(session.start(), timeout=5)

    assert captured["client_kwargs"]["url"] == SPEECHMATICS_DEFAULT_URL
    config: dict = captured["config"]
    assert config.language == "fa"
    assert config.enable_partials is True
    assert config.model == "enhanced"
    assert captured["audio_format"].sample_rate == 16000


@pytest.mark.asyncio
async def test_speechmatics_live_rejects_melia1_batch_only():
    """Melia 1 is not in the Realtime SDK Model enum; live must fail clearly."""
    import types

    async def emit(_event):
        return None

    fake_rt = types.ModuleType("speechmatics.rt")
    fake_rt.AsyncClient = object
    fake_rt.AudioEncoding = type("AudioEncoding", (), {"PCM_S16LE": "pcm_s16le"})
    fake_rt.AudioFormat = object
    fake_rt.Model = type("Model", (), {"STANDARD": "standard", "ENHANCED": "enhanced"})
    fake_rt.ServerMessageType = object
    fake_rt.TranscriptionConfig = object
    fake_rt.TranscriptResult = object
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(__import__("sys").modules, "speechmatics.rt", fake_rt)
    try:
        session = SpeechmaticsLiveSession(
            {"ASR_PROVIDER": "speechmatics", "ASR_KEY": "k", "ASR_MODEL": "melia-1"},
            emit,
        )
        with pytest.raises(ValueError, match="Batch-only"):
            await session.start()
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_fireworks_segments_are_parsed():
    """Fireworks streams `segments` deltas; the adapter must forward them."""
    events = []

    async def emit(event):
        events.append(event)

    session = FireworksLiveSession({"ASR_PROVIDER": "fireworks", "ASR_KEY": "k"}, emit)

    # delta with one pending and one finalized segment
    await session._handle_message(
        json.dumps(
            {
                "segments": [
                    {"id": 1, "text": "سلام", "is_final": True, "language": "fa"},
                    {"id": 2, "text": "دکتر", "is_final": False, "language": "fa"},
                ]
            }
        )
    )
    assert any(ev["type"] == "partial" for ev in events)
    assert "سلام" in events[-1]["text"]
    assert "دکتر" in events[-1]["text"]

    # finalized delta for the pending segment
    await session._handle_message(
        json.dumps(
            {
                "segments": [
                    {"id": 2, "text": "دکتر", "is_final": True, "language": "fa"},
                ]
            }
        )
    )
    assert events[-1]["type"] == "final"
    assert "سلام" in events[-1]["text"]
    assert "دکتر" in events[-1]["text"]

    # ordered by segment id even when deltas arrive out of order
    await session._handle_message(
        json.dumps(
            {
                "segments": [
                    {"id": 7, "text": "پایان", "is_final": True, "language": "fa"},
                    {"id": 3, "text": "قبل", "is_final": True, "language": "fa"},
                ]
            }
        )
    )
    assert events[-1]["text"] == "سلام دکتر قبل پایان"

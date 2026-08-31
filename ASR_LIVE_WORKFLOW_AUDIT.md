# Live ASR (Speechmatics & other online providers) — Full Workflow Audit & Fix

**Summary:** The live ("زنده") transcription pipeline had **three real bugs** that made
online streaming ASR (Speechmatics first, Fireworks too) produce no live text at all,
plus one UX bug that hid the reason from the user. All four are fixed in this
change set and verified against a protocol-accurate mock Speechmatics server.

---

## 1. The workflow, end to end

```
Browser (React)                                  FastAPI server (Python)                      Online ASR provider
─────────────────────────────────               ──────────────────────────────              ─────────────────────
ScribePillBox ── press record
  └─ useScribe (src/components/patient/Scribe.jsx)
       ├─ transcriptionApi.openLiveTranscription()          POST upgrade WS      ──►  GET /api/transcribe/live
       │    src/utils/api/transcriptionApi.ts                (server/api/transcribe.py)
       │    WebSocket → ws://<host>/api/transcribe/live?token=…
       │
       ├─ AudioRecorder (src/utils/audioRecorder.js)
       │    16 kHz, mono, s16le chunks (ScriptProcessor 4096)
       │    recorder.onPcm → session.sendPcm(bytes)          binary frames        ──►  session.feed_pcm(pcm)
       │    (partials displayed above the pill)              JSON: partial/final ──►  emit({type:"partial"|"final"})
       │
       └─ stopAndSendRecording()
            ├─ liveSession.stop()  → { text, authoritative }
            ├─ if authoritative text → POST /api/transcribe/reprocess (LLM field extraction)
            └─ else POST /api/transcribe/audio  (batch fallback)
                                                                            Backend adapters (server/transcription/live.py)
                                                                            ├─ SpeechmaticsLiveSession  (native WS, partials)
                                                                            ├─ FireworksLiveSession     (native WS, segments)
                                                                            └─ RollingWindowLiveSession (Whisper.cpp / OpenAI /
                                                                                custom OpenAI-compatible: re-transcribes 5 s windows)
```

Server-side session creation:

1. **`server/api/transcribe.py` → `live_transcribe`** — auth via `?token=` (desktop) or open (Docker), then
   `create_live_session(config, emit)`, `await session.start()`, loop on binary/text frames.
2. **`server/transcription/live.py` → `create_live_session`** — picks the adapter from
   `resolve_asr_connection(config)` (`server/utils/providers.py`):
   - `speechmatics` → `SpeechmaticsLiveSession`
   - `fireworks` + non-whisper model → `FireworksLiveSession`
   - everything else → `RollingWindowLiveSession` (approximate captions).
3. **`server/transcription/audio.py` → `transcribe_audio`** — batch fallback path
   (`/api/transcribe/audio`, `/dictate`). Note: for Speechmatics and Fireworks the
   "batch" path also runs through the **Realtime** websocket SDK.
4. Config comes from the DB via `config_manager` (`server/database/config/manager.py`),
   edited in **Settings → WhisperTab** (`src/components/settings/WhisperTab.jsx`).

---

## 2. Root causes found

### 🐛 Root cause #1 (the killer) — `language: "auto"` is invalid for Realtime

- The app's persisted default is `ASR_LANGUAGE="auto"` (`server/schemas/config.py`,
  `server/transcription/language.py`), and both Speechmatics adapters sent it verbatim:
  ```json
  "transcription_config": { "language": "auto", "model": "enhanced", "enable_partials": true }
  ```
- Speechmatics documents automatic language identification (`"auto"`) **for Batch only**
  ([Language identification (SaaS)](https://docs.speechmatics.com/speech-to-text/batch/language-identification);
  Virtual Appliance: *"Language Identification … is currently only available in Batch mode"*).
  The Realtime API reference requires an ISO code and says the value must be consistent with
  the endpoint URL.
- Effect: the live session is rejected at `StartRecognition` **before a single word is
  transcribed** → zero live text. Batch file transcription still worked, which matches the
  reported symptom exactly.
- **Fix:** new `streaming_asr_language()` maps `auto → fa` (the app is Persian-first;
  `en` still selectable) for **all streaming** adapters (Speechmatics + Fireworks), while
  batch `resolve_asr_language()` keeps `auto`.

### 🐛 Root cause #2 — wrong default Realtime endpoint (`eu2` legacy)

- `speechmatics-rt` SDK 1.1.1 silently defaults to `wss://eu2.rt.speechmatics.com/v2`
  (its legacy EU2 default). Current production endpoints documented by Speechmatics are:
  `wss://global.rt.speechmatics.com/v2` (auto-routes), `eu.rt.speechmatics.com`, `us.rt.speechmatics.com`.
- The app also **hid the endpoint field for Speechmatics** in the settings UI, so a
  region/user mismatch couldn't even be configured. A wrong-region key now **fails the
  WebSocket handshake (401)** → silent live failure.
- **Fix:** `speechmatics_rt_url()` resolves to `wss://global.rt.speechmatics.com/v2` by default
  (catalog + env `SPEECHMATICS_RT_URL` + config override), and the URL field is now visible
  for Speechmatics in Settings so users can pin `eu`/`us`.

### 🐛 Root cause #3 — Fireworks streaming messages were never parsed

- Fireworks streaming ASR sends `{"segments": [{id, text, is_final, language}], …}` deltas
  and finalizes with `{"checkpoint_id": "final"}`. `FireworksLiveSession._handle_message`
  only checked `transcript`/`text`/`words` and `stop()` sent `{"type":"end"}` — so the
  Fireworks live adapter ignored every transcript and never finalized.
- **Fix:** parse `segments` per segment id (finalized map + pending partials, ordered),
  and send `{"checkpoint_id":"final"}` on stop, draining trailing finals.

### 🐛 Root cause #4 — live failures were invisible to the user

- `useScribe` passed `onError: (m) => console.debug(...)` — errors like
  `401 / quota_exceeded / language error` were swallowed. Also, `start()` returned before
  Speechmatics actually accepted the session, so `{"type":"ready"}` was sent even when the
  session would fail (user sees "ready" then silence).
- **Fix:**
  - `SpeechmaticsLiveSession.start()` now waits for `RecognitionStarted` (≤ 15 s) and raises
    the provider error, so the endpoint only reports `ready` when the session is truly live.
  - Errors are shown to the user in the Scribe pill (amber banner during recording);
    socket-level errors also surface (`transcriptionApi.ts`).
  - Recording continues as before and the **batch fallback** still runs at stop, so a live
    failure never loses the audio — but now the user sees why.

### Also improved

- `max_delay: 1.0` for Speechmatics Realtime (partials are unaffected; finals arrive much
  sooner than the 4 s default) — feels properly "live".
- `server/api/config/models.py` / provider catalog: `default_base_url` filled in for
  Speechmatics so every layer (settings UI, status check, adapters) agrees.
- `server/tests/test_live_transcription.py` added: language mapping, endpoint fallback,
  StartRecognition payload (language=fa, partials on), Fireworks segments parsing.

---

## 3. Verification performed

Built a **mock Speechmatics RT v2 server** (websockets) that replays the real protocol
(StartRecognition → RecognitionStarted → AudioAdded → AddPartialTranscript → EndOfStream →
AddTranscript → EndOfTranscript) and ran the repo's actual `SpeechmaticsLiveSession`:

```
StartRecognition sent:
  audio_format: {type: raw, encoding: pcm_s16le, sample_rate: 16000}
  transcription_config: {language: fa, model: enhanced, max_delay: 1.0, enable_partials: true}
Client events: 3× partial ("سلام", "سلام این یک", "سلام این یک آزمایش است"), 1× final
stop() → "سلام این یک آزمایش است."
```

Failure scenario (mock rejecting with `Unauthorized`) now:
`start()` raises with the provider reason and emits `{"type":"error"}` — no hang, no silence.

Frontend: `eslint` clean on all changed files; `vitest run` → 23/23 tests pass.

---

## 4. What you should check (after upgrading)

1. **Settings → ASR** select **Speechmatics Realtime**, enter your API key.
   - If your account is bound to a specific region, set the endpoint explicitly, e.g.
     `wss://us.rt.speechmatics.com/v2` or `wss://eu.rt.speechmatics.com/v2`
     (the field is now visible). Default is `global` (auto-routing).
2. **Language**: leave on "تشخیص خودکار (auto)" — for live sessions the app now maps it to
   `fa` internally. Choose `fa` or `en` explicitly if you want to be sure. (Mixed fa/en
   auto-detection is a Speechmatics **Batch** feature and is not available on Realtime.)
3. **Model**: starts on `enhanced`; you can use `standard` for lower latency.
4. If live still fails, you will now see the exact reason in the orange banner under the
   record button (e.g. `401` → check key/region, `quota_exceeded` → account limit,
   `job_error` → provider-side; retry after 5–10 s as recommended by Speechmatics).
5. Restart the server so the new adapter code is loaded.

## 5. Files changed

| File | Change |
| --- | --- |
| `server/transcription/language.py` | `streaming_asr_language()` — `auto`→`fa` for streaming engines |
| `server/transcription/live.py` | Speechmatics: `fa` language, `global` endpoint resolver (+env/config), wait for `RecognitionStarted`, `max_delay=1.0`; Fireworks: `segments` parsing + `checkpoint_id` finalize; shared `speechmatics_rt_url()` |
| `server/transcription/audio.py` | Batch/whole-file Speechmatics path: same `fa` language + endpoint resolver |
| `server/utils/providers.py` | Speechmatics `default_base_url` = `wss://global.rt.speechmatics.com/v2` |
| `server/transcription/__init__.py` | export new helpers |
| `src/utils/aiProviders.js` | frontend Speechmatics default URL |
| `src/components/settings/WhisperTab.jsx` | endpoint field visible for Speechmatics + realtime language note |
| `src/components/patient/Scribe.jsx` | live errors surfaced, cleared on start/reset |
| `src/components/patient/ScribePillBox.jsx` | amber live-error banner |
| `src/pages/PatientDetails.jsx` | pass `liveError` |
| `src/utils/api/transcriptionApi.ts` | surface socket-level live connection loss |
| `server/tests/test_live_transcription.py` | new regression tests |

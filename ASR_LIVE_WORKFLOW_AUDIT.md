# Live ASR (Speechmatics & other online providers) — Full Workflow Audit & Fix

**Summary:** The live ("زنده") transcription pipeline had **three real bugs** that made
online streaming ASR (Speechmatics first, Fireworks too) produce no live text at all,
plus one UX bug that hid the reason from the user. All are fixed and verified against a
protocol-accurate mock Speechmatics server. A second pass against the **official
Speechmatics spec** (Realtime API reference, Batch API reference + `batch.yaml`,
Management "create an API key") also found that file uploads were going through the
**Realtime WebSocket instead of the official Batch REST API** — that path is now
re-implemented correctly (see §2b).

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
3. **`server/transcription/audio.py` → `transcribe_audio`** — after-the-fact file path
   (`/api/transcribe/audio`, `/dictate`). For **Speechmatics** this now uses the official
   **Batch REST API** (submit job → poll transcript), not the Realtime socket.
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
- Per the official [Languages reference](https://docs.speechmatics.com/speech-to-text/languages.md):
  *"Automatic | `auto` | … Currently supported with **Batch transcription only**."*
  The Realtime API reference requires an ISO language code and reports
  `invalid_language` otherwise.
- Effect: the live session is rejected at `StartRecognition` **before a single word is
  transcribed** → zero live text. Batch file transcription still worked, which matches the
  reported symptom exactly.
- **Fix:** new `streaming_asr_language()` maps `auto → fa` (Persian `fa` **is** a documented
  Realtime language pack) for **all streaming** adapters (Speechmatics + Fireworks), while
  batch `resolve_asr_language()` keeps `auto` — Batch supports language identification.

### 🐛 Root cause #2 — wrong default Realtime endpoint (`eu2` legacy)

- `speechmatics-rt` SDK 1.1.1 silently defaults to `wss://eu2.rt.speechmatics.com/v2`
  (its legacy EU2 default). The official [Authentication reference](https://docs.speechmatics.com/get-started/authentication.md)
  lists production Realtime SaaS endpoints as:
  `wss://global.rt.speechmatics.com/v2` (auto-routes → nearest region),
  `wss://eu.rt.speechmatics.com/v2`, `wss://us.rt.speechmatics.com/v2`.
- The app also **hid the endpoint field for Speechmatics** in settings, so a
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
  `401 / quota_exceeded / invalid_language` were swallowed. Also, `start()` returned before
  Speechmatics actually accepted the session, so `{"type":"ready"}` was sent even when the
  session would fail (user sees "ready" then silence).
- **Fix:**
  - `SpeechmaticsLiveSession.start()` now waits for `RecognitionStarted` (≤ 15 s) and raises
    the provider error, so the endpoint only reports `ready` when the session is truly live.
  - Errors are shown to the user in the Scribe pill (amber banner during recording);
    socket-level errors also surface.
  - Recording continues and the batch fallback still runs at stop, so a live failure never
    loses the audio.

---

## 2b. Second-pass recheck against the official Speechmatics docs

References checked line-by-line:
- [Realtime API reference](https://docs.speechmatics.com/api-ref/realtime-transcription-websocket)
- [Batch: create a new job](https://docs.speechmatics.com/api-ref/batch/create-a-new-job)
- [Management: create an API key](https://docs.speechmatics.com/api-ref/management/create-an-api-key)
- [batch.yaml (OpenAPI)](https://docs.speechmatics.com/batch.yaml)
- [Languages reference](https://docs.speechmatics.com/speech-to-text/languages.md)

### Spec-compliance verdict for the Realtime path ✅

| Spec field | Required/valid per docs | App now sends |
| --- | --- | --- |
| `audio_format.type` | `raw` | `raw` ✅ |
| `audio_format.encoding` | `pcm_s16le` | `pcm_s16le` ✅ |
| `audio_format.sample_rate` | integer Hz | `16000` ✅ |
| `transcription_config.language` | **required ISO code** (no `auto`) | `fa`/`en` ✅ |
| `transcription_config.model` | `standard` \| `enhanced` \| `melia-1` | `enhanced`/`standard` ✅ |
| `transcription_config.enable_partials` | bool | `true` ✅ |
| `transcription_config.max_delay` | 0.7–4 s | `1.0` ✅ |
| `EndOfStream` | `{message,last_seq_no}` | SDK emits ✅ |
| In-band `Error` (`not_authorised`, `invalid_language`, `quota_exceeded`, `job_error`…) | terminates session | surfaced via `start()`/UI ✅ |
| Close codes 4001/4005/4013 | — | surfaced through SDK error ✅ |
| Browser safety (temporary JWT query param) | recommended for browser | N/A — WS is server-side; client only talks to our `/api/transcribe/live` ✅ |
| `AudioAdded` backpressure | sending faster than engine reads can close socket "with prejudice" | browser guard: skip frames when `bufferedAmount > ~1 MB` ✅ (new) |

### ✗ What was wrong at second pass: file uploads used Realtime, not Batch

`_transcribe_speechmatics()` in `audio.py` streamed the whole recording through the
**Realtime WebSocket** (`client.transcribe(io.BytesIO(...))`). The official workflow for
recorded files is the **Batch REST API** (`batch.yaml`):

- `POST https://eu1.asr.api.speechmatics.com/v2/jobs` — multipart `config` (JSON) + `data_file`
- `GET /jobs/{jobid}/transcript?format=txt` — poll until 200 (supports `wait`)
- `GET /jobs/{jobid}` — job `duration`, `status`

It also conflated the **two product-scoped key types** from the Management reference
(`type=rt` vs `type=batch`): a Realtime key cannot call the Batch API (and vice versa).

**Fix — `_transcribe_speechmatics` is now a true Batch REST client:**
1. `speechmatics_batch_url()` → `ASR_BATCH_URL` → `SPEECHMATICS_BATCH_URL` env → an
   `https://` `ASR_BASE_URL` (custom/on-prem) → `https://eu1.asr.api.speechmatics.com/v2`
   (documented EU1 SaaS host). A `wss://` realtime URL is **never** reused.
2. Batch key: `ASR_BATCH_KEY` → `WHISPER_BATCH_KEY` → `ASR_KEY` → `WHISPER_KEY`.
   Clear error if a `type=rt` key is used against the batch API (401 → "create a key with type=batch").
3. Job config per `batch.yaml`:
   `{"type":"transcription","transcription_config":{"language":…,"model":"enhanced","enable_entities":true}}`
   and, when language is `auto`, the documented
   `language_identification_config` (`expected_languages: ["fa","en"]`,
   `low_confidence_action: "use_default_language"`, `default_language: "fa"`) so the
   Persian/English medical mix never falls back to an undesired language.
4. Poll `GET /jobs/{id}/transcript?format=txt&wait=20` (404 → keep waiting, deadline 15 min),
   then fetch `duration` from `GET /jobs/{id}` for the stats display.
5. New optional Settings fields: Batch base URL + Batch API key (only for Speechmatics);
   new config keys `ASR_BATCH_URL` / `ASR_BATCH_KEY` (sensitive → masked in `GET /global`).

### Management API / key creation (for the user)

Keys are created in the [Portal](https://portal.speechmatics.com/settings/api-keys/) or via
`POST https://mp.api.speechmatics.com/v1/api-keys` (management token auth). The
`type` query selects the product: `batch` (default), `rt`, `tts`; `region` is **deprecated** —
the region is determined by the endpoint you call. So:

- **Live transcription** → Realtime key (`type=rt`), any of the RT endpoints.
- **File upload after recording** → Batch key (`type=batch`), `https://eu1.asr.api.speechmatics.com/v2`.
- A permanent key (`project_id`, no `ttl`) or temporary key (`ttl` 60–86400 s) works; we only
  need the long-lived project key on the server (never in the browser).

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

Failure scenario (mock rejecting with `Unauthorized: region endpoint does not match API key`):
`start()` raises with the provider reason and emits `{"type":"error"}` — no hang, no silence.

**Batch REST path** (mocked httpx): submit `POST …/jobs` with the expected multipart payload,
poll `404 → 200` transcript, fetch `duration=12`; `ASR_BATCH_KEY` precedence, `ASR_KEY`
fallback; `language=auto` adds `language_identification_config`; 401 gives the
"create a key with type=batch" guidance; missing key raises clearly.

Frontend: `eslint` clean on all changed files; `vitest run` → 23/23 tests pass.

---

## 4. What you should check (after upgrading)

1. **Settings → ASR** select **Speechmatics Realtime**, enter your **Realtime API key**
   (created as `type=rt`).
   - If your account is pinned to a region, set the endpoint explicitly, e.g.
     `wss://us.rt.speechmatics.com/v2` or `wss://eu.rt.speechmatics.com/v2`
     (the field is now visible). Default is `global` (auto-routing).
2. **File uploads** use the **Batch API**: if your key is Realtime-only, create one with
   `type=batch` and paste it in the new **"کلید Batch API"** field (the base URL is
   pre-filled with `https://eu1.asr.api.speechmatics.com/v2`).
3. **Language**: leave on "تشخیص خودکار (auto)". Live sessions map it to `fa` internally;
   Batch file processing keeps true auto-detection with `fa`/`en` expected and a Persian
   fallback. Choose `fa` or `en` explicitly if you prefer.
4. **Model**: starts on `enhanced`; `standard` is lower latency.
5. If live still fails, you will see the exact reason in the orange banner under the
   record button (`not_authorised` → key/region, `quota_exceeded` → concurrent session
   limit, `job_error` → provider-side; retry after 5–10 s as Speechmatics recommends).
6. Restart the server so the new adapter code is loaded.

## 5. Files changed

| File | Change |
| --- | --- |
| `server/transcription/language.py` | `streaming_asr_language()` — `auto`→`fa` for streaming engines |
| `server/transcription/live.py` | Speechmatics: `fa` language, `global` endpoint resolver (+env/config), wait for `RecognitionStarted`, `max_delay=1.0`; Fireworks: `segments` parsing + `checkpoint_id` finalize; shared `speechmatics_rt_url()`; `SPEECHMATICS_RT_URL` env |
| `server/transcription/audio.py` | **Batch REST** file path: `POST /jobs` + poll `GET /jobs/{id}/transcript` + job duration; `speechmatics_batch_url()`; batch key precedence; lang-id config for `auto` |
| `server/schemas/config.py` | `ASR_BATCH_URL`, `ASR_BATCH_KEY` |
| `server/api/config/global_config.py` | `ASR_BATCH_KEY` treated as sensitive |
| `server/api/config/system.py` | status considers the Batch key too |
| `server/utils/providers.py` | Speechmatics catalog includes `batch_base_url` (hidden from public list) |
| `src/utils/aiProviders.js` | `batchUrl` default; stamped when switching provider |
| `src/components/settings/WhisperTab.jsx` | endpoint field for Speechmatics; Batch URL + Batch key fields; language/key-type notes |
| `src/components/patient/Scribe.jsx` | live errors surfaced, cleared on start/reset |
| `src/components/patient/ScribePillBox.jsx` | amber live-error banner |
| `src/pages/PatientDetails.jsx` | pass `liveError` |
| `src/utils/api/transcriptionApi.ts` | surface socket-level live connection loss; `bufferedAmount` backpressure guard |
| `server/tests/test_live_transcription.py` | live regression tests |
| `server/tests/test_speechmatics_batch.py` | batch REST regression tests |

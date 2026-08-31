# ASR Full Workflow Report — Persian / English / Mixed × Live / Batch × Online / Local

Date: 2026-08-31 · branch `arena/01a05726-phlox-persian`
Scope: every path that turns a recording into transcript text and feeds the
Scribe → note-fill pipeline: Speechmatics (live + batch), Fireworks (live + batch),
OpenAI/custom OpenAI-compatible, whisper.cpp server, local Whisper.cpp GGML,
local Parakeet ONNX, local Shenava ONNX.

Legend: ✅ works as intended · ⚠️ works with caveats · ❌ broken (fixed below by commit)

---

## 1. Provider × Language × Mode matrix (after fixes)

| Provider | Live (real-time) | Batch/File (after recording) |
| --- | --- | --- |
| **Speechmatics** | ✅ fa ✅ en ❌ mixed (no auto/fa_en pack on RT) | ✅ fa ✅ en ✅ mixed (`auto` + lang-identification fa/en) |
| **Fireworks** | ✅ fa ✅ en ⚠️ mixed (single language per session) | ✅ fa ✅ en ⚠️ mixed (Whisper v3 auto, now pinned `fa` on `auto`) |
| **OpenAI / custom OpenAI-compatible** | ⚠️ rolling 5 s windows | ✅ fa ✅ en ⚠️ mixed (auto-detect; depends on engine) |
| **whisper.cpp server** (external) | ⚠️ rolling windows | ✅ fa ✅ en ⚠️ mixed (auto-detect) |
| **Local Whisper.cpp GGML** | ⚠️ rolling windows | ✅ fa ✅ en ✅ mixed (`auto`) |
| **Local Parakeet v3** | ❌ (rolling over an EN-only engine) | ✅ en only ❌ fa ❌ mixed — **now guarded with a clear error** |
| **Local Shenava** | ❌ (Persian-only, manual rolling) | ✅ fa only ❌ en — **now guarded with a clear error** |

What each cell means in practice:

- **fa** = engine locked to Persian (best accuracy for Persian speech; English
  clinical terms inside Persian speech are still recognized — that's engine behavior, not a setting).
- **en** = locked to English.
- **mixed** = automatic fa↔en switching inside one recording/session.
- **rolling windows** = live captions are simulated by re-transcribing 5 s windows
  every 1.5 s (`RollingWindowLiveSession`); partials are approximate, the final
  text is re-transcribed properly when you stop.

---

## 2. What I verified against the official Speechmatics/Fireworks docs

- **Speechmatics Realtime API ref** — `language` is required (ISO code), `auto`
  is Batch-only ([Languages](https://docs.speechmatics.com/speech-to-text/languages.md)
  says *"Currently supported with Batch transcription only"*); `model` ∈
  {standard, enhanced, melia-1}; `max_delay` 0.7–4 s; `enable_partials`; in-band
  `Error` types + close codes (4001 not_authorised, 4005 quota_exceeded, 4013 job_error).
- **Speechmatics batch.yaml / create-a-new-job** — job submit is `POST /jobs`
  (multipart `config` + `data_file`), transcript is `GET /jobs/{id}/transcript?format=txt`,
  duration is `GET /jobs/{id}`; `language_identification_config.expected_languages`,
  `low_confidence_action: use_default_language`, `default_language`.
- **Speechmatics authentication / management** — Realtime and Batch are separate
  product surfaces with separate hosts; **API keys are product-scoped**
  (`type=rt` vs `type=batch`).
- **Fireworks streaming ASR docs** — `fa` **is** in the supported language list
  (~100 languages); binary PCM s16le 16 kHz mono; `segments` deltas +
  `checkpoint_id: "final"`; no `auto` value documented.
- **NVIDIA Parakeet TDT 0.6B v3** (per repo catalog + module docstring) —
  multilingual European, **not a Persian model**.
- **Shenava Koochik** — Persian-only by design.

---

## 3. Issues found in this recheck (fixed in this commit)

### ❌ 3.1 Local Parakeet + Persian/mixed → silent garbage text
`transcribe_audio()` dispatched any `parakeet-*` model regardless of the language
setting. With `ASR_LANGUAGE=fa` (or the default `auto`) the English-only Parakeet
engine ran anyway and produced nonsense Persian, with no error and no indication
to the user (the transcript would then poison the note-fill step).

**Fixed:** `_validate_local_model_language()` — Parakeet + `fa`/`auto` now throws:
> "Parakeet … cannot transcribe Persian or mixed Persian/English speech. Select a
> Whisper large-v3-turbo model, Shenava (Persian only), or an online provider."

### ❌ 3.2 Local Shenava + English → silent garbage text
Same for Shenava (Persian-only) with `ASR_LANGUAGE=en`.

**Fixed:** Shenava + `en` throws a clear "Persian-only model" error.

### ❌ 3.3 Fireworks batch with `auto` could fall back to English
Fireworks documents no `auto` language value; the code omitted `language` when
the (default) `ASR_LANGUAGE=auto`, so the provider could default to English and
mangle Persian on the file path (and on the live rolling-window fallback for
whisper models).

**Fixed:** Fireworks batch now pins `language=fa` when `auto` is selected
(Persian-first; matches the live Fireworks behavior). True mixed fa/en remains
available via Speechmatics Batch `auto` or local Whisper `auto`.

### ❌ 3.4 Melia-1 sent `language=auto` (Batch) / selected for live
- Batch: docs say Melia 1 is multilingual and does **not** support `auto` +
  lang-identification; the job would be rejected.
  **Fixed:** when `model=melia-1` and `language=auto`, send the ISO hint `fa`
  and skip `language_identification_config` (Melia switches languages itself).
- Live: `speechmatics-rt` SDK's `Model` enum has only standard/enhanced; a
  `melia-1` config was silently coerced to `enhanced`.
  **Fixed:** live now raises "Melia 1 is Batch-only…" so users know to use
  enhanced/standard live.

### ⚠️ 3.5 Remaining caveats (documented, not bugs)
- **Speechmatics live mixed fa/en is impossible** — no `fa_en` bilingual pack
  and no `auto` on Realtime. If you need true live mixed: local Whisper GGML
  (`auto`), or accept Speechmatics as one language and use Batch `auto` afterwards.
- **Fireworks live** is single-language per session (no auto) — same caveat.
- **Rolling-window live** (all non-native-streaming providers) re-detects the
  language per 5 s window; a mixed recording may flip hints between windows.
  Final text after stop is transcribed once as a whole, so the saved transcript
  is consistent.
- **Local Whisper `auto`** does genuine per-utterance detection; quality on mixed
  Persian/English is good but not as tight as Speechmatics Batch `auto`.
- **`timestamp_granularities[]=segment`** is sent to custom OpenAI-compatible
  endpoints; some strict servers reject unknown params — only affects custom
  endpoints, and the error message is already visible.

---

## 4. Current behavior per mode (what the user actually gets)

### Live recording → live captions (press record)
1. `useScribe.startRecording` opens `/api/transcribe/live` (WebSocket).
2. Backend picks adapter by provider (`server/transcription/live.py`):
   - **Speechmatics**: native WS — `language` = `fa`/`en` realtime setting
     (`auto`→`fa`), `model` = enhanced/standard, `max_delay=1.0`, partials on.
     `start()` waits for `RecognitionStarted`; errors surface in the amber banner.
   - **Fireworks** (non-whisper models): native WS with `segments` parsing;
     `language` = fa/en (auto→fa).
   - **Everything else** (OpenAI, whisper.cpp server, local Whisper/Parakeet/Shenava):
     `RollingWindowLiveSession` — 5 s windows every 1.5 s via `transcribe_audio()`.
3. On stop: Speechmatics/Fireworks final text is used directly
   (`liveIsAuthoritative` → `reprocessTranscription`); otherwise the full WAV is
   re-transcribed via the batch path (`transcribeAudio`).

### After-recording file transcription (`/api/transcribe/audio`, `/dictate`)
- **Speechmatics → Batch REST** (`POST /jobs`, poll transcript, job duration).
  `auto` = real fa/en language identification with Persian fallback.
- **Fireworks → batch Whisper v3/turbo HTTP**; `auto`→`fa`.
- **OpenAI / custom / whisper.cpp server → OpenAI-compatible**; `auto` = omit hint.
- **Local GGML → whisper.cpp sidecar**; sends `language` only when != auto.
- **Local Parakeet / Shenava → ONNX inference**; guarded by language capability.

### Fallback chain (no live text or live failure)
The full WAV is always kept and transcribed after stop — a live failure never
loses the recording; the error pill shows why + retry/download buttons.

---

## 5. Verification performed

- Live mock (protocol-accurate): StartRecognition = `language: fa, model:
  enhanced, enable_partials: true, max_delay: 1.0` → partials + final, `stop()`
  returns full text. Failure mock → `start()` raises, no hang.
- Batch mock httpx: submit payload shape, poll 404→200, duration, `auto`
  lang-identification config, key precedence, 401 guidance.
- New matrix tests: Parakeet/Shenava guards, Fireworks `auto→fa`, Melia-1 batch
  hint + live rejection.
- Frontend: eslint clean · vitest 23/23 · Python compile OK. (Full pytest still
  to be run with `make docker-test` / CI — sandbox lacks Python ≥3.12 + sqlcipher3.)

---

## 6. Recommended configuration per goal

| Goal | Recommended setting |
| --- | --- |
| Persian-only (accurate) | Speechmatics live (`fa`) + Batch (`fa`), **or** local Shenava (Persian) / Whisper Q5_0 |
| English-only | Anything; Speechmatics `en` or local Whisper `en` |
| Mixed fa/en (best quality) | **Local Whisper large-v3-turbo (auto)** for live + Speechmatics Batch `auto` for files |
| Mixed fa/en (online only) | Speechmatics **Batch** `auto` for files; live = pick `fa` (Persian-first) and accept English words inside Persian |
| Live mixed + online | Fireworks live `fa` (single-language) or OpenAI rolling `auto` — both approximate |

## 7. Files changed in this recheck

- `server/transcription/audio.py` — `_validate_local_model_language()`; Fireworks
  batch `auto→fa`; Melia-1 batch hint.
- `server/transcription/live.py` — Melia-1 live rejection.
- `server/tests/test_asr_language_matrix.py` — new matrix tests.
- `server/tests/test_live_transcription.py` — Melia-1 live test.

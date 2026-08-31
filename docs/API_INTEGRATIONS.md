# API integrations

This page records the upstream API contracts used by Phlox. Provider URLs are
entered as a **base URL**, either with or without a trailing `/v1`; Phlox
normalizes both forms and appends the endpoint exactly once.

## Provider matrix

| Provider | Phlox protocol | Requests used by Phlox | Default base URL |
| --- | --- | --- | --- |
| OpenAI | Responses API for non-streaming LLM calls; Chat Completions for streaming | `/v1/responses`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/transcriptions` | `https://api.openai.com` |
| Anthropic | Native Messages API | `/v1/messages`, `/v1/models` | `https://api.anthropic.com` |
| Ollama | OpenAI-compatible | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` | `http://127.0.0.1:11434` |
| LM Studio | OpenAI-compatible | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` | `http://127.0.0.1:1234` |
| llama.cpp server | OpenAI-compatible | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` | `http://127.0.0.1:8080` |
| 9Router / OmniRoute | OpenAI-compatible gateway | `/v1/chat/completions`, `/v1/responses`, `/v1/models`, `/v1/embeddings` | `http://127.0.0.1:20128` |
| Whisper.cpp / faster-whisper | OpenAI-compatible audio | `/v1/audio/transcriptions` | configured URL |

Anthropic's old `/v1/complete` Text Completions endpoint is intentionally not
used. Anthropic documents it as legacy and recommends Messages, which is the
native adapter in `server/llm_client/providers/anthropic.py`.

## Important compatibility behavior

- OpenAI Responses output is converted to Phlox's internal assistant message
  shape. `response.output_text` becomes `message.content`; Responses
  `function_call` items become OpenAI-style `tool_calls`; function results are
  sent back as `function_call_output` items.
- Structured output uses Responses `text.format` (not Chat Completions'
  `response_format`) for official OpenAI calls.
- Streaming keeps the Chat Completions adapter because Phlox's existing SSE
  consumer consumes delta-shaped chunks. OpenAI-compatible local gateways and
  other providers continue to use Chat Completions.
- OpenAI Responses multimodal input is translated from Chat-style
  `text`/`image_url` blocks to `input_text`/`input_image` blocks.
- ASR uses the resolved provider connection, including provider defaults. For
  example, selecting OpenAI without typing a URL now correctly targets
  `https://api.openai.com/v1/audio/transcriptions` rather than an empty host.
- API keys are sent as `Authorization: Bearer` for OpenAI-compatible services
  and `x-api-key` for Anthropic. Local services use a harmless placeholder key
  where an SDK requires one.

## Upstream references

- [OpenAI: migrate to Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Anthropic: legacy Text Completions](https://platform.claude.com/docs/en/api/python/completions/create)
- [9Router architecture](https://github.com/decolua/9router/blob/master/docs/ARCHITECTURE.md)
- [OmniRoute API reference](https://github.com/diegosouzapw/OmniRoute/wiki/API-Reference#chat-completions)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/README.md)
- [Ollama API introduction](https://docs.ollama.com/api/introduction)
- [LM Studio REST API](https://lmstudio.ai/docs/developer/rest)

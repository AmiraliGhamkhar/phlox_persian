"""Canonical catalogs for LLM, ASR, and embedding providers.

Named providers (Ollama, LM Studio, llama.cpp, 9Router, OmniRoute, OpenAI,
Anthropic, Fireworks, Speechmatics, Whisper.cpp) are first-class so the
settings UI can auto-fill endpoints, discover models, and switch cleanly.
Unknown or legacy ``openai`` values remain OpenAI-compatible.
"""

from __future__ import annotations

from typing import Any

PROVIDER_ALIASES = {
    "9router": "ninerouter",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "llama.cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "whisper.cpp": "whispercpp",
    "whisper-cpp": "whispercpp",
    "openai-compatible": "openai_compatible",
    "openai_official": "openai",
    "external": "openai_compatible",
}

# Historical default: an empty OpenAI-compatible URL talks to local Ollama.
LEGACY_OPENAI_FALLBACK_URL = "http://127.0.0.1:11434"

LLM_PROVIDERS: dict[str, dict[str, Any]] = {
    "local": {
        "id": "local",
        "name": "Local llama.cpp",
        "name_fa": "llama.cpp محلی",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "",
        "placeholder_url": "bundled llama-server",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Bundled llama.cpp server. Download a GGUF model with one click.",
        "help_fa": "سرور llama.cpp داخلی. مدل GGUF را با یک کلیک دانلود کنید.",
        "default_api_key": "not-needed",
    },
    "ollama": {
        "id": "ollama",
        "name": "Ollama",
        "name_fa": "Ollama",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:11434",
        "placeholder_url": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Local Ollama daemon. Start Ollama, then pick a pulled model.",
        "help_fa": "سرویس محلی Ollama. پس از اجرا، مدل نصب‌شده را انتخاب کنید.",
        "default_api_key": "ollama",
        "default_embedding_models": ["nomic-embed-text", "mxbai-embed-large"],
    },
    "lmstudio": {
        "id": "lmstudio",
        "name": "LM Studio",
        "name_fa": "LM Studio",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:1234",
        "placeholder_url": "http://127.0.0.1:1234",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "LM Studio local server (Developer → Start Server). Default port 1234.",
        "help_fa": "سرور محلی LM Studio (Developer ← Start Server). پورت پیش‌فرض ۱۲۳۴.",
        "default_api_key": "lm-studio",
    },
    "llamacpp": {
        "id": "llamacpp",
        "name": "llama.cpp server",
        "name_fa": "سرور llama.cpp",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:8080",
        "placeholder_url": "http://127.0.0.1:8080",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Standalone llama-server. Default OpenAI-compatible port 8080.",
        "help_fa": "سرور مستقل llama-server. پورت پیش‌فرض سازگار با OpenAI برابر ۸۰۸۰ است.",
        "default_api_key": "not-needed",
    },
    "ninerouter": {
        "id": "ninerouter",
        "name": "9Router",
        "name_fa": "9Router",
        "category": "gateway",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:20128",
        "placeholder_url": "http://127.0.0.1:20128",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "9Router OpenAI-compatible gateway. Dashboard: http://localhost:20128/dashboard",
        "help_fa": "دروازه سازگار با OpenAI برای 9Router. داشبورد: http://localhost:20128/dashboard",
        "default_api_key": "sk_9router",
    },
    "omniroute": {
        "id": "omniroute",
        "name": "OmniRoute",
        "name_fa": "OmniRoute",
        "category": "gateway",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:20128",
        "placeholder_url": "http://127.0.0.1:20128",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "OmniRoute gateway (9Router fork). Same OpenAI-compatible /v1 API.",
        "help_fa": "دروازه OmniRoute (شاخه 9Router) با همان API سازگار با OpenAI.",
        "default_api_key": "sk_omniroute",
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "name_fa": "OpenAI",
        "category": "cloud",
        "protocol": "openai_compatible",
        "default_base_url": "https://api.openai.com",
        "placeholder_url": "https://api.openai.com",
        "requires_api_key": True,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Official OpenAI API. Requires an API key (sk-...).",
        "help_fa": "API رسمی OpenAI. کلید API (sk-...) الزامی است.",
        "default_api_key": "",
        "default_models": ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "o4-mini"],
        "default_embedding_models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "default_asr_models": ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "name_fa": "Anthropic",
        "category": "cloud",
        "protocol": "anthropic",
        "default_base_url": "https://api.anthropic.com",
        "placeholder_url": "https://api.anthropic.com",
        "requires_api_key": True,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Official Anthropic Messages API. Requires an API key (sk-ant-...).",
        "help_fa": "API رسمی Anthropic Messages. کلید API (sk-ant-...) الزامی است.",
        "default_api_key": "",
        "default_models": [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ],
        "anthropic_version": "2023-06-01",
    },
    "fireworks": {
        "id": "fireworks",
        "name": "Fireworks AI",
        "name_fa": "Fireworks AI",
        "category": "cloud",
        "protocol": "openai_compatible",
        "default_base_url": "https://api.fireworks.ai/inference",
        "placeholder_url": "https://api.fireworks.ai/inference",
        "requires_api_key": True,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Fireworks OpenAI-compatible LLM endpoint.",
        "help_fa": "نقطه پایانی سازگار با OpenAI در Fireworks.",
        "default_api_key": "",
    },
    "openai_compatible": {
        "id": "openai_compatible",
        "name": "Custom OpenAI-compatible",
        "name_fa": "سفارشی سازگار با OpenAI",
        "category": "custom",
        "protocol": "openai_compatible",
        "default_base_url": "",
        "placeholder_url": "https://api.example.com",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_tools": True,
        "help": "Any OpenAI-compatible /v1 endpoint (vLLM, OpenRouter, LiteLLM, ...).",
        "help_fa": "هر نقطه پایانی سازگار با OpenAI مانند vLLM، OpenRouter یا LiteLLM.",
        "default_api_key": "",
    },
}

ASR_PROVIDERS: dict[str, dict[str, Any]] = {
    "local": {
        "id": "local",
        "name": "Local ASR",
        "name_fa": "ASR محلی",
        "category": "local",
        "protocol": "local",
        "default_base_url": "",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_live": True,
        "help": "Whisper.cpp, Parakeet, or Shenava models downloaded in the app.",
        "help_fa": "مدل‌های Whisper.cpp، Parakeet یا Shenava که در برنامه دانلود می‌شوند.",
        "default_models": [],
    },
    "openai_compatible": {
        "id": "openai_compatible",
        "name": "OpenAI-compatible ASR",
        "name_fa": "ASR سازگار با OpenAI",
        "category": "custom",
        "protocol": "openai_audio",
        "default_base_url": "",
        "placeholder_url": "https://asr.example.com",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_live": True,
        "help": "Any /v1/audio/transcriptions endpoint (faster-whisper, whisper.cpp server, ...).",
        "help_fa": "هر نقطه پایانی /v1/audio/transcriptions مانند faster-whisper یا سرور whisper.cpp.",
        "default_models": ["whisper-1"],
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI Audio",
        "name_fa": "OpenAI Audio",
        "category": "cloud",
        "protocol": "openai_audio",
        "default_base_url": "https://api.openai.com",
        "placeholder_url": "https://api.openai.com",
        "requires_api_key": True,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_live": True,
        "help": "OpenAI Whisper / gpt-4o-transcribe. Live mode uses rolling audio windows.",
        "help_fa": "مدل‌های Whisper و gpt-4o-transcribe. حالت زنده از پنجره‌های غلتان صدا استفاده می‌کند.",
        "default_models": ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
    },
    "whispercpp": {
        "id": "whispercpp",
        "name": "Whisper.cpp server",
        "name_fa": "سرور Whisper.cpp",
        "category": "local",
        "protocol": "openai_audio",
        "default_base_url": "http://127.0.0.1:2022",
        "placeholder_url": "http://127.0.0.1:2022",
        "requires_api_key": False,
        "supports_streaming": True,
        "supports_model_list": True,
        "supports_live": True,
        "help": "Standalone whisper.cpp HTTP server (OpenAI-compatible /v1/audio/transcriptions).",
        "help_fa": "سرور HTTP مستقل whisper.cpp با نقطه پایانی سازگار با OpenAI.",
        "default_models": ["whisper-1"],
    },
    "speechmatics": {
        "id": "speechmatics",
        "name": "Speechmatics Realtime",
        "name_fa": "Speechmatics؛ بلادرنگ",
        "category": "cloud",
        "protocol": "speechmatics",
        "default_base_url": "wss://global.rt.speechmatics.com/v2",
        "placeholder_url": "wss://global.rt.speechmatics.com/v2",
        "batch_base_url": "https://eu1.asr.api.speechmatics.com/v2",
        "batch_placeholder_url": "https://eu1.asr.api.speechmatics.com/v2",
        "requires_api_key": True,
        "supports_streaming": True,
        "supports_model_list": False,
        "supports_live": True,
        "help": "Speechmatics Realtime WebSocket. Partials stream while you speak.",
        "help_fa": "وب‌سوکت بلادرنگ Speechmatics. نتایج جزئی هنگام صحبت نمایش داده می‌شوند.",
        "default_models": ["enhanced", "standard"],
    },
    "fireworks": {
        "id": "fireworks",
        "name": "Fireworks AI ASR",
        "name_fa": "Fireworks AI ASR",
        "category": "cloud",
        "protocol": "fireworks",
        "default_base_url": "https://audio-prod.api.fireworks.ai",
        "placeholder_url": "https://audio-prod.api.fireworks.ai",
        "requires_api_key": True,
        "supports_streaming": True,
        "supports_model_list": False,
        "supports_live": True,
        "help": "Fireworks streaming ASR over WebSocket plus batch Whisper v3/turbo.",
        "help_fa": "ASR جریانی Fireworks روی وب‌سوکت به‌همراه Whisper v3 و نسخه Turbo.",
        "default_models": [
            "fireworks-asr-v2",
            "fireworks-asr-large",
            "whisper-v3-turbo",
            "whisper-v3",
        ],
        "streaming_url": "wss://audio-streaming.api.fireworks.ai/v1/audio/transcriptions/streaming",
        "streaming_url_v2": "wss://audio-streaming-v2.api.fireworks.ai/v1/audio/transcriptions/streaming",
        "batch_urls": {
            "whisper-v3": "https://audio-prod.api.fireworks.ai",
            "whisper-v3-turbo": "https://audio-turbo.api.fireworks.ai",
        },
    },
}

EMBEDDING_PROVIDERS: dict[str, dict[str, Any]] = {
    "local": {
        "id": "local",
        "name": "Local embedding server",
        "name_fa": "سرور بردارسازی محلی",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": ["Qwen3-Embedding-0.6B-Q8_0"],
        "help": "Bundled llama.cpp embedding server. One-click GGUF download.",
        "help_fa": "سرور بردارسازی llama.cpp داخلی با دانلود یک‌کلیکی GGUF.",
    },
    "ollama": {
        "id": "ollama",
        "name": "Ollama embeddings",
        "name_fa": "بردارسازی Ollama",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": ["nomic-embed-text", "mxbai-embed-large"],
        "help": "Ollama /v1/embeddings (nomic-embed-text is a good default).",
        "help_fa": "نقطه پایانی /v1/embeddings در Ollama؛ nomic-embed-text انتخاب مناسبی است.",
    },
    "lmstudio": {
        "id": "lmstudio",
        "name": "LM Studio embeddings",
        "name_fa": "بردارسازی LM Studio",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:1234",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": [],
        "help": "LM Studio /v1/embeddings with an embedding model loaded.",
        "help_fa": "نقطه پایانی /v1/embeddings در LM Studio با مدل بردارسازی بارگذاری‌شده.",
    },
    "llamacpp": {
        "id": "llamacpp",
        "name": "llama.cpp embeddings",
        "name_fa": "بردارسازی llama.cpp",
        "category": "local",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:8080",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": [],
        "help": "llama-server started with --embedding.",
        "help_fa": "llama-server که با گزینه --embedding اجرا شده است.",
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI embeddings",
        "name_fa": "بردارسازی OpenAI",
        "category": "cloud",
        "protocol": "openai_compatible",
        "default_base_url": "https://api.openai.com",
        "requires_api_key": True,
        "supports_model_list": True,
        "default_models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        "help": "Official OpenAI embeddings API.",
        "help_fa": "API رسمی بردارسازی OpenAI.",
    },
    "ninerouter": {
        "id": "ninerouter",
        "name": "9Router embeddings",
        "name_fa": "بردارسازی 9Router",
        "category": "gateway",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:20128",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": [],
        "help": "Embeddings through the 9Router OpenAI-compatible gateway.",
        "help_fa": "بردارسازی از طریق دروازه سازگار با OpenAI در 9Router.",
    },
    "omniroute": {
        "id": "omniroute",
        "name": "OmniRoute embeddings",
        "name_fa": "بردارسازی OmniRoute",
        "category": "gateway",
        "protocol": "openai_compatible",
        "default_base_url": "http://127.0.0.1:20128",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": [],
        "help": "Embeddings through OmniRoute.",
        "help_fa": "بردارسازی از طریق OmniRoute.",
    },
    "fireworks": {
        "id": "fireworks",
        "name": "Fireworks embeddings",
        "name_fa": "بردارسازی Fireworks",
        "category": "cloud",
        "protocol": "openai_compatible",
        "default_base_url": "https://api.fireworks.ai/inference",
        "requires_api_key": True,
        "supports_model_list": True,
        "default_models": [],
        "help": "Fireworks /inference/v1/embeddings.",
        "help_fa": "نقطه پایانی بردارسازی Fireworks.",
    },
    "openai_compatible": {
        "id": "openai_compatible",
        "name": "Custom OpenAI-compatible",
        "name_fa": "سفارشی سازگار با OpenAI",
        "category": "custom",
        "protocol": "openai_compatible",
        "default_base_url": "",
        "requires_api_key": False,
        "supports_model_list": True,
        "default_models": [],
        "help": "Any OpenAI-compatible /v1/embeddings endpoint.",
        "help_fa": "هر نقطه پایانی سازگار با OpenAI برای /v1/embeddings.",
    },
}


def normalize_provider_id(provider: str | None, kind: str = "llm") -> str:
    """Return a canonical provider id, applying aliases."""
    raw = (provider or "").strip().lower()
    raw = PROVIDER_ALIASES.get(raw, raw)
    catalogs = {
        "llm": LLM_PROVIDERS,
        "asr": ASR_PROVIDERS,
        "embedding": EMBEDDING_PROVIDERS,
    }
    catalog = catalogs.get(kind, LLM_PROVIDERS)
    if raw in catalog:
        return raw
    if kind == "llm":
        return "openai_compatible" if raw else "openai"
    if kind == "asr":
        return "openai_compatible" if raw else "openai_compatible"
    return "openai_compatible" if raw else "local"


def public_provider(info: dict[str, Any]) -> dict[str, Any]:
    """Strip internal-only keys before sending a provider to the client."""
    skip = {
        "batch_urls",
        "batch_base_url",
        "batch_placeholder_url",
        "streaming_url",
        "streaming_url_v2",
        "anthropic_version",
    }
    return {key: value for key, value in info.items() if key not in skip}


def list_providers() -> dict[str, list[dict[str, Any]]]:
    """Return the three catalogs for the settings UI."""
    return {
        "llm": [public_provider(item) for item in LLM_PROVIDERS.values()],
        "asr": [public_provider(item) for item in ASR_PROVIDERS.values()],
        "embedding": [public_provider(item) for item in EMBEDDING_PROVIDERS.values()],
    }


def detect_llm_provider(config: dict[str, Any]) -> str:
    """Resolve the LLM provider the user is actually talking to."""
    explicit = normalize_provider_id(config.get("LLM_PROVIDER"), "llm")
    if explicit in LLM_PROVIDERS and explicit not in {"openai", "openai_compatible"}:
        return explicit

    url = str(config.get("LLM_BASE_URL") or "").lower()
    if "api.openai.com" in url:
        return "openai"
    if "api.anthropic.com" in url:
        return "anthropic"
    if "fireworks.ai" in url:
        return "fireworks"
    if ":1234" in url or url.rstrip("/").endswith("1234"):
        return "lmstudio"
    if ":20128" in url or url.rstrip("/").endswith("20128"):
        return "ninerouter"
    if ":8080" in url or url.rstrip("/").endswith("8080"):
        return "llamacpp"
    if "11434" in url or not url:
        return "ollama"
    return "openai_compatible"


def resolve_llm_connection(config: dict[str, Any]) -> dict[str, Any]:
    """Return protocol, base URL, and API key for the configured LLM provider."""
    provider = normalize_provider_id(config.get("LLM_PROVIDER"), "llm")
    if provider == "openai" and not str(config.get("LLM_BASE_URL") or "").strip():
        # Preserve the historical empty-URL → Ollama behaviour.
        provider = "ollama"
        info = LLM_PROVIDERS["ollama"]
    else:
        info = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["openai_compatible"])

    if provider == "local":
        from server.utils.allocated_ports import get_llama_port

        return {
            "provider": "local",
            "protocol": "openai_compatible",
            "base_url": f"http://127.0.0.1:{get_llama_port()}",
            "api_key": "not-needed",
            "info": LLM_PROVIDERS["local"],
        }

    base_url = str(config.get("LLM_BASE_URL") or "").strip() or str(
        info.get("default_base_url") or ""
    )
    if not base_url:
        base_url = LEGACY_OPENAI_FALLBACK_URL

    api_key = str(config.get("LLM_API_KEY") or "").strip() or str(
        info.get("default_api_key") or "not-needed"
    )
    return {
        "provider": provider,
        "protocol": info.get("protocol", "openai_compatible"),
        "base_url": base_url,
        "api_key": api_key,
        "info": info,
    }


def resolve_asr_connection(config: dict[str, Any]) -> dict[str, Any]:
    """Return protocol and credentials for the configured ASR provider."""
    provider = normalize_provider_id(
        config.get("ASR_PROVIDER")
        or (
            "local"
            if config.get("LLM_PROVIDER") == "local"
            and not (config.get("ASR_BASE_URL") or config.get("WHISPER_BASE_URL"))
            else "openai_compatible"
        ),
        "asr",
    )
    info = ASR_PROVIDERS.get(provider, ASR_PROVIDERS["openai_compatible"])
    base_url = str(config.get("ASR_BASE_URL") or config.get("WHISPER_BASE_URL") or "").strip()
    if not base_url:
        base_url = str(info.get("default_base_url") or "")
    api_key = str(config.get("ASR_KEY") or config.get("WHISPER_KEY") or "").strip()
    model = str(config.get("ASR_MODEL") or config.get("WHISPER_MODEL") or "").strip()
    if not model and info.get("default_models"):
        model = info["default_models"][0]
    return {
        "provider": provider,
        "protocol": info.get("protocol", "openai_audio"),
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "info": info,
        "supports_live": bool(info.get("supports_live")),
    }


def resolve_embedding_connection(config: dict[str, Any]) -> dict[str, Any]:
    """Return base URL / key / model for embeddings, falling back to the LLM provider."""
    explicit = str(config.get("EMBEDDING_PROVIDER") or "").strip().lower()
    llm_provider = normalize_provider_id(config.get("LLM_PROVIDER"), "llm")
    source = explicit or config.get("LLM_PROVIDER")
    provider = normalize_provider_id(source, "embedding")
    # Anthropic (and any other chat-only host) has no embeddings API.
    if provider not in EMBEDDING_PROVIDERS or (
        not explicit and llm_provider not in EMBEDDING_PROVIDERS
    ):
        provider = "ollama" if not explicit else "openai_compatible"
    info = EMBEDDING_PROVIDERS[provider]

    if provider == "local" or (not explicit and config.get("LLM_PROVIDER") == "local"):
        from server.utils.allocated_ports import get_embedding_port

        return {
            "provider": "local",
            "base_url": f"http://127.0.0.1:{get_embedding_port()}",
            "api_key": "not-needed",
            "model": str(config.get("EMBEDDING_MODEL") or "Qwen3-Embedding-0.6B-Q8_0"),
            "info": info,
        }

    base_url = str(config.get("EMBEDDING_BASE_URL") or "").strip()
    if not base_url:
        base_url = str(info.get("default_base_url") or config.get("LLM_BASE_URL") or "")
    if not base_url:
        base_url = LEGACY_OPENAI_FALLBACK_URL
    api_key = str(
        config.get("EMBEDDING_API_KEY")
        or config.get("LLM_API_KEY")
        or info.get("default_api_key")
        or "not-needed"
    )
    model = str(config.get("EMBEDDING_MODEL") or "").strip()
    if not model and info.get("default_models"):
        model = info["default_models"][0]
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key or "not-needed",
        "model": model,
        "info": info,
    }


def apply_llm_provider_defaults(
    provider_id: str, current: dict[str, Any] | None = None
) -> dict[str, str]:
    """Return config keys to apply when the user switches LLM provider."""
    provider = normalize_provider_id(provider_id, "llm")
    info = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["openai_compatible"])
    current = current or {}
    updates = {"LLM_PROVIDER": provider}
    default_url = str(info.get("default_base_url") or "")
    # Always stamp a known default when switching to a named provider so the
    # UI and model listing immediately target the right host.
    if provider != "openai_compatible" or not current.get("LLM_BASE_URL"):
        updates["LLM_BASE_URL"] = default_url
    return updates


def apply_asr_provider_defaults(provider_id: str) -> dict[str, str]:
    """Return config keys to apply when the user switches ASR provider."""
    provider = normalize_provider_id(provider_id, "asr")
    info = ASR_PROVIDERS.get(provider, ASR_PROVIDERS["openai_compatible"])
    model = ""
    if info.get("default_models"):
        model = info["default_models"][0]
    return {
        "ASR_PROVIDER": provider,
        "ASR_BASE_URL": str(info.get("default_base_url") or ""),
        "WHISPER_BASE_URL": str(info.get("default_base_url") or ""),
        "ASR_MODEL": model,
        "WHISPER_MODEL": model,
    }


def apply_embedding_provider_defaults(provider_id: str) -> dict[str, str]:
    """Return config keys to apply when the user switches embedding provider."""
    provider = normalize_provider_id(provider_id, "embedding")
    info = EMBEDDING_PROVIDERS.get(provider, EMBEDDING_PROVIDERS["openai_compatible"])
    model = info["default_models"][0] if info.get("default_models") else ""
    return {
        "EMBEDDING_PROVIDER": provider,
        "EMBEDDING_BASE_URL": str(info.get("default_base_url") or ""),
        "EMBEDDING_MODEL": model,
    }


def looks_like_embedding_model(model_id: str) -> bool:
    """Heuristic used when a provider lists mixed chat/embedding models."""
    lowered = (model_id or "").lower()
    tokens = (
        "embed",
        "embedding",
        "bge",
        "e5-",
        "nomic",
        "gte-",
        "snowflake",
        "arctic-embed",
        "text-embedding",
        "multilingual-e5",
    )
    return any(token in lowered for token in tokens)

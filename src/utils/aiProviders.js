/** Frontend catalog fallbacks matching server/utils/providers.py. */

export const LLM_PROVIDER_DEFAULTS = {
  local: { url: "", requiresKey: false },
  ollama: { url: "http://127.0.0.1:11434", requiresKey: false },
  lmstudio: { url: "http://127.0.0.1:1234", requiresKey: false },
  llamacpp: { url: "http://127.0.0.1:8080", requiresKey: false },
  ninerouter: { url: "http://127.0.0.1:20128", requiresKey: false },
  omniroute: { url: "http://127.0.0.1:20128", requiresKey: false },
  openai: { url: "https://api.openai.com", requiresKey: true },
  anthropic: { url: "https://api.anthropic.com", requiresKey: true },
  fireworks: { url: "https://api.fireworks.ai/inference", requiresKey: true },
  openai_compatible: { url: "", requiresKey: false },
};

export const ASR_PROVIDER_DEFAULTS = {
  local: { url: "", models: [], requiresKey: false },
  openai_compatible: { url: "", models: ["whisper-1"], requiresKey: false },
  openai: {
    url: "https://api.openai.com",
    models: ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
    requiresKey: true,
  },
  whispercpp: { url: "http://127.0.0.1:2022", models: ["whisper-1"], requiresKey: false },
  speechmatics: {
    url: "wss://global.rt.speechmatics.com/v2",
    batchUrl: "https://eu1.asr.api.speechmatics.com/v2",
    models: ["enhanced", "standard"],
    requiresKey: true,
  },
  fireworks: {
    url: "https://audio-prod.api.fireworks.ai",
    models: ["fireworks-asr-v2", "fireworks-asr-large", "whisper-v3-turbo", "whisper-v3"],
    requiresKey: true,
  },
};

export const EMBEDDING_PROVIDER_DEFAULTS = {
  local: { url: "", models: ["Qwen3-Embedding-0.6B-Q8_0"], requiresKey: false },
  ollama: { url: "http://127.0.0.1:11434", models: ["nomic-embed-text"], requiresKey: false },
  lmstudio: { url: "http://127.0.0.1:1234", models: [], requiresKey: false },
  llamacpp: { url: "http://127.0.0.1:8080", models: [], requiresKey: false },
  openai: {
    url: "https://api.openai.com",
    models: ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
    requiresKey: true,
  },
  ninerouter: { url: "http://127.0.0.1:20128", models: [], requiresKey: false },
  omniroute: { url: "http://127.0.0.1:20128", models: [], requiresKey: false },
  fireworks: { url: "https://api.fireworks.ai/inference", models: [], requiresKey: true },
  openai_compatible: { url: "", models: [], requiresKey: false },
};

export const embeddingProviderIdForLlm = (llmProvider) => {
  if (llmProvider && EMBEDDING_PROVIDER_DEFAULTS[llmProvider]) {
    return llmProvider;
  }
  return "ollama";
};

export const isOnnxAsrModel = (modelId = "") =>
  modelId.startsWith("shenava-") || modelId.startsWith("parakeet-");

export const applyLlmProviderDefaults = (providerId, handleConfigChange) => {
  const defaults = LLM_PROVIDER_DEFAULTS[providerId] || LLM_PROVIDER_DEFAULTS.openai_compatible;
  handleConfigChange("LLM_PROVIDER", providerId);
  if (providerId !== "openai_compatible") {
    handleConfigChange("LLM_BASE_URL", defaults.url);
  }
};

export const applyAsrProviderDefaults = (providerId, handleConfigChange) => {
  const defaults = ASR_PROVIDER_DEFAULTS[providerId] || ASR_PROVIDER_DEFAULTS.openai_compatible;
  handleConfigChange("ASR_PROVIDER", providerId);
  handleConfigChange("ASR_BASE_URL", defaults.url);
  handleConfigChange("WHISPER_BASE_URL", defaults.url);
  if (defaults.batchUrl) {
    handleConfigChange("ASR_BATCH_URL", defaults.batchUrl);
  }
  const model = defaults.models[0] || "";
  handleConfigChange("ASR_MODEL", model);
  handleConfigChange("WHISPER_MODEL", model);
};

export const applyEmbeddingProviderDefaults = (providerId, handleConfigChange) => {
  const defaults =
    EMBEDDING_PROVIDER_DEFAULTS[providerId] || EMBEDDING_PROVIDER_DEFAULTS.openai_compatible;
  handleConfigChange("EMBEDDING_PROVIDER", providerId);
  handleConfigChange("EMBEDDING_BASE_URL", defaults.url);
  if (defaults.models[0]) {
    handleConfigChange("EMBEDDING_MODEL", defaults.models[0]);
  }
};

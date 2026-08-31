from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration model for the application.

    Attributes:
        LLM_PROVIDER (str): Named LLM provider (ollama, openai, anthropic, ...).
        LLM_BASE_URL (str): Base URL for the LLM endpoint.
        LLM_API_KEY (str): API key for the LLM endpoint (optional depending on provider).
        PRIMARY_MODEL (str): Primary model used for core generation tasks.
        SECONDARY_MODEL (str): Secondary model used for alternate generation tasks.
        EMBEDDING_PROVIDER (str): Independent embedding provider.
        EMBEDDING_BASE_URL (str): Base URL for embeddings.
        EMBEDDING_API_KEY (str): Optional embedding API key.
        EMBEDDING_MODEL (str): Model used for embedding generation.
        WHISPER_BASE_URL (str): Base URL for Whisper-compatible transcription endpoint.
        WHISPER_MODEL (str): Whisper model identifier.
        WHISPER_KEY (str): Legacy API key for the ASR endpoint.
        ASR_BASE_URL (str): Canonical base URL for the ASR endpoint.
        ASR_MODEL (str): Canonical ASR model identifier.
        ASR_KEY (str): Canonical API key for the ASR endpoint.
        ASR_BATCH_URL (str): Speechmatics Batch REST base URL (https://…/v2). Optional.
        ASR_BATCH_KEY (str): Speechmatics Batch API key; only needed when the primary
            key is Realtime-scoped (keys are product-scoped: ``type=rt`` vs ``type=batch``).
        ASR_LANGUAGE (str): ASR language hint: ``fa``, ``en``, or ``auto`` for mixed audio.
        ASR_PROVIDER (str): ASR provider (local, openai, fireworks, speechmatics, ...).
        REASONING_MODEL (str): Model used for reasoning/analysis tasks.
        REASONING_ENABLED (bool): Toggle to enable or disable reasoning features.
        DAILY_SUMMARY (str): Optional daily summary configuration/prompt value.
    """

    LLM_PROVIDER: str = Field(default="ollama")
    LLM_BASE_URL: str = Field(default="")
    LLM_API_KEY: str = Field(default="")

    PRIMARY_MODEL: str = Field(default="")
    SECONDARY_MODEL: str = Field(default="")
    EMBEDDING_PROVIDER: str = Field(default="")
    EMBEDDING_BASE_URL: str = Field(default="")
    EMBEDDING_API_KEY: str = Field(default="")
    EMBEDDING_MODEL: str = Field(default="")

    WHISPER_BASE_URL: str = Field(default="")
    WHISPER_MODEL: str = Field(default="")
    WHISPER_KEY: str = Field(default="")
    ASR_BASE_URL: str = Field(default="")
    ASR_MODEL: str = Field(default="")
    ASR_KEY: str = Field(default="")
    ASR_BATCH_URL: str = Field(default="")
    ASR_BATCH_KEY: str = Field(default="")
    ASR_LANGUAGE: str = Field(default="auto")
    ASR_PROVIDER: str = Field(default="openai_compatible")

    REASONING_MODEL: str = Field(default="")
    REASONING_ENABLED: bool = Field(default=False)

    DAILY_SUMMARY: str = Field(default="")


class ConfigData(BaseModel):
    """
    Container for configuration data.

    This model is used to wrap configuration data in a dictionary format.

    Attributes:
        data (dict): A dictionary containing configuration key-value pairs.
    """

    data: dict

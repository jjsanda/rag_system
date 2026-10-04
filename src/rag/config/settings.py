"""Application configuration (12-factor, pydantic-settings v2).

Every value has a safe default so an empty environment runs the fully-offline default
stack. Most settings read from ``RAG_*`` environment variables (the ``env_prefix``); secret
and standard-name settings (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``OTEL_*``) read from
their conventional un-prefixed names via explicit aliases.

Example:
    >>> import os
    >>> os.environ["RAG_TOP_K"] = "7"
    >>> Settings(_env_file=None).top_k
    7
    >>> del os.environ["RAG_TOP_K"]
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "EmbeddingProvider", "LLMProvider", "VectorStoreKind"]

EmbeddingProvider = Literal["fake", "local", "openai"]
LLMProvider = Literal["extractive", "ollama", "openai", "anthropic"]
VectorStoreKind = Literal["faiss", "chroma", "qdrant"]


class Settings(BaseSettings):
    """Typed, validated application settings loaded from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Runtime ---------------------------------------------------------------------
    env: Literal["local", "prod"] = "local"
    log_level: str = "INFO"
    log_json: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    # --- Security & limits -----------------------------------------------------------
    api_key: str | None = None
    max_question_chars: int = 4000
    max_document_chars: int = 200_000
    request_timeout_s: float = 30.0
    rate_limit: str = "60/minute"

    # --- Provider selection ----------------------------------------------------------
    embedding_provider: EmbeddingProvider = "fake"
    llm_provider: LLMProvider = "extractive"
    vector_store: VectorStoreKind = "faiss"

    # --- Chunking & retrieval --------------------------------------------------------
    tokenizer_encoding: str = "cl100k_base"
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = 4
    embedding_dim: int = 384

    # --- Embeddings: local / openai --------------------------------------------------
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    openai_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_BASE_URL")
    )

    # --- LLM: generation knobs -------------------------------------------------------
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    system_prompt: str | None = None

    # --- LLM: provider models --------------------------------------------------------
    openai_llm_model: str = "gpt-4o-mini"
    anthropic_llm_model: str = "claude-haiku-4-5"
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"

    # --- Vector store ----------------------------------------------------------------
    collection_name: str = "rag_documents"
    faiss_path: str = "./.data/faiss"
    chroma_path: str = "./.data/chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # --- Caching ---------------------------------------------------------------------
    embedding_cache: Literal["none", "memory", "disk"] = "memory"
    embedding_cache_path: str = "./.data/embcache"
    embedding_cache_size: int = 10_000
    response_cache: Literal["none", "memory"] = "none"
    response_cache_ttl_s: float = 300.0

    # --- Resilience ------------------------------------------------------------------
    provider_max_retries: int = 3
    provider_backoff_s: float = 0.5
    fallback_to_offline: bool = True

    # --- Observability ---------------------------------------------------------------
    metrics_enabled: bool = True
    otel_enabled: bool = Field(default=False, validation_alias=AliasChoices("OTEL_ENABLED"))
    otel_service_name: str = Field(
        default="rag-system", validation_alias=AliasChoices("OTEL_SERVICE_NAME")
    )
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4318",
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    # --- Corpus / UI -----------------------------------------------------------------
    corpus_dir: str = "./data/corpus"
    api_url: str = "http://localhost:8000"
    ui_port: int = 8501

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse the comma-separated ``cors_origins`` into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def public_info(self) -> dict[str, object]:
        """Non-secret configuration summary for the ``/api/v1/info`` endpoint and the UI."""
        models = {
            "fake": f"hashing-{self.embedding_dim}",
            "local": self.local_embedding_model,
            "openai": self.openai_embedding_model,
        }
        llm_models = {
            "extractive": "extractive-v1",
            "ollama": self.ollama_llm_model,
            "openai": self.openai_llm_model,
            "anthropic": self.anthropic_llm_model,
        }
        return {
            "embedding_provider": self.embedding_provider,
            "embedding_model": models[self.embedding_provider],
            "llm_provider": self.llm_provider,
            "llm_model": llm_models[self.llm_provider],
            "vector_store": self.vector_store,
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "fallback_to_offline": self.fallback_to_offline,
        }

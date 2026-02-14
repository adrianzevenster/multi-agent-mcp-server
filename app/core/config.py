from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# IMPORTANT: do not override exported env vars
load_dotenv(override=False)

def _truthy(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}

@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "mcp-local-agents")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Use a writable default (avoids "readonly database" surprises)
    # Override with DATABASE_URL if you want postgres, etc.
    database_url: str = os.getenv("DATABASE_URL", "sqlite:////tmp/mcp_logs.db")

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    # Local dev default = localhost, docker-compose should override to http://ollama:11434
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    ollama_timeout_s: int = int(os.getenv("OLLAMA_TIMEOUT_S", "180"))

    openai_compat_base_url: str = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8080/v1")
    openai_compat_model: str = os.getenv("OPENAI_COMPAT_MODEL", "gpt-oss-local")

    max_tool_steps: int = int(os.getenv("MAX_TOOL_STEPS", "6"))

    # RAG
    rag_enabled: bool = _truthy(os.getenv("RAG_ENABLED", "true"))
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "monc_rag")

    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))
    embed_cache_path: str = os.getenv("EMBED_CACHE_PATH", "./embed_cache.sqlite")

    rag_top_k: int = int(os.getenv("RAG_TOP_K", "8"))
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.25"))

    # ------------------------------------------------------------
    # Event logging to Qdrant (chat/prompt/response traces)
    # ------------------------------------------------------------
    log_events_to_qdrant: bool = _truthy(os.getenv("LOG_EVENTS_TO_QDRANT", "false"))
    qdrant_event_collection: str = os.getenv("QDRANT_EVENT_COLLECTION", "monc_chat_logs")
    qdrant_event_autocreate: bool = _truthy(os.getenv("QDRANT_EVENT_AUTOCREATE", "true"))
    qdrant_event_timeout_s: int = int(os.getenv("QDRANT_EVENT_TIMEOUT_S", "10"))

settings = Settings()

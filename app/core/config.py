from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "mcp-local-agents")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./mcp_logs.db")

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    openai_compat_base_url: str = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8080/v1")
    openai_compat_model: str = os.getenv("OPENAI_COMPAT_MODEL", "gpt-oss-local")

    max_tool_steps: int = int(os.getenv("MAX_TOOL_STEPS", "6"))

    # RAG
    QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "monc_rag")

    EMBED_MODEL = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/all-mpnet-base-v2"
    )
    EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
    EMBED_CACHE_PATH = os.getenv(
        "EMBED_CACHE_PATH",
        "./embed_cache.sqlite"
    )

    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
    RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))


settings = Settings()

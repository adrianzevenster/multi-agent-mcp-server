import inspect
import logging
import os
import time
from typing import Optional, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_db import DbLogger, QdrantEventLogger
from app.core.models import ChatRequest, ChatResponse

from app.llm.ollama_client import OllamaClient
from app.llm.openai_compat_client import OpenAICompatClient

from app.mcp.tool_types import Tool
from app.mcp.tool_registry import ToolRegistry
from app.mcp.mcp_http import mount_mcp_routes

from app.rag.embedder import LocalEmbedder
from app.rag.qdrant_store import QdrantRagStore

from app.tools import builtin_tools
from app.agents.tool_calling_agent import ToolCallingAgent


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _init_db_logger() -> Optional[DbLogger]:
    """
    Optional DB logger init with retry. Non-fatal unless REQUIRE_DB=true.
    """
    require_db = _bool_env("REQUIRE_DB", False)
    retries = int(os.getenv("DB_INIT_RETRIES", "30"))
    sleep_s = float(os.getenv("DB_INIT_SLEEP_S", "1.0"))

    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            db = DbLogger(settings.database_url)
            logging.info("DbLogger enabled")
            return db
        except Exception as e:
            last_err = e
            logging.getLogger(__name__).warning(
                "Postgres unavailable during startup (attempt %s/%s): %r",
                i + 1,
                retries,
                e,
                )
            time.sleep(sleep_s)

    if require_db and last_err is not None:
        raise last_err

    logging.getLogger(__name__).warning("DbLogger disabled (postgres not reachable)")
    return None


def _build_qdrant_event_logger(
        *,
        qdrant_url: str,
        collection: str,
        embed_dim: int,
        autocreate: bool,
) -> Optional[QdrantEventLogger]:
    """
    Create QdrantEventLogger while tolerating signature drift.
    Your runtime indicated embed_dim is required, so we map it properly.
    """
    if not qdrant_url:
        return None

    try:
        sig = inspect.signature(QdrantEventLogger.__init__)
        params = set(sig.parameters.keys())

        kwargs: dict[str, Any] = {}

        # URL param variants
        if "url" in params:
            kwargs["url"] = qdrant_url
        elif "host" in params:
            kwargs["host"] = qdrant_url
        elif "endpoint" in params:
            kwargs["endpoint"] = qdrant_url
        elif "qdrant_url" in params:
            kwargs["qdrant_url"] = qdrant_url
        else:
            logging.getLogger(__name__).warning(
                "QdrantEventLogger has no recognizable url param. params=%s", sorted(params)
            )
            return None

        # collection param variants
        if "collection" in params:
            kwargs["collection"] = collection
        elif "collection_name" in params:
            kwargs["collection_name"] = collection

        # embed dim param variants (yours requires embed_dim)
        if "embed_dim" in params:
            kwargs["embed_dim"] = embed_dim
        elif "dim" in params:
            kwargs["dim"] = embed_dim
        elif "vector_size" in params:
            kwargs["vector_size"] = embed_dim
        elif "embedding_dim" in params:
            kwargs["embedding_dim"] = embed_dim

        # autocreate param variants
        if "autocreate" in params:
            kwargs["autocreate"] = autocreate
        elif "auto_create" in params:
            kwargs["auto_create"] = autocreate
        elif "create_if_missing" in params:
            kwargs["create_if_missing"] = autocreate

        ql = QdrantEventLogger(**kwargs)  # type: ignore[arg-type]
        logging.info(
            "Qdrant event logging enabled -> %s / %s (embed_dim=%s, autocreate=%s)",
            qdrant_url,
            collection,
            embed_dim,
            autocreate,
        )
        return ql

    except Exception as e:
        logging.getLogger(__name__).warning("QdrantEventLogger failed (ignored): %r", e)
        return None


def create_app() -> FastAPI:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    log = logging.getLogger(__name__)

    log.info("ENV OLLAMA_BASE_URL=%s", os.getenv("OLLAMA_BASE_URL"))
    log.info("ENV QDRANT_URL=%s", os.getenv("QDRANT_URL"))
    log.info("ENV LOG_EVENTS_TO_QDRANT=%s", os.getenv("LOG_EVENTS_TO_QDRANT"))
    log.info("ENV QDRANT_EVENT_COLLECTION=%s", os.getenv("QDRANT_EVENT_COLLECTION"))
    log.info("ENV EMBED_DIM=%s", os.getenv("EMBED_DIM"))

    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------
    # DB logger (optional)
    # ----------------------------
    db = _init_db_logger()
    app.state.db = db

    # ----------------------------
    # Qdrant event logger (optional)
    # ----------------------------
    app.state.qdrant_event_logger = None
    if _bool_env("LOG_EVENTS_TO_QDRANT", False):
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        collection = os.getenv("QDRANT_EVENT_COLLECTION", "monc_chat_logs").strip()
        embed_dim = int(os.getenv("EMBED_DIM", "768"))
        autocreate = _bool_env("QDRANT_EVENT_AUTOCREATE", True)

        app.state.qdrant_event_logger = _build_qdrant_event_logger(
            qdrant_url=qdrant_url,
            collection=collection,
            embed_dim=embed_dim,
            autocreate=autocreate,
        )

    def safe_log_event(event_type: str, payload: dict, *, run_id: Optional[str], agent_name: str) -> None:
        # Postgres
        try:
            if app.state.db is not None:
                app.state.db.log_event(event_type, payload, run_id=run_id, agent_name=agent_name)
        except Exception:
            log.exception("DbLogger failure (ignored)")

        # Qdrant
        try:
            ql: Optional[QdrantEventLogger] = getattr(app.state, "qdrant_event_logger", None)
            if ql is not None:
                ql.log_event(event_type, payload, run_id=run_id, agent_name=agent_name)
        except Exception as e:
            log.warning("QdrantEventLogger failed (ignored): %r", e)

    app.state.safe_log_event = safe_log_event

    # ----------------------------
    # RAG / Qdrant store (optional)
    # ----------------------------
    require_qdrant = _bool_env("REQUIRE_QDRANT", False)
    rag_enabled = _bool_env("RAG_ENABLED", True)
    rag_min_score = float(os.getenv("RAG_MIN_SCORE", "0.25"))

    app.state.rag_enabled = bool(rag_enabled)
    app.state.rag_ready = False
    app.state.rag_last_error = None

    # IMPORTANT: use repo’s constructors (no unexpected kwargs)
    app.state.rag_store = QdrantRagStore()
    app.state.rag_embedder = LocalEmbedder()

    if app.state.rag_enabled:
        try:
            app.state.rag_store.ensure_collection()
            app.state.rag_ready = True
            app.state.rag_last_error = None
            log.info("Qdrant ready. RAG enabled.")
        except Exception as e:
            app.state.rag_ready = False
            app.state.rag_last_error = str(e)
            log.exception("Qdrant unavailable during startup; continuing without RAG: %r", e)
            if require_qdrant:
                raise

    def safe_rag_search(
            query: str,
            top_k: Optional[int] = None,
            brand: Optional[str] = None,
            country: Optional[str] = None,
            min_score: Optional[float] = None,
    ):
        if not getattr(app.state, "rag_enabled", True):
            return []
        if not getattr(app.state, "rag_ready", False):
            return []

        def _clean(v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            v = str(v).strip()
            if not v:
                return None
            if v.lower() in {"global", "all", "any", "none", "null"}:
                return None
            return v

        brand = _clean(brand)
        country = _clean(country)

        try:
            vec = app.state.rag_embedder.embed_query(query)

            filters = {}
            if brand:
                filters["brand"] = brand
            if country:
                filters["country"] = country

            res = app.state.rag_store.search(
                vec,
                filters=filters,
                top_k=int(top_k or settings.RAG_TOP_K),
                min_score=min_score if min_score is not None else rag_min_score,
            )

            # fall back to unfiltered if filtered returns nothing
            if not res and filters:
                res = app.state.rag_store.search(
                    vec,
                    filters={},
                    top_k=int(top_k or settings.RAG_TOP_K),
                    min_score=min_score if min_score is not None else rag_min_score,
                )
            return res

        except Exception:
            log.exception("RAG search failed (returning [])")
            return []

    # ----------------------------
    # Tools + registry
    # ----------------------------
    tools: List[Tool] = [
        Tool(
            name="ping",
            description="Health check tool; returns UTC timestamp",
            schema={"type": "object", "properties": {}, "required": []},
            fn=lambda: builtin_tools.ping(),
        ),
        Tool(
            name="add",
            description="Add two numbers",
            schema={
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            fn=lambda a, b: builtin_tools.add(a=a, b=b),
        ),
        Tool(
            name="list_capabilities",
            description="List system capabilities",
            schema={"type": "object", "properties": {}, "required": []},
            fn=lambda: builtin_tools.list_capabilities(),
        ),
        Tool(
            name="list_tools",
            description="List available tools",
            schema={"type": "object", "properties": {}, "required": []},
            fn=lambda: builtin_tools.list_tools(),
        ),
        Tool(
            name="rag_search",
            description="Search the local RAG vector database (Qdrant) for grounded context.",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "brand": {"type": "string"},
                    "country": {"type": "string"},
                    "min_score": {"type": "number"},
                },
                "required": ["query"],
            },
            fn=lambda query, top_k=None, brand=None, country=None, min_score=None: safe_rag_search(
                query=query,
                top_k=top_k,
                brand=brand,
                country=country,
                min_score=min_score,
            ),
        ),
    ]

    registry = ToolRegistry(tools)
    app.state.registry = registry

    # ✅ ensures /mcp/tools and /mcp/call exist
    app.include_router(mount_mcp_routes(registry))

    # ----------------------------
    # LLM clients
    # ----------------------------
    ollama_base = os.getenv("OLLAMA_BASE_URL", settings.ollama_base_url)
    ollama_model = os.getenv("OLLAMA_MODEL", settings.ollama_model)

    openai_base = os.getenv("OPENAI_COMPAT_BASE_URL", settings.openai_compat_base_url)
    openai_model = os.getenv("OPENAI_COMPAT_MODEL", settings.openai_compat_model)

    ollama = OllamaClient(ollama_base, ollama_model)
    openai_compat = OpenAICompatClient(openai_base, openai_model, os.getenv("OPENAI_COMPAT_API_KEY", ""))

    llm_provider = os.getenv("LLM_PROVIDER", settings.llm_provider)

    agent = ToolCallingAgent(
        registry=registry,
        db=db,
        llm_provider=llm_provider,
        ollama=ollama,
        openai_compat=openai_compat,
        max_steps=settings.max_tool_steps,
    )
    app.state.agent = agent

    # ----------------------------
    # Meta / health
    # ----------------------------
    @app.get("/", tags=["meta"])
    def root(request: Request):
        return {
            "name": settings.app_name,
            "status": "ok",
            "docs": "/docs",
            "rag": {
                "enabled": bool(getattr(request.app.state, "rag_enabled", True)),
                "ready": bool(getattr(request.app.state, "rag_ready", False)),
                "last_error": getattr(request.app.state, "rag_last_error", None),
            },
            "endpoints": {
                "chat": "/chat",
                "tools": "/mcp/tools",
                "mcp_call": "/mcp/call",
                "events": "/events",
                "healthz": "/healthz",
                "readyz": "/readyz",
            },
        }

    @app.get("/healthz", tags=["health"])
    def healthz():
        return {"ok": True}

    @app.get("/readyz", tags=["health"])
    def readyz(request: Request):
        return {
            "ok": True,
            "agent_ready": bool(getattr(request.app.state, "agent", None) is not None),
            "rag_ready": bool(getattr(request.app.state, "rag_ready", False)),
            "db_ready": bool(getattr(request.app.state, "db", None) is not None),
        }

    # ----------------------------
    # Chat + events
    # ----------------------------
    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(req: ChatRequest, request: Request) -> ChatResponse:
        agent_name = req.agent_name or "default"
        try:
            run_id, output, tool_calls = request.app.state.agent.run(
                req.message,
                run_id=req.run_id,
                agent_name=agent_name,
            )
            return ChatResponse(run_id=run_id, agent_name=agent_name, output=output, tool_calls=tool_calls)
        except Exception as e:
            safe_log = getattr(request.app.state, "safe_log_event", None)
            if callable(safe_log):
                safe_log("chat_error", {"error": str(e)}, run_id=req.run_id, agent_name=agent_name)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/events", tags=["debug"])
    def events(request: Request, limit: int = 50, run_id: Optional[str] = None):
        # Prefer DbLogger for /events (your UI expects list-like output)
        db_: Optional[DbLogger] = getattr(request.app.state, "db", None)
        if db_ is not None:
            return db_.recent_events(limit=limit, run_id=run_id)

        # Fallback if DB is not configured
        return {"ok": False, "error": "DbLogger not initialized", "events": []}

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )

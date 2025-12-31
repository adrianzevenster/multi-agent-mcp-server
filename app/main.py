from __future__ import annotations

import os
import logging
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag.embedder import LocalEmbedder
from app.rag.qdrant_store import QdrantRagStore

from app.core.config import settings
from app.core.models import ChatRequest, ChatResponse
from app.core.logging_db import DbLogger

from app.llm.ollama_client import OllamaClient
from app.llm.openai_compat_client import OpenAICompatClient

from app.mcp.tool_types import Tool
from app.mcp.tool_registry import ToolRegistry
from app.mcp.mcp_http import mount_mcp_routes

from app.agents.tool_calling_agent import ToolCallingAgent
from app.tools import builtin_tools


def create_app() -> FastAPI:
    """
    Create and configure FastAPI app.

    Sets up:
    - CORS middleware
    - Optional Postgres-backend Dblogger
    - Qdrant and local embedder for RAG search
    - tool registry and MCP routes
    - /chat endpoint backend by ToolCallingAgent
    - /events debug endpoint

    :return: Configured FastAPI app
    """
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    REQUIRE_DB = os.getenv("REQUIRE_DB", "false").lower() == "true"
    DB_INIT_RETRIES = int(os.getenv("DB_INIT_RETRIES", "30"))
    DB_INIT_SLEEP_S = float(os.getenv("DB_INIT_SLEEP_S", "1.0"))

    db: Optional[DbLogger] = None
    last_db_err: Optional[Exception] = None

    for i in range(DB_INIT_RETRIES):
        try:
            db = DbLogger(settings.database_url)
            last_db_err = None
            break
        except Exception as e:
            last_db_err = e
            logging.exception(
                f"Postgres unavailable during startup (attempt {i + 1}/{DB_INIT_RETRIES}): {e}"
            )
            time.sleep(DB_INIT_SLEEP_S)

    if db is None:
        if REQUIRE_DB:
            raise last_db_err
        logging.error("Continuing without DbLogger (events endpoint will be degraded).")

    def safe_log_event(event_type: str, payload: dict, *, run_id: Optional[str], agent_name: str) -> None:
        try:
            if db is not None:
                db.log_event(event_type, payload, run_id=run_id, agent_name=agent_name)
        except Exception as e:
            logging.exception(f"DbLogger failure (ignored): {e}")

    rag_store = QdrantRagStore()
    rag_embedder = LocalEmbedder()

    REQUIRE_QDRANT = os.getenv("REQUIRE_QDRANT", "false").lower() == "true"

    try:
        rag_store.ensure_collection()
    except Exception as e:
        logging.exception(f"Qdrant unavailable during startup: {e}")
        if REQUIRE_QDRANT:
            raise

    def safe_rag_search(query: str, top_k: Optional[int] = None):
        try:
            return rag_store.search(
                rag_embedder.embed_query(query),
                filters={},
                top_k=int(top_k or settings.RAG_TOP_K),
            )
        except Exception:
            return []

    tools = [
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
            description="List available tools (hint: /mcp/tools shows full registry)",
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
                },
                "required": ["query"],
            },
            fn=lambda query, top_k=None: safe_rag_search(query, top_k),
        ),
    ]

    registry = ToolRegistry(tools)

    ollama = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    openai_compat = OpenAICompatClient(settings.openai_compat_base_url, settings.openai_compat_model)

    agent = ToolCallingAgent(
        registry=registry,
        db=db,
        llm_provider=settings.llm_provider,
        ollama=ollama,
        openai_compat=openai_compat,
        max_steps=settings.max_tool_steps,
    )

    app.include_router(mount_mcp_routes(registry))

    @app.get("/", tags=["meta"])
    def root():
        return {
            "name": settings.app_name,
            "status": "ok",
            "docs": "/docs",
            "endpoints": {
                "chat": "/chat",
                "tools": "/mcp/tools",
                "mcp_call": "/mcp/call",
                "events": "/events",
            },
        }

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(req: ChatRequest) -> ChatResponse:
        """
        Run the tool-calling agent on user message.

        :param req: Chat request
        :return: ChatResponse including output text and tool call trace
        :raises HTTPException: 500 on unhandled errors
        """
        agent_name = req.agent_name or "default"
        try:
            run_id, output, tool_calls = agent.run(
                req.message,
                run_id=req.run_id,
                agent_name=agent_name,
            )
            return ChatResponse(
                run_id=run_id,
                agent_name=agent_name,
                output=output,
                tool_calls=tool_calls,
            )
        except Exception as e:
            safe_log_event(
                "chat_error",
                {"error": str(e)},
                run_id=req.run_id,
                agent_name=agent_name,
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/events", tags=["debug"])
    def events(limit: int = 50, run_id: Optional[str] = None):
        if db is None:
            return {"ok": False, "error": "DbLogger not initialized", "events": []}
        return db.recent_events(limit=limit, run_id=run_id)

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

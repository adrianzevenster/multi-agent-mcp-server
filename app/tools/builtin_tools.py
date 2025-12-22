from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List

def ping() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

def add(a: float, b: float) -> Dict[str, Any]:
    return {"a": a, "b": b, "sum": a + b}

def echo(text: str) -> Dict[str, Any]:
    return {"echo": text}


def list_tools() -> Dict[str, Any]:
    return {
        "tools_hint": "Use GET /mcp/tools to view tools (UI button). Builtins available: ping, add, echo, list_capabilities."
    }

def list_capabilities() -> Dict[str, Any]:
    return {
        "capabilities": [
            "Tool calling via strict JSON",
            "HTTP MCP-style tool registry (/mcp/tools, /mcp/call)",
            "Database logging of user, model, tool calls, results",
            "Works with Ollama or OpenAI-compatible local servers",
        ]
    }

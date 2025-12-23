from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List

from .tool_registry import ToolRegistry

router = APIRouter()

class ToolCallRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = {}

class ToolCallResponse(BaseModel):
    ok: bool
    result: Any = None
    error: str | None = None

def mount_mcp_routes(registry: ToolRegistry) -> APIRouter:
    r = APIRouter(prefix="/mcp", tags=["mcp"])

    @r.get("/tools")
    def list_tools() -> List[Dict[str, Any]]:
        return registry.list()

    @r.post("/call", response_model=ToolCallResponse)
    def call_tool(req: ToolCallRequest) -> ToolCallResponse:
        try:
            result = registry.call(req.tool, req.args)
            return ToolCallResponse(ok=True, result=result)
        except Exception as e:
            return ToolCallResponse(ok=False, error=str(e))

    return r

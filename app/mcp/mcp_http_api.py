# mcp_http_api.py
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

from google.adk.tools.function_tool import FunctionTool
from shared.context_tools import get_product_details, get_brand_guidelines

app = FastAPI(title="Multi-Agent MCP (HTTP)")

product_tool = FunctionTool(get_product_details)
brand_tool = FunctionTool(get_brand_guidelines)

TOOLS = {
    product_tool.name: product_tool,
    brand_tool.name: brand_tool,
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/mcp/tools")
async def list_tools() -> Dict[str, Any]:
    # lightweight JSON description for web clients
    return {
        "tools": [
            {"name": product_tool.name, "description": "get_product_details(product_id: str)"},
            {"name": brand_tool.name, "description": "get_brand_guidelines(brand: str)"},
        ]
    }

class CallToolRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}

@app.post("/mcp/call")
async def call_tool(req: CallToolRequest) -> Dict[str, Any]:
    if req.name not in TOOLS:
        return {"error": f"Unknown tool '{req.name}'"}

    try:
        result = await TOOLS[req.name].run_async(args=req.arguments, tool_context=None)
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

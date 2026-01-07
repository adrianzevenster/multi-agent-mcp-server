from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


@dataclass(frozen=True)
class SimpleTool:
    name: str
    description: str
    fn: Callable[..., Any]

    async def run_async(self, args: Dict[str, Any]) -> Any:
        out = self.fn(**args)
        if hasattr(out, "__await__"):
            return await out
        return out

def get_product_details(product_id: str) -> Dict[str, Any]:
    # TODO: replace with real implementation
    return {"product_id": product_id, "name": "Unknown", "status": "stub"}


def get_brand_guidelines(brand: str) -> Dict[str, Any]:
    # TODO: replace with real implementation
    return {"brand": brand, "guidelines": [], "status": "stub"}


TOOLS: Dict[str, SimpleTool] = {
    "get_product_details": SimpleTool(
        name="get_product_details",
        description="get_product_details(product_id: str)",
        fn=get_product_details,
    ),
    "get_brand_guidelines": SimpleTool(
        name="get_brand_guidelines",
        description="get_brand_guidelines(brand: str)",
        fn=get_brand_guidelines,
    ),
}


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def list_tools(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "tools": [
                {"name": t.name, "description": t.description}
                for t in TOOLS.values()
            ]
        }
    )


async def call_tool(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"}, status_code=400)

    name = payload.get("name")
    args = payload.get("arguments") or {}

    if not isinstance(name, str) or not name:
        return JSONResponse({"ok": False, "error": "Field 'name' must be a non-empty string"}, status_code=400)
    if not isinstance(args, dict):
        return JSONResponse({"ok": False, "error": "Field 'arguments' must be an object"}, status_code=400)

    tool = TOOLS.get(name)
    if tool is None:
        return JSONResponse({"ok": False, "error": f"Unknown tool '{name}'"}, status_code=404)

    try:
        result = await tool.run_async(args)
        return JSONResponse({"ok": True, "result": result})
    except TypeError as exc:
        return JSONResponse({"ok": False, "error": f"Bad arguments: {exc}"}, status_code=400)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/mcp/tools", list_tools, methods=["GET"]),
        Route("/mcp/call", call_tool, methods=["POST"]),
    ],
)

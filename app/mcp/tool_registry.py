from __future__ import annotations
from typing import Any, Dict, List
from .tool_types import Tool

class ToolRegistry:
    def __init__(self, tools: List[Tool]):
        self._tools = {t.name: t for t in tools}

    def list(self) -> List[Dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "schema": t.schema}
            for t in self._tools.values()
        ]

    def call(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].fn(**args)

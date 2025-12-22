from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: Dict[str, Any]
    fn: Callable[..., Any]

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    run_id: Optional[str] = None
    agent_name: Optional[str] = "default"

class ChatResponse(BaseModel):
    run_id: str
    agent_name: str
    output: str
    tool_calls: List[Dict[str, Any]] = []


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    hash: str
    metadata: Dict[str, Any]

class ChunkHit(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


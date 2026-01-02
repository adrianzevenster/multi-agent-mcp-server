# app/mcp/rag_tools.py
from __future__ import annotations
from typing import Any, Dict

from app.core.config import settings
from app.rag.embedder import LocalEmbedder
from app.rag.qdrant_store import QdrantRagStore

_embedder = LocalEmbedder()
_store = QdrantRagStore()
_store.ensure_collection()

def rag_search(args: Dict[str, Any]) -> Any:
    query = (args or {}).get("query", "").strip()
    if not query:
        return []

    top_k = int((args or {}).get("top_k") or settings.rag_top_k)

    filters = {}
    for k in ["brand", "country", "product_id", "channel", "source_type"]:
        v = (args or {}).get(k)
        if v:
            filters[k] = v

    qvec = _embedder.embed_query(query)
    hits = _store.search(qvec, filters=filters, top_k=top_k)

    out = []
    for h in hits:
        out.append({
            "chunk_id": h["chunk_id"],
            "doc_id": h["doc_id"],
            "text": h["text"],
            "score": h["score"],
            "metadata": h.get("metadata", {}),
        })
    return out

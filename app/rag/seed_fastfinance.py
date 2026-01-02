from __future__ import annotations

import os
import json
import time
import uuid
import urllib.request
from typing import Any, Dict, List

from app.rag.embedder import LocalEmbedder


def _http_json(method: str, url: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_for_qdrant(qdrant_url: str, tries: int = 60, sleep_s: float = 1.0) -> None:
    for _ in range(tries):
        try:
            urllib.request.urlopen(qdrant_url.rstrip("/") + "/readyz", timeout=2.0)
            return
        except Exception:
            time.sleep(sleep_s)
    raise RuntimeError("Qdrant not ready in time")


def ensure_collection(qdrant_url: str, collection: str, vector_size: int) -> None:
    # create collection if missing
    try:
        urllib.request.urlopen(f"{qdrant_url}/collections/{collection}", timeout=2.0)
        return
    except Exception:
        pass

    _http_json(
        "PUT",
        f"{qdrant_url}/collections/{collection}",
        {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        },
    )


def upsert_points(qdrant_url: str, collection: str, points: List[Dict[str, Any]]) -> None:
    _http_json(
        "PUT",
        f"{qdrant_url}/collections/{collection}/points?wait=true",
        {"points": points},
        timeout=30.0,
    )


def main() -> None:
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    collection = os.getenv("QDRANT_COLLECTION", "monc_rag")
    embed_dim = int(os.getenv("EMBED_DIM", "768"))

    wait_for_qdrant(qdrant_url)
    ensure_collection(qdrant_url, collection, embed_dim)

    embedder = LocalEmbedder()

    docs = [
        {
            "doc_id": "fastfinance_product",
            "chunk_id": "ff_sme_001",
            "text": "# Fast Finance — SME Credit Line (Product Facts)\nPurpose:\n- Designed to support working capital needs for small and medium-sized enterprises (SMEs).\nEligibility:\n- Subject to assessment and approval.\n- Not all applicants will qualify.\nRequired Disclaimer:\n- \"T&Cs and eligibility apply.\"",
            "metadata": {"brand": "Fast Finance", "country": "GLOBAL", "source_type": "product_facts"},
        },
        {
            "doc_id": "fastfinance_brand",
            "chunk_id": "ff_brand_001",
            "text": "# Fast Finance — Brand & Compliance Guidelines\nForbidden Claims:\n- No \"guaranteed approval\"\n- No \"instant approval\"\n- No \"immediate access\"\n- No timelines such as \"24 hours\", \"same day\", or similar.\nTone:\n- Clear, professional, and supportive.\n- Confident but not promotional or exaggerated.",
            "metadata": {"brand": "Fast Finance", "country": "GLOBAL", "source_type": "brand_guidelines"},
        },
        {
            "doc_id": "fastfinance_product",
            "chunk_id": "ff_sme_002",
            "text": "Important Messaging Rules:\n- Do NOT promise approval.\n- Do NOT mention approval timelines unless explicitly provided in official terms.\n- Do NOT mention interest rates or fees unless explicitly provided.",
            "metadata": {"brand": "Fast Finance", "country": "GLOBAL", "source_type": "product_facts"},
        },
    ]

    points = []
    for d in docs:
        vec = embedder.embed_query(d["text"])
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "doc_id": d["doc_id"],
                    "chunk_id": d["chunk_id"],
                    "text": d["text"],
                    **d["metadata"],
                },
            }
        )

    upsert_points(qdrant_url, collection, points)

    print("Seeded Fast Finance docs into Qdrant:", len(points))


if __name__ == "__main__":
    main()

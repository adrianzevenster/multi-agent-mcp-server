from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


class QdrantRagStore:
    """
    Minimal Qdrant HTTP client (no qdrant-client dependency).
    This avoids silent failures from client mismatch / named-vector mismatch.

    Payload schema assumed (from your seeder):
      {
        "doc_id": "...",
        "chunk_id": "...",
        "text": "...",
        "brand": "...",
        "country": "...",
        "source_type": "..."
      }
    """

    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
        self.collection = os.getenv("QDRANT_COLLECTION", "monc_rag")
        self.embed_dim = int(os.getenv("EMBED_DIM", "768"))

    def _http_json(
            self,
            method: str,
            path: str,
            payload: Optional[Dict[str, Any]] = None,
            timeout: float = 10.0,
    ) -> Dict[str, Any]:
        url = f"{self.qdrant_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
            except Exception:
                body = ""
            raise RuntimeError(f"Qdrant HTTPError {e.code} on {path}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"Qdrant request failed on {path}: {e}") from e

    def wait_ready(self, tries: int = 60, sleep_s: float = 1.0) -> None:
        for _ in range(tries):
            try:
                urllib.request.urlopen(f"{self.qdrant_url}/readyz", timeout=2.0)
                return
            except Exception:
                time.sleep(sleep_s)
        raise RuntimeError("Qdrant not ready in time")

    def ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.
        Uses unnamed vector (default) with cosine distance.
        """
        # exists?
        try:
            self._http_json("GET", f"/collections/{self.collection}", timeout=3.0)
            return
        except Exception:
            pass

        self._http_json(
            "PUT",
            f"/collections/{self.collection}",
            {
                "vectors": {
                    "size": self.embed_dim,
                    "distance": "Cosine",
                }
            },
            timeout=10.0,
        )

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Qdrant filter format:
          {"must":[{"key":"brand","match":{"value":"Fast Finance"}}]}
        """
        must = []
        for k, v in (filters or {}).items():
            if v is None:
                continue
            sv = str(v).strip()
            if not sv:
                continue
            must.append({"key": k, "match": {"value": sv}})
        return {"must": must} if must else None

    def search(
            self,
            query_vector: List[float],
            *,
            filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10,
            min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of hits shaped like:
          {"doc_id","chunk_id","text","score","metadata":{...}}
        """
        body: Dict[str, Any] = {
            "vector": query_vector,
            "limit": int(top_k),
            "with_payload": True,
        }

        if min_score is not None:
            body["score_threshold"] = float(min_score)

        f = self._build_filter(filters or {})
        if f:
            body["filter"] = f

        resp = self._http_json(
            "POST",
            f"/collections/{self.collection}/points/search",
            body,
            timeout=30.0,
        )

        result = resp.get("result") or []
        hits: List[Dict[str, Any]] = []

        for item in result:
            payload = item.get("payload") or {}
            score = item.get("score")
            hits.append(
                {
                    "doc_id": payload.get("doc_id", ""),
                    "chunk_id": payload.get("chunk_id", ""),
                    "text": payload.get("text", ""),
                    "score": score,
                    "metadata": {
                        "brand": payload.get("brand"),
                        "country": payload.get("country"),
                        "source_type": payload.get("source_type"),
                    },
                }
            )

        return hits

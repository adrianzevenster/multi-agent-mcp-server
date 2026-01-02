from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/debug", tags=["debug"])


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 3.0) -> Dict[str, Any]:
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
        return {"ok": False, "http_error": e.code, "body": body}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/rag")
def rag_debug() -> Dict[str, Any]:
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333").rstrip("/")
    collection = os.getenv("QDRANT_COLLECTION", "monc_rag")

    info = _http_json("GET", f"{qdrant_url}/collections/{collection}")
    count = _http_json("POST", f"{qdrant_url}/collections/{collection}/points/count", {"exact": True})

    # try a small scroll to inspect payload keys
    scroll = _http_json(
        "POST",
        f"{qdrant_url}/collections/{collection}/points/scroll",
        {"limit": 3, "with_payload": True, "with_vector": False},
    )

    # summarize payload keys if we got points
    payload_keys = []
    try:
        pts = scroll.get("result", {}).get("points", []) or []
        for p in pts:
            pl = p.get("payload") or {}
            payload_keys.append(sorted(list(pl.keys())))
    except Exception:
        pass

    return {
        "qdrant_url": qdrant_url,
        "collection": collection,
        "collection_info": info,
        "count": count,
        "scroll_sample": scroll,
        "scroll_payload_keys": payload_keys,
    }

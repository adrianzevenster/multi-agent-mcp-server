from __future__ import annotations
import requests
from app.core.config import settings

def main() -> None:
    base = settings.QDRANT_URL.rstrip("/")
    col = settings.QDRANT_COLLECTION

    r = requests.get(f"{base}/collections", timeout=10)
    r.raise_for_status()
    names = [c["name"] for c in r.json().get("result", {}).get("collections", [])]

    if col in names:
        print(f"Qdrant collection already exists: {col}")
        return

    payload = {"vectors": {"size": int(settings.EMBED_DIM), "distance": "Cosine"}}
    r = requests.put(f"{base}/collections/{col}", json=payload, timeout=20)
    r.raise_for_status()
    print(f"Created Qdrant collection: {col} (dim={settings.EMBED_DIM})")

if __name__ == "__main__":
    main()

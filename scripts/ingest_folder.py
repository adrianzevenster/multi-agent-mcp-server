from __future__ import annotations
import glob, os, hashlib
from uuid import uuid4
from typing import Dict, List

from app.rag.embedder import LocalEmbedder
from app.rag.qdrant_store import QdrantRagStore

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def infer_metadata(path: str) -> Dict:
    name = os.path.basename(path).lower()
    md: Dict = {"country": "GLOBAL"}
    if "fastfinance" in name:
        md["brand"] = "Fast Finance"
    if "brand" in name:
        md["source_type"] = "brand_guidelines"
    elif "product" in name or "facts" in name:
        md["source_type"] = "product_facts"
    else:
        md["source_type"] = "misc"
    return md

def chunk_paragraphs(doc_id: str, text: str, metadata: Dict) -> List[Dict]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out = []
    for p in paras:
        h = sha(p)
        out.append({
            "chunk_id": sha(f"{doc_id}::{h}"),
            "doc_id": doc_id,
            "text": p,
            "hash": h,
            "metadata": dict(metadata),
        })
    return out


def main() -> None:
    folder = os.getenv("RAG_SOURCES_DIR", "/app/data/rag_resources/")
    paths = sorted(glob.glob(os.path.join(folder, "*.md")) + glob.glob(os.path.join(folder, "*.txt")))
    if not paths:
        raise SystemExit(f"No files found in {folder}")

    store = QdrantRagStore()
    store.ensure_collection()
    embedder = LocalEmbedder()

    total = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()

        doc_id = os.path.splitext(os.path.basename(p))[0]
        meta = infer_metadata(p)
        chunks = chunk_paragraphs(doc_id, txt, meta)

        vectors = [embedder.embed_text_cached(c["text"], c["hash"]) for c in chunks]
        store.upsert_chunks(chunks, vectors)
        total += len(chunks)
        print(f"✅ Ingested {len(chunks)} chunks from {doc_id}")

    print(f"Done. Total upserted: {total}")

if __name__ == "__main__":
    main()

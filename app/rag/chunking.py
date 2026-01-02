import hashlib
import uuid

def chunk_text(doc_id: str, text: str, metadata: dict):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for p in paragraphs:
        h = hashlib.sha256(p.encode()).hexdigest()
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "text": p,
            "hash": h,
            "metadata": metadata
        })

    return chunks

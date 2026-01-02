from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.rag.cache import EmbeddingCache

class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBED_MODEL)
        self.cache = EmbeddingCache(settings.EMBED_CACHE_PATH)

    def embed_text_cached(self, text: str, text_hash: str) -> List[float]:
        cached = self.cache.get(text_hash)
        if cached is not None:
            return cached

        vec = self.model.encode([text], normalize_embeddings=True)[0]
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] != int(settings.EMBED_DIM):
            raise ValueError(f"Embedding dim mismatch: got {vec.shape[0]}, expected {settings.EMBED_DIM}")

        out = vec.tolist()
        self.cache.put(text_hash, out)
        return out

    def embed_query(self, query: str) -> List[float]:
        vec = self.model.encode([query], normalize_embeddings=True)[0]
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] != int(settings.EMBED_DIM):
            raise ValueError(f"Embedding dim mismatch: got {vec.shape[0]}, expected {settings.EMBED_DIM}")
        return vec.tolist()

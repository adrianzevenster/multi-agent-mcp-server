# app/rag/cache.py
import sqlite3
import json
from typing import Optional, List

class EmbeddingCache:
    def __init__(self, path: str):
        self.path = path
        with sqlite3.connect(self.path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (hash TEXT PRIMARY KEY, vec TEXT NOT NULL)"
            )
            con.commit()

    def get(self, h: str) -> Optional[List[float]]:
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT vec FROM embeddings WHERE hash=?", (h,)).fetchone()
            return None if row is None else json.loads(row[0])

    def put(self, h: str, vec: List[float]) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO embeddings(hash, vec) VALUES(?,?)", (h, json.dumps(vec)))
            con.commit()

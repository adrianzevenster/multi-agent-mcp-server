from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import requests
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, Index
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class EventLog(Base):
    __tablename__ = "event_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), nullable=False, index=True)

    run_id = Column(String(128), nullable=True, index=True)
    agent_name = Column(String(128), nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)

    payload_json = Column(Text, nullable=False)
    metadata_json = Column(Text, nullable=True)

Index("ix_event_logs_run_agent", EventLog.run_id, EventLog.agent_name)

# -------------------------------------------------------------------
# Qdrant event logger (stores prompt/response/events as points)
# -------------------------------------------------------------------
class QdrantEventLogger:
    """
    Writes events to a Qdrant collection as points.
    Each event becomes a point with:
      - vector: embedding of a compact text representation of the event
      - payload: full structured event data
    """

    def __init__(
            self,
            qdrant_url: str,
            collection: str,
            *,
            embed_dim: int,
            autocreate: bool = True,
            timeout_s: int = 10,
    ):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self.embed_dim = embed_dim
        self.autocreate = autocreate
        self.timeout_s = timeout_s

        if self.autocreate:
            self.ensure_collection()

    def ensure_collection(self) -> None:
        # Check if exists
        r = requests.get(f"{self.qdrant_url}/collections/{self.collection}", timeout=self.timeout_s)
        if r.status_code == 200:
            return

        # Create
        payload = {"vectors": {"size": self.embed_dim, "distance": "Cosine"}}
        rc = requests.put(
            f"{self.qdrant_url}/collections/{self.collection}",
            json=payload,
            timeout=self.timeout_s,
        )
        rc.raise_for_status()

    def _embed(self, text: str) -> List[float]:
        """
        Lazy import to keep 'core' light at startup.
        Falls back to a zero vector if embedding fails.
        """
        try:
            from app.rag.embedder import LocalEmbedder  # lazy
            emb = LocalEmbedder().embed_query(text)
            if not isinstance(emb, list) or len(emb) != self.embed_dim:
                return [0.0] * self.embed_dim
            return emb
        except Exception:
            return [0.0] * self.embed_dim

    def log_event(
            self,
            event_type: str,
            payload: Dict[str, Any],
            *,
            run_id: Optional[str] = None,
            agent_name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            created_at: Optional[datetime] = None,
    ) -> str:
        ts = created_at or datetime.now(timezone.utc)

        # Compact text for embedding (searchable)
        # Keep it stable and short-ish, but include key fields.
        text_for_embedding = json.dumps(
            {
                "event_type": event_type,
                "run_id": run_id,
                "agent": agent_name,
                "payload": payload,
            },
            ensure_ascii=False,
        )

        vector = self._embed(text_for_embedding)

        point_id = str(uuid.uuid4())
        q_payload = {
            "id": point_id,
            "created_at": ts.isoformat(),
            "run_id": run_id,
            "agent_name": agent_name,
            "event_type": event_type,
            "payload": payload,
            "metadata": metadata,
        }

        body = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": q_payload,
                }
            ]
        }

        r = requests.put(
            f"{self.qdrant_url}/collections/{self.collection}/points",
            json=body,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return point_id


# -------------------------------------------------------------------
# SQLite/Postgres logger (existing)
# -------------------------------------------------------------------
class DbLogger:
    def __init__(
            self,
            database_url: str,
            *,
            qdrant_event_logger: Optional[QdrantEventLogger] = None,
    ):
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        self.qdrant = qdrant_event_logger

    def log_event(
            self,
            event_type: str,
            payload: Dict[str, Any],
            *,
            run_id: Optional[str] = None,
            agent_name: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            created_at: Optional[datetime] = None,
    ) -> int:
        ts = created_at or datetime.now(timezone.utc)

        # 1) Write to SQL
        row = EventLog(
            created_at=ts,
            run_id=run_id,
            agent_name=agent_name,
            event_type=event_type,
            payload_json=json.dumps(payload, ensure_ascii=False),
            metadata_json=None if metadata is None else json.dumps(metadata, ensure_ascii=False),
        )
        with self.Session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            sql_id = int(row.id)

        # 2) Optionally mirror to Qdrant (best-effort)
        if self.qdrant is not None:
            try:
                self.qdrant.log_event(
                    event_type=event_type,
                    payload=payload,
                    run_id=run_id,
                    agent_name=agent_name,
                    metadata=metadata,
                    created_at=ts,
                )
            except Exception as e:
                # Best-effort: do not fail the request because Qdrant logging failed
                # (your app likely already has a "safe_log_event" wrapper too)
                import logging
                logging.getLogger(__name__).warning("QdrantEventLogger failed (ignored): %r", e)

        return sql_id

    def recent_events(self, limit: int = 50, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        from sqlalchemy import select, desc
        stmt = select(EventLog).order_by(desc(EventLog.id)).limit(limit)
        if run_id:
            stmt = stmt.where(EventLog.run_id == run_id)
        with self.Session() as s:
            rows = s.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "run_id": r.run_id,
                "agent_name": r.agent_name,
                "event_type": r.event_type,
                "payload": json.loads(r.payload_json),
                "metadata": None if r.metadata_json is None else json.loads(r.metadata_json),
            }
            for r in rows
        ]

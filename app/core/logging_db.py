from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from sqlalchemy import (
    Column, DateTime, Integer, String, Text, create_engine, Index
)
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

class DbLogger:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)

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
            return int(row.id)

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

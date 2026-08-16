from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS cameras (id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,source TEXT NOT NULL,enabled INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS people (id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL UNIQUE,external_ref TEXT,active INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS face_embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT,person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,embedding BLOB NOT NULL,dimension INTEGER NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT,camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,event_type TEXT NOT NULL,confidence REAL,track_id TEXT,snapshot_path TEXT,metadata_json TEXT,occurred_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_logs (id INTEGER PRIMARY KEY AUTOINCREMENT,action TEXT NOT NULL,actor TEXT,target_type TEXT,target_id TEXT,details_json TEXT,created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_person ON face_embeddings(person_id);
"""

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class SurvilDB:
    """Local persistence layer for edge deployments."""
    def __init__(self, path: str | Path = "data/survilai.db") -> None:
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path); conn.row_factory = sqlite3.Row; conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn; conn.commit()
        except Exception:
            conn.rollback(); raise
        finally: conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn: conn.executescript(SCHEMA)

    def add_camera(self, name: str, source: str) -> int:
        with self.connect() as conn:
            return int(conn.execute("INSERT INTO cameras(name,source,created_at) VALUES (?,?,?)", (name,source,utc_now())).lastrowid)

    def add_person(self, name: str, external_ref: str | None = None) -> int:
        with self.connect() as conn:
            return int(conn.execute("INSERT INTO people(name,external_ref,created_at) VALUES (?,?,?)", (name,external_ref,utc_now())).lastrowid)

    def add_embedding(self, person_id: int, embedding: bytes, dimension: int) -> int:
        with self.connect() as conn:
            return int(conn.execute("INSERT INTO face_embeddings(person_id,embedding,dimension,created_at) VALUES (?,?,?,?)", (person_id,sqlite3.Binary(embedding),dimension,utc_now())).lastrowid)

    def add_event(self, event_type: str, camera_id: int | None = None, person_id: int | None = None, confidence: float | None = None, track_id: str | None = None, snapshot_path: str | None = None, metadata_json: str | None = None) -> int:
        with self.connect() as conn:
            return int(conn.execute("""INSERT INTO events(camera_id,person_id,event_type,confidence,track_id,snapshot_path,metadata_json,occurred_at) VALUES (?,?,?,?,?,?,?,?)""", (camera_id,person_id,event_type,confidence,track_id,snapshot_path,metadata_json,utc_now())).lastrowid)

    def audit(self, action: str, actor: str | None = None, target_type: str | None = None, target_id: str | None = None, details_json: str | None = None) -> int:
        with self.connect() as conn:
            return int(conn.execute("INSERT INTO audit_logs(action,actor,target_type,target_id,details_json,created_at) VALUES (?,?,?,?,?,?)", (action,actor,target_type,target_id,details_json,utc_now())).lastrowid)

    def recent_events(self, limit: int = 100) -> list[sqlite3.Row]:
        limit=max(1,min(int(limit),1000))
        with self.connect() as conn: return list(conn.execute("SELECT * FROM events ORDER BY occurred_at DESC LIMIT ?", (limit,)))

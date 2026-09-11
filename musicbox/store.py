"""SQLite persistence for frozen, immutable versions.

A version row stores the canonical freeze request and the computed result.
Rows are never updated or deleted: versions are immutable by construction.
Freezing is idempotent — the same canonical input maps to the same
``content_hash`` and returns the existing row.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str = "musicbox.db") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(SCHEMA)

    def get_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM versions WHERE content_hash = ?", (content_hash,)
            )
            return cur.fetchone()

    def get(self, version_id: int) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM versions WHERE id = ?", (version_id,))
            return cur.fetchone()

    def insert(self, content_hash: str, request_json: str, result_json: str) -> sqlite3.Row:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO versions (content_hash, created_at, request_json, result_json)"
                " VALUES (?, ?, ?, ?)",
                (content_hash, created_at, request_json, result_json),
            )
            row = self._conn.execute(
                "SELECT * FROM versions WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            assert row is not None
            return row

    def list(self) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM versions ORDER BY id")
            return cur.fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

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
CREATE TABLE IF NOT EXISTS balance_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dynamics_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dynamics_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,
    batch_hash TEXT NOT NULL,
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
            self._conn.executescript(SCHEMA)

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

    # -- balance plans (same immutability guarantees as versions) -----------

    def get_plan_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM balance_plans WHERE content_hash = ?", (content_hash,)
            )
            return cur.fetchone()

    def get_plan(self, plan_id: int) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM balance_plans WHERE id = ?", (plan_id,)
            )
            return cur.fetchone()

    def insert_plan(
        self, content_hash: str, request_json: str, result_json: str
    ) -> sqlite3.Row:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO balance_plans (content_hash, created_at, request_json,"
                " result_json) VALUES (?, ?, ?, ?)",
                (content_hash, created_at, request_json, result_json),
            )
            row = self._conn.execute(
                "SELECT * FROM balance_plans WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            assert row is not None
            return row

    def list_plans(self) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM balance_plans ORDER BY id")
            return cur.fetchall()

    # -- dynamics trials and plans (same immutability guarantees) -----------

    def _generic_get(self, table: str, row_id: int) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (row_id,)
            )
            return cur.fetchone()

    def _generic_get_by_hash(self, table: str, content_hash: str) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT * FROM {table} WHERE content_hash = ?", (content_hash,)
            )
            return cur.fetchone()

    def _generic_insert(
        self, table: str, content_hash: str, request_json: str, result_json: str
    ) -> sqlite3.Row:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"INSERT INTO {table} (content_hash, created_at, request_json,"
                " result_json) VALUES (?, ?, ?, ?)",
                (content_hash, created_at, request_json, result_json),
            )
            row = self._conn.execute(
                f"SELECT * FROM {table} WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            assert row is not None
            return row

    def _generic_list(self, table: str) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(f"SELECT * FROM {table} ORDER BY id")
            return cur.fetchall()

    def get_trial(self, trial_id: int) -> sqlite3.Row | None:
        return self._generic_get("dynamics_trials", trial_id)

    def get_trial_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        return self._generic_get_by_hash("dynamics_trials", content_hash)

    def insert_trial(
        self, content_hash: str, request_json: str, result_json: str
    ) -> sqlite3.Row:
        return self._generic_insert(
            "dynamics_trials", content_hash, request_json, result_json
        )

    def list_trials(self) -> list[sqlite3.Row]:
        return self._generic_list("dynamics_trials")

    def get_dynamics_plan(self, plan_id: int) -> sqlite3.Row | None:
        return self._generic_get("dynamics_plans", plan_id)

    def get_dynamics_plan_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        return self._generic_get_by_hash("dynamics_plans", content_hash)

    def insert_dynamics_plan(
        self, content_hash: str, request_json: str, result_json: str
    ) -> sqlite3.Row:
        return self._generic_insert(
            "dynamics_plans", content_hash, request_json, result_json
        )

    def list_dynamics_plans(self) -> list[sqlite3.Row]:
        return self._generic_list("dynamics_plans")

    # -- calibration batches and plans (same immutability guarantees) --------
    #
    # A batch row never changes: its status (collecting -> confirmed) is
    # derived from whether a calibration plan references its content hash.

    def get_cal_batch(self, batch_id: int) -> sqlite3.Row | None:
        return self._generic_get("calibration_batches", batch_id)

    def get_cal_batch_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        return self._generic_get_by_hash("calibration_batches", content_hash)

    def insert_cal_batch(
        self, content_hash: str, request_json: str, result_json: str
    ) -> sqlite3.Row:
        return self._generic_insert(
            "calibration_batches", content_hash, request_json, result_json
        )

    def list_cal_batches(self) -> list[sqlite3.Row]:
        return self._generic_list("calibration_batches")

    def get_cal_plan(self, plan_id: int) -> sqlite3.Row | None:
        return self._generic_get("calibration_plans", plan_id)

    def get_cal_plan_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        return self._generic_get_by_hash("calibration_plans", content_hash)

    def get_cal_plan_for_batch(self, batch_hash: str) -> sqlite3.Row | None:
        """The first plan confirmed on a batch, if any (drives its status)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM calibration_plans WHERE batch_hash = ? ORDER BY id"
                " LIMIT 1",
                (batch_hash,),
            )
            return cur.fetchone()

    def insert_cal_plan(
        self, content_hash: str, batch_hash: str, request_json: str, result_json: str
    ) -> sqlite3.Row:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO calibration_plans (content_hash, batch_hash, created_at,"
                " request_json, result_json) VALUES (?, ?, ?, ?, ?)",
                (content_hash, batch_hash, created_at, request_json, result_json),
            )
            row = self._conn.execute(
                "SELECT * FROM calibration_plans WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            assert row is not None
            return row

    def list_cal_plans(self) -> list[sqlite3.Row]:
        return self._generic_list("calibration_plans")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

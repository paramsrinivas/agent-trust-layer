"""SQLite-backed atomic idempotency store (stdlib-only, multi-process safe)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_STORE_PATH = Path(__file__).resolve().parent / "data" / "idempotency.sqlite3"


class IdempotencyConflictError(ValueError):
    """Raised when a request_id is reused with a different payload/intent."""


def _store_path(db_path: Optional[Path] = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_path = os.environ.get("AGENT_TRUST_IDEMPOTENCY_DB")
    return Path(env_path) if env_path else DEFAULT_STORE_PATH


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = _store_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS idempotency_records (
            request_id   TEXT PRIMARY KEY,
            request_hash TEXT NOT NULL,
            state        TEXT NOT NULL,
            result_json  TEXT,
            created_at   TEXT NOT NULL,
            updated_at   TEXT NOT NULL
        )
        """
    )
    return conn


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def begin_request(
    request_id: str,
    request_hash: str,
    *,
    db_path: Optional[Path] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    now = _utc_now()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT request_hash, state, result_json FROM idempotency_records WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is not None:
            existing_hash, state, result_json = row
            conn.execute("COMMIT")
            if state != "DENIED" and existing_hash != request_hash:
                raise IdempotencyConflictError(
                    f"request_id '{request_id}' reused with mismatched payload."
                )
            return state, (json.loads(result_json) if result_json else None)

        conn.execute(
            """
            INSERT INTO idempotency_records
                (request_id, request_hash, state, result_json, created_at, updated_at)
            VALUES (?, ?, 'IN_PROGRESS', NULL, ?, ?)
            """,
            (request_id, request_hash, now, now),
        )
        conn.execute("COMMIT")
        return "ACQUIRED", None
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def finalize_request(
    request_id: str,
    state: str,
    result: dict[str, Any],
    *,
    db_path: Optional[Path] = None,
) -> None:
    if state not in {"COMPLETED", "UNRESOLVED", "FAILED_PERMANENT"}:
        raise ValueError(f"Invalid terminal state: {state}")
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE idempotency_records
            SET state = ?, result_json = ?, updated_at = ?
            WHERE request_id = ?
            """,
            (state, json.dumps(result), _utc_now(), request_id),
        )
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def release_request(request_id: str, *, db_path: Optional[Path] = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM idempotency_records WHERE request_id = ? AND state = 'IN_PROGRESS'",
            (request_id,),
        )
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def record_denial(
    request_id: str,
    result: dict[str, Any],
    *,
    db_path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    now = _utc_now()
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT state, result_json FROM idempotency_records WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is not None:
            state, result_json = row
            conn.execute("COMMIT")
            parsed = json.loads(result_json) if result_json else {"called": False, "status": "in_progress"}
            return state, parsed

        conn.execute(
            """
            INSERT INTO idempotency_records
                (request_id, request_hash, state, result_json, created_at, updated_at)
            VALUES (?, 'DENIED', 'DENIED', ?, ?, ?)
            """,
            (request_id, json.dumps(result), now, now),
        )
        conn.execute("COMMIT")
        return "DENIED", result
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def clear(*, db_path: Optional[Path] = None) -> None:
    path = _store_path(db_path)
    for suffix in ("", "-wal", "-shm"):
        p = Path(f"{path}{suffix}")
        if p.exists():
            p.unlink()

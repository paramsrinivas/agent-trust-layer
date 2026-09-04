"""
Idempotency store for the Trust Layer -> Razorpay boundary.

Why this exists: a purchase request can legitimately be re-submitted (a
network retry from the buyer agent, a dashboard double-click, a webhook
redelivery). Without idempotency, each resubmission of the SAME
already-authorized request would create a SEPARATE Razorpay order --
charging (in a real, non-test integration) the same purchase twice.

Design: request_id is the idempotency key. Before creating an order, check
whether this request_id already has one. If yes, return the EXISTING order
instead of creating a new one. This is the same pattern real payment
processors expect (Razorpay itself supports idempotency keys server-side;
this is a lightweight equivalent at the application layer that also works
identically in dry-run mode, so it's testable without a network call).

Storage: a single JSON file. Good enough for a hackathon prototype; a real
deployment would use a database with a unique constraint on request_id.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()
_STORE_PATH = Path("data") / "idempotency_store.json"


def _load() -> dict:
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(store: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(store, indent=2))


def get_existing_order(request_id: str) -> Optional[dict]:
    """Returns the previously-recorded result for this request_id, or None
    if this request has never been processed before."""
    with _LOCK:
        return _load().get(request_id)


def record_order(request_id: str, result: dict) -> None:
    """Record the outcome of processing this request_id, so a future
    resubmission of the same request_id short-circuits to this result
    instead of contacting Razorpay again."""
    with _LOCK:
        store = _load()
        store[request_id] = result
        _save(store)


def clear() -> None:
    """Wipe the idempotency store. Useful between demo runs / tests."""
    with _LOCK:
        if _STORE_PATH.exists():
            _STORE_PATH.unlink()

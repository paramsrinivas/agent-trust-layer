"""Razorpay payment adapter (hardened, fail-closed, reconciled retries)."""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import idempotency
from models import PaymentAuthorizationDecision

MAX_RETRIES = 2
BASE_BACKOFF_SECONDS = 1.0
MAX_RECEIPT_LEN = 40
MIN_INR_AMOUNT = Decimal("1.00")


class RazorpayNotConfigured(RuntimeError):
    """Raised when Razorpay keys are missing or invalid for the selected mode."""


@dataclass
class RazorpayResult:
    called: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    raw: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    dry_run: bool = False
    idempotent_replay: bool = False

    def to_cache_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("idempotent_replay", None)
        return d

    @classmethod
    def from_cache_dict(cls, data: dict[str, Any], *, replay: bool = True) -> "RazorpayResult":
        return cls(
            called=bool(data.get("called", False)),
            order_id=data.get("order_id"),
            status=data.get("status"),
            raw=data.get("raw"),
            error=data.get("error"),
            dry_run=bool(data.get("dry_run", False)),
            idempotent_replay=replay,
        )


def _require_keys(mode: str) -> tuple[str, str]:
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise RazorpayNotConfigured("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set.")

    if mode == "test":
        if not key_id.startswith("rzp_test_"):
            raise RazorpayNotConfigured(
                f"Expected TEST key ('rzp_test_'), got '{key_id[:12]}...'. Refusing."
            )
    elif mode == "live":
        if os.environ.get("ALLOW_LIVE_RAZORPAY") != "1" or not key_id.startswith("rzp_live_"):
            raise RazorpayNotConfigured(
                "LIVE mode requires 'rzp_live_' key and ALLOW_LIVE_RAZORPAY=1."
            )
    else:
        raise RazorpayNotConfigured(f"Unknown razorpay_mode: {mode!r}")

    return key_id, key_secret


def _get_client(mode: str):
    key_id, key_secret = _require_keys(mode)
    import razorpay
    return razorpay.Client(auth=(key_id, key_secret))


def _is_retryable_error(exc: Exception) -> bool:
    try:
        import requests
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
    except ImportError:
        pass
    try:
        import razorpay.errors as rzp_err
        retryable = tuple(
            getattr(rzp_err, name)
            for name in ("ServerError", "GatewayError")
            if hasattr(rzp_err, name)
        )
        if retryable and isinstance(exc, retryable):
            return True
    except ImportError:
        pass
    return False


def _find_existing_order_by_receipt(client, receipt: str) -> Optional[dict[str, Any]]:
    if not hasattr(client.order, "all"):
        return None
    try:
        resp = client.order.all({"receipt": receipt, "count": 5})
        items = resp.get("items", []) if isinstance(resp, dict) else []
        for item in items:
            if item.get("receipt") == receipt:
                return item
    except Exception:
        return None
    return None


def _create_order_with_retries(
    client, payload: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[str], str]:
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            existing = _find_existing_order_by_receipt(client, payload["receipt"])
            if existing is not None:
                return existing, None, "success"
        try:
            order = client.order.create(payload)
            return order, None, "success"
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc):
                return None, str(exc), "error"
            if attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0.0, 0.25)
                time.sleep(backoff)
    return None, str(last_exc), "unresolved"


def _to_paise(amount_rupees: Any) -> int:
    if isinstance(amount_rupees, bool) or amount_rupees is None:
        raise ValueError("Decision has missing/invalid 'amount' -- cannot create an order.")
    try:
        dec = Decimal(str(amount_rupees))
    except InvalidOperation as exc:
        raise ValueError(f"Non-numeric amount: {amount_rupees!r}") from exc
    if not dec.is_finite() or dec < MIN_INR_AMOUNT:
        raise ValueError(f"Amount must be finite and >= INR {MIN_INR_AMOUNT}, got {amount_rupees!r}")
    return int((dec * Decimal("100")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _build_payload(decision: PaymentAuthorizationDecision) -> dict[str, Any]:
    if not decision.request_id or len(decision.request_id) > MAX_RECEIPT_LEN:
        raise ValueError(f"request_id must be 1..{MAX_RECEIPT_LEN} chars for Razorpay receipt.")
    amount_paise = _to_paise(decision.behavior.features.get("amount"))
    return {
        "amount": amount_paise,
        "currency": "INR",
        "receipt": decision.request_id,
        "notes": {
            "agent_id": decision.agent_id,
            "session_id": decision.session_id,
            "risk_score": str(round(decision.behavior.anomaly_score, 5)),
            "source": "agent_trust_layer_demo",
        },
    }


def _process(
    decision: PaymentAuthorizationDecision,
    *,
    db_path: Optional[Path] = None,
) -> RazorpayResult:
    payload = _build_payload(decision)
    req_hash = idempotency.fingerprint_payload(
        {"amount": payload["amount"], "currency": payload["currency"], "mode": decision.razorpay_mode}
    )

    state, cached = idempotency.begin_request(decision.request_id, req_hash, db_path=db_path)
    if state == "IN_PROGRESS":
        return RazorpayResult(
            called=False,
            status="in_progress (concurrent request active)",
            idempotent_replay=True,
        )
    if state != "ACQUIRED":
        return RazorpayResult.from_cache_dict(cached or {}, replay=True)

    try:
        client = _get_client(decision.razorpay_mode)
    except RazorpayNotConfigured as exc:
        idempotency.release_request(decision.request_id, db_path=db_path)
        return RazorpayResult(
            called=False,
            dry_run=True,
            status="dry_run (Razorpay not configured)",
            error=str(exc),
            raw=payload,
        )

    try:
        order, error, outcome = _create_order_with_retries(client, payload)
    except Exception:
        idempotency.finalize_request(
            decision.request_id,
            "UNRESOLVED",
            RazorpayResult(
                called=True, status="unresolved", error="unexpected client exception", raw=payload
            ).to_cache_dict(),
            db_path=db_path,
        )
        raise

    if outcome == "success" and order is not None:
        result = RazorpayResult(
            called=True, order_id=order.get("id"), status=order.get("status"), raw=order
        )
        idempotency.finalize_request(decision.request_id, "COMPLETED", result.to_cache_dict(), db_path=db_path)
    elif outcome == "unresolved":
        result = RazorpayResult(called=True, status="unresolved", error=error, raw=payload)
        idempotency.finalize_request(decision.request_id, "UNRESOLVED", result.to_cache_dict(), db_path=db_path)
    else:
        result = RazorpayResult(called=True, status="error", error=error, raw=payload)
        idempotency.finalize_request(decision.request_id, "FAILED_PERMANENT", result.to_cache_dict(), db_path=db_path)

    return result


def create_order_for_decision(
    decision: PaymentAuthorizationDecision,
    *,
    db_path: Optional[Path] = None,
) -> RazorpayResult:
    if decision.decision != "ALLOW":
        return RazorpayResult(called=False, status=f"skipped ({decision.decision})")
    return _process(decision, db_path=db_path)


def approve_review(
    decision: PaymentAuthorizationDecision,
    *,
    db_path: Optional[Path] = None,
) -> RazorpayResult:
    if decision.decision != "REVIEW":
        return RazorpayResult(
            called=False,
            status=f"approve_review() called on a {decision.decision} decision -- refused",
        )
    return _process(decision, db_path=db_path)


def deny_review(
    decision: PaymentAuthorizationDecision,
    *,
    db_path: Optional[Path] = None,
) -> RazorpayResult:
    if decision.decision != "REVIEW":
        return RazorpayResult(
            called=False,
            status=f"deny_review() called on a {decision.decision} decision -- refused",
        )
    denial = RazorpayResult(called=False, status="denied_by_reviewer")
    state, stored = idempotency.record_denial(
        decision.request_id, denial.to_cache_dict(), db_path=db_path
    )
    if state == "DENIED" and stored == denial.to_cache_dict():
        return denial
    return RazorpayResult.from_cache_dict(stored, replay=True)

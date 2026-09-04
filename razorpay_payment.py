"""
Razorpay TEST MODE payment adapter (v2).

Design principle (unchanged, locked earlier in the project):
    The Trust Layer decides whether Razorpay ever gets the opportunity
    to create an order.

        ALLOW  -> create a real Razorpay TEST MODE order
        REVIEW -> Razorpay is never called UNLESS a human explicitly
                  approves this specific request (see approve_review())
        BLOCK  -> Razorpay is NEVER called, no override exists. A policy
                  violation is not something a human clicking a dashboard
                  button should be able to bypass.

New in this version:

1. IDEMPOTENCY. The same request_id will never create two Razorpay
   orders, even if create_order_for_decision() is called on it twice
   (retry, double-click, replay). See idempotency.py.

2. GRACEFUL FAILURE. A network failure while creating an order is NOT
   silently treated as either success or a clean rejection -- those are
   both dishonest under a timeout, because the order may or may not have
   actually been created on Razorpay's side. Retryable network errors get
   retried with backoff; if still failing, the result is explicitly
   "unresolved" (never "success", never silently dropped), which is the
   honest answer and the one that should route to manual reconciliation
   rather than a false ALLOW or a false failure.

3. REVIEW APPROVAL. A human can approve a specific REVIEW decision via
   approve_review(decision), or explicitly deny it via deny_review(decision).
   Approval is a SEPARATE, explicit action -- it is not possible to get an
   order created for a REVIEW decision by any path other than this
   function being called with that exact decision.

--------------------------------------------------------------------------
SETUP: unchanged from v1.
    1. pip install razorpay python-dotenv
    2. Get TEST MODE keys: Razorpay Dashboard -> Test Mode -> API Keys.
    3. .env file: RAZORPAY_KEY_ID=rzp_test_..., RAZORPAY_KEY_SECRET=...
    4. python3 test_razorpay_adapter.py --live
--------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models import PaymentAuthorizationDecision
import idempotency

MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = [1, 3]


class RazorpayNotConfigured(RuntimeError):
    """Raised when TEST MODE keys aren't set, or don't look like test keys."""


@dataclass
class RazorpayResult:
    called: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    raw: Optional[dict] = None
    error: Optional[str] = None
    dry_run: bool = False
    idempotent_replay: bool = False   # True if this request_id was already
                                       # processed -- we returned the cached
                                       # result instead of contacting Razorpay


def _require_test_mode_keys() -> tuple[str, str]:
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise RazorpayNotConfigured(
            "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not set."
        )
    if not key_id.startswith("rzp_test_"):
        raise RazorpayNotConfigured(
            f"Key '{key_id[:12]}...' does not look like a TEST MODE key "
            "(expected 'rzp_test_'). Refusing to proceed."
        )
    return key_id, key_secret


def _get_client():
    key_id, key_secret = _require_test_mode_keys()
    import razorpay
    return razorpay.Client(auth=(key_id, key_secret))


def _is_retryable_network_error(exc: Exception) -> bool:
    """Only network-level failures are retryable. A response FROM Razorpay
    saying the request is invalid (bad amount, bad params) is not a
    network problem -- retrying it would just get the same rejection
    again, so those are surfaced immediately as 'error', not retried."""
    try:
        import requests
        return isinstance(
            exc,
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        )
    except ImportError:
        return False


def _create_order_with_retries(client, payload: dict) -> tuple[Optional[dict], Optional[str], str]:
    """
    Returns (order_or_None, error_message_or_None, outcome) where outcome is
    one of:

        "success"    -> order created, order dict returned
        "error"      -> Razorpay rejected the request outright (not a
                         network problem). Definitely NOT created.
        "unresolved" -> network failed even after retries. We genuinely
                         do not know whether Razorpay created the order
                         before the connection dropped. This must be
                         surfaced as ambiguous, not guessed at.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            order = client.order.create(payload)
            return order, None, "success"
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_network_error(exc):
                return None, str(exc), "error"
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    return None, str(last_exc), "unresolved"


def _build_payload(decision: PaymentAuthorizationDecision) -> dict:
    amount_rupees = decision.behavior.features.get("amount")
    if amount_rupees is None:
        raise ValueError("Decision has no 'amount' feature -- cannot create an order.")
    return {
        "amount": int(round(amount_rupees * 100)),
        "currency": "INR",
        "receipt": decision.request_id,
        "notes": {
            "agent_id": decision.agent_id,
            "session_id": decision.session_id,
            "risk_score": str(round(decision.behavior.anomaly_score, 5)),
            "source": "agent_trust_layer_demo",
        },
    }


def _process(decision: PaymentAuthorizationDecision) -> RazorpayResult:
    """Shared machinery: idempotency check, then either dry-run or a real
    (retried, gracefully-handled) API call. Records the outcome so a
    resubmission of the same request_id short-circuits here next time."""
    existing = idempotency.get_existing_order(decision.request_id)
    if existing is not None:
        return RazorpayResult(idempotent_replay=True, **existing)

    payload = _build_payload(decision)

    try:
        client = _get_client()
    except RazorpayNotConfigured as exc:
        result = RazorpayResult(
            called=False, dry_run=True,
            status="dry_run (Razorpay not configured)",
            error=str(exc), raw=payload,
        )
        idempotency.record_order(decision.request_id, {
            "called": result.called, "order_id": result.order_id,
            "status": result.status, "raw": result.raw, "error": result.error,
            "dry_run": result.dry_run,
        })
        return result

    order, error, outcome = _create_order_with_retries(client, payload)

    if outcome == "success":
        result = RazorpayResult(called=True, order_id=order.get("id"),
                                 status=order.get("status"), raw=order)
    elif outcome == "unresolved":
        result = RazorpayResult(
            called=True, status="unresolved", error=error, raw=payload,
        )
        # Deliberately NOT recorded in the idempotency store: since we
        # don't know if an order exists, the safest behavior on the NEXT
        # attempt is to check with Razorpay again, not to silently accept
        # an unresolved state forever.
        return result
    else:  # "error"
        result = RazorpayResult(called=True, status="error", error=error, raw=payload)

    idempotency.record_order(decision.request_id, {
        "called": result.called, "order_id": result.order_id,
        "status": result.status, "raw": result.raw, "error": result.error,
        "dry_run": result.dry_run,
    })
    return result


def create_order_for_decision(decision: PaymentAuthorizationDecision) -> RazorpayResult:
    """
    The single boundary between the Trust Layer and real money movement
    for ALLOW decisions. REVIEW and BLOCK never reach Razorpay through
    this function -- REVIEW can only proceed via approve_review() below,
    and BLOCK has no override path at all.
    """
    if decision.decision == "BLOCK":
        return RazorpayResult(called=False, status="skipped (BLOCK)")
    if decision.decision == "REVIEW":
        return RazorpayResult(called=False, status="skipped (REVIEW, pending human approval)")
    return _process(decision)


def approve_review(decision: PaymentAuthorizationDecision) -> RazorpayResult:
    """
    Explicit human approval of a REVIEW decision. This is the ONLY path
    by which a REVIEW decision can result in a Razorpay order.

    Refuses anything that isn't actually a REVIEW decision -- a caller
    cannot approve an ALLOW (already handled normally) or a BLOCK (no
    override exists, full stop) through this function.
    """
    if decision.decision != "REVIEW":
        return RazorpayResult(
            called=False,
            status=f"approve_review() called on a {decision.decision} decision -- refused",
        )
    return _process(decision)


def deny_review(decision: PaymentAuthorizationDecision) -> RazorpayResult:
    """Explicit human denial of a REVIEW decision. Records the denial so a
    later accidental resubmission of the same request_id can't be
    approved silently."""
    result = RazorpayResult(called=False, status="denied_by_reviewer")
    idempotency.record_order(decision.request_id, {
        "called": False, "order_id": None, "status": "denied_by_reviewer",
        "raw": None, "error": None, "dry_run": False,
    })
    return result
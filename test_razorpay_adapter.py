"""
Standalone Razorpay TEST MODE test suite.

    python3 test_razorpay_adapter.py            # everything except a real
                                                   network call -- no keys,
                                                   no network needed
    python3 test_razorpay_adapter.py --live      # also makes one real
                                                   TEST MODE API call
"""

from __future__ import annotations

import argparse
import sys
import uuid

import idempotency
from models import PaymentAuthorizationDecision, PolicyResult, BehaviorResult
import razorpay_payment as rp


def fake_decision(decision: str, amount: float = 1000.0, request_id: str = None) -> PaymentAuthorizationDecision:
    return PaymentAuthorizationDecision(
        request_id=request_id or f"TEST-{uuid.uuid4().hex[:8]}",
        agent_id="TEST-AGENT",
        session_id="TEST-SESSION",
        decision=decision,
        policy=PolicyResult(passed=True, violations=[]),
        behavior=BehaviorResult(anomaly_score=0.0, is_anomalous=False, features={"amount": amount}),
        reasons=["test"],
    )


def test_gating_logic():
    print("1. Gating logic...")
    for decision_type in ["REVIEW", "BLOCK"]:
        result = rp.create_order_for_decision(fake_decision(decision_type))
        assert result.called is False, f"{decision_type} incorrectly reached Razorpay!"
        assert result.order_id is None
        print(f"   OK  {decision_type} never reaches Razorpay via create_order_for_decision()")

    allow_result = rp.create_order_for_decision(fake_decision("ALLOW"))
    assert allow_result.dry_run or allow_result.called
    print(f"   OK  ALLOW attempted to reach Razorpay (dry_run={allow_result.dry_run})")


def test_idempotency():
    print("2. Idempotency...")
    idempotency.clear()
    decision = fake_decision("ALLOW", amount=250.0, request_id="IDEMPOTENCY-TEST-1")

    first = rp.create_order_for_decision(decision)
    assert first.idempotent_replay is False, "first call should NOT be a replay"

    second = rp.create_order_for_decision(decision)
    assert second.idempotent_replay is True, "second call on the SAME request_id must be a replay"
    assert second.order_id == first.order_id, "replay must return the SAME order_id, not a new one"
    print(f"   OK  duplicate request_id returned cached result (order_id={second.order_id}), "
          "no second order created")


def test_review_approval_flow():
    print("3. REVIEW approval flow...")
    idempotency.clear()

    review_decision = fake_decision("REVIEW", amount=500.0, request_id="REVIEW-TEST-1")

    # Approve refuses non-REVIEW decisions
    allow_decision = fake_decision("ALLOW", request_id="ALLOW-MISUSE-TEST")
    misuse = rp.approve_review(allow_decision)
    assert misuse.called is False and "refused" in misuse.status
    print("   OK  approve_review() refuses to process a non-REVIEW decision")

    block_decision = fake_decision("BLOCK", request_id="BLOCK-MISUSE-TEST")
    misuse2 = rp.approve_review(block_decision)
    assert misuse2.called is False and "refused" in misuse2.status
    print("   OK  approve_review() refuses to process a BLOCK decision (no override exists)")

    # A REVIEW decision, unapproved, never reaches Razorpay
    unapproved = rp.create_order_for_decision(review_decision)
    assert unapproved.called is False
    print("   OK  unapproved REVIEW decision never reaches Razorpay")

    # Explicit approval processes it
    approved = rp.approve_review(review_decision)
    assert approved.dry_run or approved.called
    print(f"   OK  approve_review() processed the REVIEW decision (dry_run={approved.dry_run})")

    # Idempotency applies here too -- approving twice doesn't double-process
    approved_again = rp.approve_review(review_decision)
    assert approved_again.idempotent_replay is True
    print("   OK  re-approving the same REVIEW decision returns the cached result")


def test_deny_review():
    print("4. REVIEW denial...")
    idempotency.clear()
    decision = fake_decision("REVIEW", amount=750.0, request_id="DENY-TEST-1")

    denied = rp.deny_review(decision)
    assert denied.called is False and denied.status == "denied_by_reviewer"
    print("   OK  deny_review() marks the decision denied, never contacts Razorpay")

    # A later attempt to approve the SAME request_id after denial should
    # replay the denial, not silently approve it.
    late_approve = rp.approve_review(decision)
    assert late_approve.idempotent_replay is True
    assert late_approve.status == "denied_by_reviewer"
    print("   OK  a later approve_review() on an already-denied request_id "
          "returns the denial, does not silently approve it")


class _FakeConnectionError(Exception):
    """Stand-in for requests.exceptions.ConnectionError, used only to
    verify retry/backoff behavior without needing the real `requests`
    package's exception classes wired through a real network call."""


def test_graceful_network_failure():
    print("5. Graceful network failure handling (simulated, no real network)...")

    call_count = {"n": 0}

    class AlwaysFailsClient:
        class order:
            @staticmethod
            def create(payload):
                call_count["n"] += 1
                raise _FakeConnectionError("simulated connection failure")

    # Patch the retryable-error check to recognize our fake exception type,
    # and patch time.sleep so the test doesn't actually wait for backoff.
    orig_is_retryable = rp._is_retryable_network_error
    orig_sleep = rp.time.sleep
    rp._is_retryable_network_error = lambda exc: isinstance(exc, _FakeConnectionError)
    rp.time.sleep = lambda seconds: None

    try:
        order, error, outcome = rp._create_order_with_retries(
            AlwaysFailsClient(), {"amount": 100, "currency": "INR", "receipt": "x"}
        )
    finally:
        rp._is_retryable_network_error = orig_is_retryable
        rp.time.sleep = orig_sleep

    assert outcome == "unresolved", f"expected 'unresolved', got '{outcome}'"
    assert order is None
    assert call_count["n"] == rp.MAX_RETRIES + 1, (
        f"expected {rp.MAX_RETRIES + 1} attempts, got {call_count['n']}"
    )
    print(f"   OK  retried {call_count['n']} times on simulated network failure, "
          f"then surfaced outcome='unresolved' (never falsely 'success' or silently dropped)")


def test_live_connection():
    print("\n6. Live TEST MODE connection...")
    idempotency.clear()
    result = rp.create_order_for_decision(fake_decision("ALLOW", amount=100.0, request_id="LIVE-CHECK-1"))

    if result.dry_run:
        print("   FAIL  keys not configured -- see razorpay_payment.py setup instructions")
        print(f"   {result.error}")
        sys.exit(1)
    if result.error:
        print(f"   FAIL  Razorpay returned an error: {result.error}")
        sys.exit(1)

    print(f"   OK  order created: id={result.order_id}  status={result.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    test_gating_logic()
    test_idempotency()
    test_review_approval_flow()
    test_deny_review()
    test_graceful_network_failure()

    print("\nALL OFFLINE TESTS PASS")

    if args.live:
        test_live_connection()
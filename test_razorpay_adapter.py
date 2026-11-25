"""Standalone hardened test suite (isolated temp SQLite DB — never touches live data).

    python3 test_razorpay_adapter.py         # full offline suite
    python3 test_razorpay_adapter.py --live  # also runs one live Razorpay TEST MODE order call
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import threading
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

import idempotency
import razorpay_payment as rp
from authorization import TrustModel, authorize_purchase
from generator import BASE_TIME, estimate_agent_baseline, generate_agent, generate_normal_session, policy_check
from models import BehaviorResult, PaymentAuthorizationDecision, PolicyResult


def fake_decision(
    decision: str,
    amount: float = 1000.0,
    request_id: str | None = None,
    razorpay_mode: str = "test",
) -> PaymentAuthorizationDecision:
    return PaymentAuthorizationDecision(
        request_id=request_id or f"TEST-{uuid.uuid4().hex[:8]}",
        agent_id="TEST-AGENT",
        session_id="TEST-SESSION",
        decision=decision,
        policy=PolicyResult(passed=True, violations=[]),
        behavior=BehaviorResult(anomaly_score=0.0, is_anomalous=False, features={"amount": amount}),
        reasons=["test"],
        razorpay_mode=razorpay_mode,
    )


class _FakeClient:
    def __init__(self, order_id: str = "order_mock_1"):
        self.calls = 0
        self._order_id = order_id
        self.order = self

    def create(self, payload):
        self.calls += 1
        return {"id": f"{self._order_id}_{self.calls}", "status": "created", "receipt": payload["receipt"]}

    def all(self, params):
        return {"items": []}


def test_gating_fail_closed():
    print("1. Fail-closed gating logic...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.sqlite3"
        for bad_decision in ["REVIEW", "BLOCK", "allow", "UNKNOWN", ""]:
            res = rp.create_order_for_decision(fake_decision(bad_decision), db_path=db)
            assert res.called is False, f"{bad_decision!r} leaked through!"
            assert res.order_id is None

        client = _FakeClient()
        with patch.object(rp, "_get_client", return_value=client):
            res = rp.create_order_for_decision(fake_decision("ALLOW", amount=500.0), db_path=db)
            assert res.called is True and res.order_id == "order_mock_1_1"
    print("   OK  only explicit ALLOW reaches Razorpay; unknown values fail closed")


def test_idempotency_and_payload_conflict():
    print("2. Atomic idempotency & payload fingerprinting...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.sqlite3"
        client = _FakeClient()
        d1 = fake_decision("ALLOW", amount=250.0, request_id="IDEM-1")
        d1_tampered = fake_decision("ALLOW", amount=999.0, request_id="IDEM-1")

        with patch.object(rp, "_get_client", return_value=client):
            first = rp.create_order_for_decision(d1, db_path=db)
            second = rp.create_order_for_decision(d1, db_path=db)
            assert first.idempotent_replay is False
            assert second.idempotent_replay is True
            assert second.order_id == first.order_id
            assert client.calls == 1

            try:
                rp.create_order_for_decision(d1_tampered, db_path=db)
                raise AssertionError("Expected IdempotencyConflictError on mismatched amount")
            except idempotency.IdempotencyConflictError:
                pass
    print("   OK  same request_id replays once; mismatched amount raises IdempotencyConflictError")


def test_dry_run_does_not_pollute_store():
    print("3. Dry-run does not poison idempotency store...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.sqlite3"
        d = fake_decision("ALLOW", amount=300.0, request_id="DRY-1")
        with patch.object(rp, "_get_client", side_effect=rp.RazorpayNotConfigured("missing keys")):
            dry = rp.create_order_for_decision(d, db_path=db)
            assert dry.dry_run is True and dry.called is False

        client = _FakeClient()
        with patch.object(rp, "_get_client", return_value=client):
            real = rp.create_order_for_decision(d, db_path=db)
            assert real.called is True and real.idempotent_replay is False and real.order_id == "order_mock_1_1"
    print("   OK  dry-run releases claim so real keys work cleanly afterwards")


def test_review_approval_and_denial_immutability():
    print("4. REVIEW approval & immutable denial...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.sqlite3"
        client = _FakeClient()

        for bad in ["ALLOW", "BLOCK"]:
            assert "refused" in rp.approve_review(fake_decision(bad), db_path=db).status
            assert "refused" in rp.deny_review(fake_decision(bad), db_path=db).status

        d_deny = fake_decision("REVIEW", amount=750.0, request_id="DENY-FIRST")
        denied = rp.deny_review(d_deny, db_path=db)
        assert denied.status == "denied_by_reviewer"
        with patch.object(rp, "_get_client", return_value=client):
            late_app = rp.approve_review(d_deny, db_path=db)
            assert late_app.idempotent_replay is True and late_app.status == "denied_by_reviewer"
            assert client.calls == 0

        d_app = fake_decision("REVIEW", amount=500.0, request_id="APPROVE-FIRST")
        with patch.object(rp, "_get_client", return_value=client):
            approved = rp.approve_review(d_app, db_path=db)
            assert approved.order_id == "order_mock_1_1"
            late_deny = rp.deny_review(d_app, db_path=db)
            assert late_deny.idempotent_replay is True and late_deny.order_id == "order_mock_1_1"
    print("   OK  approve/deny are strictly terminal and never overwrite each other")


def test_concurrency_single_execution():
    print("5. Concurrency race protection (SQLite BEGIN IMMEDIATE)...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.sqlite3"
        calls = {"n": 0}
        barrier = threading.Barrier(2)

        class SlowClient:
            class order:
                @staticmethod
                def create(payload):
                    calls["n"] += 1
                    rp.time.sleep(0.08)
                    return {"id": "order_once", "status": "created"}

        d = fake_decision("ALLOW", amount=400.0, request_id="RACE-1")
        results = []

        def worker():
            barrier.wait()
            results.append(rp.create_order_for_decision(d, db_path=db))

        with patch.object(rp, "_get_client", return_value=SlowClient()):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert calls["n"] == 1, f"Expected 1 Razorpay call under race, got {calls['n']}"
        assert any(r.order_id == "order_once" for r in results)
        assert any(r.idempotent_replay for r in results)
    print("   OK  concurrent duplicate requests execute Razorpay exactly once")


class _FakeTimeout(Exception):
    pass


def test_retry_reconciliation_and_unresolved():
    print("6. Retry reconciliation & persistent UNRESOLVED state...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "idem.sqlite3"

        reconcile_calls = {"create": 0, "all": 0}

        class CreatedBeforeTimeoutClient:
            class order:
                @staticmethod
                def create(payload):
                    reconcile_calls["create"] += 1
                    raise _FakeTimeout("timed out after server commit")

                @staticmethod
                def all(params):
                    reconcile_calls["all"] += 1
                    return {"items": [{"id": "order_recovered", "status": "created", "receipt": params["receipt"]}]}

        with (
            patch.object(rp, "_get_client", return_value=CreatedBeforeTimeoutClient()),
            patch.object(rp, "_is_retryable_error", lambda e: isinstance(e, _FakeTimeout)),
            patch.object(rp.time, "sleep", lambda _: None),
        ):
            rec = rp.create_order_for_decision(fake_decision("ALLOW", amount=150.0, request_id="REC-1"), db_path=db)
            assert rec.order_id == "order_recovered"
            assert reconcile_calls["create"] == 1 and reconcile_calls["all"] == 1

        fail_calls = {"n": 0}

        class AlwaysFailsClient:
            class order:
                @staticmethod
                def create(payload):
                    fail_calls["n"] += 1
                    raise _FakeTimeout("network down")

                @staticmethod
                def all(params):
                    return {"items": []}

        d_fail = fake_decision("ALLOW", amount=150.0, request_id="UNRES-1")
        with (
            patch.object(rp, "_get_client", return_value=AlwaysFailsClient()),
            patch.object(rp, "_is_retryable_error", lambda e: isinstance(e, _FakeTimeout)),
            patch.object(rp.time, "sleep", lambda _: None),
        ):
            first = rp.create_order_for_decision(d_fail, db_path=db)
            assert first.status == "unresolved" and fail_calls["n"] == rp.MAX_RETRIES + 1
            second = rp.create_order_for_decision(d_fail, db_path=db)
            assert second.idempotent_replay is True and second.status == "unresolved"
            assert fail_calls["n"] == rp.MAX_RETRIES + 1
    print("   OK  reconciles existing receipts before retry; locks UNRESOLVED against blind retries")


def test_authorization_integrity_and_expiry():
    print("7. Authorization identity checks & mandate expiration...")
    rng = np.random.default_rng(7)
    agent = generate_agent("A001", "runner", rng, "test")
    other = generate_agent("A002", "student", rng, "test")
    hist = [generate_normal_session(agent, f"H{i}", BASE_TIME + timedelta(hours=i * 4), rng) for i in range(3)]
    pop = {"price_mean": 5000.0, "price_std": 1000.0, "gap_mean_sec": 240.0, "requests_per_5m_mean": 1.5}
    agent.baseline = estimate_agent_baseline(hist, pop)
    other.baseline = estimate_agent_baseline(hist, pop)

    class DummyModel:
        def decision_function(self, X):
            return np.array([0.1])

    tm = TrustModel(DummyModel(), ["amount", "price_zscore"], 0.05, 0.10)
    sess = generate_normal_session(agent, "S1", BASE_TIME + timedelta(days=1), rng, kind="evaluation")
    purchase = sess.requests[-1]

    try:
        authorize_purchase(purchase, sess, other, tm)
        raise AssertionError("Expected ValueError on mismatched agent")
    except ValueError:
        pass

    expired_sess = generate_normal_session(agent, "S2", BASE_TIME + timedelta(days=45), rng, kind="evaluation")
    pol = policy_check(expired_sess.requests[-1], expired_sess.mandate_snapshot, 1000.0, 1)
    assert pol.passed is False and any(v.rule == "mandate_expired" for v in pol.violations)
    print("   OK  agent/session mismatch rejected; expired mandate blocked by policy")


def test_live_connection():
    print("\n8. Live TEST MODE connection...")
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "live_check.sqlite3"
        result = rp.create_order_for_decision(
            fake_decision("ALLOW", amount=100.0, request_id=f"LIVE-{uuid.uuid4().hex[:8]}"),
            db_path=db,
        )
        if result.dry_run:
            print(f"   FAIL  keys not configured: {result.error}")
            sys.exit(1)
        if result.error:
            print(f"   FAIL  Razorpay error: {result.error}")
            sys.exit(1)
        print(f"   OK  order created: id={result.order_id}  status={result.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    test_gating_fail_closed()
    test_idempotency_and_payload_conflict()
    test_dry_run_does_not_pollute_store()
    test_review_approval_and_denial_immutability()
    test_concurrency_single_execution()
    test_retry_reconciliation_and_unresolved()
    test_authorization_integrity_and_expiry()

    print("\nALL 7 OFFLINE TEST SUITES PASSED")
    if args.live:
        test_live_connection()

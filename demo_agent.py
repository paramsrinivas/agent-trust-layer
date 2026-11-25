"""Thin buyer-agent simulator for the Agent Trust Layer demo."""

from __future__ import annotations

import argparse
import json
import pickle
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np

from authorization import TrustModel, audit_event, authorize_purchase
from generator import (
    BASE_TIME,
    estimate_agent_baseline,
    generate_agent,
    generate_normal_session,
)
from models import CATEGORIES, Product, PurchaseRequest, STEP_PROGRESS, Session, SessionStep
from razorpay_payment import create_order_for_decision

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"


def load_trust_model(model_variant: str = "A") -> TrustModel:
    path = DATA_DIR / "experiments" / f"trust_model_{model_variant}.pkl"
    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No persisted model found at {path}. Run "
            f"`python3 experiment.py --model {model_variant}` first."
        ) from exc

    return TrustModel(
        model=artifact["model"],
        feature_names=artifact["feature_names"],
        review_threshold=artifact["review_threshold"],
        block_threshold=artifact["block_threshold"],
    )


def load_population_baseline() -> dict:
    with open(DATA_DIR / "population_baseline.json") as f:
        return json.load(f)


def build_demo_agent(rng):
    agent = generate_agent(
        agent_id="DEMO-AGENT-01",
        persona_name="runner",
        rng=rng,
        split="test",
    )
    history = []
    t = BASE_TIME
    for i in range(10):
        session = generate_normal_session(agent, f"DEMO-HIST-{i:02d}", t, rng, kind="historical")
        history.append(session)
        t += timedelta(hours=float(rng.uniform(3, 10)))

    population = load_population_baseline()
    agent.baseline = estimate_agent_baseline(history, population)
    return agent, history


def make_normal_session(agent, rng, *, fresh_request_id: bool = False):
    sess = generate_normal_session(
        agent, "DEMO-NORMAL", BASE_TIME + timedelta(days=2), rng, kind="evaluation"
    )
    if fresh_request_id:
        sess.requests[-1].request_id = f"norm-{uuid.uuid4().hex[:8]}"
    return sess


def make_rogue_session(agent, rng, *, fresh_request_id: bool = False):
    """Deterministic behavioral drift scenario: rapid-fire velocity burst plus
    below-baseline purchase price, while remaining 100% compliant with mandate rules."""
    mandate = agent.mandate
    session_id = "DEMO-ROGUE"
    start_time = BASE_TIME + timedelta(days=2, hours=1)

    lead_in_types = ["browse", "compare", "similar"]
    tail_types = ["similar", "browse", "compare", "similar", "purchase"]
    step_types = lead_in_types + tail_types
    gaps = [260, 240, 1300, 8, 5, 3, 2]

    requests: list[PurchaseRequest] = []
    t = start_time
    for i, step_type in enumerate(step_types):
        if step_type == "purchase":
            amount = max(200.0, agent.baseline.price_mean - 1.3 * agent.baseline.price_std)
        else:
            amount = max(200.0, float(rng.normal(agent.baseline.price_mean, agent.baseline.price_std)))
        req_id = (
            f"rogue-{uuid.uuid4().hex[:8]}"
            if (fresh_request_id and step_type == "purchase")
            else f"{int(rng.integers(0, 2**32)):08x}"
        )
        req = PurchaseRequest(
            request_id=req_id,
            agent_id=agent.agent_id,
            session_id=session_id,
            timestamp=t.isoformat(),
            product=Product(
                product_id=f"{int(rng.integers(0, 2**32)):08x}",
                category=mandate.primary_category,
                amount=round(amount, 2),
                quantity=1,
            ),
            step=SessionStep(type=step_type, index=i, progress=STEP_PROGRESS[step_type]),
            gt_behavioral_anomaly=(i >= len(lead_in_types)),
        )
        requests.append(req)
        if i < len(gaps):
            t = t + timedelta(seconds=gaps[i])

    return Session(
        session_id=session_id,
        agent_id=agent.agent_id,
        started_at=start_time.isoformat(),
        mandate_snapshot=mandate,
        requests=requests,
        kind="evaluation",
        difficulty="hard",
        drift_type="velocity_compression_demo",
    )


def make_block_session(agent, rng, *, fresh_request_id: bool = False):
    """Deterministic hard-policy violation scenario (off-allowlist + over max_transaction)."""
    mandate = agent.mandate
    session_id = "DEMO-BLOCK"
    start_time = BASE_TIME + timedelta(days=2, hours=2)

    step_types = ["browse", "compare", "purchase"]
    gaps = [260, 240]

    off_allowlist = [c for c in CATEGORIES if c not in mandate.allowed_categories]
    target_category = off_allowlist[0] if off_allowlist else "electronics"

    requests = []
    t = start_time
    for i, step_type in enumerate(step_types):
        if step_type == "purchase":
            amount = round(mandate.max_transaction * 1.75, 2)
            category = target_category
        else:
            amount = round(
                max(200.0, float(rng.normal(agent.baseline.price_mean, agent.baseline.price_std))), 2
            )
            category = mandate.primary_category

        req_id = (
            f"blk-{uuid.uuid4().hex[:8]}"
            if (fresh_request_id and step_type == "purchase")
            else f"{int(rng.integers(0, 2**32)):08x}"
        )
        requests.append(
            PurchaseRequest(
                request_id=req_id,
                agent_id=agent.agent_id,
                session_id=session_id,
                timestamp=t.isoformat(),
                product=Product(
                    product_id=f"{int(rng.integers(0, 2**32)):08x}",
                    category=category,
                    amount=amount,
                    quantity=1,
                ),
                step=SessionStep(type=step_type, index=i, progress=STEP_PROGRESS[step_type]),
                gt_behavioral_anomaly=(step_type == "purchase"),
            )
        )
        if i < len(gaps):
            t = t + timedelta(seconds=gaps[i])

    return Session(
        session_id=session_id,
        agent_id=agent.agent_id,
        started_at=start_time.isoformat(),
        mandate_snapshot=mandate,
        requests=requests,
        kind="evaluation",
        difficulty="easy",
        drift_type="explicit_mandate_violation_demo",
    )


def authorize_session(session, agent, trust_model):
    purchase_requests = [r for r in session.requests if r.step.type == "purchase"]
    if not purchase_requests:
        raise RuntimeError("Demo session contains no purchase request.")
    return authorize_purchase(
        request=purchase_requests[-1],
        session=session,
        agent=agent,
        trust_model=trust_model,
    )


def print_decision(title, session, decision):
    features = decision.behavior.features
    razorpay_result = create_order_for_decision(decision)
    if razorpay_result.order_id:
        decision.razorpay_order_id = razorpay_result.order_id

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"Agent:       {decision.agent_id}")
    print(f"Session:     {decision.session_id}")
    print(f"Request:     {decision.request_id}")
    print(f"Amount:      ₹{features['amount']:,.2f}")

    print("\nPOLICY")
    print("-" * 72)
    if decision.policy.passed:
        print("✓ All explicit mandate rules passed")
    else:
        print("✗ Policy violation")
        for v in decision.policy.violations:
            print(f"  • {v.rule}: actual={v.actual}, limit={v.limit}")

    print("\nBEHAVIOR")
    print("-" * 72)
    print(f"Risk score: {decision.behavior.anomaly_score:.5f}")
    print("⚠ Behavioral anomaly detected" if decision.behavior.is_anomalous else "✓ Behavior consistent with learned baseline")
    for reason in decision.reasons:
        print(f"  • {reason}")

    print("\nPAYMENT AUTHORIZATION & RAZORPAY")
    print("-" * 72)
    print(f"Decision:    {decision.decision} (mode={decision.razorpay_mode})")
    if razorpay_result.dry_run:
        print(f"→ [DRY RUN] Razorpay keys not configured. Would create order for ₹{features['amount']:,.2f}")
    elif razorpay_result.status == "unresolved":
        print(f"→ [UNRESOLVED — RECONCILE MANUALLY] Network error: {razorpay_result.error}")
    elif not razorpay_result.called:
        print(f"→ Razorpay not contacted ({razorpay_result.status})")
    elif razorpay_result.error:
        print(f"→ Razorpay rejected request: {razorpay_result.error}")
    else:
        replay = " [idempotent replay]" if razorpay_result.idempotent_replay else ""
        print(f"→ Razorpay TEST MODE order created: {razorpay_result.order_id} ({razorpay_result.status}){replay}")

    print("\nAUDIT EVENT")
    print("-" * 72)
    print(json.dumps(audit_event(decision), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["normal", "rogue", "block", "both", "full"], default="both")
    parser.add_argument("--model", choices=["A", "B"], default="B")
    args = parser.parse_args()

    rng = np.random.default_rng(123)
    agent, _ = build_demo_agent(rng)
    trust_model = load_trust_model(args.model)

    if args.scenario in ("normal", "both", "full"):
        sess = make_normal_session(agent, rng)
        print_decision("SCENARIO 1 — NORMAL AI BUYER", sess, authorize_session(sess, agent, trust_model))
    if args.scenario in ("rogue", "both", "full"):
        sess = make_rogue_session(agent, rng)
        print_decision("SCENARIO 2 — BEHAVIORAL DRIFT", sess, authorize_session(sess, agent, trust_model))
    if args.scenario in ("block", "full"):
        sess = make_block_session(agent, rng)
        print_decision("SCENARIO 3 — MANDATE VIOLATION", sess, authorize_session(sess, agent, trust_model))


if __name__ == "__main__":
    main()

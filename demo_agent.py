"""
Thin buyer-agent simulator for the Agent Trust Layer demo.

This is intentionally NOT a second AI project.

Its only job is to emit realistic PurchaseRequest objects and demonstrate
that the same authorization engine handles both normal and abnormal behavior.
"""

from __future__ import annotations

import argparse
import json
import pickle
import uuid
from datetime import datetime, timedelta

import numpy as np

from models import (
    AgentProfile,
    PurchaseRequest,
    Product,
    SessionStep,
    Session,
    STEP_PROGRESS,
)
from generator import (
    BASE_TIME,
    generate_agent,
    generate_normal_session,
    estimate_agent_baseline,
)
from authorization import (
    TrustModel,
    authorize_purchase,
    audit_event,
    should_create_razorpay_order,
)
from razorpay_payment import create_order_for_decision


def load_trust_model(model_variant: str = "A") -> TrustModel:
    """Load the model + thresholds persisted by experiment.py.

    This does NOT re-fit IsolationForest. Re-fitting live was the source
    of a real bug: identical code, identical seed, identical data still
    produced a DIFFERENT fitted model (and a different ALLOW/REVIEW
    outcome for the same scripted scenario) across two machines, because
    IsolationForest's tree-building consumes its RNG differently across
    scikit-learn versions even with random_state fixed. Loading a single
    persisted artifact makes the demo's behavior identical everywhere it
    runs, regardless of local library versions.
    """
    path = f"data/experiments/trust_model_{model_variant}.pkl"
    try:
        with open(path, "rb") as f:
            artifact = pickle.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No persisted model found at {path}. Run "
            f"`python3 experiment.py --model {model_variant}` first -- "
            "it fits and saves the model this demo loads."
        ) from exc

    return TrustModel(
        model=artifact["model"],
        feature_names=artifact["feature_names"],
        review_threshold=artifact["review_threshold"],
        block_threshold=artifact["block_threshold"],
    )


def load_population_baseline() -> dict:
    """The ONE population baseline, computed once in generate_data.py from
    real historical sessions of the 80 train agents. Loaded here instead of
    being re-approximated from the flat feature CSV, so there's a single
    definition of 'population normal' shared by the whole pipeline."""
    with open("data/population_baseline.json") as f:
        return json.load(f)


def build_demo_agent(rng):
    """
    Create a demo agent and learn its baseline from historical behavior.
    """
    agent = generate_agent(
        agent_id="DEMO-AGENT-01",
        persona_name="runner",
        rng=rng,
        split="test",
    )

    history = []
    t = BASE_TIME
    for i in range(10):
        session = generate_normal_session(
            agent,
            f"DEMO-HIST-{i:02d}",
            t,
            rng,
            kind="historical",
        )
        history.append(session)
        t += timedelta(hours=float(rng.uniform(3, 10)))

    population = load_population_baseline()
    agent.baseline = estimate_agent_baseline(history, population)

    return agent, history


def make_normal_session(agent, rng):
    return generate_normal_session(
        agent,
        "DEMO-NORMAL",
        BASE_TIME + timedelta(days=2),
        rng,
        kind="evaluation",
    )


def make_rogue_session(agent, rng):
    """
    Behavioral drift scenario, built DETERMINISTICALLY rather than by
    perturbing a randomly-generated normal session.

    Why: the earlier version took generate_normal_session()'s output (whose
    total step count is randomly 3-8) and compressed its tail. That made the
    scenario's strength depend on how many steps the RNG happened to hand
    it that run -- with a short session, requests_last_5m is capped low no
    matter how aggressively the tail is compressed, so the "anomaly" could
    land right at the threshold boundary or even miss it entirely. A demo
    scenario needs a guaranteed, deterministic shape: a normal-paced lead-in
    (establishing this session looks ordinary at first) followed by an
    explicit rapid-fire tail (the actual behavioral change).

    Category, price, and quantity all stay within the mandate throughout --
    the ONLY thing that changes is request velocity.
    """
    mandate = agent.mandate
    session_id = "DEMO-ROGUE"
    start_time = BASE_TIME + timedelta(days=2, hours=1)

    lead_in_types = ["browse", "compare", "similar"]
    tail_types = ["similar", "browse", "compare", "similar", "purchase"]
    step_types = lead_in_types + tail_types
    # 2 normal-paced lead-in gaps, then a genuine quiet period (pushes the
    # lead-in outside both the 5- and 10-minute trailing windows), then a
    # rapid-fire tail. The quiet period is deliberate: it keeps
    # requests_in_last_10m at the purchase step UNDER the mandate's
    # explicit velocity_limit_per_10m (so the deterministic policy gate
    # passes this request) while requests_last_5m still spikes hard from
    # the tail alone -- that's the whole point of this scenario: nothing
    # here trips an explicit rule, only the behavioral model should flag it.
    gaps = [260, 240, 1300, 8, 5, 3, 2]

    requests: list[PurchaseRequest] = []
    t = start_time
    for i, step_type in enumerate(step_types):
        if step_type == "purchase":
            # Deliberately away from this agent's typical price (not
            # randomly sampled near the mean). Evidence from the actual
            # test set: velocity alone (requests_last_5m~4-5, gap~10-20s)
            # was NOT always enough to trigger detection -- the cases that
            # were reliably caught combined velocity with a price
            # meaningfully off the agent's normal range. Still fully
            # policy-compliant (well under max_transaction) -- narratively,
            # the agent grabbed a cheaper in-stock alternative while rushing.
            amount = max(200.0, agent.baseline.price_mean - 1.3 * agent.baseline.price_std)
        else:
            amount = max(200.0, float(rng.normal(agent.baseline.price_mean, agent.baseline.price_std)))
        req = PurchaseRequest(
            request_id=str(uuid.uuid4())[:8],
            agent_id=agent.agent_id,
            session_id=session_id,
            timestamp=t.isoformat(),
            product=Product(
                product_id=str(uuid.uuid4())[:8],
                category=mandate.primary_category,
                amount=round(amount, 2),
                quantity=1,
            ),
            step=SessionStep(type=step_type, index=i, progress=STEP_PROGRESS[step_type]),
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


def make_block_session(agent, rng):
    """
    Deterministic hard-policy violation scenario.

    The purchase intentionally violates the agent's explicit mandate so the
    authorization engine must BLOCK it. This demonstrates the deterministic
    safety boundary independently of the behavioral model.
    """
    mandate = agent.mandate
    session_id = "DEMO-BLOCK"
    start_time = BASE_TIME + timedelta(days=2, hours=2)

    step_types = ["browse", "compare", "purchase"]
    gaps = [260, 240]

    off_allowlist = [
        c for c in [
            "electronics", "phones", "computers", "casual_shoes",
            "sportswear", "apparel", "fitness_equipment", "running_shoes"
        ]
        if c not in mandate.allowed_categories
    ]
    target_category = off_allowlist[0] if off_allowlist else "electronics"

    requests = []
    t = start_time

    for i, step_type in enumerate(step_types):
        if step_type == "purchase":
            # Deliberately violate two explicit mandate rules.
            amount = round(mandate.max_transaction * 1.75, 2)
            category = target_category
        else:
            amount = round(
                max(
                    200.0,
                    float(rng.normal(
                        agent.baseline.price_mean,
                        agent.baseline.price_std
                    )),
                ),
                2,
            )
            category = mandate.primary_category

        req = PurchaseRequest(
            request_id=str(uuid.uuid4())[:8],
            agent_id=agent.agent_id,
            session_id=session_id,
            timestamp=t.isoformat(),
            product=Product(
                product_id=str(uuid.uuid4())[:8],
                category=category,
                amount=amount,
                quantity=1,
            ),
            step=SessionStep(
                type=step_type,
                index=i,
                progress=STEP_PROGRESS[step_type],
            ),
            gt_behavioral_anomaly=False,
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
        difficulty="easy",
        drift_type="explicit_mandate_violation_demo",
    )


def authorize_session(session, agent, trust_model):
    """
    Authorize only the final purchase request using the same authorization
    engine used by the rest of the Trust Layer.

    Razorpay order creation happens separately in print_decision() (CLI) or
    dashboard.py, and is guarded again by create_order_for_decision().
    """
    purchase_requests = [
        r for r in session.requests
        if r.step.type == "purchase"
    ]

    if not purchase_requests:
        raise RuntimeError("Demo session contains no purchase request.")

    purchase = purchase_requests[-1]

    return authorize_purchase(
        request=purchase,
        session=session,
        agent=agent,
        trust_model=trust_model,
    )


def print_decision(title, session, decision):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    features = decision.behavior.features

    print(f"Agent:       {decision.agent_id}")
    print(f"Session:     {decision.session_id}")
    print(f"Request:     {decision.request_id}")
    print(f"Amount:      ₹{features['amount']:,.2f}")
    print()

    print("POLICY")
    print("-" * 72)

    if decision.policy.passed:
        print("✓ All explicit mandate rules passed")
    else:
        print("✗ Policy violation")

        for violation in decision.policy.violations:
            print(
                f"  • {violation.rule}: "
                f"actual={violation.actual}, "
                f"limit={violation.limit}"
            )

    print()
    print("BEHAVIOR")
    print("-" * 72)

    print(
        f"Risk score: {decision.behavior.anomaly_score:.5f}"
    )

    if decision.behavior.is_anomalous:
        print("⚠ Behavioral anomaly detected")
    else:
        print("✓ Behavior consistent with learned baseline")

    print()

    for reason in decision.reasons:
        print(f"  • {reason}")

    print()
    print("PAYMENT AUTHORIZATION")
    print("-" * 72)

    if decision.decision == "ALLOW":
        print("                 ALLOW")
        print()
        print("→ This request is eligible to reach Razorpay.")
        mode_label = {
            "razorpay_test": "REAL Razorpay test-mode order created",
            "simulated": "SIMULATED order (no Razorpay credentials configured)",
            "simulated_after_error": "SIMULATED order (Razorpay API call failed, see audit log)",
        }.get(decision.razorpay_mode, decision.razorpay_mode)
        print(f"  order_id: {decision.razorpay_order_id}  [{mode_label}]")
    elif decision.decision == "REVIEW":
        print("                REVIEW")
        print()
        print("→ Payment must NOT be automatically executed.")
    else:
        print("                 BLOCK")
        print()
        print("→ Payment must NOT be created.")

    print()
    print("AUDIT EVENT")
    print("-" * 72)

    print(json.dumps(
        audit_event(decision),
        indent=2,
    ))

    print()
    print("RAZORPAY")
    print("-" * 72)

    razorpay_result = create_order_for_decision(decision)
    if razorpay_result.dry_run:
        print("→ [DRY RUN] Razorpay TEST MODE keys not configured.")
        print(f"  Would have created an order for ₹{features['amount']:,.2f}")
        print("  (see razorpay_payment.py setup instructions to make this real)")
    elif not razorpay_result.called:
        print(f"→ Razorpay not contacted ({razorpay_result.status})")
    elif razorpay_result.error:
        print(f"→ Razorpay returned an error: {razorpay_result.error}")
    else:
        print(f"→ Razorpay TEST MODE order created: {razorpay_result.order_id}")
        print(f"  status: {razorpay_result.status}")
        decision.razorpay_order_id = razorpay_result.order_id


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        choices=["normal", "rogue", "block", "both", "full"],
        default="both",
    )
    parser.add_argument(
        "--model",
        choices=["A", "B"],
        default="A",
        help="which persisted trust model to load (see experiment.py)",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(123)

    print()
    print("=" * 72)
    print("AGENT TRUST LAYER — LOCAL PAYMENT AUTHORIZATION DEMO")
    print("=" * 72)

    agent, history = build_demo_agent(rng)
    trust_model = load_trust_model(args.model)

    print()
    print("Agent profile")
    print("-" * 72)
    print(f"Agent:       {agent.agent_id}")
    print(f"Persona:     {agent.persona}")
    print(f"Intent:      {agent.mandate.intent}")
    print(
        f"Allowed:     {', '.join(agent.mandate.allowed_categories)}"
    )
    print(
        f"Max payment: ₹{agent.mandate.max_transaction:,.0f}"
    )
    print(
        f"Session cap: ₹{agent.mandate.session_budget:,.0f}"
    )
    print(
        f"Historical requests observed: "
        f"{agent.baseline.n_historical_requests}"
    )

    if args.scenario in ("normal", "both","full"):
        normal = make_normal_session(agent, rng)

        decision = authorize_session(
            normal,
            agent,
            trust_model,
        )

        print_decision(
            "SCENARIO 1 — NORMAL AI BUYER",
            normal,
            decision,
        )

    if args.scenario in ("rogue", "both", "full"):
        rogue = make_rogue_session(agent, rng)

        decision = authorize_session(
            rogue,
            agent,
            trust_model,
        )

        print_decision(
            "SCENARIO 2 — BEHAVIORAL DRIFT",
            rogue,
            decision,
        )


    if args.scenario in ("block", "full"):
        blocked = make_block_session(agent, rng)

        decision = authorize_session(
            blocked,
            agent,
            trust_model,
        )

        print_decision(
            "SCENARIO 3 — MANDATE VIOLATION",
            blocked,
            decision,
        )


if __name__ == "__main__":
    main()
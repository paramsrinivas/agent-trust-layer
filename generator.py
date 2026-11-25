"""Pure functions for agent/session generation, drift injection, baseline estimation,
policy enforcement, and shared train/serve feature extraction."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta

import numpy as np

from models import (
    AgentProfile,
    BehavioralBaseline,
    CATEGORIES,
    Mandate,
    PERSONAS,
    PolicyResult,
    PolicyViolation,
    Product,
    PurchaseRequest,
    STEP_PROGRESS,
    Session,
    SessionStep,
    TrueBehavior,
    category_distance,
)

BASE_TIME = datetime(2026, 8, 1, 9, 0, 0)
SHRINKAGE_K = 5


def _det_id(rng: np.random.Generator) -> str:
    return f"{int(rng.integers(0, 2**32)):08x}"


def generate_agent(
    agent_id: str,
    persona_name: str,
    rng: np.random.Generator,
    split: str,
) -> AgentProfile:
    template = PERSONAS[persona_name]

    price_mean = max(500.0, float(rng.normal(template["price_mean"], template["price_mean"] * 0.12)))
    price_std = max(150.0, float(rng.normal(template["price_std"], template["price_std"] * 0.15)))
    gap_mean = max(30.0, float(rng.normal(template["gap_mean_sec"], template["gap_mean_sec"] * 0.1)))
    gap_std = max(10.0, float(rng.normal(template["gap_std_sec"], template["gap_std_sec"] * 0.1)))

    true_behavior = TrueBehavior(
        price_mean=round(price_mean, 2),
        price_std=round(price_std, 2),
        gap_mean_sec=round(gap_mean, 1),
        gap_std_sec=round(gap_std, 1),
    )

    mandate = Mandate(
        intent=f"purchase {template['primary_category'].replace('_', ' ')}",
        primary_category=template["primary_category"],
        allowed_categories=list(template["allowed_categories"]),
        max_transaction=template["max_transaction"],
        session_budget=template["session_budget"],
        quantity_max=template["quantity_max"],
        velocity_limit_per_10m=template["velocity_limit_per_10m"],
        expires_at=(BASE_TIME + timedelta(days=30)).isoformat(),
    )

    return AgentProfile(
        agent_id=agent_id,
        persona=persona_name,
        split=split,
        mandate=mandate,
        true_behavior=true_behavior,
        baseline=None,
    )


def _sample_step_sequence(n_steps: int, rng: np.random.Generator) -> list[str]:
    middle_pool = ["browse", "compare", "similar"]
    middle = [str(rng.choice(middle_pool)) for _ in range(n_steps - 1)]
    if middle:
        middle[0] = "browse"
    return middle + ["purchase"]


def generate_normal_session(
    agent: AgentProfile,
    session_id: str,
    start_time: datetime,
    rng: np.random.Generator,
    kind: str = "historical",
) -> Session:
    mandate = agent.mandate
    tb = agent.true_behavior
    n_steps = int(rng.integers(3, 9))
    step_types = _sample_step_sequence(n_steps, rng)

    requests: list[PurchaseRequest] = []
    t = start_time

    for i, step_type in enumerate(step_types):
        if rng.random() < 0.7 or len(mandate.allowed_categories) == 1:
            category = mandate.primary_category
        else:
            category = str(rng.choice(mandate.allowed_categories))

        amount = max(200.0, float(rng.normal(tb.price_mean, tb.price_std)))
        if step_type == "purchase":
            amount = min(amount, mandate.max_transaction * 0.98)
        quantity = 1 if mandate.quantity_max == 1 else int(rng.choice([1, 1, 1, 2]))

        req = PurchaseRequest(
            request_id=_det_id(rng),
            agent_id=agent.agent_id,
            session_id=session_id,
            timestamp=t.isoformat(),
            product=Product(
                product_id=_det_id(rng),
                category=category,
                amount=round(amount, 2),
                quantity=quantity,
            ),
            step=SessionStep(type=step_type, index=i, progress=STEP_PROGRESS[step_type]),
            gt_behavioral_anomaly=False,
        )
        requests.append(req)

        if i < len(step_types) - 1:
            gap = max(15.0, float(rng.normal(tb.gap_mean_sec, tb.gap_std_sec)))
            t = t + timedelta(seconds=gap)

    return Session(
        session_id=session_id,
        agent_id=agent.agent_id,
        started_at=start_time.isoformat(),
        mandate_snapshot=mandate,
        requests=requests,
        kind=kind,
        difficulty="normal",
        drift_type=None,
    )


def inject_drift(
    session: Session,
    difficulty: str,
    rng: np.random.Generator,
    agent: AgentProfile,
) -> Session:
    new_session = copy.deepcopy(session)
    new_session.difficulty = difficulty
    mandate = agent.mandate
    reqs = new_session.requests

    if difficulty == "normal":
        return new_session

    if difficulty == "easy":
        new_session.drift_type = "category_and_price_spike"
        off_allowlist = [c for c in CATEGORIES if c not in mandate.allowed_categories]
        target_category = str(rng.choice(off_allowlist))
        last = reqs[-1]
        last.product.category = target_category
        last.product.amount = round(mandate.max_transaction * float(rng.uniform(3.0, 6.0)), 2)
        last.gt_behavioral_anomaly = True

    elif difficulty == "medium":
        new_session.drift_type = "category_drift"
        n_drift = min(4, len(reqs))
        tail_idx = list(range(len(reqs) - n_drift, len(reqs)))
        candidates = [c for c in mandate.allowed_categories if c != mandate.primary_category]
        target_category = max(
            candidates or [mandate.primary_category],
            key=lambda c: category_distance(mandate.primary_category, c),
        )
        for j, idx in enumerate(tail_idx):
            r = reqs[idx]
            progress = (j + 1) / n_drift
            if idx == tail_idx[-1] or rng.random() < progress:
                r.product.category = target_category
                r.gt_behavioral_anomaly = True

    elif difficulty == "hard":
        new_session.drift_type = "velocity_compression"
        max_in_window = max(3, mandate.velocity_limit_per_10m - 1)
        n_drift = min(max_in_window, len(reqs))
        tail_idx = list(range(len(reqs) - n_drift, len(reqs)))
        all_gaps = [45, 20, 10, 6]
        compressed_gaps = all_gaps[-(len(tail_idx) - 1):]

        start_t = datetime.fromisoformat(reqs[tail_idx[0]].timestamp)
        if tail_idx[0] > 0:
            prev_t = datetime.fromisoformat(reqs[tail_idx[0] - 1].timestamp)
            start_t = max(start_t, prev_t + timedelta(minutes=11))

        t = start_t
        for k, idx in enumerate(tail_idx):
            reqs[idx].timestamp = t.isoformat()
            reqs[idx].gt_behavioral_anomaly = k > 0
            if k < len(compressed_gaps):
                t = t + timedelta(seconds=compressed_gaps[k])

    return new_session


def estimate_population_baseline(
    train_agents: list[AgentProfile],
    historical_sessions: dict[str, list[Session]],
) -> dict:
    prices, gaps, rates = [], [], []
    within_stds = []
    for agent in train_agents:
        agent_prices = []
        for sess in historical_sessions[agent.agent_id]:
            reqs = sess.requests
            agent_prices.extend(r.product.amount for r in reqs)
            for i in range(1, len(reqs)):
                t0 = datetime.fromisoformat(reqs[i - 1].timestamp)
                t1 = datetime.fromisoformat(reqs[i].timestamp)
                gaps.append((t1 - t0).total_seconds())
            if len(reqs) >= 2:
                span_min = (
                    datetime.fromisoformat(reqs[-1].timestamp)
                    - datetime.fromisoformat(reqs[0].timestamp)
                ).total_seconds() / 60.0
                if span_min > 0:
                    rates.append(len(reqs) / span_min * 5.0)
        prices.extend(agent_prices)
        if len(agent_prices) >= 4:
            within_stds.append(float(np.std(agent_prices, ddof=1)))

    return dict(
        price_mean=float(np.mean(prices)),
        price_std=float(np.mean(within_stds)) if within_stds else (float(np.std(prices, ddof=1)) or 1.0),
        gap_mean_sec=float(np.mean(gaps)) if gaps else 200.0,
        requests_per_5m_mean=float(np.mean(rates)) if rates else 1.5,
    )


def estimate_agent_baseline(
    agent_historical_sessions: list[Session],
    population_baseline: dict,
    k: int = SHRINKAGE_K,
) -> BehavioralBaseline:
    prices, gaps, rates = [], [], []
    for sess in agent_historical_sessions:
        reqs = sess.requests
        prices.extend(r.product.amount for r in reqs)
        for i in range(1, len(reqs)):
            t0 = datetime.fromisoformat(reqs[i - 1].timestamp)
            t1 = datetime.fromisoformat(reqs[i].timestamp)
            gaps.append((t1 - t0).total_seconds())
        if len(reqs) >= 2:
            span_min = (
                datetime.fromisoformat(reqs[-1].timestamp)
                - datetime.fromisoformat(reqs[0].timestamp)
            ).total_seconds() / 60.0
            if span_min > 0:
                rates.append(len(reqs) / span_min * 5.0)

    n = len(prices)
    n_gaps = len(gaps)
    n_rates = len(rates)

    w_price = n / (n + k) if n else 0.0
    w_gap = n_gaps / (n_gaps + k) if n_gaps else 0.0
    w_rate = n_rates / (n_rates + k) if n_rates else 0.0

    agent_price_mean = float(np.mean(prices)) if n else population_baseline["price_mean"]
    agent_gap_mean = float(np.mean(gaps)) if n_gaps else population_baseline["gap_mean_sec"]
    agent_rate_mean = float(np.mean(rates)) if n_rates else population_baseline["requests_per_5m_mean"]

    shrunk_price_mean = w_price * agent_price_mean + (1 - w_price) * population_baseline["price_mean"]
    shrunk_gap_mean = w_gap * agent_gap_mean + (1 - w_gap) * population_baseline["gap_mean_sec"]
    shrunk_rate_mean = w_rate * agent_rate_mean + (1 - w_rate) * population_baseline["requests_per_5m_mean"]

    agent_price_std = float(np.std(prices, ddof=1)) if n >= 4 else None
    shrunk_price_std = (
        w_price * agent_price_std + (1 - w_price) * population_baseline["price_std"]
        if agent_price_std is not None
        else population_baseline["price_std"]
    )
    shrunk_price_std = max(shrunk_price_std, 1.0)

    return BehavioralBaseline(
        price_mean=round(shrunk_price_mean, 2),
        price_std=round(shrunk_price_std, 2),
        gap_mean_sec=round(shrunk_gap_mean, 1),
        requests_per_5m_mean=round(shrunk_rate_mean, 3),
        n_historical_requests=n,
    )


def requests_in_window(idx: int, session_requests: list[PurchaseRequest], minutes: float) -> int:
    t_this = datetime.fromisoformat(session_requests[idx].timestamp)
    window_start = t_this - timedelta(minutes=minutes)
    return sum(
        1 for r in session_requests[: idx + 1]
        if window_start <= datetime.fromisoformat(r.timestamp) <= t_this
    )


def policy_check(
    request: PurchaseRequest,
    mandate: Mandate,
    cumulative_spend_incl_this: float,
    requests_in_last_10m: int,
) -> PolicyResult:
    violations: list[PolicyViolation] = []
    is_purchase = request.step.type == "purchase"

    if datetime.fromisoformat(request.timestamp) > datetime.fromisoformat(mandate.expires_at):
        violations.append(PolicyViolation(
            rule="mandate_expired",
            actual=request.timestamp,
            limit=mandate.expires_at,
        ))

    if request.product.category not in mandate.allowed_categories:
        violations.append(PolicyViolation(
            rule="category_allowlist",
            actual=request.product.category,
            limit=",".join(mandate.allowed_categories),
        ))
    if requests_in_last_10m > mandate.velocity_limit_per_10m:
        violations.append(PolicyViolation(
            rule="velocity_limit_per_10m",
            actual=str(requests_in_last_10m),
            limit=str(mandate.velocity_limit_per_10m),
        ))

    if is_purchase:
        if request.product.amount > mandate.max_transaction:
            violations.append(PolicyViolation(
                rule="max_transaction",
                actual=str(request.product.amount),
                limit=str(mandate.max_transaction),
            ))
        if request.product.quantity > mandate.quantity_max:
            violations.append(PolicyViolation(
                rule="quantity_max",
                actual=str(request.product.quantity),
                limit=str(mandate.quantity_max),
            ))
        if cumulative_spend_incl_this > mandate.session_budget:
            violations.append(PolicyViolation(
                rule="session_budget",
                actual=str(round(cumulative_spend_incl_this, 2)),
                limit=str(mandate.session_budget),
            ))

    return PolicyResult(passed=len(violations) == 0, violations=violations)


def _recent_category_distance(
    idx: int,
    session_requests: list[PurchaseRequest],
    window: int = 3,
) -> float:
    if idx == 0:
        return 0.0
    start = max(0, idx - window)
    prior_categories = [r.product.category for r in session_requests[start:idx]]
    if not prior_categories:
        return 0.0
    current_category = session_requests[idx].product.category
    dists = [category_distance(current_category, c) for c in prior_categories]
    return round(float(np.mean(dists)), 3)


def extract_features(
    idx: int,
    session_requests: list[PurchaseRequest],
    mandate: Mandate,
    baseline: BehavioralBaseline,
) -> dict:
    request = session_requests[idx]
    prior_requests = session_requests[:idx]

    already_spent = sum(r.product.amount for r in prior_requests if r.step.type == "purchase")
    cumulative_spend = already_spent + (request.product.amount if request.step.type == "purchase" else 0.0)

    if prior_requests:
        t_prev = datetime.fromisoformat(prior_requests[-1].timestamp)
        t_this = datetime.fromisoformat(request.timestamp)
        time_since_previous_sec = max(0.0, (t_this - t_prev).total_seconds())
    else:
        time_since_previous_sec = baseline.gap_mean_sec

    requests_last_5m = requests_in_window(idx, session_requests, minutes=5)
    price_zscore = (request.product.amount - baseline.price_mean) / max(baseline.price_std, 1.0)
    cat_dist = category_distance(mandate.primary_category, request.product.category)
    spend_ratio = cumulative_spend / mandate.session_budget
    gap_ratio = time_since_previous_sec / max(baseline.gap_mean_sec, 1.0)
    rate_ratio = requests_last_5m / max(baseline.requests_per_5m_mean, 0.1)

    return dict(
        amount=request.product.amount,
        price_zscore=round(price_zscore, 4),
        category_distance=cat_dist,
        cumulative_spend_ratio=round(spend_ratio, 4),
        time_since_previous_sec=round(time_since_previous_sec, 1),
        requests_last_5m=requests_last_5m,
        gap_ratio=round(gap_ratio, 4),
        rate_ratio=round(rate_ratio, 4),
        quantity=request.product.quantity,
        session_progress=request.step.progress,
        recent_category_distance=_recent_category_distance(idx, session_requests),
    )

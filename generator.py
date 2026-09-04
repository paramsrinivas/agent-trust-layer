"""
Generator: pure functions that build agents, sessions, drift, baselines,
and features. Kept separate from I/O (see generate_data.py) so the same
functions can later replay a live demo request through the identical
pipeline used to build the training/eval data.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np

from models import (
    PERSONAS, CATEGORIES, category_distance, STEP_PROGRESS,
    Mandate, TrueBehavior, BehavioralBaseline, AgentProfile,
    Product, SessionStep, PurchaseRequest, Session,
    PolicyResult, PolicyViolation,
)

BASE_TIME = datetime(2026, 8, 1, 9, 0, 0)
SHRINKAGE_K = 5  # how many "pseudo-historical-requests" of population prior
                 # to blend in -- higher k = trust population more for
                 # agents with sparse history.


# ---------------------------------------------------------------------------
# 1. Agent generation
# ---------------------------------------------------------------------------

def generate_agent(agent_id: str, persona_name: str, rng: np.random.Generator,
                    split: str) -> AgentProfile:
    template = PERSONAS[persona_name]

    # per-agent jitter around the persona template so agents of the same
    # persona are similar but not identical
    price_mean = max(500.0, rng.normal(template["price_mean"], template["price_mean"] * 0.12))
    price_std = max(150.0, rng.normal(template["price_std"], template["price_std"] * 0.15))
    gap_mean = max(30.0, rng.normal(template["gap_mean_sec"], template["gap_mean_sec"] * 0.1))
    gap_std = max(10.0, rng.normal(template["gap_std_sec"], template["gap_std_sec"] * 0.1))

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
        expires_at=(BASE_TIME + timedelta(days=1)).isoformat(),
    )

    return AgentProfile(
        agent_id=agent_id,
        persona=persona_name,
        split=split,
        mandate=mandate,
        true_behavior=true_behavior,
        baseline=None,
    )


# ---------------------------------------------------------------------------
# 2. Normal session generation (state machine: browse -> compare -> similar -> purchase)
# ---------------------------------------------------------------------------

def _sample_step_sequence(n_steps: int, rng: np.random.Generator) -> list[str]:
    """Always ends in 'purchase'. Middle steps are browse/compare/similar."""
    middle_pool = ["browse", "compare", "similar"]
    middle = [rng.choice(middle_pool) for _ in range(n_steps - 1)]
    # first step is always a browse, to keep sessions readable
    if middle:
        middle[0] = "browse"
    return middle + ["purchase"]


def generate_normal_session(agent: AgentProfile, session_id: str,
                             start_time: datetime, rng: np.random.Generator,
                             kind: str = "historical") -> Session:
    mandate = agent.mandate
    tb = agent.true_behavior
    n_steps = int(rng.integers(3, 9))  # 3-8 requests
    step_types = _sample_step_sequence(n_steps, rng)

    requests: list[PurchaseRequest] = []
    t = start_time

    for i, step_type in enumerate(step_types):
        # category: mostly primary, sometimes another allowed category
        if rng.random() < 0.7 or len(mandate.allowed_categories) == 1:
            category = mandate.primary_category
        else:
            category = rng.choice(mandate.allowed_categories)

        amount = max(200.0, float(rng.normal(tb.price_mean, tb.price_std)))
        quantity = 1 if mandate.quantity_max == 1 else int(rng.choice([1, 1, 1, 2]))

        req = PurchaseRequest(
            request_id=str(uuid.uuid4())[:8],
            agent_id=agent.agent_id,
            session_id=session_id,
            timestamp=t.isoformat(),
            product=Product(
                product_id=str(uuid.uuid4())[:8],
                category=category,
                amount=round(amount, 2),
                quantity=quantity,
            ),
            step=SessionStep(type=step_type, index=i, progress=STEP_PROGRESS[step_type]),
            gt_behavioral_anomaly=False,
        )
        requests.append(req)

        if i < len(step_types) - 1:
            gap = max(5.0, float(rng.normal(tb.gap_mean_sec, tb.gap_std_sec)))
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


# ---------------------------------------------------------------------------
# 3. Drift injection -- takes a normal session and drifts its tail.
# ---------------------------------------------------------------------------

def inject_drift(session: Session, difficulty: str, rng: np.random.Generator,
                  agent: AgentProfile) -> Session:
    """Returns a NEW session (deep-ish copy) with the tail drifted.

    easy   -> single obvious violation: category leaves the allowlist AND
              amount blows past max_transaction. This SHOULD be caught by
              the deterministic policy gate alone.
    medium -> category gradually drifts toward a distant-but-still-ALLOWED
              category over the last few requests. Price/budget stay fine.
              Policy passes. Only a behavioral model can catch this.
    hard   -> category/price stay completely normal, but request velocity
              compresses drastically -- while staying under the mandate's
              explicit hard velocity limit. Policy passes. Only a
              behavioral model (aware of this agent's own normal pace)
              can catch this.
    """
    import copy
    new_session = copy.deepcopy(session)
    new_session.difficulty = difficulty
    mandate = agent.mandate
    reqs = new_session.requests

    if difficulty == "normal":
        return new_session

    if difficulty == "easy":
        new_session.drift_type = "category_and_price_spike"
        tail = reqs[-1:]  # last request only
        off_allowlist = [c for c in CATEGORIES if c not in mandate.allowed_categories]
        target_category = rng.choice(off_allowlist)
        for r in tail:
            r.product.category = target_category
            r.product.amount = round(mandate.max_transaction * float(rng.uniform(3.0, 6.0)), 2)
            r.gt_behavioral_anomaly = True

    elif difficulty == "medium":
        new_session.drift_type = "category_drift"
        # drift over the last min(4, len) requests, moving from primary
        # category toward the allowed category most distant from it
        n_drift = min(4, len(reqs))
        tail_idx = range(len(reqs) - n_drift, len(reqs))
        candidates = [c for c in mandate.allowed_categories if c != mandate.primary_category]
        if not candidates:
            candidates = [mandate.primary_category]
        target_category = max(
            candidates, key=lambda c: category_distance(mandate.primary_category, c)
        )
        for j, idx in enumerate(tail_idx):
            r = reqs[idx]
            # linearly increase probability of the target category as we
            # approach the end of the session
            progress = (j + 1) / n_drift
            if rng.random() < progress:
                r.product.category = target_category
            # keep price close to normal -- no price signal in this case
            r.gt_behavioral_anomaly = True

    elif difficulty == "hard":
        new_session.drift_type = "velocity_compression"
        n_drift = min(5, len(reqs))
        tail_idx = list(range(len(reqs) - n_drift, len(reqs)))
        # recompute timestamps for the tail with shrinking gaps, staying
        # under the mandate's hard velocity_limit_per_10m
        compressed_gaps = [90, 45, 20, 10, 6][:len(tail_idx) - 1]
        t = datetime.fromisoformat(reqs[tail_idx[0]].timestamp)
        for k, idx in enumerate(tail_idx):
            reqs[idx].timestamp = t.isoformat()
            reqs[idx].gt_behavioral_anomaly = True
            if k < len(compressed_gaps):
                t = t + timedelta(seconds=compressed_gaps[k])

    return new_session


# ---------------------------------------------------------------------------
# 4. Baseline estimation
# ---------------------------------------------------------------------------

def estimate_population_baseline(train_agents: list[AgentProfile],
                                  historical_sessions: dict[str, list[Session]]) -> dict:
    """Computed strictly from TRAIN agents' historical (normal) sessions.
    This is a real statistic over observed data -- not the hidden
    true_behavior parameters -- so there's no leakage of generator ground
    truth into what the "system" is allowed to know.
    """
    prices, gaps, rates = [], [], []
    for agent in train_agents:
        for sess in historical_sessions[agent.agent_id]:
            reqs = sess.requests
            prices.extend(r.product.amount for r in reqs)
            for i in range(1, len(reqs)):
                t0 = datetime.fromisoformat(reqs[i - 1].timestamp)
                t1 = datetime.fromisoformat(reqs[i].timestamp)
                gaps.append((t1 - t0).total_seconds())
            # approx requests per 5 min for this session
            if len(reqs) >= 2:
                span_min = (datetime.fromisoformat(reqs[-1].timestamp)
                            - datetime.fromisoformat(reqs[0].timestamp)).total_seconds() / 60.0
                if span_min > 0:
                    rates.append(len(reqs) / span_min * 5.0)

    return dict(
        price_mean=float(np.mean(prices)),
        price_std=float(np.std(prices)) or 1.0,
        gap_mean_sec=float(np.mean(gaps)) if gaps else 200.0,
        requests_per_5m_mean=float(np.mean(rates)) if rates else 1.5,
    )


def estimate_agent_baseline(agent_historical_sessions: list[Session],
                             population_baseline: dict,
                             k: int = SHRINKAGE_K) -> BehavioralBaseline:
    """Empirical-Bayes-style shrinkage: agents with more historical requests
    trust their own sample more; agents with sparse history fall back
    toward the population prior. This is what gives the system a sane
    answer for a cold-start / low-history agent instead of an overconfident,
    noisy personal estimate.
    """
    prices, gaps, rates = [], [], []
    for sess in agent_historical_sessions:
        reqs = sess.requests
        prices.extend(r.product.amount for r in reqs)
        for i in range(1, len(reqs)):
            t0 = datetime.fromisoformat(reqs[i - 1].timestamp)
            t1 = datetime.fromisoformat(reqs[i].timestamp)
            gaps.append((t1 - t0).total_seconds())
        if len(reqs) >= 2:
            span_min = (datetime.fromisoformat(reqs[-1].timestamp)
                        - datetime.fromisoformat(reqs[0].timestamp)).total_seconds() / 60.0
            if span_min > 0:
                rates.append(len(reqs) / span_min * 5.0)

    n = len(prices)
    agent_price_mean = float(np.mean(prices)) if n else population_baseline["price_mean"]
    agent_gap_mean = float(np.mean(gaps)) if gaps else population_baseline["gap_mean_sec"]
    agent_rate_mean = float(np.mean(rates)) if rates else population_baseline["requests_per_5m_mean"]

    w = n / (n + k) if n else 0.0
    shrunk_price_mean = w * agent_price_mean + (1 - w) * population_baseline["price_mean"]
    shrunk_gap_mean = w * agent_gap_mean + (1 - w) * population_baseline["gap_mean_sec"]
    shrunk_rate_mean = w * agent_rate_mean + (1 - w) * population_baseline["requests_per_5m_mean"]

    # std: use agent sample std if we have enough points, else population std
    agent_price_std = float(np.std(prices)) if n >= 4 else None
    shrunk_price_std = (
        w * agent_price_std + (1 - w) * population_baseline["price_std"]
        if agent_price_std is not None else population_baseline["price_std"]
    )
    shrunk_price_std = max(shrunk_price_std, 1.0)

    return BehavioralBaseline(
        price_mean=round(shrunk_price_mean, 2),
        price_std=round(shrunk_price_std, 2),
        gap_mean_sec=round(shrunk_gap_mean, 1),
        requests_per_5m_mean=round(shrunk_rate_mean, 3),
        n_historical_requests=n,
    )


# ---------------------------------------------------------------------------
# 5. Policy check (deterministic, no ML)
# ---------------------------------------------------------------------------

def requests_in_window(idx: int, session_requests: list[PurchaseRequest], minutes: float) -> int:
    """Count requests (including this one) within the trailing window,
    looking only at requests up to and including idx (no lookahead)."""
    t_this = datetime.fromisoformat(session_requests[idx].timestamp)
    window_start = t_this - timedelta(minutes=minutes)
    return sum(
        1 for r in session_requests[: idx + 1]
        if window_start <= datetime.fromisoformat(r.timestamp) <= t_this
    )


def policy_check(request: PurchaseRequest, mandate: Mandate,
                  cumulative_spend_incl_this: float,
                  requests_in_last_10m: int) -> PolicyResult:
    """Only 'purchase' steps move money. browse/compare/similar steps are
    the agent looking around -- they carry an indicative product/price
    (useful for behavioral signal: what is the agent looking at, at what
    price point, how fast) but they never spend anything and are never
    checked against transaction/budget/quantity limits. Only a 'purchase'
    step can violate max_transaction, session_budget, or quantity_max, and
    only a 'purchase' step ever reaches Razorpay.

    category_allowlist and velocity are checked on every step, since an
    agent browsing wildly off-mandate categories or hammering the merchant
    with requests is meaningful regardless of whether money is about to
    move.
    """
    violations: list[PolicyViolation] = []
    is_purchase = request.step.type == "purchase"

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


# ---------------------------------------------------------------------------
# 6. Feature extraction -- the SAME function used for training data and for
# a live demo request, so there's no train/serve skew.
# ---------------------------------------------------------------------------

def _recent_category_distance(idx: int, session_requests: list[PurchaseRequest],
                               window: int = 3) -> float:
    """Average category_distance between the current request's category and
    the categories of the last `window` requests in this session.

    This is deliberately different from `category_distance` (which measures
    distance from the mandate's fixed primary category). This one measures
    drift relative to the agent's own recent trajectory -- so a gradual
    category walk (footwear -> sportswear -> fitness_equipment over several
    steps) shows up here even on a step where the single-step jump from the
    immediately preceding request looks small.
    """
    if idx == 0:
        return 0.0
    start = max(0, idx - window)
    prior_categories = [r.product.category for r in session_requests[start:idx]]
    if not prior_categories:
        return 0.0
    current_category = session_requests[idx].product.category
    dists = [category_distance(current_category, c) for c in prior_categories]
    return round(float(np.mean(dists)), 3)


def extract_features(idx: int, session_requests: list[PurchaseRequest],
                      mandate: Mandate, baseline: BehavioralBaseline) -> dict:
    """idx = position of the request being scored within session_requests
    (its own session, in order). This avoids relying on dataclass equality
    to locate the request, which is fragile once fields are mutated."""
    request = session_requests[idx]
    requests_so_far = session_requests[: idx + 1]
    prior_requests = session_requests[:idx]

    # only 'purchase' steps actually spend money -- browsing/comparing
    # never adds to cumulative spend, even though those steps carry an
    # indicative price for behavioral signal.
    already_spent = sum(r.product.amount for r in prior_requests if r.step.type == "purchase")
    cumulative_spend = already_spent + (request.product.amount if request.step.type == "purchase" else 0.0)

    if prior_requests:
        t_prev = datetime.fromisoformat(prior_requests[-1].timestamp)
        t_this = datetime.fromisoformat(request.timestamp)
        time_since_previous_sec = (t_this - t_prev).total_seconds()
    else:
        time_since_previous_sec = baseline.gap_mean_sec  # neutral fallback for first request

    requests_last_5m = requests_in_window(idx, session_requests, minutes=5)

    price_zscore = (request.product.amount - baseline.price_mean) / max(baseline.price_std, 1.0)
    cat_dist = category_distance(mandate.primary_category, request.product.category)
    spend_ratio = cumulative_spend / mandate.session_budget

    return dict(
        amount=request.product.amount,
        price_zscore=round(price_zscore, 4),
        category_distance=cat_dist,
        cumulative_spend_ratio=round(spend_ratio, 4),
        time_since_previous_sec=round(time_since_previous_sec, 1),
        requests_last_5m=requests_last_5m,
        quantity=request.product.quantity,
        session_progress=request.step.progress,  # stored, excluded from Model A
        recent_category_distance=_recent_category_distance(idx, session_requests),
    )

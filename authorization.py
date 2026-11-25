"""Payment Authorization Decision engine (hardened, policy-only BLOCK by default)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from generator import extract_features, policy_check, requests_in_window
from models import (
    AgentProfile,
    BehaviorResult,
    PaymentAuthorizationDecision,
    PurchaseRequest,
    Session,
)


class TrustModel:
    """Validated adapter around a fitted sklearn anomaly model."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        review_threshold: float,
        block_threshold: float,
    ):
        if not feature_names:
            raise ValueError("feature_names must be non-empty.")
        review = float(review_threshold)
        block = float(block_threshold)
        if not (math.isfinite(review) and math.isfinite(block)):
            raise ValueError("Thresholds must be finite numbers.")
        if review > block:
            raise ValueError(
                f"review_threshold ({review}) cannot exceed block_threshold ({block})."
            )
        self.model = model
        self.feature_names = list(feature_names)
        self.review_threshold = review
        self.block_threshold = block

    def score(self, features: dict[str, float]) -> float:
        row = []
        for name in self.feature_names:
            if name not in features:
                raise KeyError(f"Missing required feature: {name!r}")
            val = float(features[name])
            if not math.isfinite(val):
                raise ValueError(f"Feature {name!r} must be finite, got {val!r}")
            row.append(val)
        vector = pd.DataFrame([row], columns=self.feature_names)
        return float(-self.model.decision_function(vector)[0])


def behavioral_reasons(
    features: dict[str, float],
    baseline,
    risk: float,
    review_threshold: float,
    block_threshold: float,
) -> list[str]:
    reasons: list[str] = []

    gap = features["time_since_previous_sec"]
    if gap < max(30.0, baseline.gap_mean_sec * 0.35):
        reasons.append(
            f"Request arrived unusually fast ({gap:.0f}s vs "
            f"{baseline.gap_mean_sec:.0f}s historical gap)"
        )

    if features["requests_last_5m"] >= 4:
        reasons.append(
            f"High request velocity ({int(features['requests_last_5m'])} "
            "requests in the last 5 minutes)"
        )

    if abs(features["price_zscore"]) >= 3.0:
        direction = "above" if features["price_zscore"] > 0 else "below"
        reasons.append(
            f"Purchase price is {direction} the agent's normal range "
            f"(z={features['price_zscore']:+.1f})"
        )

    if features["category_distance"] >= 0.67:
        reasons.append("Product category is far from the agent's primary category")

    if features.get("recent_category_distance", 0.0) >= 0.50:
        reasons.append(
            "Product category drifted away from the agent's recent session trajectory "
            f"(recent_distance={features['recent_category_distance']:.2f})"
        )

    if features["cumulative_spend_ratio"] >= 0.80:
        reasons.append(
            f"Session spend is at {features['cumulative_spend_ratio']:.0%} "
            "of the authorized budget"
        )

    if not reasons:
        if risk >= block_threshold:
            reasons.append("Behavioral risk exceeds the calibrated authorization limit")
        elif risk >= review_threshold:
            reasons.append("Behavior differs from the agent's learned behavioral baseline")

    return reasons


def authorize_purchase(
    request: PurchaseRequest,
    session: Session,
    agent: AgentProfile,
    trust_model: TrustModel,
    *,
    razorpay_mode: str = "test",
    ml_can_block: bool = False,
) -> PaymentAuthorizationDecision:
    if request.step.type != "purchase":
        raise ValueError(
            "authorize_purchase() accepts purchase steps only. "
            "Browsing/comparison requests cannot create payments."
        )
    if agent.baseline is None:
        raise ValueError(f"Agent {agent.agent_id!r} has no estimated BehavioralBaseline.")
    if request.agent_id != agent.agent_id or session.agent_id != agent.agent_id:
        raise ValueError("Agent identity mismatch between request, session, and agent profile.")
    if request.session_id != session.session_id:
        raise ValueError("Session ID mismatch between request and session.")

    matches = [i for i, r in enumerate(session.requests) if r.request_id == request.request_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected request_id {request.request_id!r} exactly once in session, found {len(matches)}."
        )
    idx = matches[0]
    canonical_request = session.requests[idx]
    if canonical_request != request:
        raise ValueError("Supplied request payload does not match canonical session request.")

    mandate = session.mandate_snapshot
    cumulative_spend = sum(
        r.product.amount for r in session.requests[: idx + 1] if r.step.type == "purchase"
    )
    requests_10m = requests_in_window(idx, session.requests, 10)

    policy = policy_check(
        request=canonical_request,
        mandate=mandate,
        cumulative_spend_incl_this=cumulative_spend,
        requests_in_last_10m=requests_10m,
    )
    features = extract_features(
        idx=idx,
        session_requests=session.requests,
        mandate=mandate,
        baseline=agent.baseline,
    )
    risk = trust_model.score(features)

    if not policy.passed:
        decision = "BLOCK"
        reasons = [f"Policy violation: {v.rule}" for v in policy.violations]
    elif ml_can_block and risk >= trust_model.block_threshold:
        decision = "BLOCK"
        reasons = behavioral_reasons(
            features, agent.baseline, risk,
            trust_model.review_threshold, trust_model.block_threshold,
        )
    elif risk >= trust_model.review_threshold:
        decision = "REVIEW"
        reasons = behavioral_reasons(
            features, agent.baseline, risk,
            trust_model.review_threshold, trust_model.block_threshold,
        )
    else:
        decision = "ALLOW"
        reasons = ["Request is within policy and consistent with agent behavior"]

    behavior = BehaviorResult(
        anomaly_score=risk,
        is_anomalous=(risk >= trust_model.review_threshold),
        features=features,
    )

    return PaymentAuthorizationDecision(
        request_id=canonical_request.request_id,
        agent_id=canonical_request.agent_id,
        session_id=canonical_request.session_id,
        decision=decision,
        policy=policy,
        behavior=behavior,
        reasons=reasons,
        razorpay_mode=razorpay_mode,
        razorpay_order_id=None,
    )


def audit_event(decision: PaymentAuthorizationDecision) -> dict[str, Any]:
    return {
        "request_id": decision.request_id,
        "agent_id": decision.agent_id,
        "session_id": decision.session_id,
        "decision": decision.decision,
        "razorpay_mode": decision.razorpay_mode,
        "razorpay_order_id": decision.razorpay_order_id,
        "risk": round(decision.behavior.anomaly_score, 5),
        "is_anomalous": decision.behavior.is_anomalous,
        "policy_passed": decision.policy.passed,
        "policy_violations": [v.rule for v in decision.policy.violations],
        "reasons": list(decision.reasons),
        "amount": decision.behavior.features.get("amount"),
        "category_distance": decision.behavior.features.get("category_distance"),
        "recent_category_distance": decision.behavior.features.get("recent_category_distance"),
        "price_zscore": decision.behavior.features.get("price_zscore"),
        "requests_last_5m": decision.behavior.features.get("requests_last_5m"),
        "time_since_previous_sec": decision.behavior.features.get("time_since_previous_sec"),
    }

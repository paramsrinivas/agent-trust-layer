"""
Payment Authorization Decision engine.

This is the product boundary between a buyer agent and Razorpay.

Design:
    1. Hard policy violations -> BLOCK.
    2. Policy passes + very high behavioral risk -> BLOCK.
    3. Policy passes + elevated behavioral risk -> REVIEW.
    4. Policy passes + low behavioral risk -> ALLOW.

The engine does NOT invent feature explanations. Reasons come from
deterministic policy evidence plus named behavioral signals derived from
the actual feature values.

Usage from the future live demo:
    decision = authorize_purchase(...)
    if decision.decision == "ALLOW":
        # only here do we create a Razorpay test-mode order/payment
        ...
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from models import (
    AgentProfile,
    BehaviorResult,
    PaymentAuthorizationDecision,
    PolicyResult,
    PurchaseRequest,
    Session,
)
from generator import extract_features, policy_check, requests_in_window


class TrustModel:
    """Small adapter around a fitted sklearn anomaly model."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        review_threshold: float,
        block_threshold: float,
    ):
        self.model = model
        self.feature_names = feature_names
        self.review_threshold = float(review_threshold)
        self.block_threshold = float(block_threshold)

    def score(self, features: dict[str, float]) -> float:
        vector = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names,
        )

        return float(-self.model.decision_function(vector)[0])


def behavioral_reasons(
    features: dict[str, float],
    baseline,
    mandate,
    risk: float,
    review_threshold: float,
    block_threshold: float,
) -> list[str]:
    """
    Human-readable evidence derived from real features.

    These are explanations, not fake SHAP values. We only show a reason when
    a concrete observed feature is materially unusual.
    """
    reasons = []

    # Velocity evidence.
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

    # Price evidence.
    if abs(features["price_zscore"]) >= 3.0:
        direction = "above" if features["price_zscore"] > 0 else "below"
        reasons.append(
            f"Purchase price is {direction} the agent's normal range "
            f"(z={features['price_zscore']:+.1f})"
        )

    # Category evidence.
    if features["category_distance"] >= 0.67:
        reasons.append(
            "Product category is far from the agent's primary category"
        )

    # Session spend evidence.
    if features["cumulative_spend_ratio"] >= 0.80:
        reasons.append(
            f"Session spend is at {features['cumulative_spend_ratio']:.0%} "
            "of the authorized budget"
        )

    # Fallback so REVIEW/BLOCK always has a comprehensible reason.
    if not reasons:
        if risk >= block_threshold:
            reasons.append(
                "Behavioral risk exceeds the calibrated authorization limit"
            )
        elif risk >= review_threshold:
            reasons.append(
                "Behavior differs from the agent's learned behavioral baseline"
            )

    return reasons


def authorize_purchase(
    request: PurchaseRequest,
    session: Session,
    agent: AgentProfile,
    trust_model: TrustModel,
) -> PaymentAuthorizationDecision:
    """
    Authorize exactly one purchase request.

    Only purchase steps can reach a payment authorization decision.
    Non-purchase requests are rejected from this function rather than
    accidentally creating a payment.
    """
    if request.step.type != "purchase":
        raise ValueError(
            "authorize_purchase() accepts purchase steps only. "
            "Browsing/comparison requests cannot create payments."
        )

    try:
        idx = next(
            i for i, r in enumerate(session.requests)
            if r.request_id == request.request_id
        )
    except StopIteration as exc:
        raise ValueError("Request is not part of the supplied session") from exc

    # Use the immutable mandate snapshot belonging to this session.
    mandate = session.mandate_snapshot

    cumulative_spend = sum(
        r.product.amount
        for r in session.requests[:idx + 1]
        if r.step.type == "purchase"
    )

    requests_10m = requests_in_window(
        idx,
        session.requests,
        10,
    )

    policy = policy_check(
        request=request,
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

    # Policy is authoritative for explicit authorization boundaries.
    # The model adds behavioral evidence; it does not override a hard rule.
    if not policy.passed:
        decision = "BLOCK"
        reasons = [
            f"Policy violation: {violation.rule}"
            for violation in policy.violations
        ]

    elif risk >= trust_model.block_threshold:
        decision = "BLOCK"
        reasons = behavioral_reasons(
            features,
            agent.baseline,
            mandate,
            risk,
            trust_model.review_threshold,
            trust_model.block_threshold,
        )

    elif risk >= trust_model.review_threshold:
        decision = "REVIEW"
        reasons = behavioral_reasons(
            features,
            agent.baseline,
            mandate,
            risk,
            trust_model.review_threshold,
            trust_model.block_threshold,
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
        request_id=request.request_id,
        agent_id=request.agent_id,
        session_id=request.session_id,
        decision=decision,
        policy=policy,
        behavior=behavior,
        reasons=reasons,
        # Authorization is payment-aware now; actual order creation happens
        # later and only after the ALLOW gate passes.
        razorpay_mode="test",
        razorpay_order_id=None,
    )


def should_create_razorpay_order(decision: PaymentAuthorizationDecision) -> bool:
    """
    The single payment-side gate.

    REVIEW and BLOCK must never automatically create a payment order.
    """
    return decision.decision == "ALLOW"


def audit_event(decision: PaymentAuthorizationDecision) -> dict:
    """Convert an authorization decision into a dashboard/audit-friendly event."""
    return {
        "request_id": decision.request_id,
        "agent_id": decision.agent_id,
        "session_id": decision.session_id,
        "decision": decision.decision,
        "razorpay_mode": decision.razorpay_mode,
        "razorpay_order_id": decision.razorpay_order_id,
        "risk": round(decision.behavior.anomaly_score, 5),
        "policy_passed": decision.policy.passed,
        "policy_violations": [
            v.rule for v in decision.policy.violations
        ],
        "reasons": decision.reasons,
        "amount": decision.behavior.features.get("amount"),
        "category_distance": decision.behavior.features.get(
            "category_distance"
        ),
        "price_zscore": decision.behavior.features.get("price_zscore"),
        "requests_last_5m": decision.behavior.features.get(
            "requests_last_5m"
        ),
        "time_since_previous_sec": decision.behavior.features.get(
            "time_since_previous_sec"
        ),
    }

"""Core data structures for the Agent Trust Layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
RazorpayMode = Literal["test", "live"]
Difficulty = Literal["normal", "easy", "medium", "hard"]
SessionKind = Literal["historical", "evaluation"]
StepType = Literal["browse", "compare", "similar", "purchase"]
Split = Literal["train", "validation", "test"]

STEP_PROGRESS = {"browse": 0.00, "compare": 0.33, "similar": 0.66, "purchase": 1.00}

CATEGORY_PATHS: dict[str, list[str]] = {
    "running_shoes":     ["running_shoes", "footwear", "fashion", "root"],
    "casual_shoes":      ["casual_shoes", "footwear", "fashion", "root"],
    "sportswear":        ["sportswear", "apparel", "fashion", "root"],
    "apparel":           ["apparel", "fashion", "root"],
    "fitness_equipment": ["fitness_equipment", "sports", "root"],
    "electronics":       ["electronics", "tech", "root"],
    "phones":            ["phones", "electronics", "tech", "root"],
    "computers":         ["computers", "electronics", "tech", "root"],
}

CATEGORIES = list(CATEGORY_PATHS.keys())


def _distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    path_a = CATEGORY_PATHS[a]
    path_b = CATEGORY_PATHS[b]
    set_b = set(path_b)
    common_depth = len(path_a) - 1
    for i, node in enumerate(path_a):
        if node in set_b:
            common_depth = i
            break
    steps_a = common_depth
    steps_b = path_b.index(path_a[common_depth])
    return round(min((steps_a + steps_b) / 6.0, 1.0), 3)


CATEGORY_DISTANCE_MATRIX: dict[tuple[str, str], float] = {
    (a, b): _distance(a, b) for a in CATEGORIES for b in CATEGORIES
}


def category_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return CATEGORY_DISTANCE_MATRIX.get((a, b), 1.0)


PERSONAS = {
    "student": dict(
        primary_category="casual_shoes",
        allowed_categories=["casual_shoes", "sportswear", "apparel"],
        price_mean=2200.0, price_std=500.0,
        gap_mean_sec=300.0, gap_std_sec=90.0,
        max_transaction=4000.0, session_budget=6000.0,
        quantity_max=1, velocity_limit_per_10m=6,
    ),
    "runner": dict(
        primary_category="running_shoes",
        allowed_categories=["running_shoes", "sportswear", "fitness_equipment"],
        price_mean=6000.0, price_std=900.0,
        gap_mean_sec=240.0, gap_std_sec=80.0,
        max_transaction=8000.0, session_budget=10000.0,
        quantity_max=1, velocity_limit_per_10m=6,
    ),
    "professional": dict(
        primary_category="apparel",
        allowed_categories=["apparel", "casual_shoes", "electronics"],
        price_mean=8500.0, price_std=1800.0,
        gap_mean_sec=200.0, gap_std_sec=70.0,
        max_transaction=12000.0, session_budget=15000.0,
        quantity_max=2, velocity_limit_per_10m=6,
    ),
    "traveler": dict(
        primary_category="apparel",
        allowed_categories=["apparel", "casual_shoes", "fitness_equipment"],
        price_mean=9000.0, price_std=3000.0,
        gap_mean_sec=260.0, gap_std_sec=100.0,
        max_transaction=15000.0, session_budget=18000.0,
        quantity_max=2, velocity_limit_per_10m=6,
    ),
}


@dataclass
class Mandate:
    intent: str
    primary_category: str
    allowed_categories: list[str]
    max_transaction: float
    session_budget: float
    quantity_max: int
    velocity_limit_per_10m: int
    expires_at: str


@dataclass
class TrueBehavior:
    price_mean: float
    price_std: float
    gap_mean_sec: float
    gap_std_sec: float


@dataclass
class BehavioralBaseline:
    price_mean: float
    price_std: float
    gap_mean_sec: float
    requests_per_5m_mean: float
    n_historical_requests: int


@dataclass
class AgentProfile:
    agent_id: str
    persona: str
    split: Split
    mandate: Mandate
    true_behavior: TrueBehavior
    baseline: Optional[BehavioralBaseline] = None


@dataclass
class Product:
    product_id: str
    category: str
    amount: float
    quantity: int


@dataclass
class SessionStep:
    type: StepType
    index: int
    progress: float


@dataclass
class PurchaseRequest:
    request_id: str
    agent_id: str
    session_id: str
    timestamp: str
    product: Product
    step: SessionStep
    gt_behavioral_anomaly: bool = False


@dataclass
class Session:
    session_id: str
    agent_id: str
    started_at: str
    mandate_snapshot: Mandate
    requests: list[PurchaseRequest]
    kind: SessionKind
    difficulty: Difficulty = "normal"
    drift_type: Optional[str] = None


@dataclass
class PolicyViolation:
    rule: str
    actual: str
    limit: str


@dataclass
class PolicyResult:
    passed: bool
    violations: list[PolicyViolation] = field(default_factory=list)


@dataclass
class BehaviorResult:
    anomaly_score: float
    is_anomalous: bool
    features: dict[str, float]


@dataclass
class PaymentAuthorizationDecision:
    request_id: str
    agent_id: str
    session_id: str
    decision: Decision
    policy: PolicyResult
    behavior: BehaviorResult
    reasons: list[str]
    razorpay_mode: RazorpayMode = "test"
    razorpay_order_id: Optional[str] = None


def to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj	

"""
Core data structures for the Agent Trust Layer synthetic data generator.

Design principles (locked from the architecture discussion):
- Mandate = authorization (what's permitted). Deterministic, never learned.
- TrueBehavior = hidden generative ground truth used ONLY by the generator
  to sample realistic requests. The estimator never sees this directly --
  it must infer a baseline from historical sessions, same as a real system
  would have to.
- BehavioralBaseline = ESTIMATED from historical sessions (personal estimate
  shrunk toward a population prior). This is what the feature extractor and
  model actually consume.
- PolicyResult / BehaviorResult / PaymentAuthorizationDecision keep the same
  separation of concerns used throughout: policy is deterministic, behavior
  is probabilistic evidence, and the decision engine combines both.

No external validation library (pydantic) is available in this environment
(no network access to install it), so we use plain dataclasses instead.
They're a fine substitute for a hackathon prototype -- if you want pydantic's
validation later, these classes translate 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from datetime import datetime, timedelta

Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
RazorpayMode = Literal["test", "live"]
Difficulty = Literal["normal", "easy", "medium", "hard"]
SessionKind = Literal["historical", "evaluation"]
StepType = Literal["browse", "compare", "similar", "purchase"]
Split = Literal["train", "validation", "test"]

STEP_PROGRESS = {"browse": 0.00, "compare": 0.33, "similar": 0.66, "purchase": 1.00}


# ---------------------------------------------------------------------------
# Category taxonomy (8 leaf categories, small fixed tree, deterministic distances)
# ---------------------------------------------------------------------------

# path from leaf to root, hand-authored -- NOT learned, NOT an embedding.
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
    """Deterministic taxonomy distance in [0, 1] based on tree path overlap.

    same category           -> 0.0
    shares immediate parent  -> ~0.2-0.3
    shares only root branch  -> ~0.6-0.8
    unrelated branches       -> 1.0

    This is a fixed rule over a fixed tree -- fully inspectable, not a
    learned embedding similarity. Computed once into a lookup matrix below.
    """
    if a == b:
        return 0.0
    path_a = CATEGORY_PATHS[a]
    path_b = CATEGORY_PATHS[b]
    set_b = set(path_b)
    # first shared ancestor walking up from the leaf
    common_depth = 0
    for i, node in enumerate(path_a):
        if node in set_b:
            common_depth = i
            break
    else:
        common_depth = len(path_a) - 1  # only 'root' shared, worst case

    steps_a = common_depth
    steps_b = path_b.index(path_a[common_depth])
    total_steps = steps_a + steps_b
    # normalize: max plausible distance in this tree is about 6 steps
    return round(min(total_steps / 6.0, 1.0), 3)


# Precomputed once -- this is the actual "dictionary" the system consumes at
# runtime. category_distance() below is just a lookup, no formula at call time.
CATEGORY_DISTANCE_MATRIX: dict[tuple[str, str], float] = {
    (a, b): _distance(a, b) for a in CATEGORIES for b in CATEGORIES
}


def category_distance(a: str, b: str) -> float:
    return CATEGORY_DISTANCE_MATRIX[(a, b)]


# ---------------------------------------------------------------------------
# Personas -- template generative parameters. Each agent samples its OWN
# true_behavior with jitter around its persona template, so agents of the
# same persona are similar but not identical.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Mandate:
    intent: str
    primary_category: str
    allowed_categories: list[str]
    max_transaction: float
    session_budget: float
    quantity_max: int
    velocity_limit_per_10m: int
    expires_at: str  # ISO timestamp


@dataclass
class TrueBehavior:
    """Hidden generative ground truth. Used by the generator only.
    The estimator/model must NEVER read this directly -- it has to infer
    a baseline from historical sessions, exactly like a real system would.
    """
    price_mean: float
    price_std: float
    gap_mean_sec: float
    gap_std_sec: float


@dataclass
class BehavioralBaseline:
    """ESTIMATED baseline -- personal estimate shrunk toward population prior.
    This is what extract_features() actually consumes.
    """
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
    true_behavior: TrueBehavior  # generator-only ground truth
    baseline: Optional[BehavioralBaseline] = None  # filled in after estimation


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
    # ground truth tags -- known only because WE generated the drift.
    # Used for evaluation only, never fed to the model as a feature.
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
    # Payment environment selected for the authorization pipeline.
    # The hackathon integration uses Test Mode; Live Mode is explicit
    # so the decision contract does not need to change later.
    razorpay_mode: RazorpayMode = "test"
    razorpay_order_id: Optional[str] = None


def to_dict(obj):
    """Recursively convert dataclasses to plain dicts for JSON export."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    return obj

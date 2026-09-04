"""
Isolation Forest kill-switch experiment for Agent Trust Layer v2.

The headline evaluation unit is PURCHASE ONLY because only a purchase can
create a Razorpay payment authorization.

Three-way methodology:
    TRAIN      80 agents, historical clean purchase behavior -> fit model
    VALIDATION 10 unseen agents -> choose REVIEW/BLOCK thresholds
    TEST       10 different unseen agents -> final report, touched once

Two behavioral feature sets:
    Model A = original 7 features
    Model B = Model A + recent_category_distance

The point is not to maximize a synthetic benchmark. The point is to test:
    "Does behavioral evidence catch anomalous payment requests that policy
     cannot express?"
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
)

DATA_DIR = Path("data")
RANDOM_STATE = 42

MODEL_A_FEATURES = [
    "amount",
    "price_zscore",
    "category_distance",
    "cumulative_spend_ratio",
    "time_since_previous_sec",
    "requests_last_5m",
    "quantity",
]

MODEL_B_FEATURES = MODEL_A_FEATURES + [
    "recent_category_distance",
]


def load_data():
    train = pd.read_csv(DATA_DIR / "features_train.csv")
    val = pd.read_csv(DATA_DIR / "features_validation.csv")
    test = pd.read_csv(DATA_DIR / "features_test.csv")

    # The payment decision is evaluated only at purchase steps.
    # Filter to the payment-authorizing step, then reset the row index.
    # Resetting is important because sklearn returns a positional NumPy array
    # while pandas keeps the original CSV index after filtering. Without this,
    # validation rows such as index 142 can try to access risk[142] when the
    # risk array only has 140 elements.
    train = train[train.step_type == "purchase"].copy().reset_index(drop=True)
    val = val[val.step_type == "purchase"].copy().reset_index(drop=True)
    test = test[test.step_type == "purchase"].copy().reset_index(drop=True)

    return train, val, test


def fit_model(train: pd.DataFrame, features: list[str]) -> IsolationForest:
    # Historical training data should be clean. Keep a defensive filter so
    # accidental policy violations never teach the model to call violations normal.
    clean = train[train.policy_passed].copy()

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(clean[features])
    return model


def anomaly_risk(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    # IsolationForest decision_function: higher = more normal.
    # We negate it so higher risk = more anomalous.
    return -model.decision_function(df[features])


def choose_thresholds(
    validation: pd.DataFrame,
    risk: np.ndarray,
) -> tuple[float, float]:
    """
    Calibrate from LEGITIMATE validation purchases only.

    REVIEW threshold:
        95th percentile of risk among policy-passing, ground-truth-normal
        validation purchases -> approximately 5% validation FPR.

    BLOCK threshold:
        99th percentile of the same legitimate distribution.

    Thresholds are selected only on validation, never test.
    """
    legitimate = validation[
        validation.policy_passed & ~validation.gt_behavioral_anomaly
    ]

    if len(legitimate) < 20:
        raise ValueError(
            "Too few legitimate validation purchases for threshold calibration."
        )

    legit_risk = risk[legitimate.index.to_numpy()]
    review = float(np.quantile(legit_risk, 0.95))
    block = float(np.quantile(legit_risk, 0.99))

    return review, block


def evaluate(
    df: pd.DataFrame,
    risk: np.ndarray,
    review_threshold: float,
    block_threshold: float,
) -> dict:
    y_true = df.gt_behavioral_anomaly.astype(int).to_numpy()

    policy_detected = (~df.policy_passed).astype(int).to_numpy()
    model_detected = (risk >= review_threshold).astype(int)
    combined_detected = ((~df.policy_passed) | (risk >= review_threshold)).astype(int)

    def metrics(pred):
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {
            "recall": recall,
            "precision": precision,
            "fpr": fpr,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    out = {
        "policy_only": metrics(policy_detected),
        "model_only": metrics(model_detected),
        "policy_plus_ml": metrics(combined_detected),
    }

    # Decisions for the actual authorization engine.
    decisions = np.where(
        ~df.policy_passed,
        "BLOCK",
        np.where(risk >= block_threshold, "BLOCK",
                 np.where(risk >= review_threshold, "REVIEW", "ALLOW"))
    )
    out["authorization_decisions"] = pd.Series(decisions).value_counts().to_dict()

    return out


def evaluate_by_difficulty(
    df: pd.DataFrame,
    risk: np.ndarray,
    review_threshold: float,
) -> pd.DataFrame:
    rows = []

    for difficulty in ["easy", "medium", "hard"]:
        sub = df[df.difficulty == difficulty].copy()
        if sub.empty:
            continue

        idx = sub.index
        y_true = sub.gt_behavioral_anomaly.astype(int).to_numpy()

        policy_pred = (~sub.policy_passed).astype(int).to_numpy()
        model_pred = (risk[idx] >= review_threshold).astype(int)
        combined_pred = (
            (~sub.policy_passed) | (risk[idx] >= review_threshold)
        ).astype(int)

        for name, pred in [
            ("Policy only", policy_pred),
            ("Model only", model_pred),
            ("Policy + ML", combined_pred),
        ]:
            rows.append({
                "difficulty": difficulty,
                "system": name,
                "recall": recall_score(
                    y_true, pred, zero_division=0
                ),
                "n_anomalous_purchases": int(y_true.sum()),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["A", "B"],
        default="A",
        help="A = original features; B = adds recent_category_distance",
    )
    args = parser.parse_args()

    train, val, test = load_data()
    features = MODEL_A_FEATURES if args.model == "A" else MODEL_B_FEATURES

    print("=" * 80)
    print(f"AGENT TRUST LAYER v2 - MODEL {args.model}")
    print("=" * 80)
    print(f"features: {features}")
    print(f"train purchase rows: {len(train)}")
    print(f"validation purchase rows: {len(val)}")
    print(f"test purchase rows: {len(test)}")
    print()

    model = fit_model(train, features)

    val_risk = anomaly_risk(model, val, features)
    review_threshold, block_threshold = choose_thresholds(val, val_risk)

    test_risk = anomaly_risk(model, test, features)

    results = evaluate(
        test,
        test_risk,
        review_threshold,
        block_threshold,
    )

    by_difficulty = evaluate_by_difficulty(
        test,
        test_risk,
        review_threshold,
    )

    print(f"Validation REVIEW threshold: {review_threshold:.5f}")
    print(f"Validation BLOCK threshold:  {block_threshold:.5f}")
    print()

    print("HELD-OUT TEST RESULTS")
    print("-" * 80)

    for name, m in results.items():
        if name == "authorization_decisions":
            continue
        print(
            f"{name:18s} "
            f"recall={m['recall']:.3f} "
            f"precision={m['precision']:.3f} "
            f"FPR={m['fpr']:.3f} "
            f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}"
        )

    print()
    print("Authorization decisions:", results["authorization_decisions"])
    print()
    print("RECALL BY DRIFT DIFFICULTY")
    print("-" * 80)
    print(by_difficulty.to_string(index=False))

    # Save artifacts so the dashboard can consume the exact same results.
    out_dir = DATA_DIR / "experiments"
    out_dir.mkdir(exist_ok=True)

    by_difficulty.to_csv(
        out_dir / f"recall_by_difficulty_model_{args.model}.csv",
        index=False,
    )

    pd.DataFrame([
        {
            "model": args.model,
            "review_threshold": review_threshold,
            "block_threshold": block_threshold,
            "system": system,
            **metrics,
        }
        for system, metrics in results.items()
        if system != "authorization_decisions"
    ]).to_csv(
        out_dir / f"summary_model_{args.model}.csv",
        index=False,
    )

    scored = test.copy()
    scored["anomaly_risk"] = test_risk
    scored["model_flag"] = test_risk >= review_threshold
    scored["combined_flag"] = (
        (~scored.policy_passed) | scored.model_flag
    )
    scored.to_csv(
        out_dir / f"scored_test_model_{args.model}.csv",
        index=False,
    )

    # Persist the fitted model + calibrated thresholds as the single
    # source of truth. demo_agent.py (and any future dashboard) should
    # LOAD this artifact rather than re-fitting IsolationForest from
    # scratch. Re-fitting live is not actually reproducible across
    # machines: identical code, identical seed, identical data can still
    # yield a different fitted model on a different scikit-learn version,
    # because IsolationForest's tree-building consumes its RNG differently
    # across sklearn releases even with random_state fixed. Persisting the
    # exact fitted model removes that source of nondeterminism entirely.
    artifact = {
        "model": model,
        "feature_names": features,
        "review_threshold": review_threshold,
        "block_threshold": block_threshold,
        "model_variant": args.model,
    }
    artifact_path = out_dir / f"trust_model_{args.model}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print()
    print(f"Artifacts saved to {out_dir}/")
    print(f"Persisted model + thresholds -> {artifact_path}")


if __name__ == "__main__":
    main()

"""Isolation Forest authorization experiment (leak-free, policy-only BLOCK)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score

DATA_DIR = Path(__file__).resolve().parent / "data"
RANDOM_STATE = 42

MODEL_A_FEATURES = [
    "amount",
    "price_zscore",
    "category_distance",
    "cumulative_spend_ratio",
    "time_since_previous_sec",
    "requests_last_5m",
    "gap_ratio",
    "rate_ratio",
    "quantity",
]

MODEL_B_FEATURES = MODEL_A_FEATURES + [
    "recent_category_distance",
]


def load_data():
    train = pd.read_csv(DATA_DIR / "features_train.csv")
    val = pd.read_csv(DATA_DIR / "features_validation.csv")
    test = pd.read_csv(DATA_DIR / "features_test.csv")

    train = train[train.step_type == "purchase"].copy().reset_index(drop=True)
    val = val[val.step_type == "purchase"].copy().reset_index(drop=True)
    test = test[test.step_type == "purchase"].copy().reset_index(drop=True)
    return train, val, test


def fit_model(train: pd.DataFrame, features: list[str]) -> IsolationForest:
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
    return -model.decision_function(df[features])


def choose_thresholds(validation: pd.DataFrame, risk: np.ndarray) -> tuple[float, float]:
    legitimate = validation[validation.policy_passed & ~validation.gt_behavioral_anomaly]
    if len(legitimate) < 20:
        raise ValueError("Too few legitimate validation purchases for threshold calibration.")
    legit_risk = risk[legitimate.index.to_numpy()]
    review = float(np.quantile(legit_risk, 0.95))
    block = float(np.quantile(legit_risk, 0.99))
    return review, block


def evaluate(
    df: pd.DataFrame,
    risk: np.ndarray,
    review_threshold: float,
) -> dict:
    y_true = (df.gt_behavioral_anomaly | ~df.policy_passed).astype(int).to_numpy()
    policy_detected = (~df.policy_passed).astype(int).to_numpy()
    model_detected = (risk >= review_threshold).astype(int)
    combined_detected = ((~df.policy_passed) | (risk >= review_threshold)).astype(int)

    def metrics(pred):
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return {
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

    decisions = np.where(
        ~df.policy_passed,
        "BLOCK",
        np.where(risk >= review_threshold, "REVIEW", "ALLOW"),
    )

    return {
        "policy_only": metrics(policy_detected),
        "model_only": metrics(model_detected),
        "policy_plus_ml": metrics(combined_detected),
        "authorization_decisions": pd.Series(decisions).value_counts().to_dict(),
    }


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
        combined_pred = ((~sub.policy_passed) | (risk[idx] >= review_threshold)).astype(int)

        for name, pred in [
            ("Policy only", policy_pred),
            ("Model only", model_pred),
            ("Policy + ML", combined_pred),
        ]:
            rows.append({
                "difficulty": difficulty,
                "system": name,
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "n_anomalous_purchases": int(y_true.sum()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B"], default="A")
    args = parser.parse_args()

    train, val, test = load_data()
    features = MODEL_A_FEATURES if args.model == "A" else MODEL_B_FEATURES

    model = fit_model(train, features)
    val_risk = anomaly_risk(model, val, features)
    review_threshold, block_threshold = choose_thresholds(val, val_risk)
    test_risk = anomaly_risk(model, test, features)

    results = evaluate(test, test_risk, review_threshold)
    by_difficulty = evaluate_by_difficulty(test, test_risk, review_threshold)

    print("=" * 80)
    print(f"AGENT TRUST LAYER v2 (HARDENED) - MODEL {args.model}")
    print("=" * 80)
    print(f"Validation REVIEW threshold: {review_threshold:.5f}  (99th pct ref: {block_threshold:.5f})")
    for name, m in results.items():
        if name == "authorization_decisions":
            continue
        print(
            f"{name:18s} recall={m['recall']:.3f} precision={m['precision']:.3f} "
            f"FPR={m['fpr']:.3f} TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}"
        )
    print("Decisions:", results["authorization_decisions"])
    print("\nRECALL BY DRIFT DIFFICULTY")
    print(by_difficulty.to_string(index=False))

    out_dir = DATA_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_difficulty.to_csv(out_dir / f"recall_by_difficulty_model_{args.model}.csv", index=False)
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
    ]).to_csv(out_dir / f"summary_model_{args.model}.csv", index=False)

    scored = test.copy()
    scored["anomaly_risk"] = test_risk
    scored["model_flag"] = test_risk >= review_threshold
    scored["combined_flag"] = (~scored.policy_passed) | scored.model_flag
    scored.to_csv(out_dir / f"scored_test_model_{args.model}.csv", index=False)

    artifact = {
        "model": model,
        "feature_names": features,
        "review_threshold": review_threshold,
        "block_threshold": block_threshold,
        "model_variant": args.model,
    }
    with open(out_dir / f"trust_model_{args.model}.pkl", "wb") as f:
        pickle.dump(artifact, f)


if __name__ == "__main__":
    main()

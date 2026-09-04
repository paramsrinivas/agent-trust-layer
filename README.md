# Agent Trust Layer — v2

Track 01: AI Growth & Agentic Commerce

## Product thesis

> We're not building another way for AI to buy things.
> We're building the reason a merchant would actually let it.

The Trust Layer sits between an AI buyer and Razorpay and produces a
**Payment Authorization Decision**.

- Policy = deterministic authorization boundary.
- Behavioral model = probabilistic evidence about whether the agent is behaving normally.
- Decision engine = ALLOW / REVIEW / BLOCK.
- Only ALLOW reaches Razorpay automatically.

## What changed in v2

1. Purchase-only headline evaluation.
2. Historical sessions create a real observed per-agent baseline.
3. Shrinkage pulls sparse personal estimates toward a train-only population prior.
4. Historical purchase generation is clipped below max transaction, avoiding contaminated "normal" training rows.
5. Train/validation/test split is by agent: 80/10/10.
6. Validation is used only to select REVIEW/BLOCK thresholds.
7. Final test set is untouched until evaluation.
8. `step_type` is exported.
9. `session_progress` is stored but not used by Model A.
10. Model B optionally adds `recent_category_distance` to test the sequence-aware
    category-drift hypothesis.
11. The same `extract_features()` function is intended for offline and live serving.
12. `mandate_snapshot` is used for policy evaluation and audit consistency.

## Files

- `models.py` — data contracts.
- `generator.py` — generation, drift injection, baseline estimation, policy and features.
- `generate_data.py` — reproducible dataset generation.
- `experiment.py` — Isolation Forest kill-switch experiment.
- `data/` — generated after running the generator.

## Run

```bash
python3 -m pip install numpy pandas scikit-learn
python3 generate_data.py
python3 experiment.py --model A
python3 experiment.py --model B
```

Model A is the locked baseline.
Model B is one controlled experiment for category drift.

## Evaluation discipline

The model is fitted only on clean historical purchase rows from training agents.

Thresholds are calibrated on legitimate validation purchases.

Final metrics are reported on different, unseen test agents.

Do not tune thresholds or features against the test set.

## Important interpretation

This is a synthetic benchmark of simulated agent-drift patterns. It is not a
claim of real-world fraud coverage.

The core question being tested is narrower:

> Can behavioral evidence detect payment requests that deterministic mandate
> policy cannot express?

If Model B does not materially improve medium drift, do not hide it. The result
means category drift likely needs a richer sequential model.

## Next product layer

Once the evaluation is credible:

AI buyer
  -> Trust Layer
  -> Payment Authorization Decision
  -> Razorpay test-mode order/payment
  -> audit trail

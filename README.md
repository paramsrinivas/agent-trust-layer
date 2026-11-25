# Agent Trust Layer — AI Payment Authorization Engine

A two-layer payment authorization engine for autonomous shopping agents that pairs **deterministic mandate policy enforcement** with an **Isolation Forest behavioral anomaly detector** over **empirical-Bayes per-agent baselines**, integrated with **Razorpay's Test Mode Orders API** and a **Streamlit merchant review console**.

---

## Why This Exists

Autonomous AI buyer agents need hard spending guardrails, but static rules alone miss subtle behavioral compromises—such as a hijacked agent executing rapid-fire purchases just under a velocity cap, or slowly drifting across allowed product categories.

Agent Trust Layer separates **authorization** from **execution**:
1. **Deterministic Policy Gate** enforces hard mandate limits (`BLOCK`, non-overridable).
2. **Behavioral Trust Model** flags policy-compliant anomalies against each agent's learned history (`REVIEW`, human-in-the-loop).
3. **Fail-Closed Payment Adapter** executes orders on Razorpay only after explicit `ALLOW` or human `Approve`, protected by atomic idempotency.

---

## Architecture

```text
PurchaseRequest + Session Context
        │
        ▼
┌─────────────────────────────────┐
│ 1. Deterministic Mandate Gate   │  category_allowlist, max_transaction,
│    (generator.policy_check)     │  session_budget, quantity_max,
└───────────────┬─────────────────┘  velocity_limit_per_10m, expires_at
                │
        ┌───────┴────────┐
     FAILED            PASSED
        │                │
        ▼                ▼
    [ BLOCK ]   ┌──────────────────────────────────────┐
  (no override) │ 2. Behavioral Trust Model            │
                │    Global IsolationForest over       │
                │    empirical-Bayes agent baselines   │
                └────────┬────────────────────┬────────┘
                         │                    │
                  risk >= review         risk < review
                         │                    │
                         ▼                    ▼
                   [ REVIEW ]            [ ALLOW ]
                (human Approve /     (fail-closed gate)
                 Deny in console)             │
                         │                    │
                Approve  └─────────┬──────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ 3. Razorpay Payment Adapter          │
                │  • SQLite WAL atomic idempotency     │
                │  • SHA-256 payload fingerprinting    │
                │  • Receipt reconciliation on retry   │
                │  • Persistent UNRESOLVED state       │
                └──────────────────────────────────────┘
```

### Core Safety & Reliability Invariants
* **Fail-closed payment boundary:** `create_order_for_decision()` only proceeds on explicit `"ALLOW"`. Any other or malformed decision string is rejected without contacting Razorpay.
* **Non-overridable policy blocks:** Mandate violations emit `BLOCK` with no override path. Probabilistic ML anomalies route to `REVIEW` so legitimate high-variance purchases are never permanently hard-blocked without human recourse.
* **Atomic idempotency & duplicate-charge prevention:** `idempotency.py` uses SQLite `WAL` mode with `BEGIN IMMEDIATE` transactions (`IN_PROGRESS`, `COMPLETED`, `DENIED`, `UNRESOLVED`, `FAILED_PERMANENT`). Concurrent duplicate submissions execute Razorpay at most once; reusing a `request_id` with a mismatched amount, currency, or mode raises `IdempotencyConflictError`.
* **Honest failure handling under network timeouts:** Before retrying a timed-out `order.create`, the adapter queries Razorpay by receipt (`client.order.all`) to reconcile orders that committed server-side. If network failures persist after retries, the request is persisted as `UNRESOLVED` for manual reconciliation rather than guessed as success or blindly retried.
* **Zero train/serve feature skew:** `generator.extract_features()` is shared identically across synthetic dataset generation, offline evaluation, CLI simulation, and the live Streamlit dashboard.

---

## Leak-Free Benchmark Results (Held-Out Test Set)

Evaluated on a **100-agent synthetic commerce benchmark** with a **persona-stratified 80 / 10 / 10 agent split** (`train`: 80 agents, `validation`: 10 unseen agents, `test`: 10 unseen agents).

* Validation and test sets contain **only unseen evaluation sessions** (100 normal + 30 drifted purchases in test; historical baseline-fitting sessions are strictly excluded to prevent in-sample baseline leakage).
* Thresholds are calibrated solely on legitimate validation purchases (95th percentile for `REVIEW`), then evaluated once on the held-out test set.

### Overall Held-Out Test Performance

| Model | System | Recall | Precision | FPR | TP | FP | TN | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Policy only | 33.3% | 100.0% | 0.0% | 10 | 0 | 100 | 20 |
| **Model A** (9 features) | Behavioral model only | 70.0% | 75.0% | 7.0% | 21 | 7 | 93 | 9 |
| **Model A** (9 features) | Policy + ML (combined) | 76.7% | 76.7% | 7.0% | 23 | 7 | 93 | 7 |
| **Model B** (+ `recent_category_distance`) | Behavioral model only | 70.0% | 77.8% | 6.0% | 21 | 6 | 94 | 9 |
| **Model B** (+ `recent_category_distance`) | **Policy + ML (combined)** | **76.7%** | **79.3%** | **6.0%** | **23** | **6** | **94** | **7** |

### Recall by Drift Difficulty (Model B, Held-Out Test)

| Difficulty | Scenario | Policy Only | Model B Only | Policy + ML |
| :--- | :--- | :---: | :---: | :---: |
| **Easy** (`n=10`) | Off-allowlist category + price spike (mandate violation) | 100% | 80% | **100%** |
| **Medium** (`n=10`) | Subtle in-allowlist category drift (policy-compliant) | 0% | 30% | **30%** |
| **Hard** (`n=10`) | Velocity compression under hard cap (policy-compliant) | 0% | 100% | **100%** |
| **Policy-compliant drift** (`n=20`) | Medium + Hard combined | 0% | 65% | **65%** |

* **Overall lift (Model B):** **2.3x recall** (`33.3% → 76.7%`) at **6.0% FPR**, with **0% false `BLOCK`s** (`ALLOW`: 101, `REVIEW`: 19, `BLOCK`: 10).

---

## Quickstart

### 1. Install dependencies & verify environment
```bash
pip install -r requirements.txt
python3 check_environment.py
```

### 2. Run the offline test suite (no API keys or network required)
Verifies fail-closed gating, SQLite WAL concurrency races, payload-fingerprint conflict detection, dry-run claim release, immutable `Approve`/`Deny` transitions, timeout receipt reconciliation, and mandate expiration checks in isolated temporary databases:
```bash
python3 test_razorpay_adapter.py
```

### 3. Regenerate the leak-free dataset & retrain models
```bash
python3 generate_data.py
python3 experiment.py --model A
python3 experiment.py --model B
```

### 4. Run the CLI demo & Streamlit merchant console
```bash
python3 demo_agent.py --scenario full --model B
streamlit run dashboard.py
```

### 5. Enable live Razorpay Test Mode orders (optional)
Create a `.env` file in the repository root with Razorpay **Test Mode** credentials (`rzp_test_...` prefix is enforced; live keys are refused unless explicitly enabled with `ALLOW_LIVE_RAZORPAY=1`):
```bash
RAZORPAY_KEY_ID=rzp_test_your_key_id
RAZORPAY_KEY_SECRET=your_test_secret
```
Then run the live Test Mode check:
```bash
python3 test_razorpay_adapter.py --live
```

---

## Repository Structure

| File | Purpose |
| :--- | :--- |
| `models.py` | Core dataclasses (`Mandate`, `BehavioralBaseline`, `PurchaseRequest`, `PaymentAuthorizationDecision`) and deterministic category taxonomy |
| `generator.py` | Pure functions for agent/session generation, drift injection, empirical-Bayes shrinkage baselines, mandate policy checks, and shared feature extraction |
| `authorization.py` | `TrustModel` adapter, `authorize_purchase()` integrity checks, feature-grounded behavioral explanations, and structured `audit_event()` logging |
| `idempotency.py` | SQLite WAL-backed atomic idempotency state machine with SHA-256 payload fingerprinting |
| `razorpay_payment.py` | Fail-closed Razorpay Orders API adapter with `Decimal` paise conversion, receipt reconciliation on retry, and human `approve_review()` / `deny_review()` |
| `generate_data.py` | Generates persona-stratified, leak-free 80/10/10 train/validation/test datasets |
| `experiment.py` | Trains Isolation Forest (Model A / Model B), calibrates validation thresholds, evaluates held-out test metrics, and persists artifacts |
| `demo_agent.py` | Deterministic CLI simulator exercising `ALLOW`, `REVIEW`, and `BLOCK` scenarios |
| `dashboard.py` | Streamlit merchant control console with live KPI reconciliation, HTML-escaped rendering, human review queue, and offline evaluation views |
| `test_razorpay_adapter.py` | 7-part offline test suite (concurrency, failure injection, integrity) + optional `--live` Razorpay Test Mode check |
| `check_environment.py` | Dependency and artifact verification script |
| `requirements.txt` | Pinned Python package requirements |

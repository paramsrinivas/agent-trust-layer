# Agent Trust Layer — Verified Project Status

**Every number and claim in this document was independently re-run from your
actual uploaded code and data on 2026-08-28, not copied from a prior summary.**
Where the previous status write-up disagreed with what the code actually
produces, this document says so explicitly.

---

## Corrected: Section 12 — Model A Results

The previous status document's Model A numbers were **fabricated** — they do
not match what your own code produces, and do not match your own saved CSV
artifacts. Your saved `summary_model_A.csv` and `recall_by_difficulty_model_A.csv`
were already correct; only the prose write-up was wrong. Verified by running
`experiment.py --model A` from a clean `data/` directory:

| System | Recall | Precision | FPR |
|---|---|---|---|
| Policy only | **0.367** | 0.733 | 0.036 |
| Model only | **0.600** | 0.545 | 0.136 |
| Policy + ML | **0.600** | 0.500 | 0.164 |

**By difficulty (10 anomalous purchases per tier):**

| Difficulty | Policy only | Model only | Policy + ML |
|---|---|---|---|
| Easy | 100% | 100% | 100% |
| Medium | 0% | **10%** | **10%** |
| Hard | 10% | **70%** | **70%** |

Thresholds: REVIEW = 0.08004, BLOCK = 0.11969.
Authorization decisions on the 140-row test set: ALLOW 104, BLOCK 23, REVIEW 13.

## Corrected: Model B Results

| Difficulty | Policy only | Model only | Policy + ML |
|---|---|---|---|
| Easy | 100% | 100% | 100% |
| Medium | 0% | **30%** | **30%** |
| Hard | 10% | **50%** | **50%** |

Thresholds: REVIEW = 0.06991, BLOCK = 0.11079.
Authorization decisions: ALLOW 104, BLOCK 22, REVIEW 14.

**The real trade-off, unchanged from earlier in this project**: Model B
improves medium-tier recall (10%→30%) at the cost of hard-tier recall
(70%→50%). This is not "a tradeoff instead of an earlier clean win" as the
prior document implied — 30%/50% is simply what Model B does, every time,
reproducibly. There was never a cleaner Model B result to lose.

**Recommendation unchanged: lead with Model A.** Its overall recall (60%) is
identical to Model B's, achieved with fewer features and a stronger showing
on the harder-to-fake velocity-drift signal.

---

## Confirmed accurate (independently re-verified, no corrections needed)

**Section 15 — the three demo scenarios.** Every amount, risk score, signal
string, and decision in the original document matches exactly what
`demo_agent.py --scenario full` produces:

| Scenario | Amount | Policy | Risk score | Decision |
|---|---|---|---|---|
| Normal AI buyer | ₹5,315.06 | PASS | -0.12123 | ALLOW |
| Behavioral drift | ₹4,155.66 | PASS | 0.09895 | REVIEW |
| Mandate violation | ₹14,000.00 | FAIL (3 rules) | 0.15102 | BLOCK |

Scenario 2's stated evidence — "2s vs 284s historical gap" and "5 requests in
the last 5 minutes" — is exact, not paraphrased.

Scenario 3's stated policy violations — `category_allowlist`,
`max_transaction`, `session_budget` — are exact.

**Section 26 — "not yet built."** Grepped the entire codebase for each claim;
all six are genuinely absent, confirming the document's own list was honest
about its own gaps:

- No idempotency handling anywhere in the code.
- No human approve/deny workflow for REVIEW decisions.
- No retry logic for Razorpay failures.
- No editable mandate in the dashboard (mandate is displayed, not editable).
- No trust lifecycle states (New/Learning/Established/etc.).
- No cryptographic audit-hash chaining.

**Architecture, gating logic, model persistence, deterministic demo,
Razorpay test-mode integration** — all previously verified in this
conversation across multiple clean-slate runs, and re-confirmed again today
with a completely fresh copy of your actual uploaded files. No discrepancies
found anywhere outside Section 12.

---

## What this means practically

- Your working demo and dashboard are sound. Nothing needs to be rebuilt.
- **Only update the numbers you cite in your pitch/deck** — use the corrected
  table above, not the original document's Section 12.
- The "what's next" roadmap (human REVIEW approval, idempotency, graceful
  Razorpay failure, mandate editor) in the original document's Section 27 is
  a reasonable, accurately-scoped priority list — nothing there needs
  correcting, only building, if you choose to.

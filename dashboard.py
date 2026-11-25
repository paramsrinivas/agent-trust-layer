from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from demo_agent import (
    authorize_session,
    build_demo_agent,
    load_trust_model,
    make_block_session,
    make_normal_session,
    make_rogue_session,
)
from razorpay_payment import approve_review, create_order_for_decision, deny_review

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

st.set_page_config(
    page_title="Stride Running · Agent Authorization",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #f7f7f5;
        --surface: #ffffff;
        --text: #171717;
        --muted: #6b6b67;
        --line: #deded8;
        --line-strong: #c8c8c0;
        --blue: #175cd3;
        --green: #16803c;
        --amber: #a15c00;
        --red: #b42318;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; }
    .block-container { max-width: 1180px !important; padding: 30px 36px 60px !important; }
    h1, h2, h3, p, label, td, th { color: var(--text); }
    h1 { font-size: 30px !important; line-height: 1.15 !important; letter-spacing: -0.035em !important; margin: 0 !important; font-weight: 720 !important; }
    h2 { color: var(--text) !important; font-size: 18px !important; line-height: 1.3 !important; letter-spacing: -0.015em !important; margin: 0 0 10px !important; font-weight: 700 !important; }
    h3 { color: var(--text) !important; font-size: 14px !important; margin: 0 0 7px !important; font-weight: 700 !important; }
    .eyebrow { color: #777771 !important; font-size: 10px; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 6px; }
    .subtle { color: var(--muted) !important; font-size: 13px; line-height: 1.45; }
    .topbar { border-bottom: 1px solid var(--line); padding-bottom: 22px; margin-bottom: 24px; }
    .merchant-status { display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--line-strong); background: var(--surface); border-radius: 5px; padding: 7px 10px; color: var(--text) !important; font-size: 12px; font-weight: 650; }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); display: inline-block; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 18px 20px; box-shadow: 0 1px 2px rgba(20,20,20,.025); }
    .panel-tight { background: var(--surface); border: 1px solid var(--line); border-radius: 7px; padding: 14px 16px; }
    .rule { height: 1px; background: var(--line); margin: 24px 0; }
    .stButton > button { min-height: 44px !important; width: 100% !important; border: 1px solid var(--line-strong) !important; border-radius: 6px !important; background: #fff !important; color: #202020 !important; font-size: 13px !important; font-weight: 650 !important; box-shadow: none !important; }
    .stButton > button:hover { border-color: #777770 !important; background: #f5f5f2 !important; color: #111 !important; }
    [data-testid="stMetric"] { background: #fff !important; border: 1px solid var(--line) !important; border-radius: 6px !important; padding: 12px 14px !important; min-height: 82px; box-shadow: none !important; }
    [data-testid="stMetricLabel"] { color: #777771 !important; font-size: 11px !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 22px !important; font-weight: 720 !important; }
    .outcome { border: 1px solid var(--line-strong); border-left: 4px solid #777; background: #fff; border-radius: 6px; padding: 16px 18px; margin: 2px 0 16px; }
    .outcome.allow { border-left-color: var(--green); }
    .outcome.review { border-left-color: #d18a00; }
    .outcome.block { border-left-color: var(--red); }
    .outcome-label { color: #777771 !important; font-size: 10px; text-transform: uppercase; letter-spacing: .1em; font-weight: 750; }
    .outcome-title { color: var(--text) !important; font-size: 25px; font-weight: 760; letter-spacing: -.025em; margin-top: 3px; }
    .outcome-copy { color: #60605c !important; font-size: 13px; margin-top: 4px; }
    .kv { display: flex; justify-content: space-between; gap: 20px; padding: 9px 0; border-bottom: 1px solid #eeeeea; font-size: 12px; }
    .kv:last-child { border-bottom: 0; }
    .kv-key { color: #777771 !important; }
    .kv-val { color: var(--text) !important; font-weight: 650; text-align: right; }
    .footer { color: #8a8a84 !important; font-size: 11px; padding-top: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
st.session_state.setdefault("audit_trail", [])
st.session_state.setdefault("pending_review_decisions", {})
st.session_state.setdefault("autonomous_enabled", True)
st.session_state.setdefault("model_variant", "B")
st.session_state.setdefault("last_result", None)


@st.cache_resource
def get_agent_and_model(model_variant: str):
    rng = np.random.default_rng(123)
    agent, _history = build_demo_agent(rng)
    model = load_trust_model(model_variant)
    return agent, model


agent, trust_model = get_agent_and_model(st.session_state.model_variant)
SCENARIO_SEEDS = {"normal": 101, "rogue": 202, "block": 303}


def _describe_outcome(decision, rp_res, amount: float) -> tuple[str, str]:
    if decision.decision == "BLOCK":
        return "Blocked", "Payment authorization denied. Razorpay was not contacted."
    if decision.decision == "REVIEW":
        return "Pending review", "Payment authorization withheld pending human review."
    if rp_res.dry_run:
        return (
            "Authorized (dry run)",
            f"₹{amount:,.2f} authorized. [DRY RUN] Razorpay keys not configured — not contacted.",
        )
    if rp_res.status == "unresolved":
        return (
            "Unresolved",
            f"₹{amount:,.2f} authorized, but Razorpay timed out. Reconcile manually before retrying.",
        )
    if rp_res.error:
        return (
            "Payment failed",
            f"₹{amount:,.2f} authorized, but Razorpay rejected the order: {rp_res.error}",
        )
    return (
        "Authorized",
        f"₹{amount:,.2f} authorized. Razorpay TEST MODE order created: {rp_res.order_id}",
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
head_left, head_right = st.columns([3.2, 1.35])
with head_left:
    st.markdown('<div class="eyebrow">Merchant control</div>', unsafe_allow_html=True)
    st.title("Stride Running")
    st.markdown('<div class="subtle">Agent authorization for autonomous purchases</div>', unsafe_allow_html=True)
with head_right:
    enabled = st.toggle("Autonomous purchases", value=st.session_state.autonomous_enabled, key="autonomous_toggle")
    st.session_state.autonomous_enabled = enabled

st.markdown(
    (
        '<span class="merchant-status"><span class="status-dot"></span>Autonomous commerce is ON</span>'
        if enabled
        else '<span class="merchant-status">Autonomous commerce is OFF</span>'
    ),
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
audit = st.session_state.audit_trail
authorized_volume = sum(
    float(x["amount"])
    for x in audit
    if x["status"] in ("Approved", "Authorized", "Authorized (dry run)")
)
request_count = len(audit)
review_queue_count = sum(x["status"] in ("Pending review", "Pending review (dry run)") for x in audit)
blocked_count = sum(x["status"] in ("Blocked", "Denied") for x in audit)

st.markdown("## Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Authorized volume", f"₹{authorized_volume:,.0f}")
m2.metric("Purchase requests", request_count)
m3.metric("Review queue", review_queue_count)
m4.metric("Blocked / Denied", blocked_count)

# ---------------------------------------------------------------------------
# Authorization workspace
# ---------------------------------------------------------------------------
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
left, right = st.columns([2.1, 1], gap="large")

with left:
    st.markdown("## Authorization")
    st.markdown(
        '<div class="subtle">Run a representative purchase through the same '
        "policy, behavioral, and payment gate used by the CLI demo.</div>",
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    normal_clicked = b1.button("Normal purchase", key="normal_purchase")
    drift_clicked = b2.button("Behavioral drift", key="behavioral_drift")
    block_clicked = b3.button("Mandate violation", key="mandate_violation")

    scenario, title = None, None
    if normal_clicked:
        scenario, title = "normal", "Normal purchase"
    elif drift_clicked:
        scenario, title = "rogue", "Behavioral drift"
    elif block_clicked:
        scenario, title = "block", "Mandate violation"

    if scenario and enabled:
        scenario_rng = np.random.default_rng(SCENARIO_SEEDS[scenario])
        if scenario == "normal":
            session = make_normal_session(agent, scenario_rng, fresh_request_id=True)
        elif scenario == "rogue":
            session = make_rogue_session(agent, scenario_rng, fresh_request_id=True)
        else:
            session = make_block_session(agent, scenario_rng, fresh_request_id=True)

        decision = authorize_session(session, agent, trust_model)
        razorpay_result = create_order_for_decision(decision)
        if razorpay_result.order_id:
            decision.razorpay_order_id = razorpay_result.order_id

        features = decision.behavior.features
        status_label, outcome_copy = _describe_outcome(decision, razorpay_result, float(features["amount"]))
        risk_label = (
            "HIGH"
            if decision.behavior.anomaly_score >= trust_model.block_threshold
            else ("ELEVATED" if decision.behavior.is_anomalous else "NORMAL")
        )

        record = {
            "time": pd.Timestamp.now().strftime("%H:%M:%S"),
            "scenario": title,
            "decision": decision.decision,
            "status": status_label,
            "risk": risk_label,
            "anomaly_score": round(float(decision.behavior.anomaly_score), 5),
            "amount": float(features["amount"]),
            "requests_last_5m": int(features["requests_last_5m"]),
            "reasons": list(decision.reasons),
            "request_id": decision.request_id,
            "session_id": decision.session_id,
            "order_id": razorpay_result.order_id,
            "outcome_copy": outcome_copy,
            "policy_passed": decision.policy.passed,
            "policy_violations": list(decision.policy.violations),
            "review_resolved": decision.decision != "REVIEW",
            "review_resolution": None,
            "features": {
                "category_distance": features["category_distance"],
                "recent_category_distance": features.get("recent_category_distance"),
                "price_zscore": features["price_zscore"],
                "time_since_previous_sec": features["time_since_previous_sec"],
                "gap_ratio": features.get("gap_ratio"),
                "rate_ratio": features.get("rate_ratio"),
            },
        }
        st.session_state.audit_trail.append(record)
        st.session_state.last_result = record
        if decision.decision == "REVIEW":
            st.session_state.pending_review_decisions[decision.request_id] = decision
        st.rerun()

    elif scenario and not enabled:
        st.warning("Autonomous purchases are OFF. No authorization was attempted.")

    result = st.session_state.last_result

    def render_review_actions(record: dict, key_suffix: str):
        pending = st.session_state.pending_review_decisions.get(record["request_id"])
        approve_col, deny_col = st.columns(2)
        if approve_col.button("✓ Approve — send to Razorpay", key=f"approve_{key_suffix}", use_container_width=True):
            if pending is not None:
                outcome = approve_review(pending)
                record["order_id"] = outcome.order_id
                if outcome.dry_run:
                    record["status"] = "Pending review (dry run)"
                    record["outcome_copy"] = (
                        f"Reviewer clicked Approve, but Razorpay keys are not configured "
                        f"(₹{record['amount']:,.2f} not submitted)."
                    )
                elif outcome.status == "unresolved":
                    record["review_resolved"] = True
                    record["review_resolution"] = "unresolved"
                    record["status"] = "Unresolved"
                    record["outcome_copy"] = (
                        f"Approved, but Razorpay connection dropped ({outcome.error}). "
                        "Reconcile manually in Razorpay Dashboard."
                    )
                    st.session_state.pending_review_decisions.pop(record["request_id"], None)
                elif outcome.error:
                    record["review_resolved"] = True
                    record["review_resolution"] = "failed"
                    record["status"] = "Payment failed"
                    record["outcome_copy"] = f"Approved, but Razorpay rejected the request: {outcome.error}"
                    st.session_state.pending_review_decisions.pop(record["request_id"], None)
                else:
                    record["review_resolved"] = True
                    record["review_resolution"] = "approved"
                    record["status"] = "Approved"
                    record["outcome_copy"] = (
                        f"Approved by reviewer. Razorpay TEST MODE order created: {outcome.order_id}"
                    )
                    st.session_state.pending_review_decisions.pop(record["request_id"], None)
                st.rerun()

        if deny_col.button("✗ Deny — block permanently", key=f"deny_{key_suffix}", use_container_width=True):
            if pending is not None:
                deny_review(pending)
                record["review_resolved"] = True
                record["review_resolution"] = "denied"
                record["status"] = "Denied"
                record["outcome_copy"] = "Denied by reviewer. Razorpay was never contacted."
                st.session_state.pending_review_decisions.pop(record["request_id"], None)
                st.rerun()

    if result:
        decision_class = escape(result["decision"].lower())
        st.markdown(
            f"""
            <div class="outcome {decision_class}">
                <div class="outcome-label">{escape(result["scenario"])} · {escape(result["status"])}</div>
                <div class="outcome-title">{escape(result["decision"])}</div>
                <div class="outcome-copy">{escape(result["outcome_copy"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["decision"] == "REVIEW" and not result["review_resolved"]:
            st.markdown("##### This request needs a human decision")
            render_review_actions(result, key_suffix=result["request_id"])
        elif result["decision"] == "REVIEW" and result["review_resolved"]:
            st.caption(f"Resolution: {result['status']}")

        d1, d2 = st.columns([1, 1], gap="large")
        with d1:
            st.markdown("### Policy")
            if result["policy_passed"]:
                st.markdown(
                    '<div class="panel-tight"><strong>PASS</strong><br>'
                    '<span class="subtle">All explicit mandate constraints passed.</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                rows = [
                    f'<div class="kv"><span class="kv-key">{escape(v.rule)}</span>'
                    f'<span class="kv-val">{escape(str(v.actual))} · limit {escape(str(v.limit))}</span></div>'
                    for v in result["policy_violations"]
                ]
                st.markdown(
                    '<div class="panel-tight"><strong>FAILED</strong>' + "".join(rows) + "</div>",
                    unsafe_allow_html=True,
                )

        with d2:
            st.markdown("### Behavior")
            behavior_reasons = [
                r for r in result["reasons"]
                if not r.startswith("Policy violation:")
                and r != "Request is within policy and consistent with agent behavior"
            ]
            reason_html = "".join(
                f'<div class="kv"><span class="kv-key">Signal</span>'
                f'<span class="kv-val">{escape(r)}</span></div>'
                for r in behavior_reasons
            )
            if not reason_html:
                reason_html = '<div class="subtle" style="margin-top:7px">No behavioral deviation detected.</div>'
            st.markdown(
                f'<div class="panel-tight"><strong>{escape(result["risk"])}</strong> '
                f'<span class="subtle">(score {result["anomaly_score"]:.4f})</span>'
                + reason_html
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("### Payment gate")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                f"""
                <div class="panel-tight">
                    <div class="eyebrow">Decision / Status</div>
                    <strong>{escape(result["decision"])} — {escape(result["status"])}</strong>
                    <div class="subtle" style="margin-top:5px">Behavioral risk: {escape(result["risk"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p2:
            if result["order_id"]:
                p2_title, p2_sub = "Order created", result["order_id"]
            elif result["status"] in ("Authorized (dry run)", "Pending review (dry run)"):
                p2_title, p2_sub = "Dry run", "Razorpay keys not configured."
            elif result["status"] == "Unresolved":
                p2_title, p2_sub = "Unresolved", "Network timeout — reconcile manually."
            elif result["status"] == "Payment failed":
                p2_title, p2_sub = "Payment rejected", "Razorpay returned an error."
            elif result["status"] == "Denied":
                p2_title, p2_sub = "Not contacted", "Denied by reviewer — payment never attempted."
            elif result["status"] == "Pending review":
                p2_title, p2_sub = "Awaiting reviewer", "Not contacted until approved."
            else:
                p2_title, p2_sub = "Not contacted", "Authorization stopped before payment."
            st.markdown(
                f"""
                <div class="panel-tight">
                    <div class="eyebrow">Razorpay test mode</div>
                    <strong>{escape(p2_title)}</strong>
                    <div class="subtle" style="margin-top:5px">{escape(p2_sub)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Decision evidence", expanded=False):
            st.json(
                {
                    "request_id": result["request_id"],
                    "session_id": result["session_id"],
                    "decision": result["decision"],
                    "status": result["status"],
                    "amount": result["amount"],
                    "anomaly_score": result["anomaly_score"],
                    "policy_passed": result["policy_passed"],
                    "behavior_signals": result["reasons"],
                    "razorpay_order_id": result["order_id"],
                    "features": result["features"],
                }
            )
    else:
        st.markdown(
            """
            <div class="panel" style="margin-top:18px">
                <div class="eyebrow">Ready</div>
                <strong>No authorization request selected.</strong>
                <div class="subtle" style="margin-top:5px">Choose a scenario above to run one purchase through the gate.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown("## Agent mandate")
    mandate = agent.mandate
    st.markdown(
        f"""
        <div class="panel">
            <div class="kv"><span class="kv-key">Agent</span>
            <span class="kv-val">{escape(agent.agent_id)}</span></div>
            <div class="kv"><span class="kv-key">Intent</span>
            <span class="kv-val">{escape(mandate.intent)}</span></div>
            <div class="kv"><span class="kv-key">Max transaction</span>
            <span class="kv-val">₹{mandate.max_transaction:,.0f}</span></div>
            <div class="kv"><span class="kv-key">Session budget</span>
            <span class="kv-val">₹{mandate.session_budget:,.0f}</span></div>
            <div class="kv"><span class="kv-key">Historical requests</span>
            <span class="kv-val">{agent.baseline.n_historical_requests}</span></div>
            <div style="margin-top:13px">
                <div class="eyebrow">Allowed categories</div>
                <div class="subtle">{escape(", ".join(mandate.allowed_categories))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Trust model")
    model_choice = st.selectbox(
        "Active model",
        ["A", "B"],
        index=0 if st.session_state.model_variant == "A" else 1,
        key="model_selector",
        help="Use the persisted trust model and its calibrated thresholds.",
    )
    if model_choice != st.session_state.model_variant:
        st.session_state.model_variant = model_choice
        st.session_state.last_result = None
        st.rerun()

    st.markdown(
        f"""
        <div class="panel-tight">
            <div class="eyebrow">Active model</div>
            <strong>Model {escape(st.session_state.model_variant)}</strong>
            <div class="subtle" style="margin-top:5px">Persisted model and validation-calibrated thresholds.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Other pending reviews
# ---------------------------------------------------------------------------
other_pending = [
    r for r in st.session_state.audit_trail
    if r["decision"] == "REVIEW" and not r["review_resolved"]
    and (not result or r["request_id"] != result["request_id"])
]
if other_pending:
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown(f"## Other pending reviews ({len(other_pending)})")
    for record in other_pending:
        st.markdown(
            f'<div class="panel-tight"><strong>{escape(record["scenario"])}</strong> · '
            f'₹{record["amount"]:,.2f} · {escape(record["status"])} · logged {escape(record["time"])}</div>',
            unsafe_allow_html=True,
        )
        render_review_actions(record, key_suffix=f"pending_{record['request_id']}")

# ---------------------------------------------------------------------------
# Authorization activity
# ---------------------------------------------------------------------------
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
activity_head, activity_action = st.columns([5, 1])
with activity_head:
    st.markdown("## Authorization activity")
with activity_action:
    if audit and st.button("Clear activity", key="clear_activity"):
        st.session_state.audit_trail = []
        st.session_state.pending_review_decisions.clear()
        st.session_state.last_result = None
        st.rerun()

if audit:
    activity = pd.DataFrame(audit)[
        ["time", "scenario", "decision", "status", "risk", "amount", "requests_last_5m", "order_id"]
    ].copy()
    activity.columns = ["Time", "Request", "Decision", "Status", "Risk", "Amount", "Requests / 5m", "Razorpay order"]
    activity["Amount"] = activity["Amount"].map(lambda x: f"₹{x:,.2f}")
    st.dataframe(activity, hide_index=True, use_container_width=True, height=min(330, 52 + len(activity) * 38))
else:
    st.markdown('<div class="panel-tight subtle">No authorization requests yet.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown("## Model evaluation")
st.markdown(
    '<div class="subtle">Held-out, leak-free test results from the experiment pipeline (unseen agents and unseen sessions only).</div>',
    unsafe_allow_html=True,
)

try:
    summary_df = pd.read_csv(DATA_DIR / "experiments" / f"summary_model_{st.session_state.model_variant}.csv")
    recall_df = pd.read_csv(DATA_DIR / "experiments" / f"recall_by_difficulty_model_{st.session_state.model_variant}.csv")

    policy_row = summary_df[summary_df["system"] == "policy_only"].iloc[0]
    combined_row = summary_df[summary_df["system"] == "policy_plus_ml"].iloc[0]

    e1, e2, e3 = st.columns(3)
    e1.metric("Policy-only recall", f'{policy_row["recall"]:.1%}')
    e2.metric("Policy + ML recall", f'{combined_row["recall"]:.1%}')
    e3.metric("Policy + ML FPR", f'{combined_row["fpr"]:.1%}')

    pivot = recall_df.pivot(index="difficulty", columns="system", values="recall").reindex(["easy", "medium", "hard"])
    pivot = pivot.rename(
        columns={
            "Policy only": "Policy only",
            "Model only": "Behavioral model only",
            "Policy + ML": "Policy + behavioral model",
        }
    )
    pivot.index = [x.title() for x in pivot.index]
    pivot = pivot.map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    st.dataframe(pivot, use_container_width=True, height=180)
except (FileNotFoundError, IndexError, KeyError):
    st.caption(f"Evaluation artifacts for Model {st.session_state.model_variant} are unavailable. Run experiment.py first.")

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Razorpay test environment · policy violations block deterministically · '
    "behavioral anomalies route to human review before payment.</div>",
    unsafe_allow_html=True,
)

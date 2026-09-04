from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from demo_agent import (
    build_demo_agent,
    load_trust_model,
    make_normal_session,
    make_rogue_session,
    make_block_session,
    authorize_session,
)
from razorpay_payment import create_order_for_decision, approve_review, deny_review

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

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--bg) !important;
    }

    .block-container {
        max-width: 1180px !important;
        padding: 30px 36px 60px !important;
    }

    /* Avoid global span/div colour rules. They fight Streamlit's widgets. */
    h1, h2, h3, p, label, td, th {
        color: var(--text);
    }

    h1 {
        font-size: 30px !important;
        line-height: 1.15 !important;
        letter-spacing: -0.035em !important;
        margin: 0 !important;
        font-weight: 720 !important;
    }

    h2 {
        color: var(--text) !important;
        font-size: 18px !important;
        line-height: 1.3 !important;
        letter-spacing: -0.015em !important;
        margin: 0 0 10px !important;
        font-weight: 700 !important;
    }

    h3 {
        color: var(--text) !important;
        font-size: 14px !important;
        margin: 0 0 7px !important;
        font-weight: 700 !important;
    }

    .eyebrow {
        color: #777771 !important;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .subtle {
        color: var(--muted) !important;
        font-size: 13px;
        line-height: 1.45;
    }

    .topbar {
        border-bottom: 1px solid var(--line);
        padding-bottom: 22px;
        margin-bottom: 24px;
    }

    .merchant-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        border: 1px solid var(--line-strong);
        background: var(--surface);
        border-radius: 5px;
        padding: 7px 10px;
        color: var(--text) !important;
        font-size: 12px;
        font-weight: 650;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
        display: inline-block;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(20,20,20,.025);
    }

    .panel-tight {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 14px 16px;
    }

    .rule {
        height: 1px;
        background: var(--line);
        margin: 24px 0;
    }

    /* Native buttons: same dimensions, strong hover/focus state. */
    .stButton > button {
        min-height: 44px !important;
        width: 100% !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 6px !important;
        background: #fff !important;
        color: #202020 !important;
        font-size: 13px !important;
        font-weight: 650 !important;
        box-shadow: none !important;
        transition: border-color .12s ease, background .12s ease;
    }

    .stButton > button:hover {
        border-color: #777770 !important;
        background: #f5f5f2 !important;
        color: #111 !important;
    }

    .stButton > button:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 2px rgba(23,92,211,.12) !important;
    }

    /* Compact selectbox. */
    div[data-baseweb="select"] > div {
        min-height: 42px !important;
        background: #fff !important;
        border-color: var(--line-strong) !important;
        border-radius: 6px !important;
    }

    /* Toggle */
    [data-testid="stToggle"] label {
        font-size: 13px !important;
        font-weight: 650 !important;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #fff !important;
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        padding: 12px 14px !important;
        min-height: 82px;
        box-shadow: none !important;
    }

    [data-testid="stMetricLabel"] {
        color: #777771 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 22px !important;
        font-weight: 720 !important;
    }

    /* Decision outcome */
    .outcome {
        border: 1px solid var(--line-strong);
        border-left: 4px solid #777;
        background: #fff;
        border-radius: 6px;
        padding: 16px 18px;
        margin: 2px 0 16px;
    }

    .outcome.allow { border-left-color: var(--green); }
    .outcome.review { border-left-color: #d18a00; }
    .outcome.block { border-left-color: var(--red); }

    .outcome-label {
        color: #777771 !important;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-weight: 750;
    }

    .outcome-title {
        color: var(--text) !important;
        font-size: 25px;
        font-weight: 760;
        letter-spacing: -.025em;
        margin-top: 3px;
    }

    .outcome-copy {
        color: #60605c !important;
        font-size: 13px;
        margin-top: 4px;
    }

    .kv {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        padding: 9px 0;
        border-bottom: 1px solid #eeeeea;
        font-size: 12px;
    }

    .kv:last-child { border-bottom: 0; }

    .kv-key { color: #777771 !important; }
    .kv-val { color: var(--text) !important; font-weight: 650; text-align: right; }

    .decision-note {
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fafaf8;
        padding: 11px 13px;
        font-size: 12px;
        color: #555550 !important;
        line-height: 1.45;
        margin-top: 12px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line) !important;
        border-radius: 6px !important;
        overflow: hidden;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
        box-shadow: none !important;
    }

    .footer {
        color: #8a8a84 !important;
        font-size: 11px;
        padding-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "pending_review_decisions" not in st.session_state:
    st.session_state.pending_review_decisions = {}
if "autonomous_enabled" not in st.session_state:
    st.session_state.autonomous_enabled = True
if "model_variant" not in st.session_state:
    st.session_state.model_variant = "A"
if "last_result" not in st.session_state:
    st.session_state.last_result = None


@st.cache_resource
def get_agent_and_model(model_variant: str):
    rng = np.random.default_rng(123)
    agent, _history = build_demo_agent(rng)
    model = load_trust_model(model_variant)
    return agent, model, rng


agent, trust_model, _agent_rng = get_agent_and_model(st.session_state.model_variant)

# Canonical demo fixtures: each scenario gets its own fixed RNG stream.
# This keeps the demo reproducible while still exercising the real session
# generators and authorization pipeline.
SCENARIO_SEEDS = {
    "normal": 101,
    "rogue": 202,
    "block": 303,
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="topbar">', unsafe_allow_html=True)
head_left, head_right = st.columns([3.2, 1.35], vertical_alignment="center")

with head_left:
    st.markdown('<div class="eyebrow">Merchant control</div>', unsafe_allow_html=True)
    st.title("Stride Running")
    st.markdown(
        '<div class="subtle">Agent authorization for autonomous purchases</div>',
        unsafe_allow_html=True,
    )

with head_right:
    enabled = st.toggle(
        "Autonomous purchases",
        value=st.session_state.autonomous_enabled,
        key="autonomous_toggle",
    )
    st.session_state.autonomous_enabled = enabled

st.markdown(
    (
        '<span class="merchant-status"><span class="status-dot"></span>'
        'Autonomous commerce is ON</span>'
        if enabled
        else
        '<span class="merchant-status">Autonomous commerce is OFF</span>'
    ),
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
audit = st.session_state.audit_trail
authorized_volume = sum(
    float(x["amount"]) for x in audit if x["decision"] == "ALLOW"
)
request_count = len(audit)
review_count = sum(x["decision"] == "REVIEW" for x in audit)
block_count = sum(x["decision"] == "BLOCK" for x in audit)

st.markdown("## Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Authorized volume", f"₹{authorized_volume:,.0f}")
m2.metric("Purchase requests", request_count)
m3.metric("Review queue", review_count)
m4.metric("Blocked", block_count)

# ---------------------------------------------------------------------------
# Authorization workspace
# ---------------------------------------------------------------------------
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

left, right = st.columns([2.1, 1], gap="large")

with left:
    st.markdown("## Authorization")
    st.markdown(
        '<div class="subtle">Run a representative purchase through the same '
        'policy, behavioral, and payment gate used by the demo.</div>',
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    normal_clicked = b1.button("Normal purchase", key="normal_purchase")
    drift_clicked = b2.button("Behavioral drift", key="behavioral_drift")
    block_clicked = b3.button("Mandate violation", key="mandate_violation")

    scenario = None
    title = None
    if normal_clicked:
        scenario, title = "normal", "Normal purchase"
    elif drift_clicked:
        scenario, title = "rogue", "Behavioral drift"
    elif block_clicked:
        scenario, title = "block", "Mandate violation"

    if scenario and enabled:
        # Never share RNG state between demo scenarios. Otherwise repeated
        # clicks can change the "same" representative purchase.
        scenario_rng = np.random.default_rng(SCENARIO_SEEDS[scenario])

        if scenario == "normal":
            session = make_normal_session(agent, scenario_rng)
        elif scenario == "rogue":
            session = make_rogue_session(agent, scenario_rng)
        else:
            session = make_block_session(agent, scenario_rng)

        decision = authorize_session(session, agent, trust_model)

        # Payment is contacted only after the authorization decision.
        razorpay_result = create_order_for_decision(decision)
        if razorpay_result.error:
            decision.razorpay_order_id = None
        elif razorpay_result.called:
            decision.razorpay_order_id = razorpay_result.order_id

        features = decision.behavior.features
        risk_label = RISK_LABELS = {
            "ALLOW": "LOW",
            "REVIEW": "ELEVATED",
            "BLOCK": "HIGH",
        }[decision.decision]

        if decision.decision == "ALLOW":
            outcome_copy = (
                f"₹{features['amount']:,.2f} is eligible for payment. "
                "Razorpay was contacted in test mode."
            )
        elif decision.decision == "REVIEW":
            outcome_copy = (
                "Payment authorization withheld. Razorpay was not contacted."
            )
        else:
            outcome_copy = (
                "Payment authorization denied. Razorpay was not contacted."
            )

        record = {
            "time": pd.Timestamp.now().strftime("%H:%M:%S"),
            "scenario": title,
            "decision": decision.decision,
            "risk": risk_label,
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
                "price_zscore": features["price_zscore"],
                "time_since_previous_sec": features["time_since_previous_sec"],
            },
        }
        st.session_state.audit_trail.append(record)
        st.session_state.last_result = record
        # Keep the raw decision object (not just the flattened display
        # dict) so a later Approve/Deny click can call approve_review()/
        # deny_review() on the EXACT decision that was authorized --
        # never reconstructed from display strings.
        if decision.decision == "REVIEW":
            st.session_state.pending_review_decisions[decision.request_id] = decision

        # Overview (Authorized volume / Purchase requests / Review queue /
        # Blocked) is rendered EARLIER in this script, before this handler
        # runs -- so without forcing a fresh rerun here, it would draw
        # using audit_trail as it stood BEFORE this record was appended,
        # showing counts exactly one interaction behind. The approve/deny
        # handlers already do this correctly; this handler didn't, which
        # is why Overview under-counted on every new scenario run.
        st.rerun()

    elif scenario and not enabled:
        st.warning("Autonomous purchases are OFF. No authorization was attempted.")

    result = st.session_state.last_result

    def render_review_actions(record: dict, key_suffix: str):
        """Renders Approve/Deny for one REVIEW record and handles the
        click. Shared by the main outcome card and the 'other pending
        reviews' list below, so any unresolved REVIEW -- not just the
        most recently run one -- has a working action, not just a
        visible-but-unactionable row in the activity table."""
        pending = st.session_state.pending_review_decisions.get(record["request_id"])
        approve_col, deny_col = st.columns(2)
        if approve_col.button("✓ Approve — send to Razorpay", key=f"approve_{key_suffix}",
                               use_container_width=True):
            if pending is not None:
                outcome = approve_review(pending)
                record["review_resolved"] = True
                record["review_resolution"] = "approved"
                record["order_id"] = outcome.order_id
                if outcome.dry_run:
                    record["outcome_copy"] = (
                        "Approved by reviewer. [DRY RUN] Razorpay not configured -- "
                        f"would have created an order for ₹{record['amount']:,.2f}."
                    )
                elif outcome.error:
                    record["outcome_copy"] = f"Approved, but Razorpay returned an error: {outcome.error}"
                else:
                    record["outcome_copy"] = (
                        f"Approved by reviewer. Razorpay TEST MODE order created: {outcome.order_id}"
                    )
                st.rerun()
        if deny_col.button("✗ Deny — block permanently", key=f"deny_{key_suffix}",
                            use_container_width=True):
            if pending is not None:
                deny_review(pending)
                record["review_resolved"] = True
                record["review_resolution"] = "denied"
                record["outcome_copy"] = "Denied by reviewer. Razorpay was never contacted."
                st.rerun()

    if result:
        decision_class = result["decision"].lower()
        st.markdown(
            f"""
            <div class="outcome {decision_class}">
                <div class="outcome-label">{result["scenario"]}</div>
                <div class="outcome-title">{result["decision"]}</div>
                <div class="outcome-copy">{result["outcome_copy"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result["decision"] == "REVIEW" and not result["review_resolved"]:
            st.markdown("##### This request needs a human decision")
            render_review_actions(result, key_suffix=result["request_id"])
        elif result["decision"] == "REVIEW" and result["review_resolved"]:
            resolution_label = ("✓ Approved by reviewer" if result["review_resolution"] == "approved"
                                 else "✗ Denied by reviewer")
            st.caption(resolution_label)

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
                rows = []
                for v in result["policy_violations"]:
                    rows.append(
                        f'<div class="kv"><span class="kv-key">{v.rule}</span>'
                        f'<span class="kv-val">{v.actual} · limit {v.limit}</span></div>'
                    )
                st.markdown(
                    '<div class="panel-tight"><strong>FAILED</strong>'
                    + "".join(rows)
                    + "</div>",
                    unsafe_allow_html=True,
                )

        with d2:
            st.markdown("### Behavior")
            # Decision state is the source of truth for the merchant-facing
            # risk label. Do not infer "anomalous" merely from the existence
            # of a reason string; ALLOW includes a positive explanation.
            behavior_status = {
                "ALLOW": "NORMAL",
                "REVIEW": "ELEVATED",
                "BLOCK": "HIGH",
            }[result["decision"]]

            behavior_reasons = [
                r for r in result["reasons"]
                if not r.startswith("Policy violation:")
                and r != "Request is within policy and consistent with agent behavior"
            ]

            if result["decision"] == "BLOCK" and not behavior_reasons:
                behavior_reasons = [
                    "Authorization risk is high; policy violations independently require a block."
                ]

            reason_html = "".join(
                f'<div class="kv"><span class="kv-key">Signal</span>'
                f'<span class="kv-val">{r}</span></div>'
                for r in behavior_reasons
            )

            if not reason_html:
                reason_html = (
                    '<div class="subtle" style="margin-top:7px">'
                    'No behavioral deviation detected.'
                    '</div>'
                )

            st.markdown(
                f'<div class="panel-tight"><strong>{behavior_status}</strong>'
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
                    <div class="eyebrow">Decision</div>
                    <strong>{result["decision"]}</strong>
                    <div class="subtle" style="margin-top:5px">
                        Risk level: {result["risk"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p2:
            # An order can exist for two different original decisions:
            # a direct ALLOW, or a REVIEW that a human later approved.
            # What actually matters here is whether order_id is set, not
            # whether the ORIGINAL decision label was "ALLOW" -- checking
            # decision=="ALLOW" alone meant an approved REVIEW (whose
            # decision label correctly stays "REVIEW", by design) always
            # fell through to "Not contacted" even after a real order was
            # created, which is what the screenshots caught.
            if result["order_id"]:
                st.markdown(
                    f"""
                    <div class="panel-tight">
                        <div class="eyebrow">Razorpay test mode</div>
                        <strong>Order created</strong>
                        <div class="subtle" style="margin-top:5px">{result["order_id"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif result["decision"] == "ALLOW":
                st.markdown(
                    '<div class="panel-tight"><div class="eyebrow">Razorpay test mode</div>'
                    '<strong>Order not created</strong><div class="subtle" style="margin-top:5px">'
                    'Adapter returned no order ID.</div></div>',
                    unsafe_allow_html=True,
                )
            elif result["decision"] == "REVIEW" and result["review_resolved"] and result["review_resolution"] == "denied":
                st.markdown(
                    '<div class="panel-tight"><div class="eyebrow">Razorpay test mode</div>'
                    '<strong>Not contacted</strong><div class="subtle" style="margin-top:5px">'
                    'Denied by reviewer -- payment never attempted.</div></div>',
                    unsafe_allow_html=True,
                )
            elif result["decision"] == "REVIEW" and not result["review_resolved"]:
                st.markdown(
                    '<div class="panel-tight"><div class="eyebrow">Razorpay test mode</div>'
                    '<strong>Awaiting reviewer</strong><div class="subtle" style="margin-top:5px">'
                    'Not contacted until approved.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="panel-tight"><div class="eyebrow">Razorpay test mode</div>'
                    '<strong>Not contacted</strong><div class="subtle" style="margin-top:5px">'
                    'Authorization stopped before payment.</div></div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Decision evidence", expanded=False):
            st.json(
                {
                    "request_id": result["request_id"],
                    "session_id": result["session_id"],
                    "decision": result["decision"],
                    "amount": result["amount"],
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
                <div class="subtle" style="margin-top:5px">
                    Choose a scenario above to run one purchase through the gate.
                </div>
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
            <span class="kv-val">DEMO-AGENT-01</span></div>
            <div class="kv"><span class="kv-key">Intent</span>
            <span class="kv-val">{mandate.intent}</span></div>
            <div class="kv"><span class="kv-key">Max transaction</span>
            <span class="kv-val">₹{mandate.max_transaction:,.0f}</span></div>
            <div class="kv"><span class="kv-key">Session budget</span>
            <span class="kv-val">₹{mandate.session_budget:,.0f}</span></div>
            <div class="kv"><span class="kv-key">Historical requests</span>
            <span class="kv-val">{agent.baseline.n_historical_requests}</span></div>
            <div style="margin-top:13px">
                <div class="eyebrow">Allowed categories</div>
                <div class="subtle">{", ".join(mandate.allowed_categories)}</div>
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
    st.markdown(
        f"""
        <div class="panel-tight">
            <div class="eyebrow">Active model</div>
            <strong>Model {st.session_state.model_variant}</strong>
            <div class="subtle" style="margin-top:5px">
                Persisted model and calibrated thresholds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if model_choice != st.session_state.model_variant:
        st.session_state.model_variant = model_choice
        st.session_state.last_result = None
        st.rerun()

# ---------------------------------------------------------------------------
# Other pending reviews -- any unresolved REVIEW that isn't the one shown
# above. Without this, running a second scenario before resolving the
# first REVIEW orphaned it: the row stayed in the activity table forever
# with no way to act on it. Every unresolved REVIEW gets its own
# Approve/Deny here, not just the most recently run one.
# ---------------------------------------------------------------------------
other_pending = [
    r for r in st.session_state.audit_trail
    if r["decision"] == "REVIEW" and not r["review_resolved"]
    and (not result or r["request_id"] != result["request_id"])
]
if other_pending:
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown(f"## Other pending reviews ({len(other_pending)})")
    st.caption("Unresolved from earlier in this session -- each needs its own decision.")
    for record in other_pending:
        st.markdown(
            f'<div class="panel-tight"><strong>{record["scenario"]}</strong> · '
            f'₹{record["amount"]:,.2f} · logged {record["time"]}</div>',
            unsafe_allow_html=True,
        )
        render_review_actions(record, key_suffix=f"pending_{record['request_id']}")

# ---------------------------------------------------------------------------
# Authorization activity
# ---------------------------------------------------------------------------
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
activity_head, activity_action = st.columns([5, 1], vertical_alignment="bottom")
with activity_head:
    st.markdown("## Authorization activity")
    st.markdown(
        '<div class="subtle">Requests run in this browser session. '
        'Amounts and decisions are fixed demo fixtures; Razorpay order IDs are generated by test mode.</div>',
        unsafe_allow_html=True,
    )
with activity_action:
    if audit and st.button("Clear activity", key="clear_activity"):
        st.session_state.audit_trail = []
        st.session_state.last_result = None
        st.rerun()

if audit:
    activity = pd.DataFrame(audit)[
        ["time", "scenario", "decision", "risk", "amount", "requests_last_5m", "order_id"]
    ].copy()
    activity.columns = [
        "Time", "Request", "Decision", "Risk", "Amount",
        "Requests / 5m", "Razorpay order"
    ]
    activity["Amount"] = activity["Amount"].map(lambda x: f"₹{x:,.2f}")
    st.dataframe(activity, hide_index=True, width="stretch", height=min(330, 52 + len(activity) * 38))
else:
    st.markdown(
        '<div class="panel-tight subtle">No authorization requests yet.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown("## Model evaluation")
st.markdown(
    '<div class="subtle">Held-out test results from the experiment pipeline. '
    'These figures are measured offline and are not regenerated by the dashboard.</div>',
    unsafe_allow_html=True,
)

try:
    summary_path = f"data/experiments/summary_model_{st.session_state.model_variant}.csv"
    recall_path = f"data/experiments/recall_by_difficulty_model_{st.session_state.model_variant}.csv"

    summary_df = pd.read_csv(summary_path)
    recall_df = pd.read_csv(recall_path)

    policy_row = summary_df[summary_df["system"] == "policy_only"].iloc[0]
    combined_row = summary_df[summary_df["system"] == "policy_plus_ml"].iloc[0]

    e1, e2, e3 = st.columns(3)
    e1.metric("Policy-only recall", f'{policy_row["recall"]:.1%}')
    e2.metric("Policy + ML recall", f'{combined_row["recall"]:.1%}')
    e3.metric("Policy + ML FPR", f'{combined_row["fpr"]:.1%}')

    pivot = recall_df.pivot(
        index="difficulty", columns="system", values="recall"
    ).reindex(["easy", "medium", "hard"])

    pivot = pivot.rename(
        columns={
            "policy_only": "Policy only",
            "model_only": "Behavioral model only",
            "policy_plus_ml": "Policy + behavioral model",
        }
    )
    pivot.index = [x.title() for x in pivot.index]
    pivot = pivot.map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")

    st.dataframe(pivot, width="stretch", height=180)

except (FileNotFoundError, IndexError, KeyError) as exc:
    st.caption(f"Evaluation artifacts for Model {st.session_state.model_variant} are unavailable.")

st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Razorpay test environment · authorization is policy + behavioral '
    'trust · review and block decisions stop before payment.</div>',
    unsafe_allow_html=True,
)
"""
Milestone 1: generate synthetic agent-trust data end to end.

Run:
    python3 generate_data.py

Produces (in ./data/):
    agents.json               -- all 100 agent profiles (mandate, hidden
                                  true_behavior clearly marked as generator-
                                  only ground truth, and the ESTIMATED
                                  baseline actually used by the system)
    sessions.json              -- raw request streams, human-inspectable
    features_train.csv         -- 80 agents' historical (normal) requests
                                  -> used to fit the population baseline
                                  and later the Isolation Forest
    features_validation.csv    -- 10 unseen agents' historical + evaluation
                                  requests, labeled -> used to pick
                                  REVIEW/BLOCK thresholds
    features_test.csv          -- 10 different unseen agents, same shape,
                                  touched exactly once for final metrics
    inspection_report.txt      -- a few representative sessions printed
                                  human-readably, so you can sanity-check
                                  the generator before trusting any model
                                  built on top of it

Split is BY AGENT, never by row: 80 / 10 / 10.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from models import PERSONAS, to_dict
from generator import (
    generate_agent, generate_normal_session, inject_drift,
    estimate_population_baseline, estimate_agent_baseline,
    policy_check, extract_features, requests_in_window,
)

SEED = 42
N_AGENTS_TRAIN = 80
N_AGENTS_VAL = 10
N_AGENTS_TEST = 10
N_HISTORICAL_SESSIONS = 10   # clean/normal, used to estimate baseline
EVAL_DIFFICULTIES = ["normal", "easy", "medium", "hard"]  # 1 eval session each, for val/test agents

BASE_TIME = datetime(2026, 8, 1, 9, 0, 0)
OUT_DIR = "data"


def main():
    rng = np.random.default_rng(SEED)
    random.seed(SEED)

    # -----------------------------------------------------------------
    # 1. Build 100 agents, persona-balanced, then split BY AGENT (80/10/10)
    # -----------------------------------------------------------------
    persona_names = list(PERSONAS.keys())
    total_agents = N_AGENTS_TRAIN + N_AGENTS_VAL + N_AGENTS_TEST  # 100
    agent_ids = [f"A{idx:03d}" for idx in range(1, total_agents + 1)]

    shuffled_ids = agent_ids[:]
    random.shuffle(shuffled_ids)
    train_ids = set(shuffled_ids[:N_AGENTS_TRAIN])
    val_ids = set(shuffled_ids[N_AGENTS_TRAIN:N_AGENTS_TRAIN + N_AGENTS_VAL])
    test_ids = set(shuffled_ids[N_AGENTS_TRAIN + N_AGENTS_VAL:])

    def split_of(agent_id: str) -> str:
        if agent_id in train_ids:
            return "train"
        if agent_id in val_ids:
            return "validation"
        return "test"

    agents = {}
    for i, agent_id in enumerate(agent_ids):
        persona = persona_names[i % len(persona_names)]  # balanced round robin
        agents[agent_id] = generate_agent(agent_id, persona, rng, split_of(agent_id))

    # -----------------------------------------------------------------
    # 2. Historical (normal) sessions for every agent -- used to estimate
    #    that agent's own baseline. NEVER drifted.
    # -----------------------------------------------------------------
    historical_sessions: dict[str, list] = {aid: [] for aid in agent_ids}
    for agent_id, agent in agents.items():
        t = BASE_TIME
        for s in range(N_HISTORICAL_SESSIONS):
            session_id = f"{agent_id}-H{s:02d}"
            sess = generate_normal_session(agent, session_id, t, rng, kind="historical")
            historical_sessions[agent_id].append(sess)
            t = t + timedelta(hours=float(rng.uniform(3, 10)))  # spread sessions over "days"

    # -----------------------------------------------------------------
    # 3. Population baseline -- computed ONLY from TRAIN agents' historical
    #    data (never from validation/test agents, and never from hidden
    #    true_behavior).
    # -----------------------------------------------------------------
    train_agents = [agents[aid] for aid in agent_ids if agents[aid].split == "train"]
    population_baseline = estimate_population_baseline(train_agents, historical_sessions)

    # -----------------------------------------------------------------
    # 4. Every agent (train/val/test) gets its own estimated baseline,
    #    shrunk toward the (train-only) population prior.
    # -----------------------------------------------------------------
    for agent_id, agent in agents.items():
        agent.baseline = estimate_agent_baseline(
            historical_sessions[agent_id], population_baseline
        )

    # -----------------------------------------------------------------
    # 5. Evaluation sessions -- ONLY for validation/test agents.
    #    Train agents are never evaluated; their historical data alone
    #    feeds Isolation Forest fitting (it should learn "normal" from
    #    genuinely normal data).
    # -----------------------------------------------------------------
    evaluation_sessions: dict[str, list] = {aid: [] for aid in agent_ids}
    for agent_id, agent in agents.items():
        if agent.split == "train":
            continue
        t = historical_sessions[agent_id][-1].requests[-1].timestamp
        t = datetime.fromisoformat(t) + timedelta(hours=2)
        for difficulty in EVAL_DIFFICULTIES:
            session_id = f"{agent_id}-E-{difficulty}"
            base_sess = generate_normal_session(agent, session_id, t, rng, kind="evaluation")
            drifted = inject_drift(base_sess, difficulty, rng, agent)
            evaluation_sessions[agent_id].append(drifted)
            t = t + timedelta(hours=float(rng.uniform(1, 4)))

    # -----------------------------------------------------------------
    # 6. Feature extraction -> three CSVs
    # -----------------------------------------------------------------
    def sessions_to_rows(session_list, agent) -> list[dict]:
        rows = []
        for sess in session_list:
            reqs = sess.requests
            for idx, req in enumerate(reqs):
                feats = extract_features(idx, reqs, sess.mandate_snapshot, agent.baseline)
                cumulative_spend = sum(
                    r.product.amount for r in reqs[: idx + 1] if r.step.type == "purchase"
                )
                req_10m = requests_in_window(idx, reqs, minutes=10)
                policy = policy_check(req, sess.mandate_snapshot, cumulative_spend, req_10m)

                row = dict(
                    agent_id=agent.agent_id,
                    persona=agent.persona,
                    session_id=sess.session_id,
                    request_id=req.request_id,
                    request_index=idx,
                    session_kind=sess.kind,
                    difficulty=sess.difficulty,
                    drift_type=sess.drift_type,
                    step_type=req.step.type,
                    category=req.product.category,
                    **feats,
                    policy_passed=policy.passed,
                    policy_violations=";".join(v.rule for v in policy.violations),
                    gt_behavioral_anomaly=req.gt_behavioral_anomaly,
                )
                rows.append(row)
        return rows

    train_rows, val_rows, test_rows = [], [], []
    for agent_id, agent in agents.items():
        if agent.split == "train":
            train_rows.extend(sessions_to_rows(historical_sessions[agent_id], agent))
        elif agent.split == "validation":
            val_rows.extend(sessions_to_rows(historical_sessions[agent_id], agent))
            val_rows.extend(sessions_to_rows(evaluation_sessions[agent_id], agent))
        else:
            test_rows.extend(sessions_to_rows(historical_sessions[agent_id], agent))
            test_rows.extend(sessions_to_rows(evaluation_sessions[agent_id], agent))

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    df_train.to_csv(f"{OUT_DIR}/features_train.csv", index=False)
    df_val.to_csv(f"{OUT_DIR}/features_validation.csv", index=False)
    df_test.to_csv(f"{OUT_DIR}/features_test.csv", index=False)

    # -----------------------------------------------------------------
    # 7. Export agents.json and sessions.json for inspection
    # -----------------------------------------------------------------
    agents_export = {aid: to_dict(a) for aid, a in agents.items()}
    with open(f"{OUT_DIR}/agents.json", "w") as f:
        json.dump(agents_export, f, indent=2)

    # Save the population baseline as its own artifact. This is the ONE
    # place population baseline is computed -- experiment.py and
    # demo_agent.py should both load this file rather than each
    # recomputing their own approximation from the flat feature CSVs.
    with open(f"{OUT_DIR}/population_baseline.json", "w") as f:
        json.dump(population_baseline, f, indent=2)

    sessions_export = {
        aid: dict(
            historical=[to_dict(s) for s in historical_sessions[aid]],
            evaluation=[to_dict(s) for s in evaluation_sessions[aid]],
        )
        for aid in agent_ids
    }
    with open(f"{OUT_DIR}/sessions.json", "w") as f:
        json.dump(sessions_export, f, indent=2)

    # -----------------------------------------------------------------
    # 8. Inspection report
    # -----------------------------------------------------------------
    write_inspection_report(agents, historical_sessions, evaluation_sessions)

    # -----------------------------------------------------------------
    # 9. Console summary
    # -----------------------------------------------------------------
    print("Generated:")
    print(f"  agents: {len(agents)} "
          f"(train={sum(a.split=='train' for a in agents.values())}, "
          f"val={sum(a.split=='validation' for a in agents.values())}, "
          f"test={sum(a.split=='test' for a in agents.values())})")
    print(f"  features_train.csv:      {len(df_train)} rows")
    print(f"  features_validation.csv: {len(df_val)} rows "
          f"(behavioral_anomaly={df_val['gt_behavioral_anomaly'].sum()}, "
          f"policy_violation={(~df_val['policy_passed']).sum()})")
    print(f"  features_test.csv:       {len(df_test)} rows "
          f"(behavioral_anomaly={df_test['gt_behavioral_anomaly'].sum()}, "
          f"policy_violation={(~df_test['policy_passed']).sum()})")
    print(f"  -> {OUT_DIR}/agents.json, {OUT_DIR}/sessions.json, "
          f"{OUT_DIR}/inspection_report.txt")


def write_inspection_report(agents, historical_sessions, evaluation_sessions):
    lines = []
    lines.append("=" * 78)
    lines.append("INSPECTION REPORT -- representative sessions")
    lines.append("=" * 78)

    def fmt_session(sess, agent):
        out = [f"\nSession {sess.session_id}  (agent={agent.agent_id}, "
               f"persona={agent.persona}, kind={sess.kind}, "
               f"difficulty={sess.difficulty}, drift_type={sess.drift_type})"]
        out.append(f"  mandate: primary={sess.mandate_snapshot.primary_category}  "
                   f"allowed={sess.mandate_snapshot.allowed_categories}  "
                   f"max_txn=Rs.{sess.mandate_snapshot.max_transaction}  "
                   f"session_budget=Rs.{sess.mandate_snapshot.session_budget}  "
                   f"velocity_limit={sess.mandate_snapshot.velocity_limit_per_10m}/10m")
        out.append(f"  baseline: price_mean=Rs.{agent.baseline.price_mean:.0f}  "
                   f"price_std=Rs.{agent.baseline.price_std:.0f}  "
                   f"gap_mean={agent.baseline.gap_mean_sec:.0f}s  "
                   f"n_hist_requests={agent.baseline.n_historical_requests}")
        for idx, r in enumerate(sess.requests):
            feats = extract_features(idx, sess.requests, sess.mandate_snapshot, agent.baseline)
            cum = sum(
                x.product.amount for x in sess.requests[: idx + 1] if x.step.type == "purchase"
            )
            req_10m = requests_in_window(idx, sess.requests, minutes=10)
            pol = policy_check(r, sess.mandate_snapshot, cum, req_10m)
            flag = "  <-- GT ANOMALY" if r.gt_behavioral_anomaly else ""
            pol_flag = f"  [POLICY VIOLATION: {','.join(v.rule for v in pol.violations)}]" if not pol.passed else ""
            out.append(
                f"    [{idx}] {r.timestamp}  {r.step.type:8s}  {r.product.category:18s} "
                f"Rs.{r.product.amount:8.0f}  z={feats['price_zscore']:+.2f}  "
                f"cat_dist={feats['category_distance']:.2f}  "
                f"spend_ratio={feats['cumulative_spend_ratio']:.2f}  "
                f"req_5m={feats['requests_last_5m']}  "
                f"gap={feats['time_since_previous_sec']:.0f}s{pol_flag}{flag}"
            )
        return "\n".join(out)

    # one normal historical session from a train agent
    train_agent = next(a for a in agents.values() if a.split == "train")
    lines.append(fmt_session(historical_sessions[train_agent.agent_id][0], train_agent))

    # one validation agent's full set of evaluation sessions: normal/easy/medium/hard
    val_agent = next(a for a in agents.values() if a.split == "validation")
    lines.append("\n" + "-" * 78)
    lines.append(f"Validation agent {val_agent.agent_id} ({val_agent.persona}) -- "
                 f"all four evaluation sessions side by side:")
    for sess in evaluation_sessions[val_agent.agent_id]:
        lines.append(fmt_session(sess, val_agent))

    with open(f"{OUT_DIR}/inspection_report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

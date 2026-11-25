"""Generate leak-free synthetic agent-trust data end to end.

Split is persona-stratified BY AGENT (80 train / 10 val / 10 test).
Validation and test CSVs contain ONLY unseen evaluation sessions (never the
historical sessions used to fit each agent's baseline), eliminating in-sample
baseline leakage.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from generator import (
    BASE_TIME,
    estimate_agent_baseline,
    estimate_population_baseline,
    extract_features,
    generate_agent,
    generate_normal_session,
    inject_drift,
    policy_check,
    requests_in_window,
)
from models import PERSONAS, to_dict

SEED = 42
N_HISTORICAL_SESSIONS = 10
N_NORMAL_EVAL_SESSIONS = 10
DRIFT_DIFFICULTIES = ["easy", "medium", "hard"]

OUT_DIR = Path(__file__).resolve().parent / "data"


def main():
    rng = np.random.default_rng(SEED)
    persona_names = list(PERSONAS.keys())

    val_per_persona = [3, 3, 2, 2]
    test_per_persona = [2, 2, 3, 3]

    agents = {}
    agent_ids = []
    idx_counter = 1
    for p_idx, persona in enumerate(persona_names):
        splits = (
            ["train"] * 20
            + ["validation"] * val_per_persona[p_idx]
            + ["test"] * test_per_persona[p_idx]
        )
        rng.shuffle(splits)
        for split in splits:
            aid = f"A{idx_counter:03d}"
            idx_counter += 1
            agent_ids.append(aid)
            agents[aid] = generate_agent(aid, persona, rng, split)

    historical_sessions: dict[str, list] = {aid: [] for aid in agent_ids}
    for agent_id, agent in agents.items():
        t = BASE_TIME
        for s in range(N_HISTORICAL_SESSIONS):
            sess = generate_normal_session(agent, f"{agent_id}-H{s:02d}", t, rng, kind="historical")
            historical_sessions[agent_id].append(sess)
            t = t + timedelta(hours=float(rng.uniform(3, 10)))

    train_agents = [agents[aid] for aid in agent_ids if agents[aid].split == "train"]
    population_baseline = estimate_population_baseline(train_agents, historical_sessions)

    for agent_id, agent in agents.items():
        agent.baseline = estimate_agent_baseline(historical_sessions[agent_id], population_baseline)

    evaluation_sessions: dict[str, list] = {aid: [] for aid in agent_ids}
    for agent_id, agent in agents.items():
        if agent.split == "train":
            continue
        t = datetime.fromisoformat(historical_sessions[agent_id][-1].requests[-1].timestamp) + timedelta(hours=2)
        for k in range(N_NORMAL_EVAL_SESSIONS):
            sess = generate_normal_session(agent, f"{agent_id}-E-normal-{k:02d}", t, rng, kind="evaluation")
            evaluation_sessions[agent_id].append(sess)
            t = t + timedelta(hours=float(rng.uniform(1, 4)))
        for difficulty in DRIFT_DIFFICULTIES:
            base_sess = generate_normal_session(agent, f"{agent_id}-E-{difficulty}", t, rng, kind="evaluation")
            drifted = inject_drift(base_sess, difficulty, rng, agent)
            evaluation_sessions[agent_id].append(drifted)
            t = t + timedelta(hours=float(rng.uniform(1, 4)))

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
                rows.append(dict(
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
                ))
        return rows

    train_rows, val_rows, test_rows = [], [], []
    for agent_id, agent in agents.items():
        if agent.split == "train":
            train_rows.extend(sessions_to_rows(historical_sessions[agent_id], agent))
        elif agent.split == "validation":
            val_rows.extend(sessions_to_rows(evaluation_sessions[agent_id], agent))
        else:
            test_rows.extend(sessions_to_rows(evaluation_sessions[agent_id], agent))

    df_train = pd.DataFrame(train_rows)
    df_val = pd.DataFrame(val_rows)
    df_test = pd.DataFrame(test_rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    df_train.to_csv(OUT_DIR / "features_train.csv", index=False)
    df_val.to_csv(OUT_DIR / "features_validation.csv", index=False)
    df_test.to_csv(OUT_DIR / "features_test.csv", index=False)

    with open(OUT_DIR / "agents.json", "w") as f:
        json.dump({aid: to_dict(a) for aid, a in agents.items()}, f, indent=2)
    with open(OUT_DIR / "population_baseline.json", "w") as f:
        json.dump(population_baseline, f, indent=2)
    with open(OUT_DIR / "sessions.json", "w") as f:
        json.dump(
            {
                aid: dict(
                    historical=[to_dict(s) for s in historical_sessions[aid]],
                    evaluation=[to_dict(s) for s in evaluation_sessions[aid]],
                )
                for aid in agent_ids
            },
            f,
            indent=2,
        )

    print("Generated leak-free dataset:")
    print(f"  agents: {len(agents)} (train=80, val=10, test=10, persona-stratified)")
    print(f"  features_train.csv:      {len(df_train)} rows")
    print(f"  features_validation.csv: {len(df_val)} rows (unseen eval only)")
    print(f"  features_test.csv:       {len(df_test)} rows (unseen eval only)")


if __name__ == "__main__":
    main()

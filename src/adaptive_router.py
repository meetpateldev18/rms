"""
Adaptive Routing Controller.

Motivation (see report/main.tex conclusion): multi-agent orchestration only
sometimes beats a single-prompt baseline — the win is conditional on model
capability and bug structure. This module turns that observation into a
system: a lightweight classifier, trained on bug-structure features
(features.py), predicts whether a given bug needs the expensive 6-agent
pipeline or whether the cheap single-prompt baseline will already fix it.

Two use modes:
  1. Offline cross-validated policy evaluation (`evaluate_offline`) — reuses
     the existing historical local-backend CSVs (baseline + multi-agent,
     already run for every problem in QuixBugs/HumanEvalFix/DebugBench) to
     measure, honestly and without leakage, how well a held-out router
     would have routed each bug. This is the headline result.
  2. Live routing (`predict_route` + `train_router`/`load_router`) — a
     trained model that can route a brand-new, never-seen problem at
     inference time, wired into run_adaptive.py.
"""

import os
import csv
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from config import RESULTS_DIR
import features as feat_mod

MODEL_PATH = os.path.join(RESULTS_DIR, "adaptive_router_model.joblib")

# (dataset, loader_module, loader_kwargs) — local backend only: every one of
# these already has both a multi-agent and a baseline CSV on disk.
DATASETS = ["quixbugs", "humanevalfix", "debugbench"]


def _load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {row["program"]: row for row in csv.DictReader(f)}


def _load_llm_calls(name):
    """Exact per-bug LLM call counts from the *_results_detailed.json sibling
    of a multi-agent results CSV. NOTE: `llm_calls` in that file is
    coordinator.py's running total across the whole experiment (self.llm.
    total_calls, not total_calls-at-entry), so later bugs carry every
    earlier bug's calls too — recover the per-bug count by diffing
    consecutive cumulative values in file order."""
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        detailed = json.load(f)
    out = {}
    prev_cum = 0
    for r in detailed:
        cum = r.get("llm_calls", prev_cum + 4 * int(r.get("attempts", 1)))
        out[r["program"]] = max(cum - prev_cum, 1)
        prev_cum = cum
    return out


def _load_problems(dataset):
    if dataset == "quixbugs":
        import load_quixbugs
        return load_quixbugs.load_problems()
    elif dataset == "humanevalfix":
        import load_humanevalfix
        return load_humanevalfix.load_problems("python")
    elif dataset == "debugbench":
        import load_debugbench
        return load_debugbench.load_problems()
    raise ValueError(dataset)


def build_training_table() -> pd.DataFrame:
    """One row per problem that has both a local multi-agent and local
    baseline result on disk, with static features + the routing label:
    needs_multiagent = 1 iff multi-agent succeeded AND baseline did NOT
    (i.e. the extra machinery was necessary, not just along for the ride)."""
    rows = []
    for dataset in DATASETS:
        multi = _load_csv(f"{dataset}_local_results.csv")
        base = _load_csv(f"{dataset}_local_baseline_results.csv")
        calls = _load_llm_calls(f"{dataset}_local_results_detailed.json")
        if not multi or not base:
            continue
        problems = {p.name: p for p in _load_problems(dataset)}
        for name, prob in problems.items():
            if name not in multi or name not in base:
                continue
            multi_success = multi[name]["success"] == "True"
            base_success = base[name]["success"] == "True"
            row = feat_mod.extract_features(prob)
            row["program"] = name
            row["multi_success"] = multi_success
            row["base_success"] = base_success
            row["multi_time"] = float(multi[name]["time_seconds"])
            row["base_time"] = float(base[name]["time_seconds"])
            row["multi_calls"] = calls.get(name, 4 * int(multi[name].get("attempts", 1)))
            row["base_calls"] = 1
            row["needs_multiagent"] = int(multi_success and not base_success)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_pipeline() -> Pipeline:
    # RandomForest chosen over LogisticRegression after comparing both by
    # cross-validated routed success rate / cost (see report): a shallow
    # forest exploits the nonlinear "buggy_pass_rate" signal far better
    # (routed success rate 57.8% @ 1.39 calls/bug vs. logreg's 55.1% @
    # 3.86) while logistic regression over-predicts the rare "needs
    # multiagent" class and barely beats chance (precision ~0.10 vs. the
    # 12.5% base rate).
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=5, class_weight="balanced", random_state=42,
        )),
    ])


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feat_mod.FEATURE_COLUMNS_NUMERIC].copy()
    for col in feat_mod.FEATURE_COLUMNS_CATEGORICAL:
        dummies = pd.get_dummies(df[col], prefix=col)
        X = pd.concat([X, dummies], axis=1)
    return X


def evaluate_offline(n_folds: int = 5, seed: int = 42) -> dict:
    """Stratified K-fold cross-validated offline policy evaluation. For each
    held-out fold, a router trained ONLY on the other folds predicts the
    route for every bug in the fold; we then look up that bug's ALREADY-
    MEASURED outcome under the predicted route (no re-running the LLM — this
    is standard offline policy evaluation, not live execution). Reports
    routed vs. always-baseline vs. always-multiagent vs. oracle success
    rate and total LLM-call cost, plus router classification quality."""
    df = build_training_table()
    if df.empty:
        raise RuntimeError("No historical local-backend results found to train on.")

    X = _feature_matrix(df)
    y = df["needs_multiagent"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    routed_success = 0
    routed_calls = 0
    fold_accs, fold_precs, fold_recs, fold_f1s = [], [], [], []
    per_dataset_routed = {d: {"success": 0, "total": 0, "calls": 0} for d in DATASETS}

    for train_idx, test_idx in skf.split(X, y):
        pipe = _build_pipeline()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds = pipe.predict(X.iloc[test_idx])

        fold_accs.append(accuracy_score(y[test_idx], preds))
        fold_precs.append(precision_score(y[test_idx], preds, zero_division=0))
        fold_recs.append(recall_score(y[test_idx], preds, zero_division=0))
        fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))

        for row_pos, pred in zip(test_idx, preds):
            row = df.iloc[row_pos]
            if pred == 1:
                success = row["multi_success"]
                calls = row["multi_calls"]
            else:
                success = row["base_success"]
                calls = row["base_calls"]
            routed_success += int(success)
            routed_calls += calls
            per_dataset_routed[row["dataset"]]["total"] += 1
            per_dataset_routed[row["dataset"]]["success"] += int(success)
            per_dataset_routed[row["dataset"]]["calls"] += calls

    n = len(df)
    always_base_success = df["base_success"].sum()
    always_multi_success = df["multi_success"].sum()
    always_base_calls = df["base_calls"].sum()
    always_multi_calls = df["multi_calls"].sum()
    oracle_success = (df["base_success"] | df["multi_success"]).sum()

    # Feature importances from a model fit on all data (reporting only —
    # the CV numbers above never let this model see held-out test rows).
    full_pipe = _build_pipeline()
    X_all, y_all = X, y
    full_pipe.fit(X_all, y_all)
    importances = dict(zip(X_all.columns, full_pipe.named_steps["clf"].feature_importances_))
    top_importances = dict(sorted(importances.items(), key=lambda kv: -kv[1])[:8])

    per_dataset_all = {}
    for d in DATASETS:
        sub = df[df["dataset"] == d]
        if sub.empty:
            continue
        per_dataset_all[d] = {
            "n": len(sub),
            "baseline_success_rate": round(sub["base_success"].mean() * 100, 1),
            "multiagent_success_rate": round(sub["multi_success"].mean() * 100, 1),
            "oracle_success_rate": round((sub["base_success"] | sub["multi_success"]).mean() * 100, 1),
            "routed_success_rate": per_dataset_routed[d]["success"] / per_dataset_routed[d]["total"] * 100,
            "routed_avg_calls": round(per_dataset_routed[d]["calls"] / per_dataset_routed[d]["total"], 2),
        }
        per_dataset_all[d]["routed_success_rate"] = round(per_dataset_all[d]["routed_success_rate"], 1)

    results = {
        "n_problems": int(n),
        "n_folds": n_folds,
        "router": {
            "accuracy": round(float(np.mean(fold_accs)), 3),
            "precision_needs_multiagent": round(float(np.mean(fold_precs)), 3),
            "recall_needs_multiagent": round(float(np.mean(fold_recs)), 3),
            "f1_needs_multiagent": round(float(np.mean(fold_f1s)), 3),
            "positive_rate": round(float(y.mean()), 3),
        },
        "routed": {
            "success": int(routed_success), "total": n,
            "success_rate": round(routed_success / n * 100, 1),
            "total_llm_calls": int(routed_calls),
            "avg_llm_calls": round(routed_calls / n, 2),
        },
        "always_baseline": {
            "success": int(always_base_success), "total": n,
            "success_rate": round(always_base_success / n * 100, 1),
            "total_llm_calls": int(always_base_calls),
            "avg_llm_calls": round(always_base_calls / n, 2),
        },
        "always_multiagent": {
            "success": int(always_multi_success), "total": n,
            "success_rate": round(always_multi_success / n * 100, 1),
            "total_llm_calls": int(always_multi_calls),
            "avg_llm_calls": round(always_multi_calls / n, 2),
        },
        "oracle": {
            "success": int(oracle_success), "total": n,
            "success_rate": round(oracle_success / n * 100, 1),
        },
        "per_dataset": per_dataset_all,
        "feature_importances": {k: round(float(v), 4) for k, v in top_importances.items()},
    }
    return results


def train_router(save: bool = True) -> Pipeline:
    """Train a single router on ALL historical data (for live deployment,
    not for the reported offline CV numbers above)."""
    df = build_training_table()
    X = _feature_matrix(df)
    y = df["needs_multiagent"].values
    pipe = _build_pipeline()
    pipe.fit(X, y)
    if save:
        joblib.dump({"pipeline": pipe, "columns": list(X.columns)}, MODEL_PATH)
    return pipe


def load_router():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained router at {MODEL_PATH} — run `python adaptive_router.py --train` first.")
    return joblib.load(MODEL_PATH)


def predict_route(problem, bundle=None) -> str:
    """Return 'multiagent' or 'baseline' for a live Problem instance."""
    bundle = bundle or load_router()
    row = feat_mod.extract_features(problem)
    X = pd.DataFrame([row])
    X_feat = _feature_matrix(X)
    # Align to training columns (missing categorical dummies -> 0, e.g. an
    # unseen dataset/language value at inference time).
    X_aligned = X_feat.reindex(columns=bundle["columns"], fill_value=0)
    pred = bundle["pipeline"].predict(X_aligned)[0]
    return "multiagent" if pred == 1 else "baseline"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save the live router model")
    parser.add_argument("--evaluate", action="store_true", help="Run the offline cross-validated policy evaluation")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if args.train:
        train_router()
        print(f"Router trained and saved to {MODEL_PATH}")

    if args.evaluate or not (args.train or args.evaluate):
        results = evaluate_offline(n_folds=args.folds)
        print(json.dumps(results, indent=2))
        out_path = os.path.join(RESULTS_DIR, "adaptive_router_evaluation.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""
Charts and comparison tables for the two new extensions: Adaptive Routing
Controller (adaptive_router.py) and Patch Tournament (tournament_coordinator.py
/ run_tournament.py). Mirrors analyze_results.py's palette and style for
visual consistency with the existing report figures.

Usage:
    cd src
    python analyze_new_methods.py
"""

import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

BLUE, ORANGE, GREEN, RED, PURPLE = "#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_csv(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


# ============================================================
# Adaptive Router charts
# ============================================================
def plot_router_overall(ev: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Always\nBaseline", "Always\nMulti-Agent", "Adaptive\nRouter", "Oracle\n(best of both)"]
    success = [ev["always_baseline"]["success_rate"], ev["always_multiagent"]["success_rate"],
               ev["routed"]["success_rate"], ev["oracle"]["success_rate"]]
    colors = [ORANGE, BLUE, GREEN, PURPLE]

    ax = axes[0]
    bars = ax.bar(labels, success, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, success):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val}%",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Repair Success Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    calls = [ev["always_baseline"]["avg_llm_calls"], ev["always_multiagent"]["avg_llm_calls"],
             ev["routed"]["avg_llm_calls"], None]
    bars = ax.bar(labels[:3], calls[:3], color=colors[:3], edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, calls[:3]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{val}",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Avg. LLM Calls / Bug", fontsize=12)
    ax.set_title("Cost (5-fold CV offline policy eval, n=303)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Adaptive Routing Controller vs. Fixed Strategies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_router_overall.png"), dpi=150)
    plt.close()
    print("  Saved: adaptive_router_overall.png")


def plot_router_per_dataset(ev: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    per_ds = ev["per_dataset"]
    datasets = list(per_ds.keys())
    disp = {"quixbugs": "QuixBugs", "humanevalfix": "HumanEvalFix", "debugbench": "DebugBench"}

    x = np.arange(len(datasets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    series = [
        ("Baseline", [per_ds[d]["baseline_success_rate"] for d in datasets], ORANGE),
        ("Multi-Agent", [per_ds[d]["multiagent_success_rate"] for d in datasets], BLUE),
        ("Adaptive Router", [per_ds[d]["routed_success_rate"] for d in datasets], GREEN),
        ("Oracle", [per_ds[d]["oracle_success_rate"] for d in datasets], PURPLE),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (label, vals, color), offset in zip(series, offsets):
        bars = ax.bar(x + offset, vals, width, label=label, color=color, edgecolor="black", linewidth=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([disp.get(d, d) for d in datasets])
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Adaptive Router: Per-Dataset Breakdown — Not a Uniform Win", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_router_per_dataset.png"), dpi=150)
    plt.close()
    print("  Saved: adaptive_router_per_dataset.png")


def plot_feature_importances(ev: dict):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    importances = ev["feature_importances"]
    names = list(importances.keys())
    vals = list(importances.values())
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color=BLUE, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Adaptive Router: Feature Importances", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "adaptive_router_feature_importances.png"), dpi=150)
    plt.close()
    print("  Saved: adaptive_router_feature_importances.png")


# ============================================================
# Patch Tournament charts
# ============================================================
TOURNAMENT_DATASETS = ["quixbugs", "humanevalfix", "debugbench"]


def tournament_metrics(rows):
    total = len(rows)
    if total == 0:
        return None
    success = sum(1 for r in rows if r["success"] == "True")
    agreements = [float(r["avg_agreement"]) for r in rows if r.get("avg_agreement")]
    any_pass = [float(r["any_candidate_passed_rate"]) for r in rows if r.get("any_candidate_passed_rate")]
    return {
        "total": total, "success": success,
        "success_rate": round(success / total * 100, 1),
        "avg_agreement": round(sum(agreements) / len(agreements), 3) if agreements else None,
        "any_candidate_passed_rate": round(sum(any_pass) / len(any_pass) * 100, 1) if any_pass else None,
    }


def plot_tournament_comparison():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    disp = {"quixbugs": "QuixBugs", "humanevalfix": "HumanEvalFix", "debugbench": "DebugBench"}

    datasets, tourn_vals, multi_vals, base_vals = [], [], [], []
    for d in TOURNAMENT_DATASETS:
        tourn_rows = load_csv(f"{d}_local_tournament_results.csv")
        multi_rows = load_csv(f"{d}_local_results.csv")
        base_rows = load_csv(f"{d}_local_baseline_results.csv")
        if not tourn_rows:
            continue
        t = tournament_metrics(tourn_rows)
        # Compare against the ORIGINAL multi-agent/baseline results restricted
        # to the same subset the tournament actually covered. NOTE: matched
        # by POSITION, not by "program" name — DebugBench's loader does not
        # guarantee unique names (the same LeetCode slug can be injected with
        # different bugs, e.g. "closest-dessert-cost__quadruple" appears 3x),
        # so a name-based join silently mismatches rows for ~9/119 DebugBench
        # problems. Both run_tournament.py and run_experiment.py iterate the
        # same deterministic, seeded problem list in the same order (verified:
        # 0 positional mismatches across all 3 datasets), so positional
        # slicing is the correct join here.
        n = len(tourn_rows)
        m_sub = multi_rows[:n]
        b_sub = base_rows[:n]
        m_succ = sum(1 for r in m_sub if r["success"] == "True")
        b_succ = sum(1 for r in b_sub if r["success"] == "True")
        datasets.append(d)
        tourn_vals.append(t["success_rate"])
        multi_vals.append(round(m_succ / len(m_sub) * 100, 1) if m_sub else None)
        base_vals.append(round(b_succ / len(b_sub) * 100, 1) if b_sub else None)

    if not datasets:
        print("  [skip] tournament comparison: no tournament results found yet")
        return

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, base_vals, width, label="Baseline", color=ORANGE, edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x, multi_vals, width, label="Multi-Agent (single draw)", color=BLUE, edgecolor="black", linewidth=0.6)
    bars3 = ax.bar(x + width, tourn_vals, width, label=f"Patch Tournament (K=3)", color=GREEN, edgecolor="black", linewidth=0.6)
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            if bar.get_height() is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([disp.get(d, d) for d in datasets])
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Patch Tournament vs. Single-Draw Multi-Agent vs. Baseline (local model)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "tournament_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: tournament_comparison.png")

    # Agreement-score distribution across all datasets combined
    all_agreements = []
    for d in datasets:
        rows = load_csv(f"{d}_local_tournament_results.csv")
        all_agreements.extend(float(r["avg_agreement"]) for r in rows if r.get("avg_agreement"))
    if all_agreements:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_agreements, bins=10, range=(0, 1), color=PURPLE, edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Candidate Agreement Score (1.0 = full consensus)", fontsize=12)
        ax.set_ylabel("Number of Bugs", fontsize=12)
        ax.set_title("Patch Tournament: Candidate Agreement Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "tournament_agreement_distribution.png"), dpi=150)
        plt.close()
        print("  Saved: tournament_agreement_distribution.png")

    return {d: {"tournament": t, "multi": m, "baseline": b}
            for d, t, m, b in zip(datasets, tourn_vals, multi_vals, base_vals)}


def main():
    print("=" * 60)
    print("NEW METHODS ANALYSIS: Adaptive Router + Patch Tournament")
    print("=" * 60)

    ev_path = os.path.join(RESULTS_DIR, "adaptive_router_evaluation.json")
    if os.path.exists(ev_path):
        with open(ev_path) as f:
            ev = json.load(f)
        print("\n-- Adaptive Router --")
        plot_router_overall(ev)
        plot_router_per_dataset(ev)
        plot_feature_importances(ev)
    else:
        print("\n[skip] adaptive router evaluation not found — run adaptive_router.py --evaluate first")

    print("\n-- Patch Tournament --")
    summary = plot_tournament_comparison()
    if summary:
        print(json.dumps(summary, indent=2))

    print("\nDone. Check results/figures/ for new charts.")


if __name__ == "__main__":
    main()

"""
Results analysis and chart generation.
Run after both run_experiment.py and baseline.py have completed.

Usage:
    cd src
    python analyze_results.py
"""

import os
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR


def load_csv(filename):
    """Load results from CSV file."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_metrics(results, label="System"):
    """Compute all evaluation metrics."""
    total = len(results)
    if total == 0:
        return {}

    success = sum(1 for r in results if r["success"] == "True" or r["success"] is True)
    success_rate = success / total * 100

    pass_at_1 = sum(1 for r in results
                    if (r["success"] == "True" or r["success"] is True)
                    and int(r["attempts"]) == 1)

    fixed = [r for r in results if r["success"] == "True" or r["success"] is True]
    avg_time = sum(float(r["time_seconds"]) for r in fixed) / len(fixed) if fixed else 0
    avg_attempts = sum(int(r["attempts"]) for r in fixed) / len(fixed) if fixed else 0
    total_time = sum(float(r["time_seconds"]) for r in results)

    metrics = {
        "label": label,
        "total": total,
        "success": success,
        "success_rate": round(success_rate, 1),
        "pass_at_1": pass_at_1,
        "pass_at_1_rate": round(pass_at_1 / total * 100, 1),
        "avg_time_fixed": round(avg_time, 1),
        "avg_attempts": round(avg_attempts, 1),
        "total_time": round(total_time, 1),
    }

    print(f"\n--- {label} ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def plot_comparison(multi_metrics, baseline_metrics):
    """Generate comparison bar charts."""
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Chart 1: Success Rate Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Multi-Agent\n(Proposed)", "Single-Prompt\n(Baseline)"]
    values = [multi_metrics["success_rate"], baseline_metrics["success_rate"]]
    colors = ["#2196F3", "#FF9800"]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Repair Success Rate Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "success_rate_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: success_rate_comparison.png")

    # Chart 2: Pass@1 Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [multi_metrics["pass_at_1_rate"], baseline_metrics["pass_at_1_rate"]]
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Pass@1 Rate (%)", fontsize=12)
    ax.set_title("Pass@1 Comparison (Fixed on First Attempt)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "pass_at_1_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: pass_at_1_comparison.png")

    # Chart 3: Per-program results (multi-agent)
    fig, ax = plt.subplots(figsize=(14, 5))
    multi_results = load_csv("results.csv")
    programs = [r["program"] for r in multi_results]
    successes = [1 if r["success"] == "True" else 0 for r in multi_results]
    color_map = ["#4CAF50" if s else "#F44336" for s in successes]
    ax.bar(range(len(programs)), [1]*len(programs), color=color_map, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(programs)))
    ax.set_xticklabels(programs, rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title("Per-Program Recovery Results (Green=Fixed, Red=Failed)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "per_program_results.png"), dpi=150)
    plt.close()
    print("  Saved: per_program_results.png")

    # Chart 4: Metrics Summary Table as figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table_data = [
        ["Metric", "Multi-Agent (Proposed)", "Single-Prompt (Baseline)"],
        ["Programs Tested", str(multi_metrics["total"]), str(baseline_metrics["total"])],
        ["Bugs Fixed", str(multi_metrics["success"]), str(baseline_metrics["success"])],
        ["Success Rate", f"{multi_metrics['success_rate']}%", f"{baseline_metrics['success_rate']}%"],
        ["Pass@1", f"{multi_metrics['pass_at_1']}/{multi_metrics['total']}", f"{baseline_metrics['pass_at_1']}/{baseline_metrics['total']}"],
        ["Avg Time (fixed)", f"{multi_metrics['avg_time_fixed']}s", f"{baseline_metrics['avg_time_fixed']}s"],
        ["Avg Attempts", str(multi_metrics["avg_attempts"]), "1.0"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Experimental Results Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "results_table.png"), dpi=150)
    plt.close()
    print("  Saved: results_table.png")


LOCAL_DATASETS = ["quixbugs", "humanevalfix", "debugbench"]


def plot_cross_dataset(per_dataset_metrics):
    """Grouped bar chart + summary table across all local-backend datasets."""
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    datasets = [d for d in LOCAL_DATASETS if d in per_dataset_metrics]
    multi_vals = [per_dataset_metrics[d]["multi"]["success_rate"] for d in datasets]
    base_vals = [per_dataset_metrics[d]["baseline"]["success_rate"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width/2, multi_vals, width, label="Multi-Agent (Proposed)", color="#2196F3", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, base_vals, width, label="Single-Prompt (Baseline)", color="#FF9800", edgecolor="black", linewidth=0.8)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("humanevalfix", "HumanEvalFix").replace("quixbugs", "QuixBugs").replace("debugbench", "DebugBench") for d in datasets])
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Repair Success Rate Across Benchmarks (local model)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cross_dataset_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: cross_dataset_comparison.png")

    # Summary table across all datasets
    fig, ax = plt.subplots(figsize=(11, 2 + len(datasets) * 0.8))
    ax.axis("off")
    table_data = [["Dataset", "Multi-Agent Success", "Baseline Success", "Multi-Agent Pass@1", "Baseline Pass@1"]]
    for d in datasets:
        m, b = per_dataset_metrics[d]["multi"], per_dataset_metrics[d]["baseline"]
        table_data.append([
            d, f"{m['success']}/{m['total']} ({m['success_rate']}%)", f"{b['success']}/{b['total']} ({b['success_rate']}%)",
            f"{m['pass_at_1_rate']}%", f"{b['pass_at_1_rate']}%",
        ])
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.7)
    for j in range(len(table_data[0])):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Cross-Dataset Results Summary (local model)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "cross_dataset_results_table.png"), dpi=150)
    plt.close()
    print("  Saved: cross_dataset_results_table.png")


def analyze_local():
    """Analyze the new local-backend runs across QuixBugs, HumanEvalFix, DebugBench."""
    per_dataset_metrics = {}
    for dataset in LOCAL_DATASETS:
        multi = load_csv(f"{dataset}_local_results.csv")
        base = load_csv(f"{dataset}_local_baseline_results.csv")
        if not multi or not base:
            print(f"  [skip] {dataset}: missing local results (run run_experiment.py/baseline.py --dataset {dataset})")
            continue
        per_dataset_metrics[dataset] = {
            "multi": compute_metrics(multi, f"{dataset} — Multi-Agent (local)"),
            "baseline": compute_metrics(base, f"{dataset} — Baseline (local)"),
        }

    if not per_dataset_metrics:
        print("\nNo local-backend results found yet.")
        return

    print("\nGenerating cross-dataset charts...")
    plot_cross_dataset(per_dataset_metrics)
    return per_dataset_metrics


def analyze_gemini():
    """Analyze the Gemini-backend runs across HumanEvalFix + DebugBench
    (QuixBugs/Gemini is the original results.csv/baseline_results.csv)."""
    per_dataset_metrics = {}
    for dataset in ["humanevalfix", "debugbench"]:
        multi = load_csv(f"{dataset}_gemini_results.csv")
        base = load_csv(f"{dataset}_gemini_baseline_results.csv")
        if not multi or not base:
            print(f"  [skip] {dataset}: missing gemini results")
            continue
        per_dataset_metrics[dataset] = {
            "multi": compute_metrics(multi, f"{dataset} — Multi-Agent (gemini)"),
            "baseline": compute_metrics(base, f"{dataset} — Baseline (gemini)"),
        }
    return per_dataset_metrics


def plot_backend_comparison(local_metrics, gemini_metrics):
    """Grouped bar chart: local vs Gemini, multi-agent vs baseline, per dataset."""
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    quixbugs_gemini = {
        "multi": compute_metrics(load_csv("results.csv"), "quixbugs — Multi-Agent (gemini, original)"),
        "baseline": compute_metrics(load_csv("baseline_results.csv"), "quixbugs — Baseline (gemini, original)"),
    }
    gemini_all = {"quixbugs": quixbugs_gemini, **gemini_metrics}

    datasets = [d for d in LOCAL_DATASETS if d in local_metrics and d in gemini_all]
    if not datasets:
        print("  [skip] backend comparison: need both local and gemini results for at least one dataset")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.2
    x = np.arange(len(datasets))
    series = [
        ("Local Multi-Agent", [local_metrics[d]["multi"]["success_rate"] for d in datasets], "#2196F3", 0.55),
        ("Local Baseline", [local_metrics[d]["baseline"]["success_rate"] for d in datasets], "#FF9800", 0.55),
        ("Gemini Multi-Agent", [gemini_all[d]["multi"]["success_rate"] for d in datasets], "#2196F3", 1.0),
        ("Gemini Baseline", [gemini_all[d]["baseline"]["success_rate"] for d in datasets], "#FF9800", 1.0),
    ]
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (label, vals, color, alpha), offset in zip(series, offsets):
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=alpha, edgecolor="black", linewidth=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("humanevalfix", "HumanEvalFix").replace("quixbugs", "QuixBugs").replace("debugbench", "DebugBench") for d in datasets])
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Multi-Agent vs Baseline — Local Model vs Gemini", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "backend_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: backend_comparison.png")


def main():
    print("=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)

    multi_results = load_csv("results.csv")
    baseline_results = load_csv("baseline_results.csv")

    if multi_results and baseline_results:
        multi_metrics = compute_metrics(multi_results, "Multi-Agent System (QuixBugs/Gemini)")
        baseline_metrics = compute_metrics(baseline_results, "Single-Prompt Baseline (QuixBugs/Gemini)")
        print("\nGenerating QuixBugs/Gemini charts...")
        plot_comparison(multi_metrics, baseline_metrics)
    else:
        print("\nNo original QuixBugs/Gemini results found — skipping those charts.")

    print("\n" + "-" * 60)
    print("LOCAL-BACKEND CROSS-DATASET ANALYSIS")
    print("-" * 60)
    local_metrics = analyze_local()

    print("\n" + "-" * 60)
    print("GEMINI-BACKEND ANALYSIS (HumanEvalFix + DebugBench)")
    print("-" * 60)
    gemini_metrics = analyze_gemini()

    if local_metrics and gemini_metrics:
        print("\nGenerating local-vs-Gemini backend comparison chart...")
        plot_backend_comparison(local_metrics, gemini_metrics)

    print("\nDone! Check results/figures/ for all charts.")


if __name__ == "__main__":
    main()

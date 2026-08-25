"""
Main experiment runner.
Runs the multi-agent recovery system on a dataset (QuixBugs, HumanEvalFix, or DebugBench).

Usage:
    cd src
    python run_experiment.py --dataset quixbugs
    python run_experiment.py --dataset humanevalfix
    python run_experiment.py --dataset debugbench --limit 10   # pilot run
"""

import os
import json
import csv
import argparse

from config import RESULTS_DIR, LLM_BACKEND
from llm_factory import get_llm_client
from coordinator import RecoveryCoordinator
import executor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["quixbugs", "humanevalfix", "debugbench"], default="quixbugs")
    parser.add_argument("--language", choices=["python", "java", "cpp", "javascript"], default="python",
                         help="Only humanevalfix currently supports non-python languages")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems (pilot runs)")
    return parser.parse_args()


def load_problems(dataset, language="python"):
    if dataset == "quixbugs":
        import load_quixbugs
        return load_quixbugs.load_problems()
    elif dataset == "humanevalfix":
        import load_humanevalfix
        return load_humanevalfix.load_problems(language)
    elif dataset == "debugbench":
        import load_debugbench
        return load_debugbench.load_problems()
    raise ValueError(f"Unknown dataset: {dataset}")


def run_experiment(dataset, limit=None, language="python"):
    print("=" * 60)
    print(f"FAILURE RECOVERY SYSTEM - Multi-Agent Experiment ({dataset}, {language}, backend={LLM_BACKEND})")
    print("=" * 60)

    problems = load_problems(dataset, language)
    if limit:
        problems = problems[:limit]
    print(f"\nLoaded {len(problems)} problems.\n")

    llm = get_llm_client()
    coordinator = RecoveryCoordinator(llm)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Distinct filenames per dataset+backend — never collides with the original
    # Gemini/QuixBugs results.csv / baseline_results.csv already cited in the paper.
    stem = f"{dataset}_{LLM_BACKEND}" if language == "python" else f"{dataset}_{language}_{LLM_BACKEND}"
    csv_path = os.path.join(RESULTS_DIR, f"{stem}_results.csv")
    json_path = os.path.join(RESULTS_DIR, f"{stem}_results_detailed.json")
    fieldnames = ["program", "success", "attempts", "time_seconds", "error_type"]

    completed = set()
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["program"])
                results.append(row)
        print(f"\nResuming: {len(completed)} problems already completed, skipping them.")

    total_success = sum(1 for r in results if r.get("success") == "True" or r.get("success") is True)

    for i, prob in enumerate(problems):
        if prob.name in completed:
            print(f"\n[{i+1}/{len(problems)}] {prob.name} — SKIPPED (already done)")
            continue

        print(f"\n[{i+1}/{len(problems)}] {prob.name}")
        # NOTE: coordinator.recover() must run on the main thread — its agents call
        # executor.run_with_timeout(), which uses signal.alarm() and only works there.
        # Each LLM call is already bounded by LocalClient's (connect, read) timeout,
        # and each test execution by EXEC_TIMEOUT, so no outer watchdog is needed here.
        result = coordinator.recover(
            program_name=prob.name,
            buggy_code=prob.buggy_code,
            problem=prob,
        )

        if result["success"]:
            total_success += 1

        print(f"  Running total: {total_success}/{i+1} fixed")

        row = {
            "program": result["program"],
            "success": result["success"],
            "attempts": result["attempts"],
            "time_seconds": result["time_seconds"],
            "error_type": result["error_type"],
        }
        results.append(row)

        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        detailed = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    detailed = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                detailed = []
        detailed.append(result)
        with open(json_path, "w") as f:
            json.dump(detailed, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset} | Backend: {LLM_BACKEND}")
    print(f"Total problems: {len(results)}")
    print(f"Successfully fixed: {total_success}")
    if results:
        print(f"Success rate: {total_success/len(results)*100:.1f}%")

    pass_at_1 = sum(1 for r in results if str(r["success"]) == "True" and str(r["attempts"]) == "1")
    if results:
        print(f"Pass@1: {pass_at_1}/{len(results)} ({pass_at_1/len(results)*100:.1f}%)")

    fixed = [r for r in results if str(r["success"]) == "True"]
    if fixed:
        avg_time = sum(float(r["time_seconds"]) for r in fixed) / len(fixed)
        avg_attempts = sum(int(r["attempts"]) for r in fixed) / len(fixed)
        print(f"Avg time (fixed bugs): {avg_time:.1f}s")
        print(f"Avg attempts (fixed bugs): {avg_attempts:.1f}")

    print(f"\nTotal LLM calls: {llm.total_calls}")
    print(f"Results saved to: {csv_path}")
    print(f"Detailed log saved to: {json_path}")


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.dataset, args.limit, args.language)

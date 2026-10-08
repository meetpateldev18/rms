"""
Experiment runner for the Patch Tournament coordinator (tournament_coordinator.py).
Mirrors run_experiment.py's CLI/resume/CSV-writing conventions exactly, so
results are directly comparable to the existing multi-agent/baseline runs.

Usage:
    cd src
    python run_tournament.py --dataset quixbugs --k 3
    python run_tournament.py --dataset humanevalfix --limit 40 --k 3
"""

import os
import json
import csv
import argparse

from config import RESULTS_DIR, LLM_BACKEND
from llm_factory import get_llm_client
from tournament_coordinator import TournamentCoordinator
import executor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["quixbugs", "humanevalfix", "debugbench"], default="quixbugs")
    parser.add_argument("--language", choices=["python", "java", "cpp", "javascript"], default="python")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems (pilot/budget runs)")
    parser.add_argument("--k", type=int, default=3, help="Number of patch candidates per attempt")
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


def run_tournament(dataset, limit=None, language="python", k=3):
    print("=" * 60)
    print(f"PATCH TOURNAMENT - K={k} Multi-Candidate Experiment ({dataset}, {language}, backend={LLM_BACKEND})")
    print("=" * 60)

    problems = load_problems(dataset, language)
    if limit:
        problems = problems[:limit]
    print(f"\nLoaded {len(problems)} problems.\n")

    llm = get_llm_client()
    coordinator = TournamentCoordinator(llm, k=k)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = f"{dataset}_{LLM_BACKEND}" if language == "python" else f"{dataset}_{language}_{LLM_BACKEND}"
    csv_path = os.path.join(RESULTS_DIR, f"{stem}_tournament_results.csv")
    json_path = os.path.join(RESULTS_DIR, f"{stem}_tournament_results_detailed.json")
    fieldnames = ["program", "success", "attempts", "time_seconds", "error_type", "avg_agreement", "any_candidate_passed_rate"]

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
        result = coordinator.recover(program_name=prob.name, buggy_code=prob.buggy_code, problem=prob)

        if result["success"]:
            total_success += 1
        print(f"  Running total: {total_success}/{i+1} fixed")

        agreements = result.get("agreement_per_attempt", [])
        passed_counts = result.get("candidates_passed_per_attempt", [])
        row = {
            "program": result["program"],
            "success": result["success"],
            "attempts": result["attempts"],
            "time_seconds": result["time_seconds"],
            "error_type": result["error_type"],
            "avg_agreement": round(sum(agreements) / len(agreements), 3) if agreements else "",
            "any_candidate_passed_rate": round(sum(1 for c in passed_counts if c > 0) / len(passed_counts), 3) if passed_counts else "",
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
    print("TOURNAMENT SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset} | Backend: {LLM_BACKEND} | K: {k}")
    print(f"Total problems: {len(results)}")
    print(f"Successfully fixed: {total_success}")
    if results:
        print(f"Success rate: {total_success/len(results)*100:.1f}%")
    print(f"Total LLM calls: {llm.total_calls}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_tournament(args.dataset, args.limit, args.language, args.k)

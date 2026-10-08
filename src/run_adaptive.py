"""
Live execution of the Adaptive Routing Controller (adaptive_router.py).

Unlike adaptive_router.py's --evaluate mode (which backtests routing
decisions against ALREADY-MEASURED historical outcomes for honest, leak-free
reported numbers — see report), this script actually drives the system: for
each problem, it extracts zero-LLM-cost features, asks the trained router
for a route, and then genuinely executes that route — a single-prompt call
(mirroring baseline.py) for "baseline", or the full 6-agent coordinator for
"multiagent" — end to end, writing fresh results. This is the real
production entry point a user would call on brand-new, never-before-seen
buggy programs; the CV backtest is what makes the *reported* success-rate
and cost numbers trustworthy without re-spending thousands of LLM calls
re-deriving a result the historical data already contains.

Usage:
    cd src
    python adaptive_router.py --train          # fit + save the router once
    python run_adaptive.py --dataset quixbugs --limit 10
"""

import os
import csv
import time
import argparse

from config import RESULTS_DIR, LLM_BACKEND
from llm_factory import get_llm_client
from coordinator import RecoveryCoordinator
import adaptive_router
import executor
from baseline import _example_call_hint, _LANG_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["quixbugs", "humanevalfix", "debugbench"], default="quixbugs")
    parser.add_argument("--language", choices=["python", "java", "cpp", "javascript"], default="python")
    parser.add_argument("--limit", type=int, default=None)
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
    raise ValueError(dataset)


def _run_baseline_once(llm, problem):
    """Single-prompt repair — same prompt construction as baseline.py."""
    lang_name = _LANG_NAMES.get(problem.language, "Python")
    prompt = f"""Fix the bug in this {lang_name} function. Return ONLY the corrected {lang_name} code, nothing else. No explanations, no markdown, no backticks.

Buggy code:
{problem.buggy_code}

{_example_call_hint(problem)}"""
    patched, hard_timeout_err = executor.run_with_hard_timeout(llm.generate, args=(prompt,), timeout=90)
    patched = patched or ""
    cleaned = patched.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    validation = executor.run_tests(problem, cleaned)
    return validation["all_passed"], 1


def run_adaptive(dataset, limit=None, language="python"):
    print("=" * 60)
    print(f"ADAPTIVE ROUTING - Live Execution ({dataset}, {language}, backend={LLM_BACKEND})")
    print("=" * 60)

    bundle = adaptive_router.load_router()
    problems = load_problems(dataset, language)
    if limit:
        problems = problems[:limit]
    print(f"\nLoaded {len(problems)} problems.\n")

    llm = get_llm_client()
    coordinator = RecoveryCoordinator(llm)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = f"{dataset}_{LLM_BACKEND}" if language == "python" else f"{dataset}_{language}_{LLM_BACKEND}"
    csv_path = os.path.join(RESULTS_DIR, f"{stem}_adaptive_results.csv")
    fieldnames = ["program", "route", "success", "attempts", "time_seconds", "llm_calls"]

    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            completed = {row["program"] for row in csv.DictReader(f)}
        print(f"Resuming: {len(completed)} already done.")

    total_success, total_calls_used = 0, 0
    for i, prob in enumerate(problems):
        if prob.name in completed:
            continue
        route = adaptive_router.predict_route(prob, bundle)
        start = time.time()
        calls_before = llm.total_calls

        if route == "baseline":
            success, attempts = _run_baseline_once(llm, prob)
        else:
            result = coordinator.recover(program_name=prob.name, buggy_code=prob.buggy_code, problem=prob)
            success, attempts = result["success"], result["attempts"]

        elapsed = round(time.time() - start, 2)
        calls_used = llm.total_calls - calls_before
        total_success += int(success)
        total_calls_used += calls_used

        print(f"[{i+1}/{len(problems)}] {prob.name}: routed={route}, success={success}, calls={calls_used}, time={elapsed}s")

        row = {"program": prob.name, "route": route, "success": success,
               "attempts": attempts, "time_seconds": elapsed, "llm_calls": calls_used}
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

    print(f"\nDone. {total_success} fixed, {total_calls_used} total LLM calls. Saved to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_adaptive(args.dataset, args.limit, args.language)

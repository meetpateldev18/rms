"""
Baseline: Single-prompt LLM repair (no multi-agent, no iteration).
Used for comparison against the multi-agent approach.

Usage:
    cd src
    python baseline.py --dataset quixbugs
    python baseline.py --dataset humanevalfix
    python baseline.py --dataset debugbench --limit 10   # pilot run
"""

import os
import json
import csv
import time
import argparse

from config import RESULTS_DIR, LLM_BACKEND
from llm_factory import get_llm_client
import executor


_LANG_NAMES = {"python": "Python", "java": "Java", "cpp": "C++", "javascript": "JavaScript"}


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


def _example_call_hint(problem):
    """Build a one-line hint showing the LLM how the entry point is called,
    since the prompt format differs per dataset/test_style."""
    if problem.test_style == "tuple":
        inputs, expected = problem.test_cases[0]
        return f"Example: {problem.name}({', '.join(map(repr, inputs))}) should return {repr(expected)}"
    elif problem.test_style == "assert":
        return f"The function must be named `{problem.entry_point}` and satisfy the provided test suite."
    elif problem.test_style == "example_io":
        kwargs, expected = problem.examples[0]
        args_str = ", ".join(repr(v) for v in kwargs.values())
        return f"Example: {problem.class_name}().{problem.entry_point}({args_str}) should return {repr(expected)}"
    return ""


def run_baseline(dataset, limit=None, language="python"):
    print("=" * 60)
    print(f"BASELINE - Single-Prompt LLM Repair ({dataset}, {language}, backend={LLM_BACKEND})")
    print("=" * 60)

    problems = load_problems(dataset, language)
    if limit:
        problems = problems[:limit]
    print(f"\nLoaded {len(problems)} problems.\n")

    llm = get_llm_client()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = f"{dataset}_{LLM_BACKEND}" if language == "python" else f"{dataset}_{language}_{LLM_BACKEND}"
    csv_path = os.path.join(RESULTS_DIR, f"{stem}_baseline_results.csv")
    fieldnames = ["program", "success", "attempts", "time_seconds", "passed", "total"]

    completed = set()
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["program"])
                results.append(row)
        print(f"\nResuming: {len(completed)} problems already completed, skipping them.")

    total_success = sum(1 for r in results if str(r.get("success")) == "True")

    for i, prob in enumerate(problems):
        if prob.name in completed:
            print(f"\n[{i+1}/{len(problems)}] {prob.name} — SKIPPED (already done)")
            continue

        start_time = time.time()
        print(f"\n[{i+1}/{len(problems)}] {prob.name}")

        lang_name = _LANG_NAMES.get(prob.language, "Python")
        prompt = f"""Fix the bug in this {lang_name} function. Return ONLY the corrected {lang_name} code, nothing else. No explanations, no markdown, no backticks.

Buggy code:
{prob.buggy_code}

{_example_call_hint(prob)}"""

        patched, hard_timeout_err = executor.run_with_hard_timeout(llm.generate, args=(prompt,), timeout=90)
        if hard_timeout_err:
            print(f"  [HARD TIMEOUT] {hard_timeout_err}")
            patched = ""
        cleaned = patched.strip() if patched else ""
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        validation = executor.run_tests(prob, cleaned)
        elapsed = round(time.time() - start_time, 2)

        success = validation["all_passed"]
        if success:
            total_success += 1

        row = {
            "program": prob.name,
            "success": success,
            "attempts": 1,
            "time_seconds": elapsed,
            "passed": validation["passed"],
            "total": validation["total_tests"],
        }
        results.append(row)

        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        status = "FIXED" if success else "FAILED"
        print(f"  [{status}] {validation['passed']}/{validation['total_tests']} tests passed ({elapsed}s)")
        print(f"  Running total: {total_success}/{i+1}")

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset} | Backend: {LLM_BACKEND}")
    print(f"Total problems: {len(results)}")
    print(f"Successfully fixed: {total_success}")
    if results:
        print(f"Success rate: {total_success/len(results)*100:.1f}%")
    print(f"Total LLM calls: {llm.total_calls}")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_baseline(args.dataset, args.limit, args.language)

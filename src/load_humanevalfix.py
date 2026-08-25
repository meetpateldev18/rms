"""
Loads the HumanEvalFix benchmark (bigcode/humanevalpack) into Problem objects.
164 hand-written programs per language, each with one implanted bug and a
full assert-based test suite (test_style="assert").

Supports python (in-process exec), and java/cpp/javascript (subprocess
compile-and-run via executor.py) — same dataset schema across all four
configs, just different source syntax and a different HF config name.
"""

import os
import json

from config import HUMANEVALFIX_DIR

# Our internal language key -> bigcode/humanevalpack config name
_HF_CONFIG = {
    "python": "python",
    "java": "java",
    "cpp": "cpp",
    "javascript": "js",
}


def _cache_file(language):
    return os.path.join(HUMANEVALFIX_DIR, f"humanevalfix_{language}.json")


def _download(language):
    from datasets import load_dataset
    ds = load_dataset("bigcode/humanevalpack", _HF_CONFIG[language], split="test")
    rows = []
    for row in ds:
        rows.append({
            "name": row["task_id"].replace("/", "_"),
            "buggy_code": row["import"] + row["declaration"] + row["buggy_solution"],
            "test_code": row["test"],
            "entry_point": row["entry_point"],
            "bug_type": row["bug_type"],
        })
    cache_file = _cache_file(language)
    os.makedirs(HUMANEVALFIX_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def load_problems(language="python"):
    """Load all 164 HumanEvalFix problems for the given language
    (downloads + caches on first run per language)."""
    from problem import Problem

    if language not in _HF_CONFIG:
        raise ValueError(f"Unsupported language for HumanEvalFix: {language!r} (supported: {list(_HF_CONFIG)})")

    cache_file = _cache_file(language)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            rows = json.load(f)
    else:
        print(f"[load_humanevalfix] Downloading bigcode/humanevalpack ({_HF_CONFIG[language]}) from Hugging Face...")
        rows = _download(language)
        print(f"[load_humanevalfix] Cached {len(rows)} problems to {cache_file}")

    problems = []
    for row in rows:
        problems.append(Problem(
            name=row["name"],
            buggy_code=row["buggy_code"],
            test_style="assert",
            dataset="humanevalfix",
            test_code=row["test_code"],
            entry_point=row["entry_point"],
            language=language,
        ))
    return problems


if __name__ == "__main__":
    import sys
    lang = sys.argv[1] if len(sys.argv) > 1 else "python"
    problems = load_problems(lang)
    print(f"Loaded {len(problems)} problems ({lang}).")
    p = problems[0]
    print(f"\nExample: {p.name}")
    print(f"Buggy code:\n{p.buggy_code}")
    print(f"Entry point: {p.entry_point}")

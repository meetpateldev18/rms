"""
Loads the DebugBench Python subset (Rtian/DebugBench on Hugging Face) into
Problem objects (test_style="example_io").

DebugBench's public release only ships 1-3 example Input/Output pairs per
problem (taken from the original LeetCode problem statement), not the full
hidden test suite used by LeetCode's live judge (which would require an
authenticated LeetCode session to query — not something we automate here).
We parse those public examples into executable test cases against the
LeetCode-style `Solution` class. This is a known fidelity limitation,
documented in the results/report.

Problems whose examples can't be parsed into valid Python literals are
skipped (logged). The remaining pool is stratified-sampled by bug `category`
down to ~120 problems with a fixed seed for reproducibility.
"""

import os
import re
import ast
import json
import random

from config import DEBUGBENCH_DIR

CACHE_FILE = os.path.join(DEBUGBENCH_DIR, "debugbench_parsed.json")
SAMPLE_SIZE = 120
SEED = 42


def _split_top_level(s, sep=","):
    """Split on `sep` but only at bracket/quote depth 0."""
    parts = []
    depth = 0
    current = ""
    in_str = None
    for i, c in enumerate(s):
        if in_str:
            current += c
            if c == in_str and (i == 0 or s[i - 1] != "\\"):
                in_str = None
        elif c in ("'", '"'):
            in_str = c
            current += c
        elif c in "([{":
            depth += 1
            current += c
        elif c in ")]}":
            depth -= 1
            current += c
        elif c == sep and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += c
    if current.strip():
        parts.append(current)
    return parts


def _normalize_literal(s):
    s = s.strip()
    s = re.sub(r'\btrue\b', 'True', s)
    s = re.sub(r'\bfalse\b', 'False', s)
    s = re.sub(r'\bnull\b', 'None', s)
    return s


def _parse_example(example_str):
    """Parse one 'Input: ...\\nOutput: ...' string into (kwargs, expected)."""
    m_in = re.search(r'Input:\s*(.*?)(?:\n\s*Output:|$)', example_str, re.S)
    m_out = re.search(r'Output:\s*(.*?)(?:\n\s*Explanation:|$)', example_str, re.S)
    if not m_in or not m_out:
        return None

    kwargs = {}
    for chunk in _split_top_level(m_in.group(1).strip(), ","):
        if "=" not in chunk:
            return None
        key, val = chunk.split("=", 1)
        try:
            kwargs[key.strip()] = ast.literal_eval(_normalize_literal(val))
        except (ValueError, SyntaxError):
            return None

    try:
        expected = ast.literal_eval(_normalize_literal(m_out.group(1).strip()))
    except (ValueError, SyntaxError):
        return None

    return kwargs, expected


def _extract_methods(code):
    """Return (class_name, [(method_name, [param_names]), ...])."""
    class_match = re.search(r'class\s+(\w+)', code)
    class_name = class_match.group(1) if class_match else None

    methods = []
    for m in re.finditer(r'def\s+(\w+)\s*\(\s*self\s*(?:,\s*([^)]*))?\)', code):
        name = m.group(1)
        params = []
        if m.group(2) and m.group(2).strip():
            for p in m.group(2).split(","):
                p = p.strip()
                if p:
                    params.append(p.split(":")[0].split("=")[0].strip())
        methods.append((name, params))
    return class_name, methods


def _select_entry_point(methods, kwarg_keys):
    kwarg_set = set(kwarg_keys)
    non_init = [(n, p) for n, p in methods if n != "__init__"]

    exact = [n for n, p in non_init if set(p) == kwarg_set]
    if len(exact) == 1:
        return exact[0]

    same_count = [n for n, p in non_init if len(p) == len(kwarg_keys)]
    if same_count:
        return same_count[-1]

    return non_init[-1][0] if non_init else None


def _download_and_parse():
    from datasets import load_dataset

    ds = load_dataset("Rtian/DebugBench")["test"]
    py_rows = [r for r in ds if r["language"] == "python3"]
    print(f"[load_debugbench] {len(py_rows)} python3 rows in the full dataset")

    parsed, skipped = [], 0
    for row in py_rows:
        # TreeNode/ListNode params are serialized as plain lists in the example text
        # (e.g. "root = [1,2,3]") — we can't reliably reconstruct the actual node
        # structure LeetCode expects, so these problems are excluded rather than
        # silently mis-scored.
        if "TreeNode" in row["buggy_code"] or "ListNode" in row["buggy_code"]:
            skipped += 1
            continue

        examples = []
        for ex_str in row["examples"]:
            parsed_ex = _parse_example(ex_str)
            if parsed_ex:
                examples.append(parsed_ex)
        if not examples:
            skipped += 1
            continue

        class_name, methods = _extract_methods(row["buggy_code"])
        if not class_name or not methods:
            skipped += 1
            continue

        entry_point = _select_entry_point(methods, examples[0][0].keys())
        if not entry_point:
            skipped += 1
            continue

        parsed.append({
            "name": row["slug"] + "__" + row["subtype"].replace(" ", "_"),
            "buggy_code": row["buggy_code"],
            "class_name": class_name,
            "entry_point": entry_point,
            "category": row["category"],
            "examples": [[kw, exp] for kw, exp in examples],
        })

    print(f"[load_debugbench] Parsed {len(parsed)}/{len(py_rows)} problems "
          f"({skipped} skipped — examples/class/entry_point unparsable)")

    os.makedirs(DEBUGBENCH_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(parsed, f, indent=2)
    return parsed


def _stratified_sample(rows, size, seed):
    rng = random.Random(seed)
    by_category = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r)

    for bucket in by_category.values():
        bucket.sort(key=lambda r: r["name"])
        rng.shuffle(bucket)

    total = len(rows)
    sample = []
    for category, bucket in by_category.items():
        quota = max(1, round(size * len(bucket) / total))
        sample.extend(bucket[:quota])

    sample.sort(key=lambda r: r["name"])
    return sample[:size]


def load_problems(sample_size=SAMPLE_SIZE):
    from problem import Problem

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            rows = json.load(f)
    else:
        print("[load_debugbench] Downloading Rtian/DebugBench from Hugging Face...")
        rows = _download_and_parse()

    sample = _stratified_sample(rows, sample_size, SEED)
    print(f"[load_debugbench] Stratified sample: {len(sample)}/{len(rows)} problems")

    problems = []
    for row in sample:
        problems.append(Problem(
            name=row["name"],
            buggy_code=row["buggy_code"],
            test_style="example_io",
            dataset="debugbench",
            examples=[(kw, exp) for kw, exp in row["examples"]],
            entry_point=row["entry_point"],
            class_name=row["class_name"],
        ))
    return problems


if __name__ == "__main__":
    problems = load_problems()
    print(f"\nLoaded {len(problems)} problems.")
    p = problems[0]
    print(f"\nExample: {p.name}")
    print(f"Class: {p.class_name}, entry_point: {p.entry_point}")
    print(f"Examples: {p.examples}")

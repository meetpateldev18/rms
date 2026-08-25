"""
Loads QuixBugs programs (Python, with JSON test cases) into Problem objects.
Shared by run_experiment.py and baseline.py (previously duplicated in both).
"""

import os
import json

from config import DATASET_DIR
from problem import Problem


def load_problems():
    """Load all QuixBugs programs that have JSON test cases."""
    problems = []

    json_dir = os.path.join(DATASET_DIR, "json_testcases")
    buggy_dir = os.path.join(DATASET_DIR, "python_programs")

    for json_file in sorted(os.listdir(json_dir)):
        if not json_file.endswith(".json"):
            continue

        program_name = json_file.replace(".json", "")
        buggy_file = os.path.join(buggy_dir, f"{program_name}.py")

        if not os.path.exists(buggy_file):
            print(f"[Skip] {program_name}: no buggy file found")
            continue

        with open(buggy_file, "r") as f:
            buggy_code = f.read()

        # Extract just the function (before the docstring/comments)
        code_lines = []
        for line in buggy_code.split("\n"):
            if line.startswith('"""') or line.startswith("'''"):
                break
            code_lines.append(line)
        buggy_code_clean = "\n".join(code_lines).strip()

        with open(os.path.join(json_dir, json_file), "r") as f:
            content = f.read().strip()

        # Each line in the JSON file is a test case: [[inputs], expected_output]
        test_cases = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                tc = json.loads(line)
                if isinstance(tc, list) and len(tc) == 2:
                    inputs = tc[0] if isinstance(tc[0], list) else [tc[0]]
                    expected = tc[1]
                    test_cases.append([inputs, expected])
            except json.JSONDecodeError:
                continue

        if not test_cases:
            print(f"[Skip] {program_name}: no valid test cases")
            continue

        problems.append(Problem(
            name=program_name,
            buggy_code=buggy_code_clean,
            test_style="tuple",
            dataset="quixbugs",
            test_cases=test_cases,
        ))

    return problems

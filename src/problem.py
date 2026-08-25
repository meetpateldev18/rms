"""
Uniform representation of a buggy-code problem, regardless of source dataset.

test_style determines which fields are populated and how executor.py runs tests:
  "tuple"      (QuixBugs)      — test_cases: list of [[inputs...], expected]
  "assert"     (HumanEvalFix)  — test_code (defines check(candidate)) + entry_point
  "example_io" (DebugBench)    — examples: list of (kwargs, expected) + entry_point + class_name
"""


class Problem:
    def __init__(self, name, buggy_code, test_style, dataset,
                 test_cases=None, test_code=None, entry_point=None,
                 examples=None, class_name=None, language="python"):
        self.name = name
        self.buggy_code = buggy_code
        self.test_style = test_style
        self.dataset = dataset
        self.test_cases = test_cases
        self.test_code = test_code
        self.entry_point = entry_point
        self.examples = examples
        self.class_name = class_name
        self.language = language  # "python" (default) | "java" | "cpp" | "javascript"

    def num_tests(self):
        if self.test_style == "tuple":
            return len(self.test_cases)
        if self.test_style == "example_io":
            return len(self.examples)
        return 1  # "assert" — one check() call covers all internal assertions

    def __repr__(self):
        return f"Problem(name={self.name!r}, dataset={self.dataset!r}, test_style={self.test_style!r})"

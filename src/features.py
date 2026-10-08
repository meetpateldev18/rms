"""
Static, pre-repair feature extraction for a Problem.

These features are computable BEFORE any LLM call is made — they describe
the bug's surface structure (size, branching, test coverage, dataset/
language origin), not anything about how hard it turns out to be to fix.
They're the input to the adaptive routing controller in adaptive_router.py:
the premise (see report/main.tex's own conclusion) is that whether
multi-agent orchestration beats a single-prompt baseline is conditional on
bug structure, so a classifier over bug-structure features should be able
to predict which side of that line a new bug falls on.
"""

import ast
import executor

_BRANCH_KEYWORDS_PY = ("if ", "elif ", "for ", "while ", "except", " and ", " or ", "case ")
_BRANCH_KEYWORDS_GENERIC = ("if ", "else if", "for ", "while ", "catch", "&&", "||", "case ")


def _cyclomatic_complexity_python(code: str) -> int:
    """McCabe cyclomatic complexity via AST decision-point counting.
    Falls back to keyword counting (see _branch_count_generic) if the code
    doesn't parse — a buggy program may itself be a syntax error."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _branch_count_generic(code, _BRANCH_KEYWORDS_PY)

    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, (ast.comprehension,)):
            complexity += 1
    return complexity


def _branch_count_generic(code: str, keywords) -> int:
    """Keyword-counting proxy for cyclomatic complexity, used for non-Python
    languages (Java/C++/JS) where we don't parse an AST."""
    complexity = 1
    lowered = code
    for kw in keywords:
        complexity += lowered.count(kw)
    return complexity


def extract_features(problem) -> dict:
    """Return a flat dict of numeric/categorical features for `problem`.
    No LLM calls, no test execution — purely static analysis of buggy_code."""
    code = problem.buggy_code or ""
    lines = [l for l in code.split("\n") if l.strip()]
    language = getattr(problem, "language", "python")

    if language == "python":
        complexity = _cyclomatic_complexity_python(code)
    else:
        complexity = _branch_count_generic(code, _BRANCH_KEYWORDS_GENERIC)

    num_defs = code.count("def ") if language == "python" else (
        code.count("public ") + code.count("private ") + code.count("function ")
    )

    # Zero-LLM-cost signal: how badly is the *unmodified* buggy code already
    # failing? Pure Python test execution (same call coordinator.py already
    # makes before attempt 1) — no LLM involved, so it's free to use for
    # routing. A bug that already passes most tests is a plausibly small,
    # single-line fix; one that passes none may be more structurally broken.
    try:
        baseline_run = executor.run_tests(problem, code)
        n_tests = max(baseline_run.get("total_tests", 1), 1)
        buggy_pass_rate = baseline_run.get("passed", 0) / n_tests
    except Exception:
        buggy_pass_rate = 0.0

    return {
        "loc": len(lines),
        "chars": len(code),
        "cyclomatic_complexity": complexity,
        "num_tests": problem.num_tests(),
        "num_functions": max(num_defs, 1),
        "buggy_pass_rate": buggy_pass_rate,
        "dataset": problem.dataset,
        "language": language,
        "test_style": problem.test_style,
    }


FEATURE_COLUMNS_NUMERIC = ["loc", "chars", "cyclomatic_complexity", "num_tests", "num_functions", "buggy_pass_rate"]
FEATURE_COLUMNS_CATEGORICAL = ["dataset", "language", "test_style"]

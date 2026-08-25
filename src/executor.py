"""
Dataset-agnostic test execution.

Dispatches on Problem.test_style so agents.py / coordinator.py / baseline.py
don't need to know whether a problem came from QuixBugs (tuple I/O),
HumanEvalFix (assert-based check()), or DebugBench (example I/O against a
LeetCode-style Solution class).
"""

import signal
import threading
import subprocess
import tempfile
import os

EXEC_TIMEOUT = 5  # seconds — kill any test that takes longer than this (catches infinite loops)
PROBLEM_TIMEOUT = 360  # seconds — hard ceiling per problem, so one stuck LLM call can't stall a multi-hour run
COMPILE_TIMEOUT = 15  # seconds — javac/g++ startup + compile of a small file


class _AlarmTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _AlarmTimeout()


def run_with_timeout(func, args=(), kwargs=None, timeout=EXEC_TIMEOUT):
    """Run a function with a timeout. Returns (result, error).

    Uses SIGALRM instead of a thread-with-join: a stuck infinite loop is
    genuinely interrupted (the signal fires inside it and unwinds via
    exception), rather than being abandoned as a live thread that keeps
    burning CPU/GIL time for the rest of the process's life. With hundreds
    of test calls per experiment run, orphaned busy threads compound and
    stall everything after them — this doesn't.
    """
    kwargs = kwargs or {}
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        if hasattr(result, '__next__'):
            result = list(result)
        return result, None
    except _AlarmTimeout:
        return None, TimeoutError(f"Execution timed out after {timeout}s (likely infinite loop)")
    except Exception as e:
        return None, e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_with_hard_timeout(func, args=(), kwargs=None, timeout=PROBLEM_TIMEOUT):
    """Best-effort OUTER watchdog around a whole problem's LLM+recovery pipeline.

    Deliberately thread-based (not SIGALRM): run_with_timeout's per-test-case
    alarms fire routinely *inside* func, and signal.alarm is a single global
    per-process timer — nesting an outer alarm around code that also sets
    inner alarms would silently cancel whichever one disarms second. A leaked
    daemon thread here is rare (one per genuinely stuck problem, not one per
    test case), so the tradeoff is acceptable where it wasn't for EXEC_TIMEOUT.

    Returns (result, error). On timeout, error is a TimeoutError and the
    underlying call is abandoned (still running in the background) rather
    than killed — Python has no safe way to force-kill another thread.
    """
    kwargs = kwargs or {}
    result = [None]
    error = [None]
    done = threading.Event()

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
        finally:
            done.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if not done.wait(timeout):
        return None, TimeoutError(f"Problem exceeded the {timeout}s hard timeout (stuck LLM call?) — skipping")
    if error[0] is not None:
        return None, error[0]
    return result[0], None


def _extract_first_func(code):
    """Execute code and extract the first callable function (QuixBugs style)."""
    namespace = {}
    exec(code, namespace)
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith("_"):
            return obj
    return None


# ============================================================
# "tuple" — QuixBugs: test_cases = [[[inputs...], expected], ...]
# ============================================================
def _run_tuple_tests(code, test_cases, limit=None):
    func = _extract_first_func(code)
    if func is None:
        return {"total_tests": len(test_cases), "passed": 0, "failed": len(test_cases),
                "all_passed": False, "errors": ["No callable function found in code"]}

    cases = test_cases if limit is None else test_cases[:limit]
    passed, failed, errors = 0, 0, []
    for i, tc in enumerate(cases):
        inputs, expected = tc
        actual, exc = run_with_timeout(func, args=inputs)
        if exc is not None:
            failed += 1
            errors.append(f"Test {i+1}: {type(exc).__name__}: {exc}")
        elif actual != expected:
            failed += 1
            errors.append(f"Test {i+1}: returned {repr(actual)}, expected {repr(expected)}")
        else:
            passed += 1
    return {"total_tests": len(cases), "passed": passed, "failed": failed,
            "all_passed": failed == 0, "errors": errors}


# ============================================================
# "assert" — HumanEvalFix: test_code defines check(candidate) AND calls it
# at module level, so the actual test run happens inside exec(test_code, ...)
# itself — that exec call must be inside the timeout, not just a would-be
# separate check(candidate) call afterward (that call may never be reached,
# and even when it is, exec(test_code) alone can already hang forever on a
# genuinely buggy infinite loop).
# ============================================================
def _run_assert_tests(code, test_code, entry_point, language="python"):
    if language == "java":
        return _run_java_assert(code, test_code)
    elif language == "cpp":
        return _run_cpp_assert(code, test_code)
    elif language == "javascript":
        return _run_js_assert(code, test_code)
    elif language != "python":
        raise ValueError(f"Unsupported language: {language!r}")

    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return {"total_tests": 1, "passed": 0, "failed": 1, "all_passed": False,
                "errors": [f"{type(e).__name__}: {e}"]}

    candidate = namespace.get(entry_point)
    if candidate is None:
        return {"total_tests": 1, "passed": 0, "failed": 1, "all_passed": False,
                "errors": [f"entry_point '{entry_point}' not found in code"]}

    def _run_test_code():
        exec(test_code, namespace)
        # Some datasets' test fields only define check() without calling it —
        # call it explicitly if it wasn't already invoked above.
        check_fn = namespace.get("check")
        if check_fn is not None:
            check_fn(candidate)

    _, exc = run_with_timeout(_run_test_code)
    if exc is not None:
        return {"total_tests": 1, "passed": 0, "failed": 1, "all_passed": False,
                "errors": [f"{type(exc).__name__}: {exc}"]}
    return {"total_tests": 1, "passed": 1, "failed": 0, "all_passed": True, "errors": []}


def _single(passed, error=None):
    return {"total_tests": 1, "passed": 1 if passed else 0, "failed": 0 if passed else 1,
            "all_passed": passed, "errors": [] if passed else [error]}


# HumanEvalFix's Java `test` field is a standalone `public class Main { ... }` that
# references java.util types (List, Arrays, ArrayList) without importing them itself —
# the dataset relies on those imports existing in a shared compilation context that
# doesn't exist once Main.java is compiled as its own file, so we supply them here.
_JAVA_TEST_IMPORTS = "import java.util.*;\nimport java.lang.*;\n\n"


def _run_java_assert(code, test_code):
    """HumanEvalFix Java: code is a complete `class Solution {...}`; test_code is a
    complete `public class Main {...}` that throws AssertionError on failure —
    a nonzero exit code from `java Main` reliably signals a failing test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with open(os.path.join(tmpdir, "Solution.java"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "Main.java"), "w") as f:
                f.write(_JAVA_TEST_IMPORTS + test_code)
        except OSError as e:
            return _single(False, f"IOError: {e}")

        try:
            compiled = subprocess.run(
                ["javac", "Solution.java", "Main.java"],
                cwd=tmpdir, capture_output=True, text=True, timeout=COMPILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return _single(False, "CompileTimeout: javac did not finish in time")
        if compiled.returncode != 0:
            return _single(False, f"CompileError: {compiled.stderr[-500:]}")

        try:
            run = subprocess.run(
                ["java", "-cp", tmpdir, "Main"],
                capture_output=True, text=True, timeout=EXEC_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return _single(False, "TimeoutError: Execution timed out after {}s (likely infinite loop)".format(EXEC_TIMEOUT))
        if run.returncode != 0:
            return _single(False, f"RuntimeError: {run.stderr[-500:] or run.stdout[-500:]}")
        return _single(True)


def _run_cpp_assert(code, test_code):
    """HumanEvalFix C++: code is #include's + a function definition; test_code is
    `#undef NDEBUG\\n#include<assert.h>\\nint main(){...}` — a failed assert()
    aborts the process with a nonzero exit code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "main.cpp")
        binary_path = os.path.join(tmpdir, "prog")
        try:
            with open(src_path, "w") as f:
                f.write(code + "\n" + test_code)
        except OSError as e:
            return _single(False, f"IOError: {e}")

        try:
            compiled = subprocess.run(
                ["g++", "-std=c++17", "-o", binary_path, src_path],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return _single(False, "CompileTimeout: g++ did not finish in time")
        if compiled.returncode != 0:
            return _single(False, f"CompileError: {compiled.stderr[-500:]}")

        try:
            run = subprocess.run([binary_path], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            return _single(False, "TimeoutError: Execution timed out after {}s (likely infinite loop)".format(EXEC_TIMEOUT))
        if run.returncode != 0:
            return _single(False, f"RuntimeError (exit {run.returncode}): {run.stderr[-500:] or run.stdout[-500:]}")
        return _single(True)


def _run_js_assert(code, test_code):
    """HumanEvalFix JavaScript: test_code uses console.assert(), which — unlike
    Python/Java/C++ assertions — does NOT raise or set a nonzero exit code on
    failure in Node.js; it only prints "Assertion failed: ..." to stderr and
    keeps running (verified empirically). Exit code alone is therefore not a
    reliable pass/fail signal here; stderr content must be checked too."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "test.js")
        try:
            with open(script_path, "w") as f:
                f.write(code + "\n" + test_code)
        except OSError as e:
            return _single(False, f"IOError: {e}")

        try:
            run = subprocess.run(["node", script_path], capture_output=True, text=True, timeout=EXEC_TIMEOUT)
        except subprocess.TimeoutExpired:
            return _single(False, "TimeoutError: Execution timed out after {}s (likely infinite loop)".format(EXEC_TIMEOUT))
        if run.returncode != 0:
            return _single(False, f"RuntimeError: {run.stderr[-500:] or run.stdout[-500:]}")
        if "Assertion failed" in run.stderr:
            return _single(False, f"AssertionError: {run.stderr[-500:]}")
        return _single(True)


# ============================================================
# "example_io" — DebugBench: examples = [(kwargs, expected), ...] against a Solution class
# ============================================================
# LeetCode's platform auto-imports these for submitted solutions, so the raw
# DebugBench code snippets rely on them being in scope without importing them.
_LEETCODE_PREAMBLE = """
from typing import *
from collections import *
from functools import reduce, lru_cache, cache
from math import ceil, floor, sqrt, gcd, inf
import collections
import math
import heapq
import itertools
import functools
import bisect
import string
import re
import random
"""


def _run_example_io_tests(code, examples, entry_point, class_name):
    namespace = {}
    try:
        exec(_LEETCODE_PREAMBLE + code, namespace)
    except Exception as e:
        return {"total_tests": len(examples), "passed": 0, "failed": len(examples), "all_passed": False,
                "errors": [f"{type(e).__name__}: {e}"]}

    cls = namespace.get(class_name)
    if cls is None:
        return {"total_tests": len(examples), "passed": 0, "failed": len(examples), "all_passed": False,
                "errors": [f"class '{class_name}' not found in code"]}

    try:
        instance = cls()
        method = getattr(instance, entry_point)
    except Exception as e:
        return {"total_tests": len(examples), "passed": 0, "failed": len(examples), "all_passed": False,
                "errors": [f"could not resolve {class_name}.{entry_point}: {e}"]}

    passed, failed, errors = 0, 0, []
    for i, (kwargs, expected) in enumerate(examples):
        # Call positionally (in the order params appear in the example text) rather than
        # by keyword — the example's input variable names don't always match the actual
        # parameter names in the Solution method, but the order is always call-compatible.
        actual, exc = run_with_timeout(method, args=tuple(kwargs.values()))
        if exc is not None:
            failed += 1
            errors.append(f"Example {i+1}: {type(exc).__name__}: {exc}")
        elif actual != expected:
            failed += 1
            errors.append(f"Example {i+1}: returned {repr(actual)}, expected {repr(expected)}")
        else:
            passed += 1
    return {"total_tests": len(examples), "passed": passed, "failed": failed,
            "all_passed": failed == 0, "errors": errors}


# ============================================================
# Public API
# ============================================================
def run_tests(problem, code):
    """Full validation run (Agent 6 / baseline). Returns
    {total_tests, passed, failed, all_passed, errors}."""
    language = getattr(problem, "language", "python")
    if language == "python":
        # Other languages compile via their own toolchain inside the runners below
        # (javac/g++), which already reports syntax errors — this pre-check is
        # Python-specific and would misfire as a false "SyntaxError" on other languages.
        try:
            compile(code, "<patch>", "exec")
        except SyntaxError as e:
            n = problem.num_tests()
            return {"total_tests": n, "passed": 0, "failed": n, "all_passed": False,
                    "errors": [f"SyntaxError: {e}"]}

    if problem.test_style == "tuple":
        return _run_tuple_tests(code, problem.test_cases)
    elif problem.test_style == "assert":
        return _run_assert_tests(code, problem.test_code, problem.entry_point, language)
    elif problem.test_style == "example_io":
        return _run_example_io_tests(code, problem.examples, problem.entry_point, problem.class_name)
    else:
        raise ValueError(f"Unknown test_style: {problem.test_style!r}")


def get_failure_output(problem, code, limit=5):
    """Quick failure capture for Agent 1 (Failure Detection). Returns a
    human-readable string describing what fails, capped to `limit` cases
    for tuple/example_io styles (assert style is a single check() call)."""
    language = getattr(problem, "language", "python")
    if language == "python":
        try:
            compile(code, "<patch>", "exec")
        except SyntaxError as e:
            return f"SyntaxError: {e}"

    if problem.test_style == "tuple":
        result = _run_tuple_tests(code, problem.test_cases, limit=limit)
    elif problem.test_style == "assert":
        result = _run_assert_tests(code, problem.test_code, problem.entry_point, language)
    elif problem.test_style == "example_io":
        result = _run_example_io_tests(code, problem.examples[:limit], problem.entry_point, problem.class_name)
    else:
        raise ValueError(f"Unknown test_style: {problem.test_style!r}")

    return "\n".join(result["errors"]) if result["errors"] else "Unknown failure"

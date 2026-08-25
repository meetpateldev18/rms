import json
import executor

# Per-language display name, code-fence marker, and the term for "the line
# that starts the function/method" — used to keep prompts language-aware
# without duplicating each agent's prompt text per language.
_LANG_INFO = {
    "python": {"name": "Python", "fence": "python", "def_line": "the def line"},
    "java": {"name": "Java", "fence": "java", "def_line": "the method signature"},
    "cpp": {"name": "C++", "fence": "cpp", "def_line": "the function signature"},
    "javascript": {"name": "JavaScript", "fence": "javascript", "def_line": "the function declaration"},
}


def _lang(language: str) -> dict:
    return _LANG_INFO.get(language, _LANG_INFO["python"])


def _parse_json_response(response: str) -> dict:
    """Try to parse JSON from LLM response, handling markdown code blocks."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


# ============================================================
# AGENT 1: Failure Detection Agent (LLM)
# ============================================================
class FailureDetectionAgent:
    """Analyzes test failures and classifies the error type using LLM."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, program_name: str, buggy_code: str, problem) -> dict:
        print(f"\n[Agent 1: FailureDetection] Analyzing failure...")

        lang = _lang(getattr(problem, "language", "python"))

        # First, run the tests to get raw error output
        test_output_str = executor.get_failure_output(problem, buggy_code, limit=5)
        print(f"[Agent 1]   {test_output_str[:300]}")

        # Now ask LLM to classify the failure
        prompt = f"""You are a failure detection agent. Analyze the following buggy {lang['name']} function and its test failures.

Function name: {program_name}

Buggy code:
```{lang['fence']}
{buggy_code}
```

Test results:
{test_output_str}

Classify this failure. Respond in this exact JSON format (no markdown, no code blocks, just raw JSON):
{{"error_type": "one of: WrongOutput, RuntimeError, InfiniteLoop, SyntaxError, IndexError, TypeError, Other", "description": "brief description of what is going wrong", "key_observation": "the most important clue about what the bug might be"}}"""

        print(f"[Agent 1] Asking LLM to classify failure...")
        response = self.llm.generate(prompt)
        print(f"[Agent 1] LLM response: {response[:200]}")

        try:
            parsed = _parse_json_response(response)
            print(f"[Agent 1] Error type: {parsed.get('error_type', '?')}")
            print(f"[Agent 1] Description: {parsed.get('description', '?')[:100]}")
            print(f"[Agent 1] Key observation: {parsed.get('key_observation', '?')[:100]}")
            parsed["test_output"] = test_output_str
            return parsed
        except (json.JSONDecodeError, IndexError):
            print(f"[Agent 1] JSON parse failed, using raw test output")
            return {
                "error_type": "Unknown",
                "description": test_output_str[:200],
                "key_observation": test_output_str[:200],
                "test_output": test_output_str,
            }


# ============================================================
# AGENT 2: Code Localization Agent (LLM)
# ============================================================
class CodeLocalizationAgent:
    """Identifies the exact faulty line in the code using LLM."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, program_name: str, buggy_code: str, failure_info: dict, language: str = "python") -> dict:
        print(f"\n[Agent 2: CodeLocalization] Localizing fault(s)...")
        lang = _lang(language)

        # Number the lines for the LLM
        numbered_code = ""
        for i, line in enumerate(buggy_code.strip().split("\n"), 1):
            numbered_code += f"  {i}: {line}\n"

        prompt = f"""You are a code localization agent. Given a buggy {lang['name']} function and failure analysis, identify EVERY faulty line — there may be ONE bug or SEVERAL independent bugs in this code. Do not stop at the first one you find; keep reading the whole function.

Function: {program_name}

Code (with line numbers):
{numbered_code}

Failure analysis:
- Error type: {failure_info.get('error_type', 'Unknown')}
- Description: {failure_info.get('description', 'Unknown')}
- Key observation: {failure_info.get('key_observation', 'Unknown')}
- Test output: {failure_info.get('test_output', 'Unknown')[:300]}

Respond in this exact JSON format (no markdown, no code blocks, just raw JSON). List every faulty line you find as a separate entry in "faulty_lines" — one entry if there's only one bug, multiple entries if there are multiple independent bugs:
{{"faulty_lines": [{{"line_number": <int>, "line_content": "the exact line of code that is buggy", "reason": "why this line is incorrect"}}]}}"""

        print(f"[Agent 2] Asking LLM to localize fault(s)...")
        response = self.llm.generate(prompt)
        print(f"[Agent 2] LLM response: {response[:200]}")

        try:
            parsed = _parse_json_response(response)
            faulty_lines = self._normalize(parsed)
            parsed["faulty_lines"] = faulty_lines
            print(f"[Agent 2] Found {len(faulty_lines)} faulty line(s):")
            for fl in faulty_lines:
                print(f"[Agent 2]   Line #{fl.get('line_number', '?')}: {fl.get('line_content', '?')} — {fl.get('reason', '?')[:80]}")
            return parsed
        except (json.JSONDecodeError, IndexError):
            print(f"[Agent 2] JSON parse failed, returning raw response")
            return {
                "faulty_lines": [{"line_number": -1, "line_content": "unknown", "reason": response[:200]}],
            }

    @staticmethod
    def _normalize(parsed: dict) -> list:
        """Accept either the new {"faulty_lines": [...]} shape or the old
        single-line {"faulty_line_number", "faulty_line_content", "reason"}
        shape, always returning a list of {"line_number", "line_content", "reason"}."""
        if isinstance(parsed.get("faulty_lines"), list) and parsed["faulty_lines"]:
            lines = []
            for fl in parsed["faulty_lines"]:
                lines.append({
                    "line_number": fl.get("line_number", fl.get("faulty_line_number", -1)),
                    "line_content": fl.get("line_content", fl.get("faulty_line_content", "unknown")),
                    "reason": fl.get("reason", "unknown"),
                })
            return lines
        if "faulty_line_number" in parsed or "faulty_line_content" in parsed:
            return [{
                "line_number": parsed.get("faulty_line_number", -1),
                "line_content": parsed.get("faulty_line_content", "unknown"),
                "reason": parsed.get("reason", "unknown"),
            }]
        return [{"line_number": -1, "line_content": "unknown", "reason": "no faulty line identified"}]


# ============================================================
# AGENT 3: Debugging Agent (LLM)
# ============================================================
class DebuggingAgent:
    """Analyzes root cause and proposes a fix strategy using LLM."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, program_name: str, buggy_code: str, failure_info: dict, localization_info: dict, prev_patch: str = None, language: str = "python") -> dict:
        print(f"\n[Agent 3: Debugging] Analyzing root cause(s)...")
        lang = _lang(language)

        retry_context = ""
        if prev_patch:
            retry_context = f"""

IMPORTANT: A previous fix attempt FAILED. The previous patch was:
```{lang['fence']}
{prev_patch}
```
That did NOT work — either it missed a bug entirely, or introduced a new problem. Re-examine the whole function for ANY remaining issue, not just the one you focused on last time."""

        faulty_lines = localization_info.get("faulty_lines", [])
        localization_str = "\n".join(
            f"- Line {fl.get('line_number', '?')}: {fl.get('line_content', 'unknown')} — {fl.get('reason', 'unknown')}"
            for fl in faulty_lines
        ) or "- unknown"

        prompt = f"""You are a debugging agent. Analyze the root cause of EACH bug identified below and propose a fix for each. There may be one bug or several independent bugs — address all of them.

Function: {program_name}

Buggy code:
```{lang['fence']}
{buggy_code}
```

Failure analysis:
- Error type: {failure_info.get('error_type', 'Unknown')}
- Description: {failure_info.get('description', 'Unknown')}

Fault localization ({len(faulty_lines)} faulty line(s) identified):
{localization_str}
{retry_context}

Respond in this exact JSON format (no markdown, no code blocks, just raw JSON). Include one entry per bug in "fixes":
{{"fixes": [{{"root_cause": "why this specific bug occurs", "old_code": "the exact buggy expression or line", "new_code": "the exact corrected expression or line"}}], "fix_strategy": "overall summary of what needs to change and why"}}"""

        print(f"[Agent 3] Asking LLM for root cause analysis...")
        response = self.llm.generate(prompt)
        print(f"[Agent 3] LLM response: {response[:200]}")

        try:
            parsed = _parse_json_response(response)
            fixes = self._normalize(parsed)
            parsed["fixes"] = fixes
            print(f"[Agent 3] {len(fixes)} fix(es) proposed:")
            for fx in fixes:
                print(f"[Agent 3]   {fx.get('old_code', '?')} → {fx.get('new_code', '?')} ({fx.get('root_cause', '?')[:80]})")
            return parsed
        except (json.JSONDecodeError, IndexError):
            print(f"[Agent 3] JSON parse failed, returning raw response")
            return {
                "fixes": [{"root_cause": response[:200], "old_code": "unknown", "new_code": "unknown"}],
                "fix_strategy": "unknown",
            }

    @staticmethod
    def _normalize(parsed: dict) -> list:
        """Accept either the new {"fixes": [...]} shape or the old single-fix
        {"root_cause", "old_code", "new_code"} shape, always returning a list."""
        if isinstance(parsed.get("fixes"), list) and parsed["fixes"]:
            return [{
                "root_cause": fx.get("root_cause", "unknown"),
                "old_code": fx.get("old_code", "unknown"),
                "new_code": fx.get("new_code", "unknown"),
            } for fx in parsed["fixes"]]
        if "old_code" in parsed or "root_cause" in parsed:
            return [{
                "root_cause": parsed.get("root_cause", "unknown"),
                "old_code": parsed.get("old_code", "unknown"),
                "new_code": parsed.get("new_code", "unknown"),
            }]
        return [{"root_cause": "unknown", "old_code": "unknown", "new_code": "unknown"}]


# ============================================================
# AGENT 4: Patch Generation Agent (LLM)
# ============================================================
class PatchGenerationAgent:
    """Generates the complete corrected function using LLM."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, program_name: str, buggy_code: str, debugging_info: dict, language: str = "python") -> dict:
        lang = _lang(language)
        fixes = debugging_info.get("fixes") or [{
            "root_cause": debugging_info.get("root_cause", "unknown"),
            "old_code": debugging_info.get("old_code", "unknown"),
            "new_code": debugging_info.get("new_code", "unknown"),
        }]
        print(f"\n[Agent 4: PatchGeneration] Generating corrected code ({len(fixes)} fix(es) to apply)...")

        fixes_str = "\n".join(
            f"{i+1}. {fx.get('old_code', '?')} → {fx.get('new_code', '?')} ({fx.get('root_cause', '?')})"
            for i, fx in enumerate(fixes)
        )

        prompt = f"""You are a patch generation agent. Generate the COMPLETE corrected {lang['name']} function, applying EVERY fix listed below — do not apply only one if several are listed.

Function: {program_name}

Buggy code:
```{lang['fence']}
{buggy_code}
```

Debugging analysis:
- Fix strategy: {debugging_info.get('fix_strategy', 'unknown')}
- Fixes to apply ({len(fixes)} total):
{fixes_str}

Respond in this exact JSON format (no markdown, no code blocks, just raw JSON):
{{"corrected_code": "the COMPLETE corrected function with ALL fixes applied", "changes_made": "brief summary of everything that was changed"}}

IMPORTANT: In corrected_code, use \\n for newlines. Return the COMPLETE function including {lang['def_line']}, not just the fixed lines. Apply ALL listed fixes, not just the first one."""

        print(f"[Agent 4] Asking LLM to generate patch...")
        response = self.llm.generate(prompt)
        print(f"[Agent 4] LLM response: {response[:300]}")

        try:
            parsed = _parse_json_response(response)
            code = parsed.get("corrected_code", "")
            # Unescape newlines/tabs if they're literal strings
            if "\\n" in code and "\n" not in code:
                code = code.replace("\\n", "\n")
            if "\\t" in code:
                code = code.replace("\\t", "\t")
            # Fix indentation if tabs got lost
            lines = code.split("\n")
            if len(lines) > 1 and lines[1] and not lines[1][0].isspace():
                code = lines[0] + "\n" + "\n".join("    " + l if l.strip() else l for l in lines[1:])
            parsed["corrected_code"] = code

            print(f"[Agent 4] Changes: {parsed.get('changes_made', '?')[:100]}")
            print(f"[Agent 4] Corrected code:")
            for i, line in enumerate(code.strip().split('\n'), 1):
                print(f"[Agent 4]   {i:3d} | {line}")
            return parsed

        except (json.JSONDecodeError, IndexError):
            print(f"[Agent 4] JSON parse failed, extracting code from response")
            code = response.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            if "def " in code:
                start = code.index("def ")
                code = code[start:]
            return {
                "corrected_code": code,
                "changes_made": "Extracted from raw LLM response",
            }


# ============================================================
# AGENT 5: Patch Application Agent (Pure Python — no LLM)
# ============================================================
class PatchApplicationAgent:
    """Applies the generated patch. Pure Python — no LLM needed."""

    def run(self, original_code: str, patched_code: str, language: str = "python") -> dict:
        print(f"\n[Agent 5: PatchApplication] Applying patch...")

        if not patched_code or not patched_code.strip():
            print(f"[Agent 5] ERROR: Empty patch")
            return {"success": False, "applied_code": original_code, "error": "Empty patch"}

        if language != "python":
            # Other languages compile via their own toolchain (javac/g++) inside
            # executor.run_tests — Python's compile() can't check their syntax,
            # and a false-positive rejection here would silently discard valid
            # patches before Agent 6 ever gets to run them.
            print(f"[Agent 5] Non-Python language ({language}) — compile check deferred to Validation")
            print(f"[Agent 5] Applied {len(patched_code)} chars of patched code")
            return {"success": True, "applied_code": patched_code, "error": None}

        # Verify patched code compiles
        try:
            compile(patched_code, "<patch>", "exec")
            print(f"[Agent 5] Patch compiles successfully")
            print(f"[Agent 5] Applied {len(patched_code)} chars of patched code")
            return {"success": True, "applied_code": patched_code, "error": None}
        except SyntaxError as e:
            print(f"[Agent 5] ERROR: Patch has syntax error: {e}")
            return {"success": False, "applied_code": original_code, "error": f"SyntaxError: {e}"}


# ============================================================
# AGENT 6: Validation Agent (Pure Python — no LLM)
# ============================================================
class ValidationAgent:
    """Validates patched code against test cases. Pure Python — no LLM needed."""

    def run(self, program_name: str, patched_code: str, problem) -> dict:
        print(f"\n[Agent 6: Validation] Running {problem.num_tests()} test(s) on patched code...")

        results = executor.run_tests(problem, patched_code)

        for err in results["errors"]:
            print(f"[Agent 6]   FAIL — {err}")

        status = "ALL PASSED" if results["all_passed"] else f"{results['passed']}/{results['total_tests']} passed"
        print(f"[Agent 6] Result: {status}")
        return results

import time
from agents import (
    FailureDetectionAgent,
    CodeLocalizationAgent,
    DebuggingAgent,
    PatchGenerationAgent,
    PatchApplicationAgent,
    ValidationAgent,
)
from config import MAX_RETRY_ATTEMPTS
import executor


class RecoveryCoordinator:
    """Orchestrates the 6-agent failure recovery loop.

    Agent flow per attempt:
      1. FailureDetectionAgent    (LLM)    — classify the failure
      2. CodeLocalizationAgent    (LLM)    — find the faulty line
      3. DebuggingAgent           (LLM)    — root cause + fix strategy
      4. PatchGenerationAgent     (LLM)    — generate corrected code
      5. PatchApplicationAgent    (Python)  — apply and compile-check
      6. ValidationAgent          (Python)  — run tests

    4 LLM calls per attempt. Max 3 attempts = max 12 LLM calls per bug.
    """

    def __init__(self, llm):
        self.llm = llm
        self.failure_detector = FailureDetectionAgent(llm)
        self.code_localizer = CodeLocalizationAgent(llm)
        self.debugger = DebuggingAgent(llm)
        self.patch_generator = PatchGenerationAgent(llm)
        self.patch_applier = PatchApplicationAgent()
        self.validator = ValidationAgent()
        print("[Coordinator] Initialized with 6 agents:")
        print("[Coordinator]   1. FailureDetectionAgent   (LLM)")
        print("[Coordinator]   2. CodeLocalizationAgent   (LLM)")
        print("[Coordinator]   3. DebuggingAgent          (LLM)")
        print("[Coordinator]   4. PatchGenerationAgent    (LLM)")
        print("[Coordinator]   5. PatchApplicationAgent   (Python)")
        print("[Coordinator]   6. ValidationAgent         (Python)")
        print(f"[Coordinator] Max retry attempts: {MAX_RETRY_ATTEMPTS}")

    def recover(self, program_name: str, buggy_code: str, problem) -> dict:
        """Run the full 6-agent recovery loop for a single buggy program."""
        start_time = time.time()
        result = {
            "program": program_name,
            "success": False,
            "attempts": 0,
            "time_seconds": 0,
            "error_type": "",
            "llm_calls": 0,
            "patch": "",
            "agent_log": [],
        }

        print(f"\n{'='*70}")
        print(f"[Coordinator] ========== PROCESSING: {program_name} ==========")
        print(f"{'='*70}")
        print(f"\n[Coordinator] Buggy code:")
        for i, line in enumerate(buggy_code.strip().split('\n'), 1):
            print(f"[Coordinator]   {i:3d} | {line}")
        print(f"\n[Coordinator] Test style: {problem.test_style} ({problem.num_tests()} test unit(s))")

        prev_patch = None

        # current_code is what the agents analyze/patch this attempt — starts as the
        # original buggy code, but ratchets forward to a partially-fixed version if a
        # failed attempt still made genuine progress (more tests passing than before).
        # This lets later attempts focus on remaining bugs instead of re-discovering
        # ones already fixed, which matters most for problems with multiple independent
        # bugs (see DebugBench's "multiple error" category).
        current_code = buggy_code
        baseline = executor.run_tests(problem, buggy_code)
        best_code = buggy_code
        best_passed = baseline.get("passed", 0)
        print(f"[Coordinator] Baseline: {best_passed}/{problem.num_tests()} tests already passing on the unmodified buggy code")

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            result["attempts"] = attempt
            llm_before = self.llm.total_calls

            print(f"\n{'~'*70}")
            print(f"[Coordinator] ===== ATTEMPT {attempt} of {MAX_RETRY_ATTEMPTS} =====")
            print(f"{'~'*70}")
            if current_code != buggy_code:
                print(f"[Coordinator] Building on prior partial progress ({best_passed} tests passing) rather than the original buggy code")

            # ---- AGENT 1: Failure Detection (LLM) ----
            print(f"\n[Coordinator] >>> STEP 1/6: Failure Detection")
            failure_info = self.failure_detector.run(program_name, current_code, problem)
            result["error_type"] = failure_info.get("error_type", "Unknown")
            result["agent_log"].append({"agent": "FailureDetection", "attempt": attempt, "output": failure_info})

            # ---- AGENT 2: Code Localization (LLM) ----
            print(f"\n[Coordinator] >>> STEP 2/6: Code Localization")
            localization_info = self.code_localizer.run(program_name, current_code, failure_info, problem.language)
            result["agent_log"].append({"agent": "CodeLocalization", "attempt": attempt, "output": localization_info})

            # ---- AGENT 3: Debugging (LLM) ----
            print(f"\n[Coordinator] >>> STEP 3/6: Debugging & Root Cause Analysis")
            debugging_info = self.debugger.run(program_name, current_code, failure_info, localization_info, prev_patch, problem.language)
            result["agent_log"].append({"agent": "Debugging", "attempt": attempt, "output": debugging_info})

            # ---- AGENT 4: Patch Generation (LLM) ----
            print(f"\n[Coordinator] >>> STEP 4/6: Patch Generation")
            patch_info = self.patch_generator.run(program_name, current_code, debugging_info, problem.language)
            patched_code = patch_info.get("corrected_code", "")
            result["agent_log"].append({"agent": "PatchGeneration", "attempt": attempt, "output": {
                "changes_made": patch_info.get("changes_made", ""),
                "code_length": len(patched_code),
            }})

            # ---- AGENT 5: Patch Application ----
            print(f"\n[Coordinator] >>> STEP 5/6: Patch Application")
            application_result = self.patch_applier.run(current_code, patched_code, problem.language)
            result["agent_log"].append({"agent": "PatchApplication", "attempt": attempt, "output": application_result})

            if not application_result["success"]:
                print(f"\n[Coordinator] Patch application failed — skipping to next attempt")
                prev_patch = patched_code
                result["llm_calls"] = self.llm.total_calls - llm_before + result.get("llm_calls", 0)
                continue

            # ---- AGENT 6: Validation (Python) ----
            print(f"\n[Coordinator] >>> STEP 6/6: Validation")
            validation = self.validator.run(program_name, application_result["applied_code"], problem)
            result["agent_log"].append({"agent": "Validation", "attempt": attempt, "output": validation})

            llm_this_attempt = self.llm.total_calls - llm_before
            print(f"\n[Coordinator] Attempt {attempt} summary: {validation['passed']}/{validation['total_tests']} passed, {llm_this_attempt} LLM calls")

            if validation["all_passed"]:
                result["success"] = True
                result["patch"] = application_result["applied_code"]
                result["time_seconds"] = round(time.time() - start_time, 2)
                result["llm_calls"] = self.llm.total_calls
                print(f"\n{'*'*70}")
                print(f"[Coordinator] SUCCESS! Fixed '{program_name}' on attempt {attempt}!")
                print(f"[Coordinator] Time: {result['time_seconds']}s | LLM calls this bug: {llm_this_attempt}")
                print(f"{'*'*70}")
                return result

            # Failed — decide whether this attempt made genuine progress worth
            # building on, or whether to fall back to the best code seen so far.
            passed_now = validation.get("passed", 0)
            if passed_now > best_passed:
                print(f"\n[Coordinator] Partial progress: {passed_now} tests passing now (was {best_passed}) — continuing from this patch next attempt")
                best_passed = passed_now
                best_code = application_result["applied_code"]
                current_code = best_code
            else:
                if current_code != best_code:
                    print(f"\n[Coordinator] No improvement ({passed_now} vs best {best_passed}) — reverting to best-known code for next attempt")
                current_code = best_code
            prev_patch = patched_code
            print(f"\n[Coordinator] Fix failed — will retry with error feedback")

        result["time_seconds"] = round(time.time() - start_time, 2)
        result["llm_calls"] = self.llm.total_calls
        print(f"\n{'X'*70}")
        print(f"[Coordinator] FAILED '{program_name}' after {MAX_RETRY_ATTEMPTS} attempts ({result['time_seconds']}s)")
        print(f"{'X'*70}")
        return result

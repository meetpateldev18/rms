"""
Patch Tournament Coordinator.

Extends RecoveryCoordinator (coordinator.py) with self-consistency: instead
of generating one patch candidate per attempt and living with whatever the
LLM happened to produce, this coordinator samples K candidate patches at
different temperatures from the SAME debugging analysis, validates all K
against the test suite, and keeps the best one. This directly targets the
failure mode where the 6-agent chain's diagnosis (steps 1-3) was fine but a
single stochastic patch-generation draw (step 4) missed — a different
sample from the same prompt might not have.

Candidate selection: prefer any candidate that passes every test; among
those, take the first (lowest temperature); if none pass, take whichever
candidate passed the most tests. Ties broken by generation order.

Also tracks a disagreement/consensus signal — how many of the K candidates
produced textually distinct corrected code — as a cheap proxy for model
uncertainty on this bug, reported per-attempt in the result log.
"""

import time
import executor
from coordinator import RecoveryCoordinator
from config import MAX_RETRY_ATTEMPTS

# Temperatures sampled per attempt, in priority order (lowest first = the
# coordinator's usual deterministic-ish draw, then increasingly diverse
# resamples). K is len(this) unless overridden.
DEFAULT_TEMPERATURES = [0.2, 0.5, 0.9]


class TournamentCoordinator(RecoveryCoordinator):
    def __init__(self, llm, k: int = 3, temperatures=None):
        super().__init__(llm)
        self.k = k
        self.temperatures = (temperatures or DEFAULT_TEMPERATURES)[:k]
        print(f"[TournamentCoordinator] Patch tournament enabled: K={self.k}, temperatures={self.temperatures}")

    def _generate_candidates(self, program_name, current_code, debugging_info, language):
        """Run Agent 4 (Patch Generation) K times at different temperatures,
        then Agent 5 (Patch Application) + Agent 6 (Validation) on each.
        Returns the list of candidate dicts, in generation order."""
        candidates = []
        for i, temp in enumerate(self.temperatures):
            print(f"\n[TournamentCoordinator] --- Candidate {i+1}/{self.k} (temperature={temp}) ---")
            patch_info = self.patch_generator.run(program_name, current_code, debugging_info, language, temperature=temp)
            patched_code = patch_info.get("corrected_code", "")

            application_result = self.patch_applier.run(current_code, patched_code, language)
            candidates.append({
                "temperature": temp,
                "patch_info": patch_info,
                "application_result": application_result,
            })
        return candidates

    def _validate_candidates(self, candidates, problem):
        for c in candidates:
            if c["application_result"]["success"]:
                c["validation"] = executor.run_tests(problem, c["application_result"]["applied_code"])
            else:
                n = problem.num_tests()
                c["validation"] = {
                    "total_tests": n, "passed": 0, "failed": n, "all_passed": False,
                    "errors": [c["application_result"].get("error", "patch application failed")],
                }
        return candidates

    @staticmethod
    def _pick_winner(candidates):
        passing = [c for c in candidates if c["validation"]["all_passed"]]
        if passing:
            return passing[0], True
        # Among non-passing candidates, prefer any that at least applied
        # (compiled) successfully — a tie at passed=0 between a candidate
        # that failed to apply and one that applied but is simply wrong
        # should not favor the one that failed outright.
        applied = [c for c in candidates if c["application_result"]["success"]]
        pool = applied or candidates
        best = max(pool, key=lambda c: c["validation"].get("passed", 0))
        return best, False

    @staticmethod
    def _agreement_score(candidates):
        """Fraction of candidate pairs that produced identical corrected
        code — 1.0 means every sample converged on the same patch (high
        confidence), 0.0 means every sample differed (high disagreement)."""
        codes = [c["patch_info"].get("corrected_code", "").strip() for c in candidates]
        if len(codes) <= 1:
            return 1.0
        unique = len(set(codes))
        return round(1.0 - (unique - 1) / (len(codes) - 1), 3)

    def recover(self, program_name: str, buggy_code: str, problem) -> dict:
        """Same overall attempt/retry structure as RecoveryCoordinator.recover,
        but Agent 4 fans out into a K-candidate tournament each attempt."""
        start_time = time.time()
        result = {
            "program": program_name, "success": False, "attempts": 0,
            "time_seconds": 0, "error_type": "", "llm_calls": 0, "patch": "",
            "agent_log": [], "tournament_k": self.k, "agreement_per_attempt": [],
            "candidates_passed_per_attempt": [],
        }

        print(f"\n{'='*70}\n[TournamentCoordinator] ========== PROCESSING: {program_name} ==========\n{'='*70}")
        print(f"[TournamentCoordinator] Test style: {problem.test_style} ({problem.num_tests()} test unit(s))")

        prev_patch = None
        current_code = buggy_code
        baseline = executor.run_tests(problem, buggy_code)
        best_code = buggy_code
        best_passed = baseline.get("passed", 0)

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            result["attempts"] = attempt
            llm_before = self.llm.total_calls
            print(f"\n{'~'*70}\n[TournamentCoordinator] ===== ATTEMPT {attempt} of {MAX_RETRY_ATTEMPTS} =====\n{'~'*70}")

            failure_info = self.failure_detector.run(program_name, current_code, problem)
            result["error_type"] = failure_info.get("error_type", "Unknown")
            result["agent_log"].append({"agent": "FailureDetection", "attempt": attempt, "output": failure_info})

            localization_info = self.code_localizer.run(program_name, current_code, failure_info, problem.language)
            result["agent_log"].append({"agent": "CodeLocalization", "attempt": attempt, "output": localization_info})

            debugging_info = self.debugger.run(program_name, current_code, failure_info, localization_info, prev_patch, problem.language)
            result["agent_log"].append({"agent": "Debugging", "attempt": attempt, "output": debugging_info})

            print(f"\n[TournamentCoordinator] >>> STEP 4/6: Patch Generation TOURNAMENT (K={self.k})")
            candidates = self._generate_candidates(program_name, current_code, debugging_info, problem.language)
            candidates = self._validate_candidates(candidates, problem)
            winner, any_passed = self._pick_winner(candidates)
            agreement = self._agreement_score(candidates)
            n_passing = sum(1 for c in candidates if c["validation"]["all_passed"])

            result["agreement_per_attempt"].append(agreement)
            result["candidates_passed_per_attempt"].append(n_passing)
            result["agent_log"].append({
                "agent": "PatchTournament", "attempt": attempt,
                "output": {
                    "k": self.k, "temperatures": self.temperatures,
                    "n_passing": n_passing, "agreement": agreement,
                    "winner_temperature": winner["temperature"],
                },
            })

            application_result = winner["application_result"]
            validation = winner["validation"]
            print(f"[TournamentCoordinator] Tournament result: {n_passing}/{self.k} candidates passed all tests, "
                  f"agreement={agreement}, winner temp={winner['temperature']}")

            if not application_result["success"]:
                prev_patch = winner["patch_info"].get("corrected_code", "")
                result["llm_calls"] = self.llm.total_calls - llm_before + result.get("llm_calls", 0)
                continue

            llm_this_attempt = self.llm.total_calls - llm_before
            print(f"[TournamentCoordinator] Attempt {attempt} summary: {validation['passed']}/{validation['total_tests']} passed, {llm_this_attempt} LLM calls")

            if validation["all_passed"]:
                result["success"] = True
                result["patch"] = application_result["applied_code"]
                result["time_seconds"] = round(time.time() - start_time, 2)
                result["llm_calls"] = self.llm.total_calls
                print(f"\n{'*'*70}\n[TournamentCoordinator] SUCCESS! Fixed '{program_name}' on attempt {attempt}!\n{'*'*70}")
                return result

            passed_now = validation.get("passed", 0)
            if passed_now > best_passed:
                best_passed = passed_now
                best_code = application_result["applied_code"]
                current_code = best_code
            else:
                current_code = best_code
            prev_patch = winner["patch_info"].get("corrected_code", "")

        result["time_seconds"] = round(time.time() - start_time, 2)
        result["llm_calls"] = self.llm.total_calls
        print(f"\n{'X'*70}\n[TournamentCoordinator] FAILED '{program_name}' after {MAX_RETRY_ATTEMPTS} attempts ({result['time_seconds']}s)\n{'X'*70}")
        return result

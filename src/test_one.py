"""
Quick test: Run the multi-agent system on just ONE bug (bitcount).
Use this to verify everything works before running the full experiment.

Usage:
    cd src
    python test_one.py
"""

from llm_factory import get_llm_client
from coordinator import RecoveryCoordinator
from problem import Problem

# The buggy bitcount program (uses ^ instead of &)
buggy_code = """def bitcount(n):
    count = 0
    while n:
        n ^= n - 1
        count += 1
    return count"""

# Test cases: [inputs, expected_output]
test_cases = [
    [[127], 7],
    [[128], 1],
    [[3005], 9],
    [[13], 3],
    [[14], 3],
]

problem = Problem(name="bitcount", buggy_code=buggy_code, test_style="tuple",
                   dataset="quixbugs", test_cases=test_cases)

print("=" * 60)
print("TEST RUN: bitcount (single bug)")
print("=" * 60)
print(f"\nBuggy code:\n{buggy_code}\n")
print(f"Test cases: {len(test_cases)}")
print(f"Expected: bitcount(127)=7, bitcount(128)=1\n")

# Initialize
llm = get_llm_client()
coordinator = RecoveryCoordinator(llm)

# Run recovery
result = coordinator.recover("bitcount", buggy_code, problem)

# Print result
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"  Success: {result['success']}")
print(f"  Attempts: {result['attempts']}")
print(f"  Time: {result['time_seconds']}s")
print(f"  Error type: {result['error_type']}")
print(f"  API calls: {llm.total_calls}")

if result['success']:
    print(f"\n  Corrected code:\n{result['patch']}")

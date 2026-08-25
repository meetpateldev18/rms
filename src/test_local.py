"""
Quick test to verify the local Ollama model is working.

Usage:
    cd src
    python test_local.py
"""

from config import OLLAMA_HOST, OLLAMA_MODEL

print("=" * 50)
print("LOCAL MODEL (OLLAMA) TEST")
print("=" * 50)
print(f"\n[OK] Host: {OLLAMA_HOST}")
print(f"[OK] Model: {OLLAMA_MODEL}")

print("\n[...] Importing local_client...")
try:
    from local_client import LocalClient
    print("[OK] local_client imported successfully")
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    print("  → Run: pip install requests")
    exit(1)

print(f"\n[...] Connecting to Ollama and sending test request...")
try:
    client = LocalClient()
    response = client.generate("Reply with exactly: HELLO_TEST_SUCCESS")
    print(f"[OK] Response received: {response.strip()}")
except Exception as e:
    print(f"[FAIL] Call failed: {e}")
    print("\n  Common fixes:")
    print("  → Ollama not running: brew services start ollama")
    print(f"  → Model not pulled: ollama pull {OLLAMA_MODEL}")
    exit(1)

print("\n" + "=" * 50)
print("ALL TESTS PASSED — Your local model is working!")
print("=" * 50)
print("\nNext step: python test_one.py")

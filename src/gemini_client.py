import time
from google import genai
from google.genai import types
from config import GEMINI_API_KEYS, GEMINI_MODEL, DELAY_BETWEEN_CALLS


REQUEST_TIMEOUT_MS = 90_000  # google-genai's HttpOptions.timeout has NO default —
# an unbounded request will hang forever on a stuck connection (observed directly:
# a single call sat with no response for 59+ minutes after a transient 503, blocking
# the whole experiment). This bounds every request to a real ceiling.


def _is_quota_exhausted(exc: Exception) -> bool:
    """Distinguish a daily-quota 429 (rotate key, backoff is pointless) from
    a transient error (retry the same key)."""
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg and "PerDay" in msg


def _make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


class GeminiClient:
    """Wrapper for Gemini API with rate limiting and multi-key rotation.

    On a daily-quota 429, rotates to the next key in GEMINI_API_KEYS instead
    of burning exponential backoff against a key that won't recover until
    the next reset. Keys are shared, independent quota pools (verified
    empirically per-key, since Google doesn't document this cleanly) — a
    key that's exhausted stays marked exhausted for the rest of the process.
    """

    def __init__(self):
        if not GEMINI_API_KEYS:
            raise RuntimeError("No Gemini API key(s) configured — set GEMINI_API_KEY or GEMINI_API_KEYS in .env")
        self.keys = GEMINI_API_KEYS
        self.key_index = 0
        self.exhausted = set()
        self.client = _make_client(self.keys[self.key_index])
        self.model = GEMINI_MODEL
        self.last_call_time = 0
        self.total_calls = 0
        print(f"[GeminiClient] Initialized with model: {self.model}")
        print(f"[GeminiClient] {len(self.keys)} API key(s) available for rotation")
        print(f"[GeminiClient] Rate limit delay: {DELAY_BETWEEN_CALLS}s between calls")

    def _rate_limit(self):
        """Enforce delay between API calls to respect free tier limits."""
        elapsed = time.time() - self.last_call_time
        if elapsed < DELAY_BETWEEN_CALLS:
            wait = round(DELAY_BETWEEN_CALLS - elapsed, 1)
            print(f"  [GeminiClient] Rate limiting... waiting {wait}s")
            time.sleep(DELAY_BETWEEN_CALLS - elapsed)

    def _rotate_key(self) -> bool:
        """Mark the current key exhausted and switch to the next unexhausted
        one. Returns False if every key is exhausted."""
        self.exhausted.add(self.key_index)
        for offset in range(1, len(self.keys) + 1):
            candidate = (self.key_index + offset) % len(self.keys)
            if candidate not in self.exhausted:
                self.key_index = candidate
                self.client = _make_client(self.keys[self.key_index])
                print(f"  [GeminiClient] Quota exhausted on key #{sorted(self.exhausted)} — "
                      f"switched to key #{self.key_index + 1}/{len(self.keys)}")
                return True
        print(f"  [GeminiClient] All {len(self.keys)} keys are quota-exhausted.")
        return False

    def generate(self, prompt: str, max_retries: int = 4, temperature: float = 0.2) -> str:
        """Send a prompt to Gemini and return the text response."""
        self._rate_limit()

        prompt_preview = prompt[:100].replace('\n', ' ')
        print(f"\n  [GeminiClient] === API Call #{self.total_calls + 1} (key #{self.key_index + 1}/{len(self.keys)}) ===")
        print(f"  [GeminiClient] Prompt preview: {prompt_preview}...")
        print(f"  [GeminiClient] Prompt length: {len(prompt)} chars")

        attempt = 0
        while attempt <= max_retries:
            try:
                print(f"  [GeminiClient] Sending request to {self.model} (attempt {attempt+1}/{max_retries+1})...", flush=True)
                start = time.time()

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=2048,
                    ),
                )

                elapsed = round(time.time() - start, 1)
                self.last_call_time = time.time()
                self.total_calls += 1

                if response.text:
                    resp_preview = response.text[:200].replace('\n', ' ')
                    print(f"  [GeminiClient] Response received in {elapsed}s ({len(response.text)} chars)")
                    print(f"  [GeminiClient] Response preview: {resp_preview}...")
                    return response.text.strip()

                print(f"  [GeminiClient] WARNING: Empty response received in {elapsed}s")
                return ""

            except Exception as e:
                elapsed = round(time.time() - start, 1)

                if _is_quota_exhausted(e):
                    if self._rotate_key():
                        continue  # retry immediately on the new key, doesn't count against max_retries
                    else:
                        print(f"  [GeminiClient] FAILED — no keys with remaining quota ({elapsed}s): {e}")
                        return f"ERROR: All API keys quota-exhausted: {e}"

                if attempt < max_retries:
                    # Exponential backoff: 10s, 20s, 40s, 80s
                    wait = 10 * (2 ** attempt)
                    print(f"  [GeminiClient] ERROR after {elapsed}s: {e}")
                    print(f"  [GeminiClient] Retrying in {wait}s (attempt {attempt+2}/{max_retries+1})...")
                    time.sleep(wait)
                    attempt += 1
                else:
                    print(f"  [GeminiClient] FAILED after {max_retries + 1} attempts ({elapsed}s): {e}")
                    return f"ERROR: {e}"

        return "ERROR: Unknown failure"

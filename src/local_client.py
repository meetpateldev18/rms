import time
import requests
from config import OLLAMA_HOST, OLLAMA_MODEL


class LocalClient:
    """Wrapper for a local Ollama model. Same interface as GeminiClient."""

    def __init__(self):
        self.host = OLLAMA_HOST
        self.model = OLLAMA_MODEL
        self.total_calls = 0
        self._ensure_model_available()
        print(f"[LocalClient] Initialized with model: {self.model} @ {self.host}")

    def _ensure_model_available(self):
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", [])]
            if not any(n == self.model or n.startswith(self.model.split(":")[0]) for n in names):
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. Available: {names}. "
                    f"Run: ollama pull {self.model}"
                )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it with: brew services start ollama"
            )

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Send a prompt to the local model and return the text response."""
        prompt_preview = prompt[:100].replace('\n', ' ')
        print(f"\n  [LocalClient] === Call #{self.total_calls + 1} ===")
        print(f"  [LocalClient] Prompt preview: {prompt_preview}...")
        print(f"  [LocalClient] Prompt length: {len(prompt)} chars")

        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                print(f"  [LocalClient] Generating with {self.model} (attempt {attempt+1}/{max_retries+1})...", flush=True)

                resp = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 2048,
                        },
                    },
                    timeout=(10, 90),  # (connect, read) — fail well before an outer per-problem watchdog would
                )
                resp.raise_for_status()
                text = resp.json().get("response", "")

                elapsed = round(time.time() - start, 1)
                self.total_calls += 1

                if text:
                    resp_preview = text[:200].replace('\n', ' ')
                    print(f"  [LocalClient] Response received in {elapsed}s ({len(text)} chars)")
                    print(f"  [LocalClient] Response preview: {resp_preview}...")
                    return text.strip()

                print(f"  [LocalClient] WARNING: Empty response received in {elapsed}s")
                return ""

            except Exception as e:
                elapsed = round(time.time() - start, 1)
                if attempt < max_retries:
                    wait = 3 * (attempt + 1)
                    print(f"  [LocalClient] ERROR after {elapsed}s: {e}")
                    print(f"  [LocalClient] Retrying in {wait}s (attempt {attempt+2}/{max_retries+1})...")
                    time.sleep(wait)
                else:
                    print(f"  [LocalClient] FAILED after {max_retries + 1} attempts ({elapsed}s): {e}")
                    return f"ERROR: {e}"

        return "ERROR: Unknown failure"

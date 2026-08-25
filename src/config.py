import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ============================================================
# Set GEMINI_API_KEY (single) or GEMINI_API_KEYS (comma-separated, for
# rotation across quota-exhausted keys) in .env in the project root
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

_keys_raw = os.environ.get("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in _keys_raw.split(",") if k.strip()] or (
    [GEMINI_API_KEY] if GEMINI_API_KEY else []
)

# Model: Gemini 3.1 Flash Lite — fastest, most generous free tier
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# Rate limiting
REQUESTS_PER_MINUTE = 10
DELAY_BETWEEN_CALLS = 4  # seconds (10 RPM = 6s min, 4s with burst tolerance)

# ============================================================
# LLM backend selection: "local" (Ollama, on-device) or "gemini" (API)
# ============================================================
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct")

# Recovery settings
MAX_RETRY_ATTEMPTS = 3

# Paths
ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "quixbugs")
HUMANEVALFIX_DIR = os.path.join(ROOT_DIR, "dataset", "humanevalfix")
DEBUGBENCH_DIR = os.path.join(ROOT_DIR, "dataset", "debugbench")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

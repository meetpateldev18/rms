from config import LLM_BACKEND


def get_llm_client():
    """Return the configured LLM client (local Ollama or Gemini API)."""
    if LLM_BACKEND == "gemini":
        from gemini_client import GeminiClient
        return GeminiClient()
    elif LLM_BACKEND == "local":
        from local_client import LocalClient
        return LocalClient()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r} (expected 'local' or 'gemini')")

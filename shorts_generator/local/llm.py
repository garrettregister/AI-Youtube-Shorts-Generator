"""Local LLM backend — OpenAI, Gemini, or OpenAI-compatible (Ollama / llama.cpp)."""
from ..config import (
    GEMINI_MODEL,
    LLM_PROVIDER,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_MODEL,
    OPENAI_MODEL,
    require_gemini_key,
    require_openai_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 8192,
        },
    )
    return response.text or ""


def call_openai_compatible_llm(prompt: str) -> str:
    """Any OpenAI-compatible /v1 server — Ollama, llama-server, LM Studio, vLLM.

    Selected when LLM_PROVIDER=openai-compatible (alias: ollama). Needs no API key.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for LLM_PROVIDER=openai-compatible. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key=LOCAL_LLM_API_KEY or "ollama")
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=LOCAL_LLM_MODEL,
            temperature=0.2,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Older Ollama / llama-server builds may not support response_format.
        response = client.chat.completions.create(
            model=LOCAL_LLM_MODEL,
            temperature=0.2,
            messages=messages,
        )
    return response.choices[0].message.content or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    if provider in ("openai-compatible", "ollama"):
        return call_openai_compatible_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. "
        "Use 'openai', 'gemini', or 'openai-compatible' (alias 'ollama')."
    )

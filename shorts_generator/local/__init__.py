"""Local-mode backends — no MuAPI calls, runs on your machine.

Used when the pipeline is invoked with mode="local". Requires the optional
deps in requirements-local.txt (yt-dlp, faster-whisper, openai, google-genai,
opencv, moviepy). For the LLM step, either an API key (OpenAI / Gemini) or a
local OpenAI-compatible server via LLM_PROVIDER=openai-compatible (Ollama,
llama-server, LM Studio, vLLM) — the latter needs no API key.
"""

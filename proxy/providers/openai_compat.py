import logging

from fastapi import Request

from proxy.config import OPENAI_API_KEY, OPENAI_BASE_URL
from proxy.providers.openai import OpenAIProvider

logger = logging.getLogger("proxy")


class OpenAICompatProvider(OpenAIProvider):
    """OpenAI-compatible API provider (Ollama, LM Studio, vLLM, etc.).

    Inherits all behavior from OpenAIProvider — same content flattening
    and request setup, just uses the openai-compat config keys (which are
    already resolved to OPENAI_API_KEY / OPENAI_BASE_URL by config.py).
    """
    pass

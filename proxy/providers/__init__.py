import logging

from proxy.config import USE_GEMINI_OAUTH, USE_VERTEX_AUTH

from proxy.providers.gemini_format import (
    clean_gemini_schema,
    _openai_messages_to_gemini,
    _openai_tools_to_gemini,
    _gemini_response_to_litellm_format,
)
from proxy.providers.code_assist import (
    _code_assist_stream,
    _code_assist_stream_as_openai,
    _code_assist_generate,
)
from proxy.providers.base import BaseProvider

logger = logging.getLogger("proxy")


def get_provider(model: str) -> BaseProvider:
    """Return the appropriate provider instance based on the model prefix."""
    if model.startswith("kimi/"):
        from proxy.providers.kimi import KimiProvider
        return KimiProvider()
    elif model.startswith("openai/"):
        from proxy.providers.openai import OpenAIProvider
        return OpenAIProvider()
    elif model.startswith("gemini/"):
        if USE_GEMINI_OAUTH:
            from proxy.providers.gemini_oauth import GeminiOAuthProvider
            return GeminiOAuthProvider()
        elif USE_VERTEX_AUTH:
            from proxy.providers.vertex import VertexProvider
            return VertexProvider()
        else:
            from proxy.providers.gemini_api import GeminiAPIProvider
            return GeminiAPIProvider()
    else:
        # Default: Anthropic passthrough
        from proxy.providers.anthropic import AnthropicProvider
        return AnthropicProvider()

import logging

from fastapi import Request

from proxy.config import GEMINI_API_KEY
from proxy.providers.base import BaseProvider

logger = logging.getLogger("proxy")


class GeminiAPIProvider(BaseProvider):
    """Gemini API key provider (Google AI Studio)."""

    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        litellm_request["api_key"] = GEMINI_API_KEY
        logger.debug(f"Using Gemini API key for model: {litellm_request.get('model')}")
        return litellm_request

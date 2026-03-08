import logging

from fastapi import Request

from proxy.config import ANTHROPIC_API_KEY
from proxy.providers.base import BaseProvider

logger = logging.getLogger("proxy")


class AnthropicProvider(BaseProvider):
    """Anthropic transparent proxy — model names passed through unchanged."""

    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        litellm_request["api_key"] = ANTHROPIC_API_KEY
        logger.debug(f"Using Anthropic API key for model: {litellm_request.get('model')}")
        return litellm_request

import logging

from fastapi import Request

from proxy.config import KIMI_API_KEY, KIMI_BASE_URL
from proxy.providers.base import BaseProvider

logger = logging.getLogger("proxy")


class KimiProvider(BaseProvider):
    """Kimi Code provider."""

    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        raw_kimi_model = litellm_request["model"][5:]  # strip "kimi/"
        litellm_request["model"] = f"openai/{raw_kimi_model}"
        litellm_request["api_key"] = KIMI_API_KEY
        litellm_request["base_url"] = KIMI_BASE_URL
        # Pass through the original User-Agent so Kimi recognises this as a coding agent
        user_agent = raw_request.headers.get("user-agent", "claude-code/1.0.0")
        litellm_request["extra_headers"] = {"User-Agent": user_agent}
        logger.debug(f"Using Kimi API key and base URL {KIMI_BASE_URL} for model: {raw_kimi_model}")
        return litellm_request

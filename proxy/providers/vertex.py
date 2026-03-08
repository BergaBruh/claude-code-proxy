import logging

from fastapi import Request

from proxy.config import VERTEX_PROJECT, VERTEX_LOCATION
from proxy.providers.base import BaseProvider

logger = logging.getLogger("proxy")


class VertexProvider(BaseProvider):
    """Google Vertex AI provider (Application Default Credentials)."""

    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        litellm_request["vertex_project"] = VERTEX_PROJECT
        litellm_request["vertex_location"] = VERTEX_LOCATION
        litellm_request["custom_llm_provider"] = "vertex_ai"
        logger.debug(
            f"Using Gemini ADC with project={VERTEX_PROJECT}, "
            f"location={VERTEX_LOCATION} and model: {litellm_request.get('model')}"
        )
        return litellm_request

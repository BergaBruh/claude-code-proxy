import logging
from typing import Any, AsyncGenerator, Dict

from fastapi import Request, HTTPException

from proxy.auth.oauth import get_gemini_oauth_access_token
from proxy.auth.code_assist import _get_code_assist_project
from proxy.providers.base import BaseProvider
from proxy.providers.gemini_format import _openai_messages_to_gemini, _openai_tools_to_gemini, _gemini_response_to_litellm_format
from proxy.providers.code_assist import _code_assist_stream_as_openai, _code_assist_generate

logger = logging.getLogger("proxy")


class GeminiOAuthProvider(BaseProvider):
    """Gemini OAuth / Code Assist provider (bypasses litellm)."""

    def __init__(self):
        self._oauth_token = None
        self._raw_model = None

    @property
    def uses_litellm(self) -> bool:
        return False

    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        self._oauth_token = get_gemini_oauth_access_token()
        if not self._oauth_token:
            raise HTTPException(
                status_code=401,
                detail="Gemini OAuth: could not obtain access token. Check ~/.gemini/oauth_creds.json"
            )
        self._raw_model = litellm_request["model"].replace("gemini/", "", 1)
        logger.debug(f"Using Gemini OAuth (Code Assist endpoint) for model: {self._raw_model}")
        return litellm_request

    def _build_gemini_request(self, litellm_request: dict) -> tuple:
        """Build Gemini-native request body. Returns (project, model, gemini_req)."""
        project = _get_code_assist_project(self._oauth_token)
        gemini_req = _openai_messages_to_gemini(litellm_request["messages"])
        gemini_tools = _openai_tools_to_gemini(litellm_request.get("tools"))
        if gemini_tools:
            gemini_req["tools"] = gemini_tools
        gen_config: Dict[str, Any] = {}
        if litellm_request.get("max_tokens"):
            gen_config["maxOutputTokens"] = litellm_request["max_tokens"]
        if litellm_request.get("temperature") is not None:
            gen_config["temperature"] = litellm_request["temperature"]
        if litellm_request.get("top_p") is not None:
            gen_config["topP"] = litellm_request["top_p"]
        if gen_config:
            gemini_req["generationConfig"] = gen_config
        return project, self._raw_model, gemini_req

    async def generate(self, litellm_request: dict) -> Any:
        project, model, gemini_req = self._build_gemini_request(litellm_request)
        raw_resp = await _code_assist_generate(self._oauth_token, project, model, gemini_req)
        return _gemini_response_to_litellm_format(raw_resp, model)

    async def stream(self, litellm_request: dict) -> AsyncGenerator:
        project, model, gemini_req = self._build_gemini_request(litellm_request)
        return _code_assist_stream_as_openai(self._oauth_token, project, model, gemini_req)

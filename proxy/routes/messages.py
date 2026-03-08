import json
import logging
import time

import litellm
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

from proxy.models import MessagesRequest
from proxy.model_mapping import map_model_for_provider
from proxy.converters import convert_anthropic_to_litellm, convert_litellm_to_anthropic
from proxy.streaming import handle_streaming
from proxy.providers import get_provider
from proxy.logging_config import log_request_beautifully, log_error_beautifully

logger = logging.getLogger("proxy")

KNOWN_PROVIDERS = {"openai", "openai-compat", "google", "google-api", "google-oauth", "google-vertex", "anthropic", "kimi"}


def _extract_provider_override(raw_request: Request):
    """Return a provider name if the Authorization bearer token is a known provider, else None."""
    auth = raw_request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip().lower()
        if token in KNOWN_PROVIDERS:
            return token
    return None


def _sanitize_for_json(obj, _depth=0):
    """Safely serialize objects for JSON error responses."""
    if _depth > 8:
        return str(obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item, _depth + 1) for item in obj]
    if hasattr(obj, '__dict__'):
        return _sanitize_for_json(obj.__dict__, _depth + 1)
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def register_messages_routes(app):
    """Register the /v1/messages endpoint."""

    @app.post("/v1/messages")
    async def create_message(
        request: MessagesRequest,
        raw_request: Request
    ):
        try:
            body = await raw_request.body()
            body_json = json.loads(body.decode('utf-8'))
            original_model = body_json.get("model", "unknown")

            # Per-request provider override via Authorization header
            provider_override = _extract_provider_override(raw_request)
            if provider_override:
                request.model = map_model_for_provider(
                    request.original_model or original_model, provider_override
                )
                logger.debug(f"🔀 Provider override '{provider_override}': remapped model to '{request.model}'")

            display_model = original_model
            if "/" in display_model:
                display_model = display_model.split("/")[-1]

            logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

            # Convert Anthropic request to LiteLLM format
            litellm_request = convert_anthropic_to_litellm(request)

            # Get the provider and configure the request
            provider = get_provider(request.model)
            litellm_request = provider.configure_request(litellm_request, raw_request)

            logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200
            )

            # Handle streaming mode
            if request.stream:
                if provider.uses_litellm:
                    response_generator = await litellm.acompletion(**litellm_request)
                else:
                    response_generator = await provider.stream(litellm_request)

                return StreamingResponse(
                    handle_streaming(response_generator, request),
                    media_type="text/event-stream"
                )
            else:
                start_time = time.time()
                if provider.uses_litellm:
                    litellm_response = litellm.completion(**litellm_request)
                else:
                    litellm_response = await provider.generate(litellm_request)
                logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")

                return convert_litellm_to_anthropic(litellm_response, request)

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()

            error_details = {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": error_traceback
            }

            for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
                if hasattr(e, attr):
                    error_details[attr] = getattr(e, attr)

            if hasattr(e, '__dict__'):
                for key, value in e.__dict__.items():
                    if key not in error_details and key not in ['args', '__traceback__']:
                        error_details[key] = str(value)

            log_error_beautifully(e)

            sanitized_details = _sanitize_for_json(error_details)
            logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")

            error_message = f"Error: {str(e)}"
            if 'message' in error_details and error_details['message']:
                error_message += f"\nMessage: {error_details['message']}"
            if 'response' in error_details and error_details['response']:
                error_message += f"\nResponse: {error_details['response']}"

            status_code = error_details.get('status_code', 500)
            raise HTTPException(status_code=status_code, detail=error_message)

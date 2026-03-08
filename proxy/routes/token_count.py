import logging

from fastapi import Request, HTTPException

from proxy.models import MessagesRequest, TokenCountRequest, TokenCountResponse
from proxy.model_mapping import map_model_for_provider
from proxy.converters import convert_anthropic_to_litellm
from proxy.routes.messages import _extract_provider_override
from proxy.logging_config import log_request_beautifully

logger = logging.getLogger("proxy")


def register_token_count_routes(app):
    """Register the /v1/messages/count_tokens endpoint."""

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(
        request: TokenCountRequest,
        raw_request: Request
    ):
        try:
            original_model = request.original_model or request.model

            # Per-request provider override
            provider_override = _extract_provider_override(raw_request)
            if provider_override:
                request.model = map_model_for_provider(
                    request.original_model or original_model, provider_override
                )
                logger.debug(f"🔀 Provider override '{provider_override}': remapped model to '{request.model}'")

            display_model = original_model
            if "/" in display_model:
                display_model = display_model.split("/")[-1]

            # Convert the messages to a format LiteLLM can understand
            converted_request = convert_anthropic_to_litellm(
                MessagesRequest(
                    model=request.model,
                    max_tokens=100,
                    messages=request.messages,
                    system=request.system,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    thinking=request.thinking
                )
            )

            try:
                from litellm import token_counter

                num_tools = len(request.tools) if request.tools else 0
                log_request_beautifully(
                    "POST",
                    raw_request.url.path,
                    display_model,
                    converted_request.get('model'),
                    len(converted_request['messages']),
                    num_tools,
                    200
                )

                token_counter_args = {
                    "model": converted_request["model"],
                    "messages": converted_request["messages"],
                }

                token_count = token_counter(**token_counter_args)
                return TokenCountResponse(input_tokens=token_count)

            except ImportError:
                logger.error("Could not import token_counter from litellm")
                return TokenCountResponse(input_tokens=1000)

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
            raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

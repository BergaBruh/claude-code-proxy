import json
import logging
import time
from typing import Dict, Any

import litellm
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

from proxy.config import (
    OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY,
    OPENAI_BASE_URL, USE_GEMINI_OAUTH, USE_VERTEX_AUTH,
    VERTEX_PROJECT, VERTEX_LOCATION,
)
from proxy.models import MessagesRequest, TokenCountRequest, TokenCountResponse, map_model_for_provider
from proxy.converters import convert_anthropic_to_litellm, convert_litellm_to_anthropic
from proxy.streaming import handle_streaming
from proxy.auth.oauth import get_gemini_oauth_access_token
from proxy.auth.code_assist import _get_code_assist_project
from proxy.providers.gemini_format import (
    _openai_messages_to_gemini,
    _openai_tools_to_gemini,
    _gemini_response_to_litellm_format,
)
from proxy.providers.code_assist import _code_assist_stream_as_openai, _code_assist_generate
from proxy.logging_config import log_request_beautifully, log_error_beautifully

logger = logging.getLogger("proxy")

KNOWN_PROVIDERS = {"openai", "openai-compat", "google", "google-api", "google-oauth", "google-vertex", "anthropic"}


def _extract_provider_override(raw_request: Request):
    """Return a provider name if the Authorization bearer token is a known provider, else None."""
    auth = raw_request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip().lower()
        if token in KNOWN_PROVIDERS:
            return token
    return None


def register_routes(app):
    """Register all route handlers on the FastAPI app."""

    @app.post("/v1/messages")
    async def create_message(
        request: MessagesRequest,
        raw_request: Request
    ):
        try:
            # print the body here
            body = await raw_request.body()

            # Parse the raw body as JSON since it's bytes
            body_json = json.loads(body.decode('utf-8'))
            original_model = body_json.get("model", "unknown")

            # Per-request provider override via Authorization header
            provider_override = _extract_provider_override(raw_request)
            if provider_override:
                request.model = map_model_for_provider(
                    request.original_model or original_model, provider_override
                )
                logger.debug(f"🔀 Provider override '{provider_override}': remapped model to '{request.model}'")

            # Get the display name for logging, just the model name without provider prefix
            display_model = original_model
            if "/" in display_model:
                display_model = display_model.split("/")[-1]

            # Clean model name for capability check
            clean_model = request.model
            if clean_model.startswith("anthropic/"):
                clean_model = clean_model[len("anthropic/"):]
            elif clean_model.startswith("openai/"):
                clean_model = clean_model[len("openai/"):]

            logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

            # Convert Anthropic request to LiteLLM format
            litellm_request = convert_anthropic_to_litellm(request)

            # Determine which API key to use based on the model
            if request.model.startswith("openai/"):
                litellm_request["api_key"] = OPENAI_API_KEY
                # Use custom OpenAI base URL if configured
                if OPENAI_BASE_URL:
                    litellm_request["base_url"] = OPENAI_BASE_URL
                    logger.debug(f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL} for model: {request.model}")
                else:
                    logger.debug(f"Using OpenAI API key for model: {request.model}")
            elif request.model.startswith("gemini/"):
                if USE_GEMINI_OAUTH:
                    oauth_token = get_gemini_oauth_access_token()
                    if not oauth_token:
                        raise HTTPException(status_code=401, detail="Gemini OAuth: could not obtain access token. Check ~/.gemini/oauth_creds.json")
                    # Bypass litellm — use Code Assist API (cloudcode-pa.googleapis.com)
                    raw_model = request.model.replace("gemini/", "", 1)
                    litellm_request["_gemini_oauth_token"] = oauth_token
                    litellm_request["_gemini_oauth_model"] = raw_model
                    logger.debug(f"Using Gemini OAuth (Code Assist endpoint) for model: {raw_model}")
                elif USE_VERTEX_AUTH:
                    litellm_request["vertex_project"] = VERTEX_PROJECT
                    litellm_request["vertex_location"] = VERTEX_LOCATION
                    litellm_request["custom_llm_provider"] = "vertex_ai"
                    logger.debug(f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION} and model: {request.model}")
                else:
                    litellm_request["api_key"] = GEMINI_API_KEY
                    logger.debug(f"Using Gemini API key for model: {request.model}")
            else:
                litellm_request["api_key"] = ANTHROPIC_API_KEY
                logger.debug(f"Using Anthropic API key for model: {request.model}")

            # For OpenAI models - modify request format to work with limitations
            if "openai" in litellm_request["model"] and "messages" in litellm_request:
                logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

                # For OpenAI models, we need to convert content blocks to simple strings
                for i, msg in enumerate(litellm_request["messages"]):
                    # Special case - handle message content directly when it's a list of tool_result
                    if "content" in msg and isinstance(msg["content"], list):
                        is_only_tool_result = True
                        for block in msg["content"]:
                            if not isinstance(block, dict) or block.get("type") != "tool_result":
                                is_only_tool_result = False
                                break

                        if is_only_tool_result and len(msg["content"]) > 0:
                            logger.warning(f"Found message with only tool_result content - special handling required")
                            # Extract the content from all tool_result blocks
                            all_text = ""
                            for block in msg["content"]:
                                all_text += "Tool Result:\n"
                                result_content = block.get("content", [])

                                # Handle different formats of content
                                if isinstance(result_content, list):
                                    for item in result_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            all_text += item.get("text", "") + "\n"
                                        elif isinstance(item, dict):
                                            try:
                                                item_text = item.get("text", json.dumps(item))
                                                all_text += item_text + "\n"
                                            except:
                                                all_text += str(item) + "\n"
                                elif isinstance(result_content, str):
                                    all_text += result_content + "\n"
                                else:
                                    try:
                                        all_text += json.dumps(result_content) + "\n"
                                    except:
                                        all_text += str(result_content) + "\n"

                            # Replace the list with extracted text
                            litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                            logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                            continue  # Skip normal processing for this message

                    # 1. Handle content field - normal case
                    if "content" in msg:
                        # Check if content is a list (content blocks)
                        if isinstance(msg["content"], list):
                            # Convert complex content blocks to simple string
                            text_content = ""
                            for block in msg["content"]:
                                if isinstance(block, dict):
                                    # Handle different content block types
                                    if block.get("type") == "text":
                                        text_content += block.get("text", "") + "\n"

                                    # Handle tool_result content blocks - extract nested text
                                    elif block.get("type") == "tool_result":
                                        tool_id = block.get("tool_use_id", "unknown")
                                        text_content += f"[Tool Result ID: {tool_id}]\n"

                                        # Extract text from the tool_result content
                                        result_content = block.get("content", [])
                                        if isinstance(result_content, list):
                                            for item in result_content:
                                                if isinstance(item, dict) and item.get("type") == "text":
                                                    text_content += item.get("text", "") + "\n"
                                                elif isinstance(item, dict):
                                                    if "text" in item:
                                                        text_content += item.get("text", "") + "\n"
                                                    else:
                                                        try:
                                                            text_content += json.dumps(item) + "\n"
                                                        except:
                                                            text_content += str(item) + "\n"
                                        elif isinstance(result_content, dict):
                                            if result_content.get("type") == "text":
                                                text_content += result_content.get("text", "") + "\n"
                                            else:
                                                try:
                                                    text_content += json.dumps(result_content) + "\n"
                                                except:
                                                    text_content += str(result_content) + "\n"
                                        elif isinstance(result_content, str):
                                            text_content += result_content + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"

                                    # Handle tool_use content blocks
                                    elif block.get("type") == "tool_use":
                                        tool_name = block.get("name", "unknown")
                                        tool_id = block.get("id", "unknown")
                                        tool_input = json.dumps(block.get("input", {}))
                                        text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"

                                    # Handle image content blocks
                                    elif block.get("type") == "image":
                                        text_content += "[Image content - not displayed in text format]\n"

                            # Make sure content is never empty for OpenAI models
                            if not text_content.strip():
                                text_content = "..."

                            litellm_request["messages"][i]["content"] = text_content.strip()
                        # Also check for None or empty string content
                        elif msg["content"] is None:
                            litellm_request["messages"][i]["content"] = "..."  # Empty content not allowed

                    # 2. Remove any fields OpenAI doesn't support in messages
                    for key in list(msg.keys()):
                        if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                            logger.warning(f"Removing unsupported field from message: {key}")
                            del msg[key]

                # 3. Final validation - check for any remaining invalid values
                for i, msg in enumerate(litellm_request["messages"]):
                    logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")

                    if isinstance(msg.get("content"), list):
                        logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                        litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                    elif msg.get("content") is None:
                        logger.warning(f"Message {i} has None content - replacing with placeholder")
                        litellm_request["messages"][i]["content"] = "..."

            # Only log basic info about the request
            logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

            # Extract Gemini OAuth metadata (not valid litellm params)
            gemini_oauth_token = litellm_request.pop("_gemini_oauth_token", None)
            gemini_oauth_model = litellm_request.pop("_gemini_oauth_model", None)

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
                if gemini_oauth_token:
                    project = _get_code_assist_project(gemini_oauth_token)
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

                    response_generator = _code_assist_stream_as_openai(
                        gemini_oauth_token, project, gemini_oauth_model, gemini_req
                    )
                else:
                    response_generator = await litellm.acompletion(**litellm_request)

                return StreamingResponse(
                    handle_streaming(response_generator, request),
                    media_type="text/event-stream"
                )
            else:
                start_time = time.time()
                if gemini_oauth_token:
                    project = _get_code_assist_project(gemini_oauth_token)
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

                    raw_resp = await _code_assist_generate(
                        gemini_oauth_token, project, gemini_oauth_model, gemini_req
                    )
                    # Convert to a dict that convert_litellm_to_anthropic can handle
                    litellm_response = _gemini_response_to_litellm_format(raw_resp, gemini_oauth_model)
                else:
                    litellm_response = litellm.completion(**litellm_request)
                logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")

                # Convert response to Anthropic format
                anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

                return anthropic_response

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()

            # Capture as much info as possible about the error
            error_details = {
                "error": str(e),
                "type": type(e).__name__,
                "traceback": error_traceback
            }

            # Check for LiteLLM-specific attributes
            for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
                if hasattr(e, attr):
                    error_details[attr] = getattr(e, attr)

            # Check for additional exception details in dictionaries
            if hasattr(e, '__dict__'):
                for key, value in e.__dict__.items():
                    if key not in error_details and key not in ['args', '__traceback__']:
                        error_details[key] = str(value)

            # Helper function to safely serialize objects for JSON
            def sanitize_for_json(obj, _depth=0):
                if _depth > 8:
                    return str(obj)
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                if isinstance(obj, dict):
                    return {k: sanitize_for_json(v, _depth + 1) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [sanitize_for_json(item, _depth + 1) for item in obj]
                if hasattr(obj, '__dict__'):
                    return sanitize_for_json(obj.__dict__, _depth + 1)
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

            # Pretty-print to proxy console
            log_error_beautifully(e)

            # Log all error details with safe serialization
            sanitized_details = sanitize_for_json(error_details)
            logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")

            # Format error for response
            error_message = f"Error: {str(e)}"
            if 'message' in error_details and error_details['message']:
                error_message += f"\nMessage: {error_details['message']}"
            if 'response' in error_details and error_details['response']:
                error_message += f"\nResponse: {error_details['response']}"

            # Return detailed error
            status_code = error_details.get('status_code', 500)
            raise HTTPException(status_code=status_code, detail=error_message)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(
        request: TokenCountRequest,
        raw_request: Request
    ):
        try:
            # Log the incoming token count request
            original_model = request.original_model or request.model

            # Per-request provider override via Authorization header
            provider_override = _extract_provider_override(raw_request)
            if provider_override:
                request.model = map_model_for_provider(
                    request.original_model or original_model, provider_override
                )
                logger.debug(f"🔀 Provider override '{provider_override}': remapped model to '{request.model}'")

            # Get the display name for logging, just the model name without provider prefix
            display_model = original_model
            if "/" in display_model:
                display_model = display_model.split("/")[-1]

            # Clean model name for capability check
            clean_model = request.model
            if clean_model.startswith("anthropic/"):
                clean_model = clean_model[len("anthropic/"):]
            elif clean_model.startswith("openai/"):
                clean_model = clean_model[len("openai/"):]

            # Convert the messages to a format LiteLLM can understand
            converted_request = convert_anthropic_to_litellm(
                MessagesRequest(
                    model=request.model,
                    max_tokens=100,  # Arbitrary value not used for token counting
                    messages=request.messages,
                    system=request.system,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    thinking=request.thinking
                )
            )

            # Use LiteLLM's token_counter function
            try:
                from litellm import token_counter

                # Log the request beautifully
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

                # Prepare token counter arguments
                token_counter_args = {
                    "model": converted_request["model"],
                    "messages": converted_request["messages"],
                }

                # Add custom base URL for OpenAI models if configured
                if request.model.startswith("openai/") and OPENAI_BASE_URL:
                    token_counter_args["base_url"] = OPENAI_BASE_URL

                # Count tokens
                token_count = token_counter(**token_counter_args)

                # Return Anthropic-style response
                return TokenCountResponse(input_tokens=token_count)

            except ImportError:
                logger.error("Could not import token_counter from litellm")
                # Fallback to a simple approximation
                return TokenCountResponse(input_tokens=1000)

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
            raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

    @app.get("/")
    async def root():
        return {"message": "Anthropic Proxy for LiteLLM"}

import json
import logging

from fastapi import Request

from proxy.config import OPENAI_API_KEY, OPENAI_BASE_URL
from proxy.providers.base import BaseProvider

logger = logging.getLogger("proxy")


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def configure_request(self, litellm_request: dict, raw_request: Request) -> dict:
        litellm_request["api_key"] = OPENAI_API_KEY
        if OPENAI_BASE_URL:
            litellm_request["base_url"] = OPENAI_BASE_URL
            logger.debug(f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL}")
        else:
            logger.debug(f"Using OpenAI API key for model: {litellm_request.get('model')}")

        _flatten_content_blocks(litellm_request)
        return litellm_request


def _flatten_content_blocks(litellm_request: dict):
    """Convert Anthropic content blocks to plain strings for OpenAI-compatible models."""
    if "messages" not in litellm_request:
        return

    if "openai" not in litellm_request.get("model", ""):
        return

    logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

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
                all_text = ""
                for block in msg["content"]:
                    all_text += "Tool Result:\n"
                    result_content = block.get("content", [])

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

                litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                continue

        # Normal case: handle content field
        if "content" in msg:
            if isinstance(msg["content"], list):
                text_content = ""
                for block in msg["content"]:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_content += block.get("text", "") + "\n"
                        elif block.get("type") == "tool_result":
                            tool_id = block.get("tool_use_id", "unknown")
                            text_content += f"[Tool Result ID: {tool_id}]\n"

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
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            tool_id = block.get("id", "unknown")
                            tool_input = json.dumps(block.get("input", {}))
                            text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                        elif block.get("type") == "image":
                            text_content += "[Image content - not displayed in text format]\n"

                if not text_content.strip():
                    text_content = "..."

                litellm_request["messages"][i]["content"] = text_content.strip()
            elif msg["content"] is None:
                litellm_request["messages"][i]["content"] = "..."

        # Remove unsupported fields from messages
        for key in list(msg.keys()):
            if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                logger.warning(f"Removing unsupported field from message: {key}")
                del msg[key]

    # Final validation
    for i, msg in enumerate(litellm_request["messages"]):
        logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")

        if isinstance(msg.get("content"), list):
            logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
            litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
        elif msg.get("content") is None:
            logger.warning(f"Message {i} has None content - replacing with placeholder")
            litellm_request["messages"][i]["content"] = "..."
